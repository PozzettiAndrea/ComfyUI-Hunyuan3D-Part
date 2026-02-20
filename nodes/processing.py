"""
Processing Nodes for Hunyuan3D-Part.

X-Part generation node that loads models from config dicts
passed by loader nodes.
"""

import torch
import numpy as np
import trimesh
import folder_paths
import os
import tempfile
import time
import concurrent.futures

# Import utilities from core
from .mesh_utils import load_mesh, save_mesh, colorize_segmentation, get_temp_mesh_path
from .schedulers import FlowMatchEulerDiscreteScheduler

# Worker-process model cache for X-Part models
_xpart_model_cache = {}


def _fix_meta_buffers(model, device):
    """Reinitialize any buffers left on meta device after assign=True loading."""
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)


def _xpart_arch_config():
    """Architecture config for X-Part models (matches pretrained weights)."""
    num_latents = 1024
    z_scale_factor = 1.0039506158752403
    num_tokens_cond = 2048

    _shared_vae_params = dict(
        embed_dim=64, num_freqs=8, include_pi=False, heads=16,
        width=1024, num_encoder_layers=8, num_decoder_layers=16,
        qkv_bias=False, qk_norm=True, scale_factor=z_scale_factor,
        geo_decoder_mlp_expand_ratio=4, geo_decoder_downsample_ratio=1,
        geo_decoder_ln_post=True, point_feats=4,
    )

    return {
        "shapevae": {"params": {
            **_shared_vae_params,
            "num_latents": num_latents,
            "pc_size": 40960,
            "pc_sharpedge_size": 0,
        }},
        "conditioner": {"params": {
            "use_geo": True, "use_obj": True, "use_seg_feat": True,
            "geo_cfg": {
                "output_dim": 1024,
                "params": {
                    "use_local": True,
                    "local_feat_type": "latents_shape",
                    "num_tokens_cond": num_tokens_cond,
                    "local_geo_cfg": {"params": {
                        **_shared_vae_params,
                        "num_latents": num_tokens_cond,
                        "pc_size": 40960,
                        "pc_sharpedge_size": 0,
                    }},
                },
            },
            "obj_encoder_cfg": {
                "output_dim": 1024,
                "params": {
                    **_shared_vae_params,
                    "num_latents": 4096,
                    "pc_size": 40960,
                    "pc_sharpedge_size": 0,
                },
            },
            "seg_feat_cfg": {"params": {}},
        }},
        "model": {"params": {
            "use_self_attention": True, "use_cross_attention": True,
            "use_cross_attention_2": True, "use_bbox_cond": False,
            "num_freqs": 8, "use_part_embed": True, "valid_num": 50,
            "input_size": num_latents, "in_channels": 64,
            "hidden_size": 2048,
            "encoder_hidden_dim": 1024,
            "encoder_hidden2_dim": 1024,
            "depth": 21, "num_heads": 16,
            "qk_norm": True, "qkv_bias": False, "qk_norm_type": "rms",
            "with_decoupled_ca": False, "decoupled_ca_dim": num_tokens_cond,
            "decoupled_ca_weight": 1.0, "use_attention_pooling": False,
            "use_pos_emb": False,
            "num_moe_layers": 6, "num_experts": 8, "moe_top_k": 2,
        }},
        "scheduler": {"params": {"num_train_timesteps": 1000}},
    }


def _get_xpart_models(config):
    """Load or return cached X-Part models (ComfyUI-native, meta-device pattern)."""
    cache_key = f"{config['precision']}_{config['enable_flash']}_{config['pc_size']}"

    if cache_key in _xpart_model_cache:
        cached = _xpart_model_cache[cache_key]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached['dit'].to(device)
        cached['vae'].to(device)
        cached['conditioner'].to(device)
        return cached

    from .hunyuan3d_part.model import PartFormerDITPlain
    from .hunyuan3d_part.vae import VolumeDecoderShapeVAE
    from .hunyuan3d_part.conditioner import Conditioner
    import comfy.ops
    from safetensors.torch import load_file

    ops = comfy.ops.manual_cast

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = config['precision']
    enable_flash = config['enable_flash']
    pc_size = config['pc_size']
    ckpt_path = config['ckpt_path']

    dtype = torch.float16 if precision == "float16" else torch.bfloat16

    # Get architecture config and apply runtime overrides
    xpart_config = _xpart_arch_config()

    print(f"[X-Part Models] Overriding pc_size in config: {pc_size}")
    xpart_config["shapevae"]["params"]["pc_size"] = pc_size
    xpart_config["shapevae"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_size"] = pc_size
    xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_size"] = pc_size
    xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["seg_feat_cfg"]["params"]["enable_flash"] = enable_flash

    print(f"[X-Part Models] Loading DiT, VAE, and Conditioner ({precision}, meta-device)...")
    t0 = time.time()

    def load_dit():
        model_params = dict(xpart_config["model"]["params"])
        with torch.device("meta"):
            model = PartFormerDITPlain(**model_params, dtype=dtype, device="meta", operations=ops)
        sd = load_file(config['model_file'], device=device)
        model.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(model, device)
        model = model.to(device=device).eval()
        print(f"[X-Part Models] DiT loaded")
        return model

    def load_vae():
        vae_params = dict(xpart_config["shapevae"]["params"])
        with torch.device("meta"):
            vae = VolumeDecoderShapeVAE(**vae_params, dtype=dtype, device="meta", operations=ops)
        sd = load_file(config['vae_file'], device=device)
        vae.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(vae, device)
        vae = vae.to(device=device).eval()
        print(f"[X-Part Models] VAE loaded")
        return vae

    def load_cond():
        # Extract params directly (no instantiate_from_config)
        cond_cfg = dict(xpart_config["conditioner"]["params"])

        # Build geo_encoder_params from geo_cfg
        geo_encoder_params = None
        geo_output_dim = None
        if cond_cfg.get("use_geo") and "geo_cfg" in cond_cfg:
            raw_geo = dict(cond_cfg["geo_cfg"])
            geo_output_dim = raw_geo.get("output_dim")
            raw_geo_params = dict(raw_geo.get("params", {}))
            # Convert local_geo_cfg from nested {target, params} to flat params dict
            if "local_geo_cfg" in raw_geo_params:
                raw_local = raw_geo_params["local_geo_cfg"]
                raw_geo_params["local_geo_cfg"] = dict(raw_local.get("params", {}))
            geo_encoder_params = raw_geo_params

        # Build obj_encoder_params from obj_encoder_cfg
        obj_encoder_params = None
        obj_output_dim = None
        if cond_cfg.get("use_obj") and "obj_encoder_cfg" in cond_cfg:
            raw_obj = dict(cond_cfg["obj_encoder_cfg"])
            obj_output_dim = raw_obj.get("output_dim")
            obj_encoder_params = dict(raw_obj.get("params", {}))

        # Build seg_feat_encoder_params from seg_feat_cfg
        seg_feat_encoder_params = None
        seg_feat_output_dim = None
        if cond_cfg.get("use_seg_feat") and "seg_feat_cfg" in cond_cfg:
            raw_seg = dict(cond_cfg["seg_feat_cfg"])
            seg_feat_output_dim = raw_seg.get("output_dim")
            seg_feat_encoder_params = dict(raw_seg.get("params", {}))

        # No meta-device for Conditioner (Sonata uses spconv which needs real device)
        conditioner = Conditioner(
            use_image=cond_cfg.get("use_image", False),
            use_geo=cond_cfg.get("use_geo", True),
            use_obj=cond_cfg.get("use_obj", True),
            use_seg_feat=cond_cfg.get("use_seg_feat", False),
            geo_encoder_params=geo_encoder_params,
            geo_output_dim=geo_output_dim,
            obj_encoder_params=obj_encoder_params,
            obj_output_dim=obj_output_dim,
            seg_feat_encoder_params=seg_feat_encoder_params,
            seg_feat_output_dim=seg_feat_output_dim,
            dtype=dtype, device=device, operations=ops,
        )
        sd = load_file(config['cond_file'], device=device)
        conditioner.load_state_dict(sd, strict=False)

        # Keep seg_feat_encoder in float32 (Sonata backbone)
        if hasattr(conditioner, 'seg_feat_encoder') and conditioner.seg_feat_encoder is not None:
            conditioner.seg_feat_encoder = conditioner.seg_feat_encoder.to(dtype=torch.float32)

        conditioner = conditioner.to(device=device).eval()
        print(f"[X-Part Models] Conditioner loaded")
        return conditioner

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_dit = executor.submit(load_dit)
        future_vae = executor.submit(load_vae)
        future_cond = executor.submit(load_cond)
        dit = future_dit.result()
        vae = future_vae.result()
        cond = future_cond.result()

    total_time = time.time() - t0
    print(f"[X-Part Models] All models loaded in {total_time:.2f}s")

    result = {'dit': dit, 'vae': vae, 'conditioner': cond, 'config': xpart_config}

    if config.get('cache_on_gpu', True):
        _xpart_model_cache[cache_key] = result

    return result


class XPartGenerateParts:
    """
    Generate high-quality part meshes using X-Part.

    Takes mesh_with_features (from ComputeMeshFeatures with all_points=True),
    bounding boxes, and X-Part config as inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_with_features": ("MESH_FEATURES",),  # REQUIRED - use ComputeMeshFeatures with all_points=True
                "bounding_boxes": ("BBOXES_3D",),
                "xpart_config": ("XPART_CONFIG",),
                "octree_resolution": ("INT", {
                    "default": 256,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Mesh quality. 256=~8GB, 512=~12-16GB, 1024=~24GB+ VRAM"
                }),
                "num_inference_steps": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Diffusion steps. 25=fast default, 50=high quality"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "-1.0 = disabled (fastest). 0-10 = enabled (slower, doubles compute)."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducibility."
                }),
                "num_chunks": ("INT", {
                    "default": 10000,
                    "min": 10000,
                    "max": 500000,
                    "step": 50000,
                    "tooltip": "Extraction batch size. Higher = more VRAM but faster."
                }),
                "output_coordinate_system": (["Y-up (default)", "Z-up"], {
                    "default": "Y-up (default)",
                    "tooltip": "Output coordinate system. Use Z-up if your input mesh is Z-up (CAD convention like STL)."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "parts_path", "bbox_path", "exploded_path")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D/Processing"

    def generate(self, mesh_with_features, bounding_boxes, xpart_config, octree_resolution, num_inference_steps,
                guidance_scale, seed, num_chunks, output_coordinate_system):
        """Generate part meshes."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            # Extract mesh and Sonata features from mesh_with_features
            mesh_obj = mesh_with_features['mesh']
            sonata_features = mesh_with_features['features']
            sonata_points = mesh_with_features['points']
            sonata_normals = mesh_with_features['normals']
            all_points_mode = mesh_with_features.get('all_points', False)

            if not all_points_mode:
                print(f"[X-Part Generate] WARNING: mesh_with_features was computed with all_points=False. "
                      f"For best results, use ComputeMeshFeatures with all_points=True for X-Part generation.")

            print(f"[X-Part Generate] Using pre-computed Sonata features ({len(sonata_features)} points)")

            # Save mesh to temp file for pipeline
            mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
            save_mesh(mesh_obj, mesh_path)

            # Extract bounding boxes
            if isinstance(bounding_boxes, dict) and 'bboxes' in bounding_boxes:
                aabb = bounding_boxes['bboxes']
                print(f"[X-Part Generate] Using {len(aabb)} bounding boxes")
            else:
                aabb = None
                print(f"[X-Part Generate] No bounding boxes, will auto-detect")

            # Load models from config
            models = _get_xpart_models(xpart_config)
            dit = models['dit']
            vae = models['vae']
            conditioner = models['conditioner']
            xpart_cfg = models['config']

            dtype_str = xpart_config.get("precision", "float32")
            cache_on_gpu = xpart_config.get("cache_on_gpu", True)
            dtype = torch.float32 if dtype_str == "float32" else torch.float16

            print(f"[X-Part Generate] Using {dtype_str} precision")

            # Import from core
            from .xpart_pipeline import PartFormerPipeline
            from pathlib import Path

            # Create scheduler directly
            scheduler_params = dict(xpart_cfg["scheduler"]["params"])
            scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_params)

            # Create minimal pipeline instance
            pipeline = PartFormerPipeline(
                vae=vae,
                model=dit,
                scheduler=scheduler,
                conditioner=conditioner,
                bbox_predictor=None,
                device=device,
                dtype=dtype,
                verbose=True
            )

            print(f"[X-Part Generate] Running generation...")
            print(f"  - Octree resolution: {octree_resolution}")
            print(f"  - Inference steps: {num_inference_steps}")
            print(f"  - Guidance scale: {guidance_scale}")
            print(f"  - Seed: {seed}")

            # Convert aabb to tensor if provided
            if aabb is not None:
                aabb_tensor = torch.from_numpy(aabb).to(device).float()
            else:
                aabb_tensor = None

            # Prepare precomputed Sonata features for conditioner
            precomputed_sonata = {
                'features': torch.as_tensor(sonata_features).to(device).float(),
                'points': torch.as_tensor(sonata_points).to(device).float(),
                'normals': torch.as_tensor(sonata_normals).to(device).float(),
            }

            # Run generation with precomputed Sonata features
            # obj_mesh is now a list of trimesh.Trimesh (not a Scene)
            result = pipeline(
                mesh_path=mesh_path,
                aabb=aabb_tensor,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_chunks=num_chunks,
                output_type="trimesh",
                precomputed_sonata_features=precomputed_sonata
            )
            parts_list = result[0]  # list[trimesh.Trimesh]
            viz_tuple = result[1]   # (out_bbox, mesh_gt_bbox, explode_object) or None

            print(f"[X-Part Generate] Generation complete! {len(parts_list)} parts")

            # Apply coordinate system transformation if requested
            if output_coordinate_system == "Z-up":
                print("[X-Part Generate] Converting output to Z-up coordinate system...")
                rotation_matrix = np.array([
                    [1,  0,  0, 0],
                    [0,  0,  1, 0],
                    [0, -1,  0, 0],
                    [0,  0,  0, 1]
                ])
                for part in parts_list:
                    part.apply_transform(rotation_matrix)
                print("[X-Part Generate] Converted to Z-up")

            # Save outputs
            output_dir = folder_paths.get_output_directory()

            # Build temp Scene for saving parts GLB
            parts_scene = trimesh.Scene()
            for p in parts_list:
                parts_scene.add_geometry(p)
            parts_path = os.path.join(output_dir, f"xpart_parts_{seed}.glb")
            save_mesh(parts_scene, parts_path)
            print(f"[X-Part Generate] Saved parts to: {parts_path}")

            # Save bbox viz and exploded view if available
            bbox_path = ""
            exploded_path = ""
            if viz_tuple is not None:
                out_bbox, mesh_gt_bbox, explode_object = viz_tuple
                if output_coordinate_system == "Z-up":
                    for name in list(out_bbox.geometry.keys()):
                        geom = out_bbox.geometry[name]
                        if hasattr(geom, 'apply_transform'):
                            geom.apply_transform(rotation_matrix)
                    for name in list(explode_object.geometry.keys()):
                        geom = explode_object.geometry[name]
                        if hasattr(geom, 'apply_transform'):
                            geom.apply_transform(rotation_matrix)

                bbox_path = os.path.join(output_dir, f"xpart_bbox_{seed}.glb")
                save_mesh(out_bbox, bbox_path)
                print(f"[X-Part Generate] Saved bbox viz to: {bbox_path}")

                exploded_path = os.path.join(output_dir, f"xpart_exploded_{seed}.glb")
                save_mesh(explode_object, exploded_path)
                print(f"[X-Part Generate] Saved exploded view to: {exploded_path}")

            # Clean up temp file
            if mesh_path and mesh_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(mesh_path)
                except (OSError, IOError) as cleanup_err:
                    print(f"[X-Part Generate] Warning: Failed to clean temp file: {cleanup_err}")

            # Auto-unload models if cache_on_gpu is False
            if not cache_on_gpu:
                print("[X-Part Generate] Auto-unloading models from GPU...")
                dit.to("cpu")
                vae.to("cpu")
                conditioner.to("cpu")
                torch.cuda.empty_cache()
                print("[X-Part Generate] Models unloaded, VRAM freed")

            return (parts_list, parts_path, bbox_path, exploded_path)

        except Exception as e:
            print(f"[X-Part Generate] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "XPartGenerateParts": XPartGenerateParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XPartGenerateParts": "X-Part Generate Parts",
}

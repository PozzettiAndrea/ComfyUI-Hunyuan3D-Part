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
from .core.mesh_utils import load_mesh, save_mesh, colorize_segmentation, get_temp_mesh_path
from .core.models.diffusion.schedulers import FlowMatchEulerDiscreteScheduler

# Worker-process model cache for X-Part models
_xpart_model_cache = {}


def _get_xpart_models(config):
    """Load or return cached X-Part models within the worker process."""
    cache_key = f"{config['precision']}_{config['enable_flash']}_{config['pc_size']}"

    if cache_key in _xpart_model_cache:
        cached = _xpart_model_cache[cache_key]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cached['dit'].to(device)
        cached['vae'].to(device)
        cached['conditioner'].to(device)
        return cached

    from .core.models.partformer_dit import PartFormerDITPlain
    from .core.models.autoencoders import VolumeDecoderShapeVAE
    from .core.models.conditioner.condioner_release import Conditioner
    from omegaconf import OmegaConf
    from pathlib import Path
    from safetensors.torch import load_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = config['precision']
    enable_flash = config['enable_flash']
    pc_size = config['pc_size']
    ckpt_path = config['ckpt_path']

    dtype = torch.float16 if precision == "float16" else torch.bfloat16

    # Load shared config
    config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
    xpart_config = OmegaConf.load(str(config_path))

    # Override pc_size values
    print(f"[X-Part Models] Overriding pc_size in config: {pc_size}")
    xpart_config["shapevae"]["params"]["pc_size"] = pc_size
    xpart_config["shapevae"]["params"]["pc_sharpedge_size"] = 0

    # Override in conditioner geo_cfg (local_geo_cfg)
    if "conditioner" in xpart_config and "params" in xpart_config["conditioner"]:
        if "geo_cfg" in xpart_config["conditioner"]["params"]:
            if "params" in xpart_config["conditioner"]["params"]["geo_cfg"]:
                if "local_geo_cfg" in xpart_config["conditioner"]["params"]["geo_cfg"]["params"]:
                    if "params" in xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]:
                        xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_size"] = pc_size
                        xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_sharpedge_size"] = 0

        if "obj_encoder_cfg" in xpart_config["conditioner"]["params"]:
            if "params" in xpart_config["conditioner"]["params"]["obj_encoder_cfg"]:
                xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_size"] = pc_size
                xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_sharpedge_size"] = 0

        if "seg_feat_cfg" in xpart_config["conditioner"]["params"]:
            if "params" not in xpart_config["conditioner"]["params"]["seg_feat_cfg"]:
                xpart_config["conditioner"]["params"]["seg_feat_cfg"]["params"] = {}
            xpart_config["conditioner"]["params"]["seg_feat_cfg"]["params"]["enable_flash"] = enable_flash

    print(f"[X-Part Models] Loading DiT, VAE, and Conditioner in parallel ({precision})...")
    t0 = time.time()

    def load_dit():
        model_params = dict(xpart_config["model"]["params"])
        model = PartFormerDITPlain(**model_params)
        state_dict = load_file(config['model_file'], device=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device=device, dtype=dtype).eval()
        print(f"[X-Part Models] DiT loaded")
        return model

    def load_vae():
        vae_params = dict(xpart_config["shapevae"]["params"])
        vae = VolumeDecoderShapeVAE(**vae_params)
        state_dict = load_file(config['vae_file'], device=device)
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(device=device, dtype=dtype).eval()
        print(f"[X-Part Models] VAE loaded")
        return vae

    def load_cond():
        cond_cfg = xpart_config["conditioner"]["params"]
        if "geo_cfg" in cond_cfg:
            cond_cfg["geo_cfg"]["target"] = ".models.conditioner.part_encoders.PartEncoder"
            if "local_geo_cfg" in cond_cfg["geo_cfg"]["params"]:
                cond_cfg["geo_cfg"]["params"]["local_geo_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"
        if "obj_encoder_cfg" in cond_cfg:
            cond_cfg["obj_encoder_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"
        if "seg_feat_cfg" in cond_cfg:
            cond_cfg["seg_feat_cfg"]["target"] = ".models.conditioner.sonata_extractor.SonataFeatureExtractor"

        conditioner = Conditioner(**cond_cfg)
        state_dict = load_file(config['cond_file'], device=device)
        conditioner.load_state_dict(state_dict, strict=False)

        # Selective dtype conversion (keep seg_feat_encoder in float32)
        if hasattr(conditioner, 'geo_encoder') and conditioner.geo_encoder is not None:
            conditioner.geo_encoder = conditioner.geo_encoder.to(dtype=dtype)
        if hasattr(conditioner, 'obj_encoder') and conditioner.obj_encoder is not None:
            conditioner.obj_encoder = conditioner.obj_encoder.to(dtype=dtype)
        if hasattr(conditioner, 'geo_out_proj') and conditioner.geo_out_proj is not None:
            conditioner.geo_out_proj = conditioner.geo_out_proj.to(dtype=dtype)
        if hasattr(conditioner, 'obj_out_proj') and conditioner.obj_out_proj is not None:
            conditioner.obj_out_proj = conditioner.obj_out_proj.to(dtype=dtype)
        if hasattr(conditioner, 'seg_feat_outproj') and conditioner.seg_feat_outproj is not None:
            conditioner.seg_feat_outproj = conditioner.seg_feat_outproj.to(dtype=dtype)
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
            from .core.xpart_pipeline import PartFormerPipeline
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

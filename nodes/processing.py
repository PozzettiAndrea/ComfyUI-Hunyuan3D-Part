"""
Processing Nodes for Hunyuan3D-Part.

P3-SAM segmentation and X-Part generation nodes.
Loads models from config dicts passed by loader nodes.
"""

import torch
import numpy as np
import trimesh
import hashlib
import folder_paths
import os
import tempfile
import time
import concurrent.futures
import comfy.model_management
import comfy.utils

# Import utilities from core
from .mesh_utils import load_mesh, save_mesh, colorize_segmentation, get_temp_mesh_path
from .schedulers import FlowMatchEulerDiscreteScheduler

# Worker-process model caches (persist across node executions, same as TRELLIS2 pattern)
_p3sam_model_cache = {}
_xpart_model_cache = {}


def _get_p3sam_model(config):
    """Load or return cached P3-SAM model (ComfyUI-native: auto-dtype, ModelPatcher, load_models_gpu)."""
    import comfy.model_patcher

    precision = config.get('precision', 'auto')
    attn_backend = config.get('attn_backend', 'auto')
    enable_flash = attn_backend in ('auto', 'flash_attn')

    cache_key = f"{precision}_{attn_backend}"

    if cache_key in _p3sam_model_cache:
        patcher = _p3sam_model_cache[cache_key]
        comfy.model_management.load_models_gpu([patcher])
        return patcher.model

    from .p3sam.model import MultiHeadSegment
    from safetensors.torch import load_file

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    print(f"[P3-SAM] Building model (precision={precision}, attn_backend={attn_backend})...")
    model = MultiHeadSegment(
        in_channel=512,
        head_num=3,
        ignore_label=-100,
        enable_flash=enable_flash
    )

    state_dict = load_file(config['ckpt_path'], device="cpu")
    # Strip 'dit.' prefix from checkpoint keys (legacy checkpoint format)
    state_dict = {k.removeprefix("dit."): v for k, v in state_dict.items()}

    # Resolve dtype: "auto" = auto-detect, otherwise explicit override
    weight_dtype = next(iter(state_dict.values())).dtype
    if precision == 'auto':
        model_dtype = comfy.model_management.unet_dtype(
            device=device,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=weight_dtype,
        )
    else:
        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"[P3-SAM] precision={precision}, dtype={model_dtype}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(model_dtype)

    # Wrap in ModelPatcher for ComfyUI VRAM management
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=device, offload_device=offload_device,
    )
    comfy.model_management.load_models_gpu([patcher])

    print(f"[P3-SAM] Model loaded on {device}")

    _p3sam_model_cache[cache_key] = patcher

    return patcher.model


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


def _get_xpart_models(config, pc_size=40960):
    """Load or return cached X-Part models (ComfyUI-native: auto-dtype, ModelPatcher, load_models_gpu)."""
    import comfy.ops
    import comfy.model_patcher

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    precision = config.get('precision', 'auto')
    attn_backend = config.get('attn_backend', 'auto')
    enable_flash = attn_backend in ('auto', 'flash_attn')

    # Detect weight dtype from safetensors
    from safetensors import safe_open
    with safe_open(config['model_file'], framework="pt", device="cpu") as f:
        weight_dtype = f.get_tensor(list(f.keys())[0]).dtype

    # Resolve model dtype: "auto" = auto-detect, otherwise explicit override
    if precision == 'auto':
        model_dtype = comfy.model_management.unet_dtype(
            device=device,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=weight_dtype,
        )
    else:
        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        model_dtype, device, [torch.bfloat16, torch.float32],
    )
    ops = comfy.ops.pick_operations(model_dtype, manual_cast_dtype, load_device=device)

    print(f"[X-Part Models] precision={precision}, dtype={model_dtype}, attn_backend={attn_backend}, ops={ops.__name__}")

    cache_key = f"{precision}_{attn_backend}_{pc_size}"

    if cache_key in _xpart_model_cache:
        cached = _xpart_model_cache[cache_key]
        comfy.model_management.load_models_gpu(
            [cached['dit'], cached['vae'], cached['conditioner']]
        )
        print(f"[X-Part Models] Using cached models")
        return cached

    from .hunyuan3d_part.model import PartFormerDITPlain
    from .hunyuan3d_part.vae import VolumeDecoderShapeVAE
    from .hunyuan3d_part.conditioner import Conditioner
    from safetensors.torch import load_file

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

    print(f"[X-Part Models] Loading DiT, VAE, and Conditioner ({model_dtype}, meta-device)...")
    t0 = time.time()

    def load_dit():
        model_params = dict(xpart_config["model"]["params"])
        with torch.device("meta"):
            model = PartFormerDITPlain(**model_params, dtype=model_dtype, device="meta", operations=ops)
        sd = load_file(config['model_file'], device="cpu")
        model.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(model, device)
        model.to(dtype=model_dtype)
        model.eval()
        print(f"[X-Part Models] DiT loaded")
        return model

    def load_vae():
        vae_params = dict(xpart_config["shapevae"]["params"])
        with torch.device("meta"):
            vae = VolumeDecoderShapeVAE(**vae_params, dtype=model_dtype, device="meta", operations=ops)
        sd = load_file(config['vae_file'], device="cpu")
        vae.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(vae, device)
        vae.to(dtype=model_dtype)
        vae.eval()
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
            dtype=model_dtype, device=device, operations=ops,
        )
        sd = load_file(config['cond_file'], device="cpu")
        conditioner.load_state_dict(sd, strict=False)

        # Keep seg_feat_encoder in float32 (Sonata backbone)
        if hasattr(conditioner, 'seg_feat_encoder') and conditioner.seg_feat_encoder is not None:
            conditioner.seg_feat_encoder = conditioner.seg_feat_encoder.to(dtype=torch.float32)

        conditioner.eval()
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

    # Wrap in ModelPatcher for ComfyUI VRAM management
    dit_patcher = comfy.model_patcher.ModelPatcher(dit, load_device=device, offload_device=offload_device)
    vae_patcher = comfy.model_patcher.ModelPatcher(vae, load_device=device, offload_device=offload_device)
    cond_patcher = comfy.model_patcher.ModelPatcher(cond, load_device=device, offload_device=offload_device)

    comfy.model_management.load_models_gpu([dit_patcher, vae_patcher, cond_patcher])

    result = {
        'dit': dit_patcher,
        'vae': vae_patcher,
        'conditioner': cond_patcher,
        'config': xpart_config,
        'dtype': model_dtype,
    }

    _xpart_model_cache[cache_key] = result

    return result


class ComputeMeshFeatures:
    """
    Compute mesh features for P3-SAM segmentation.

    Performs:
    - Mesh cleaning and adjacent faces computation (~1.4s)
    - Point cloud sampling (~0.03s)
    - Sonata feature extraction (~0.5s)

    Uses the Sonata encoder built into P3-SAM model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "p3sam_config": ("P3SAM_CONFIG",),
                "all_points": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ALL mesh vertices instead of sampling. Enable for X-Part generation."
                }),
                "point_num": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample (ignored if all_points=True)."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "tooltip": "Random seed for point sampling (ignored if all_points=True)."
                }),
            },
        }

    RETURN_TYPES = ("MESH_FEATURES",)
    RETURN_NAMES = ("mesh_with_features",)
    FUNCTION = "compute_features"
    CATEGORY = "Hunyuan3D/Processing"

    def compute_features(self, mesh, p3sam_config, all_points, point_num, seed):
        """Compute mesh features using P3-SAM's internal Sonata encoder."""
        try:
            # Load mesh if needed
            if isinstance(mesh, dict) and 'trimesh' in mesh:
                mesh_obj = mesh['trimesh']
            elif isinstance(mesh, str):
                mesh_obj = load_mesh(mesh)
            elif isinstance(mesh, trimesh.Trimesh):
                mesh_obj = mesh
            else:
                mesh_obj = load_mesh(mesh)

            print(f"[Compute Features] Computing mesh features...")

            # Load model from config
            p3sam = _get_p3sam_model(p3sam_config)

            # Import required functions from core
            from .p3sam_processing import (
                build_adjacent_faces_numba,
                get_feat,
                clean_mesh,
                normalize_pc
            )

            # Clean mesh if needed
            print(f"[Compute Features] Cleaning mesh...")
            mesh_loaded = clean_mesh(mesh_obj)
            mesh_loaded = trimesh.Trimesh(vertices=mesh_loaded.vertices, faces=mesh_loaded.faces)

            # Build adjacent faces
            print(f"[Compute Features] Building adjacent faces...")
            face_adjacency = mesh_loaded.face_adjacency
            adjacent_faces = build_adjacent_faces_numba(face_adjacency)

            # Get points: either sample or use ALL vertices
            if all_points:
                # Use ALL mesh vertices
                print(f"[Compute Features] Using ALL {len(mesh_loaded.vertices)} mesh vertices...")
                _points = mesh_loaded.vertices
                normals = mesh_loaded.vertex_normals
                face_idx = None  # No face mapping when using vertices
            else:
                # Sample point cloud
                print(f"[Compute Features] Sampling {point_num} points...")
                _points, face_idx = trimesh.sample.sample_surface(mesh_loaded, point_num, seed=seed)
                normals = mesh_loaded.face_normals[face_idx]

            _points_normalized = normalize_pc(_points)

            # Extract features using P3-SAM's internal Sonata
            print(f"[Compute Features] Extracting Sonata features...")
            points = _points_normalized.astype(np.float32)
            normals = normals.astype(np.float32)

            # Get features (p3sam has .sonata and .transform attributes)
            t0 = time.time()
            feats = get_feat(p3sam, points, normals)
            feat_time = time.time() - t0
            print(f"[Compute Features] Features computed ({feat_time:.2f}s)")

            # Prepare output data
            # np.asarray() strips trimesh TrackedArray wrappers so comfy_env can serialize
            cache_data = {
                'mesh': mesh_loaded,
                'face_idx': np.asarray(face_idx) if face_idx is not None else None,
                'adjacent_faces': np.asarray(adjacent_faces),
                'features': feats.detach().cpu(),
                'points': np.asarray(points),
                'normals': np.asarray(normals),
                'all_points': all_points,
                'point_num': point_num if not all_points else len(_points),
                'seed': seed,
            }

            return (cache_data,)

        except Exception as e:
            print(f"[Compute Features] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class P3SAMSegmentMesh:
    """
    Segment mesh into semantic parts using P3-SAM.

    Takes pre-computed mesh features and runs P3-SAM inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_with_features": ("MESH_FEATURES",),
                "p3sam_config": ("P3SAM_CONFIG",),
                "prompt_num": ("INT", {
                    "default": 400,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Number of prompt points."
                }),
                "prompt_bs": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 128,
                    "step": 1,
                    "tooltip": "Prompt batch size. Higher = more VRAM but faster."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.7,
                    "max": 0.999,
                    "step": 0.01,
                    "tooltip": "Merge threshold."
                }),
                "post_process": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable post-processing."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "BBOXES_3D", "FACE_IDS", "STRING")
    RETURN_NAMES = ("mesh", "bounding_boxes", "face_ids", "preview_path")
    FUNCTION = "segment"
    CATEGORY = "Hunyuan3D/Processing"

    def segment(self, mesh_with_features, p3sam_config, prompt_num, prompt_bs, threshold, post_process):
        """Segment mesh into parts using P3-SAM."""
        try:
            import fpsample
            from tqdm import tqdm
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor

            # Extract feature data
            mesh_loaded = mesh_with_features['mesh']
            face_idx = mesh_with_features['face_idx']
            adjacent_faces = mesh_with_features['adjacent_faces']
            feats = mesh_with_features['features']
            points = mesh_with_features['points']
            seed = mesh_with_features['seed']

            print(f"[P3-SAM Segment] Running segmentation with {prompt_num} prompts...")

            # Load model from config (ModelPatcher handles GPU placement)
            p3sam = _get_p3sam_model(p3sam_config)
            device = comfy.model_management.get_torch_device()

            # Import functions from core
            from .p3sam_processing import (
                get_mask,
                cal_iou,
                fix_label,
                get_aabb_from_face_ids,
                do_post_process
            )

            # FPS sample prompt points
            fps_idx = fpsample.fps_sampling(points, prompt_num)
            _point_prompts = points[fps_idx]

            # Get masks (inference step)
            bs = prompt_bs
            step_num = prompt_num // bs + 1
            mask_res = []
            iou_res = []
            comfy_pbar = comfy.utils.ProgressBar(step_num)
            for i in tqdm(range(step_num), desc="P3-SAM Inference"):
                cur_prompt = _point_prompts[bs * i : bs * (i + 1)]
                if len(cur_prompt) == 0:
                    continue
                pred_mask_1, pred_mask_2, pred_mask_3, pred_iou = get_mask(
                    p3sam, feats, points, cur_prompt
                )
                pred_mask = np.stack([pred_mask_1, pred_mask_2, pred_mask_3], axis=-1)
                max_idx = np.argmax(pred_iou, axis=-1)
                for j in range(max_idx.shape[0]):
                    mask_res.append(pred_mask[:, j, max_idx[j]])
                    iou_res.append(pred_iou[j, max_idx[j]])
                comfy_pbar.update(1)

            mask_res = np.stack(mask_res, axis=-1)

            # Sort by IOU
            mask_iou = [[mask_res[:, i], iou_res[i]] for i in range(len(iou_res))]
            mask_iou_sorted = sorted(mask_iou, key=lambda x: x[1], reverse=True)
            mask_sorted = [mask_iou_sorted[i][0] for i in range(len(iou_res))]
            iou_sorted = [mask_iou_sorted[i][1] for i in range(len(iou_res))]

            # NMS
            clusters = defaultdict(list)
            with ThreadPoolExecutor(max_workers=20) as executor:
                for i in tqdm(range(len(mask_sorted)), desc="NMS"):
                    _mask = mask_sorted[i]
                    futures = []
                    for j in clusters.keys():
                        futures.append(executor.submit(cal_iou, _mask, mask_sorted[j]))

                    for j, future in zip(clusters.keys(), futures):
                        if future.result() > 0.9:
                            clusters[j].append(i)
                            break
                    else:
                        clusters[i].append(i)

            print(f"[P3-SAM Segment] NMS complete: {len(clusters)} clusters")

            # Filter single mask clusters
            filtered_clusters = [i for i in clusters.keys() if len(clusters[i]) > 2]

            # Merge similar clusters
            merged_clusters = []
            for i in filtered_clusters:
                merged = False
                for j in range(len(merged_clusters)):
                    if cal_iou(mask_sorted[i], mask_sorted[merged_clusters[j]]) > threshold:
                        merged = True
                        break
                if not merged:
                    merged_clusters.append(i)

            # Calculate point labels
            point_labels = np.zeros(len(points), dtype=np.int32) - 1
            for idx, cluster_id in enumerate(merged_clusters):
                mask = mask_sorted[cluster_id] > 0.5
                point_labels[mask] = idx

            # Project to mesh faces
            face_labels = np.zeros(len(mesh_loaded.faces), dtype=np.int32) - 1
            if face_idx is not None:
                for i, fid in enumerate(face_idx):
                    if point_labels[i] >= 0 and face_labels[fid] < 0:
                        face_labels[fid] = point_labels[i]

            # Fix unlabeled faces
            if post_process:
                face_ids = fix_label(face_labels, adjacent_faces, use_aabb=True, mesh=mesh_loaded, show_info=True)
            else:
                face_ids = face_labels

            # Get AABBs
            aabb = get_aabb_from_face_ids(mesh_loaded, face_ids)

            print(f"[P3-SAM Segment] Segmentation complete: found {len(aabb)} parts")

            # Add metadata
            processed_mesh = mesh_loaded
            processed_mesh.metadata['face_part_ids'] = face_ids
            processed_mesh.metadata['part_bboxes'] = aabb
            processed_mesh.metadata['num_parts'] = len(aabb)

            # Create colored visualization
            colored_mesh = colorize_segmentation(processed_mesh, face_ids, seed=seed)

            # Save preview
            output_dir = folder_paths.get_output_directory()
            preview_path = os.path.join(output_dir, f"p3sam_segmentation_{seed}.glb")
            save_mesh(colored_mesh, preview_path)
            print(f"[P3-SAM Segment] Saved preview to: {preview_path}")

            # Prepare outputs
            bboxes_output = {
                'bboxes': aabb,
                'num_parts': len(aabb)
            }

            face_ids_output = {
                'face_ids': face_ids,
                'num_parts': len(np.unique(face_ids[face_ids >= 0]))
            }

            return (processed_mesh, bboxes_output, face_ids_output, preview_path)

        except Exception as e:
            print(f"[P3-SAM Segment] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


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
                "pc_size": ("INT", {
                    "default": 40960,
                    "min": 1024,
                    "max": 81920,
                    "step": 1024,
                    "tooltip": "Points per object/part. 40960=trained default, <20480=quality loss, <5120=very poor. Model reloads on change."
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
                guidance_scale, seed, pc_size, output_coordinate_system):
        """Generate part meshes."""
        device = comfy.model_management.get_torch_device()
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

            # Load models from config (returns ModelPatcher-wrapped models)
            models = _get_xpart_models(xpart_config, pc_size=pc_size)
            dit = models['dit'].model
            vae = models['vae'].model
            conditioner = models['conditioner'].model
            xpart_cfg = models['config']
            dtype = models['dtype']

            print(f"[X-Part Generate] Using {dtype} precision")

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
                aabb_tensor = torch.from_numpy(aabb).to(device=device, dtype=dtype)
            else:
                aabb_tensor = None

            # Prepare precomputed Sonata features for conditioner
            precomputed_sonata = {
                'features': torch.as_tensor(sonata_features).to(device=device, dtype=dtype),
                'points': torch.as_tensor(sonata_points).to(device=device, dtype=dtype),
                'normals': torch.as_tensor(sonata_normals).to(device=device, dtype=dtype),
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

            # VRAM is managed by ComfyUI via ModelPatcher/load_models_gpu

            return (parts_list, parts_path, bbox_path, exploded_path)

        except Exception as e:
            print(f"[X-Part Generate] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComputeMeshFeatures": ComputeMeshFeatures,
    "P3SAMSegmentMesh": P3SAMSegmentMesh,
    "XPartGenerateParts": XPartGenerateParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComputeMeshFeatures": "Compute Mesh Features",
    "P3SAMSegmentMesh": "P3-SAM Segment Mesh",
    "XPartGenerateParts": "X-Part Generate Parts",
}

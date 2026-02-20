"""
Feature Caching Nodes for Hunyuan3D-Part.

Implements caching for expensive preprocessing operations to speed up
repeated runs on the same mesh. Handles model loading internally
from config dicts passed by loader nodes.
"""

import torch
import numpy as np
import trimesh
import hashlib
from collections import OrderedDict
import comfy.model_management

from .mesh_utils import load_mesh

# Maximum number of cached feature sets (to prevent unbounded memory growth)
MAX_CACHE_SIZE = 10

# Worker-process model cache (shared across all node instances)
_p3sam_model_cache = {}


def _get_p3sam_model(config):
    """Load or return cached P3-SAM model within the worker process."""
    cache_key = f"{config['enable_flash']}"

    if cache_key in _p3sam_model_cache:
        model = _p3sam_model_cache[cache_key]
        device = comfy.model_management.get_torch_device()
        model.to(device)
        return model

    from .p3sam.model import MultiHeadSegment
    from safetensors.torch import load_file

    device = comfy.model_management.get_torch_device()

    print(f"[P3-SAM] Building model with enable_flash={config['enable_flash']}...")
    model = MultiHeadSegment(
        in_channel=512,
        head_num=3,
        ignore_label=-100,
        enable_flash=config['enable_flash']
    )

    state_dict = load_file(config['ckpt_path'], device=device)
    # Strip 'dit.' prefix from checkpoint keys (legacy checkpoint format)
    state_dict = {k.removeprefix("dit."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    print(f"[P3-SAM] Model loaded on {device}")

    if config.get('cache_on_gpu', True):
        _p3sam_model_cache[cache_key] = model

    return model


class ComputeMeshFeatures:
    """
    Compute mesh features for P3-SAM segmentation.

    Performs:
    - Mesh cleaning and adjacent faces computation (~1.4s)
    - Point cloud sampling (~0.03s)
    - Sonata feature extraction (~0.5s)

    Uses the Sonata encoder built into P3-SAM model.
    Features are cached internally for identical mesh+params combinations.
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

    # Global cache storage with LRU eviction
    _feature_cache = OrderedDict()

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

            # Generate cache key based on mesh + parameters
            cache_key = self._generate_cache_key(mesh_obj, all_points, point_num, seed)

            # Check if already cached
            if cache_key in ComputeMeshFeatures._feature_cache:
                print(f"[Compute Features] Using cached features")
                ComputeMeshFeatures._feature_cache.move_to_end(cache_key)
                cached_data = ComputeMeshFeatures._feature_cache[cache_key]
                return (cached_data,)

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
            import time
            t0 = time.time()
            feats = get_feat(p3sam, points, normals)
            feat_time = time.time() - t0
            print(f"[Compute Features] Features computed ({feat_time:.2f}s)")

            # Auto-unload if not caching
            if not p3sam_config.get('cache_on_gpu', True):
                p3sam.to("cpu")
                comfy.model_management.soft_empty_cache()

            # Prepare cache data
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

            # Store in cache with LRU eviction
            ComputeMeshFeatures._feature_cache[cache_key] = cache_data

            # Evict oldest entries if cache exceeds max size
            while len(ComputeMeshFeatures._feature_cache) > MAX_CACHE_SIZE:
                evicted_key, _ = ComputeMeshFeatures._feature_cache.popitem(last=False)
                print(f"[Compute Features] Cache full, evicted oldest entry: {evicted_key[:32]}...")

            return (cache_data,)

        except Exception as e:
            print(f"[Compute Features] Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _generate_cache_key(mesh, all_points, point_num, seed):
        """Generate unique cache key for mesh + parameters."""
        # Hash mesh vertices and faces
        vertices_hash = hashlib.md5(mesh.vertices.tobytes()).hexdigest()[:16]
        faces_hash = hashlib.md5(mesh.faces.tobytes()).hexdigest()[:16]
        if all_points:
            params_hash = "all_points"
        else:
            params_hash = f"{point_num}_{seed}"
        return f"{vertices_hash}_{faces_hash}_{params_hash}"


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
            import folder_paths
            import os
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

            # Load model from config
            p3sam = _get_p3sam_model(p3sam_config)
            device = comfy.model_management.get_torch_device()
            p3sam = p3sam.to(device).eval()

            # Import functions from core
            from .p3sam_processing import (
                get_mask,
                cal_iou,
                fix_label,
                get_aabb_from_face_ids,
                do_post_process
            )
            from .mesh_utils import colorize_segmentation, save_mesh

            # FPS sample prompt points
            fps_idx = fpsample.fps_sampling(points, prompt_num)
            _point_prompts = points[fps_idx]

            # Get masks (inference step)
            bs = prompt_bs
            step_num = prompt_num // bs + 1
            mask_res = []
            iou_res = []
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

            # Auto-unload P3-SAM if cache_on_gpu is False
            if not p3sam_config.get('cache_on_gpu', True):
                print("[P3-SAM Segment] Auto-unloading P3-SAM model from GPU...")
                p3sam.to("cpu")
                comfy.model_management.soft_empty_cache()

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


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComputeMeshFeatures": ComputeMeshFeatures,
    "P3SAMSegmentMesh": P3SAMSegmentMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComputeMeshFeatures": "Compute Mesh Features",
    "P3SAMSegmentMesh": "P3-SAM Segment Mesh",
}

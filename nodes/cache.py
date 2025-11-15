"""
Feature Caching Nodes for Hunyuan3D-Part.

Implements caching for expensive preprocessing operations to speed up
repeated runs on the same mesh.
"""

import torch
import numpy as np
import trimesh
import hashlib

from .core.mesh_utils import load_mesh


class CacheMeshFeatures:
    """
    Cache mesh preprocessing results for faster subsequent runs.

    Caches:
    - Adjacent faces computation (~3.4s)
    - Sonata feature extraction (~15s)

    Total savings: ~18 seconds on cache hit.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "sonata_model": ("MODEL",),
                "point_num": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample. Must match segmentation settings."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "tooltip": "Random seed for point sampling. Must match segmentation settings."
                }),
            },
        }

    RETURN_TYPES = ("FEATURE_CACHE",)
    RETURN_NAMES = ("feature_cache",)
    FUNCTION = "cache_features"
    CATEGORY = "Hunyuan3D/Optimization"

    # Global cache storage
    _feature_cache = {}

    def cache_features(self, mesh, sonata_model, point_num, seed):
        """Cache mesh features."""
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
            cache_key = self._generate_cache_key(mesh_obj, point_num, seed)

            # Check if already cached
            if cache_key in CacheMeshFeatures._feature_cache:
                print(f"[Cache Features] ✓ Using cached features (saves ~18s)")
                cached_data = CacheMeshFeatures._feature_cache[cache_key]
                return (cached_data,)

            print(f"[Cache Features] Computing features (first time for this mesh+params)...")

            # Extract Sonata model
            sonata = sonata_model["model"]
            sonata = sonata.to(self.device).eval()

            # Import required functions from core
            from .core.p3sam_processing import (
                load_mesh_file,
                build_adjacent_faces_numba,
                sample_point_cloud,
                get_feat
            )

            # Load and process mesh (includes adjacent faces computation)
            print(f"[Cache Features] Loading mesh and computing adjacent faces...")
            mesh_loaded, adjacent_faces = load_mesh_file(mesh_obj, clean_mesh_flag=True)

            # Sample point cloud
            print(f"[Cache Features] Sampling {point_num} points...")
            sampled_mesh = sample_point_cloud(mesh_loaded, point_num, seed)

            # Extract features using Sonata
            print(f"[Cache Features] Extracting features with Sonata...")
            points = np.asarray(sampled_mesh.vertices, dtype=np.float32)
            normals = np.asarray(sampled_mesh.vertex_normals, dtype=np.float32)

            # Get features
            import time
            t0 = time.time()
            feats = get_feat(sonata, points, normals)
            feat_time = time.time() - t0
            print(f"[Cache Features] ✓ Feature extraction complete ({feat_time:.2f}s)")

            # Prepare cache data
            cache_data = {
                'mesh': mesh_loaded,
                'sampled_mesh': sampled_mesh,
                'adjacent_faces': adjacent_faces,
                'features': feats,
                'points': points,
                'normals': normals,
                'point_num': point_num,
                'seed': seed,
            }

            # Store in cache
            CacheMeshFeatures._feature_cache[cache_key] = cache_data
            print(f"[Cache Features] ✓ Features cached for future use")

            return (cache_data,)

        except Exception as e:
            print(f"[Cache Features] Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _generate_cache_key(mesh, point_num, seed):
        """Generate unique cache key for mesh + parameters."""
        # Hash mesh vertices and faces
        vertices_hash = hashlib.md5(mesh.vertices.tobytes()).hexdigest()[:16]
        faces_hash = hashlib.md5(mesh.faces.tobytes()).hexdigest()[:16]
        params_hash = f"{point_num}_{seed}"
        return f"{vertices_hash}_{faces_hash}_{params_hash}"


class P3SAMSegmentMeshCached:
    """
    P3-SAM segmentation that uses cached features.

    Requires feature cache from CacheMeshFeatures node.
    Significantly faster than regular segmentation on cache hit (~18s savings).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature_cache": ("FEATURE_CACHE",),
                "p3sam_model": ("MODEL",),
                "prompt_num": ("INT", {
                    "default": 400,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Number of prompt points."
                }),
                "prompt_bs": ("INT", {
                    "default": 4,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Prompt batch size."
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
    FUNCTION = "segment_cached"
    CATEGORY = "Hunyuan3D/Optimization"

    def segment_cached(self, feature_cache, p3sam_model, prompt_num, prompt_bs, threshold, post_process):
        """Segment using cached features."""
        try:
            import folder_paths
            import os

            # Extract cached data
            mesh_loaded = feature_cache['mesh']
            sampled_mesh = feature_cache['sampled_mesh']
            adjacent_faces = feature_cache['adjacent_faces']
            feats = feature_cache['features']
            points = feature_cache['points']
            normals = feature_cache['normals']
            point_num = feature_cache['point_num']
            seed = feature_cache['seed']

            print(f"[P3-SAM Cached] Using cached features (saved ~18s)")
            print(f"[P3-SAM Cached] Running segmentation with {prompt_num} prompts...")

            # Extract P3-SAM model
            p3sam = p3sam_model["model"]
            p3sam = p3sam.to(self.device).eval()
            p3sam_parallel = torch.nn.DataParallel(p3sam)

            # Import segmentation functions from core
            from .core.p3sam_processing import (
                fps_sample_prompt_points,
                get_mask,
                sort_by_iou,
                nms,
                filter_single_mask_clusters,
                merge_again,
                calculate_points_without_mask,
                missed_masks,
                calculate_final_point_cloud_masks,
                color_point_cloud,
                project_mesh_and_count_labels,
                merge_mesh,
                fix_face_ids,
                calculate_connected_regions,
                sort_connected_regions,
                remove_small_area_regions,
                remove_labels_from_small_areas,
                add_large_missing_parts,
                assign_new_face_ids,
                calculate_part_and_label_aabb,
                calculate_part_neighbors,
                merge_no_mask_regions,
                calculate_final_aabb
            )
            from .core.mesh_utils import colorize_segmentation, save_mesh

            # FPS sample prompt points
            prompt_points_indices = fps_sample_prompt_points(points, prompt_num, seed)

            # Get masks (inference step)
            all_masks, all_ious = get_mask(
                [p3sam, p3sam_parallel],
                feats,
                points,
                prompt_points_indices,
                prompt_bs=prompt_bs
            )

            # Post-process masks
            sorted_indices = sort_by_iou(all_ious)
            nms_masks = nms(all_masks, sorted_indices, threshold=threshold)

            # Continue with full post-processing pipeline
            nms_masks = filter_single_mask_clusters(nms_masks)
            nms_masks = merge_again(nms_masks, threshold=threshold)

            points_without_mask = calculate_points_without_mask(nms_masks, points)
            if len(points_without_mask) > 0:
                nms_masks = missed_masks(
                    nms_masks,
                    all_masks,
                    sorted_indices,
                    points_without_mask,
                    threshold=threshold
                )

            point_labels = calculate_final_point_cloud_masks(nms_masks, len(points))
            colored_pc = color_point_cloud(sampled_mesh, point_labels, seed)

            # Project to mesh
            face_ids = project_mesh_and_count_labels(
                mesh_loaded,
                adjacent_faces,
                colored_pc,
                point_labels
            )

            if post_process:
                # Full post-processing
                merged_mesh, merged_labels = merge_mesh(mesh_loaded, face_ids)
                face_ids = fix_face_ids(merged_mesh, merged_labels, mesh_loaded)

                # Calculate connected regions
                connected_meshes, connected_labels = calculate_connected_regions(
                    mesh_loaded,
                    face_ids
                )

                # Sort and filter
                connected_meshes, connected_labels = sort_connected_regions(
                    connected_meshes,
                    connected_labels
                )
                connected_meshes, connected_labels = remove_small_area_regions(
                    connected_meshes,
                    connected_labels,
                    min_area_ratio=0.005
                )

                face_ids = remove_labels_from_small_areas(
                    mesh_loaded,
                    face_ids,
                    connected_labels
                )

                # Second merge
                merged_mesh, merged_labels = merge_mesh(mesh_loaded, face_ids)
                face_ids = fix_face_ids(merged_mesh, merged_labels, mesh_loaded)

                # Recalculate connected regions
                connected_meshes, connected_labels = calculate_connected_regions(
                    mesh_loaded,
                    face_ids
                )

                # Add missing parts
                face_ids = add_large_missing_parts(
                    mesh_loaded,
                    face_ids,
                    connected_labels
                )

                # Assign final IDs
                face_ids = assign_new_face_ids(face_ids, connected_labels)

                # Calculate AABBs
                part_aabb, label_aabb = calculate_part_and_label_aabb(
                    mesh_loaded,
                    face_ids
                )

                # Calculate neighbors
                neighbors = calculate_part_neighbors(
                    mesh_loaded,
                    face_ids,
                    part_aabb
                )

                # Final merge of no-mask regions
                face_ids = merge_no_mask_regions(face_ids, neighbors)
                face_ids = assign_new_face_ids(face_ids, connected_labels)

                # Final AABB
                aabb = calculate_final_aabb(mesh_loaded, face_ids)
            else:
                # Simple AABB without post-processing
                aabb = calculate_part_and_label_aabb(mesh_loaded, face_ids)[0]

            print(f"[P3-SAM Cached] Segmentation complete: found {len(aabb)} parts")

            # Add metadata
            processed_mesh = mesh_loaded
            processed_mesh.metadata['face_part_ids'] = face_ids
            processed_mesh.metadata['part_bboxes'] = aabb
            processed_mesh.metadata['num_parts'] = len(aabb)

            # Create colored visualization
            colored_mesh = colorize_segmentation(processed_mesh, face_ids, seed=seed)

            # Save preview
            output_dir = folder_paths.get_output_directory()
            preview_path = os.path.join(output_dir, f"p3sam_cached_{seed}.glb")
            save_mesh(colored_mesh, preview_path)
            print(f"[P3-SAM Cached] Saved preview to: {preview_path}")

            # Prepare outputs
            bboxes_output = {
                'bboxes': aabb,
                'num_parts': len(aabb)
            }

            face_ids_output = {
                'face_ids': face_ids,
                'num_parts': len(np.unique(face_ids))
            }

            return (processed_mesh, bboxes_output, face_ids_output, preview_path)

        except Exception as e:
            print(f"[P3-SAM Cached] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "CacheMeshFeatures": CacheMeshFeatures,
    "P3SAMSegmentMeshCached": P3SAMSegmentMeshCached,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CacheMeshFeatures": "Cache Mesh Features",
    "P3SAMSegmentMeshCached": "P3-SAM Segment (Cached)",
}

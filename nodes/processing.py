"""
Processing Nodes for Hunyuan3D-Part.

X-Part generation node that accepts model inputs
instead of loading models internally.
"""

import torch
import numpy as np
import trimesh
import folder_paths
import os
import tempfile

# Import utilities from core
from .core.mesh_utils import load_mesh, save_mesh, colorize_segmentation, get_temp_mesh_path
from .core.models.diffusion.schedulers import FlowMatchEulerDiscreteScheduler


class XPartGenerateParts:
    """
    Generate high-quality part meshes using X-Part.

    Takes mesh_with_features (from ComputeMeshFeatures with all_points=True),
    bounding boxes, and combined X-Part models as inputs.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_with_features": ("MESH_FEATURES",),  # REQUIRED - use ComputeMeshFeatures with all_points=True
                "bounding_boxes": ("BBOXES_3D",),
                "xpart_models": ("XPART_MODELS",),
                "octree_resolution": ("INT", {
                    "default": 256,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Mesh quality. 256=~8GB, 512=~12-16GB, 1024=~24GB+ VRAM"
                }),
                "num_inference_steps": ("INT", {
                    "default": 25,  # Changed from 50 to 25 as requested
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

    RETURN_TYPES = ("TRIMESH", "TRIMESH", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "bbox_viz", "parts_path", "bbox_path")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D/Processing"

    def generate(self, mesh_with_features, bounding_boxes, xpart_models, octree_resolution, num_inference_steps,
                guidance_scale, seed, num_chunks, output_coordinate_system):
        """Generate part meshes."""
        try:
            # Extract mesh and Sonata features from mesh_with_features cache
            mesh_obj = mesh_with_features['mesh']
            sonata_features = mesh_with_features['features']  # Pre-computed Sonata features
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

            # Extract models from combined input
            dit = xpart_models["dit"]
            vae = xpart_models["vae"]
            conditioner = xpart_models["conditioner"]
            dtype_str = xpart_models.get("dtype", "float32")
            cache_on_gpu = xpart_models.get("cache_on_gpu", True)

            dtype = torch.float32 if dtype_str == "float32" else torch.float16

            print(f"[X-Part Generate] Using {dtype_str} precision")

            # Ensure models on GPU
            dit = dit.to(self.device).eval()
            vae = vae.to(self.device).eval()
            conditioner = conditioner.to(self.device).eval()

            # Import from core
            from .core.xpart_pipeline import PartFormerPipeline
            from omegaconf import OmegaConf
            from pathlib import Path

            # Load config for scheduler
            config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
            config = OmegaConf.load(str(config_path))

            # Create scheduler directly
            scheduler_params = dict(config["scheduler"]["params"])
            scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_params)

            # Create minimal pipeline instance
            pipeline = PartFormerPipeline(
                vae=vae,
                model=dit,
                scheduler=scheduler,
                conditioner=conditioner,
                bbox_predictor=None,  # Not used when aabb is provided
                device=self.device,
                dtype=dtype,
                verbose=True  # Required to return full tuple format
            )

            print(f"[X-Part Generate] Running generation...")
            print(f"  - Octree resolution: {octree_resolution}")
            print(f"  - Inference steps: {num_inference_steps}")
            print(f"  - Guidance scale: {guidance_scale}")
            print(f"  - Seed: {seed}")

            # Convert aabb to tensor if provided
            if aabb is not None:
                aabb_tensor = torch.from_numpy(aabb).to(self.device).float()
            else:
                aabb_tensor = None

            # Prepare precomputed Sonata features for conditioner
            precomputed_sonata = {
                'features': torch.as_tensor(sonata_features).to(self.device).float(),
                'points': torch.as_tensor(sonata_points).to(self.device).float(),
                'normals': torch.as_tensor(sonata_normals).to(self.device).float(),
            }

            # Run generation with precomputed Sonata features
            obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
                mesh_path=mesh_path,
                aabb=aabb_tensor,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_chunks=num_chunks,
                output_type="trimesh",
                precomputed_sonata_features=precomputed_sonata  # Pass pre-computed features
            )

            print(f"[X-Part Generate] Generation complete!")

            # Apply coordinate system transformation if requested
            if output_coordinate_system == "Z-up":
                print("[X-Part Generate] Converting output to Z-up coordinate system...")
                # Y-up to Z-up: rotate -90° around X axis (trimesh row-major convention)
                # This transforms: x' = x, y' = z, z' = -y
                rotation_matrix = np.array([
                    [1,  0,  0, 0],
                    [0,  0,  1, 0],
                    [0, -1,  0, 0],
                    [0,  0,  0, 1]
                ])
                # Apply to scene meshes
                if hasattr(obj_mesh, 'geometry'):  # It's a Scene
                    for key in obj_mesh.geometry.keys():
                        obj_mesh.geometry[key].apply_transform(rotation_matrix)
                else:  # Single Trimesh
                    obj_mesh.apply_transform(rotation_matrix)

                if hasattr(out_bbox, 'geometry'):
                    for key in out_bbox.geometry.keys():
                        out_bbox.geometry[key].apply_transform(rotation_matrix)
                else:
                    out_bbox.apply_transform(rotation_matrix)
                print("[X-Part Generate] ✓ Converted to Z-up")

            # Save outputs
            output_dir = folder_paths.get_output_directory()

            parts_path = os.path.join(output_dir, f"xpart_parts_{seed}.glb")
            save_mesh(obj_mesh, parts_path)
            print(f"[X-Part Generate] Saved parts to: {parts_path}")

            bbox_path = os.path.join(output_dir, f"xpart_bbox_{seed}.glb")
            save_mesh(out_bbox, bbox_path)
            print(f"[X-Part Generate] Saved bbox viz to: {bbox_path}")

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
                print("[X-Part Generate] ✓ Models unloaded, VRAM freed")

            return (obj_mesh, out_bbox, parts_path, bbox_path)

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

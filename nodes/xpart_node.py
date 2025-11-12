"""
X-Part Generation Node for ComfyUI.
Generates high-fidelity 3D parts using X-Part diffusion model.
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
import folder_paths
import os
import tempfile

# Import utilities from parent package using relative imports
from ..node_utils.mesh_utils import load_mesh, save_mesh, get_temp_mesh_path
from ..node_utils.model_loader import ModelCache


class Hunyuan3D_XPart_Generation:
    """
    ComfyUI node for X-Part 3D part generation.

    Takes a mesh and bounding boxes, generates high-quality part meshes
    using diffusion-based generation.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),  # Accept TRIMESH objects from GeometryPack or other nodes
                "bounding_boxes": ("BBOXES_3D",),  # Accept BBOXES_3D from P3-SAM
                "octree_resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "display": "number"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "display": "number"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "TRIMESH", "TRIMESH", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "exploded_view", "bbox_visualization", "parts_path", "exploded_path", "bbox_path")
    FUNCTION = "generate_parts"
    CATEGORY = "Hunyuan3D"

    def generate_parts(self, mesh, bounding_boxes, octree_resolution,
                      num_inference_steps, guidance_scale, seed):
        """
        Generate high-quality part meshes using X-Part.

        Args:
            mesh: trimesh.Trimesh object or file path
            bounding_boxes: BBOXES_3D dict or "auto" for auto-detection
            octree_resolution: Resolution for marching cubes
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            Tuple of (part_meshes, exploded_view, bbox_viz, parts_path, exploded_path, bbox_path)
        """
        try:
            # Load mesh - handles file paths, trimesh objects, and custom types
            mesh_path = None
            if isinstance(mesh, dict) and 'trimesh' in mesh:
                # Handle custom MESH_3D dict type
                mesh_obj = mesh['trimesh']
                # Save to temp file for pipeline
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)
                print(f"[X-Part] Using trimesh object from MESH_3D dict, saved to temp: {mesh_path}")
            elif isinstance(mesh, str):
                # Handle file path string
                print(f"[X-Part] Loading mesh from file: {mesh}")
                mesh_obj = load_mesh(mesh)
                mesh_path = mesh
            elif isinstance(mesh, trimesh.Trimesh):
                # Handle raw trimesh object (from GeometryPack, etc.)
                mesh_obj = mesh
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)
                print(f"[X-Part] Using raw trimesh object (from GeometryPack or similar), saved to temp: {mesh_path}")
            else:
                # Try to handle other types
                print(f"[X-Part] Attempting to load mesh from type: {type(mesh)}")
                mesh_obj = load_mesh(mesh)
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)

            # Handle bounding boxes
            aabb = None
            if isinstance(bounding_boxes, str) and bounding_boxes.lower() == "auto":
                print(f"[X-Part] Auto-detecting bounding boxes")
                aabb = None  # Let pipeline auto-detect
            elif isinstance(bounding_boxes, dict) and 'bboxes' in bounding_boxes:
                aabb = bounding_boxes['bboxes']
                print(f"[X-Part] Using provided bounding boxes: {len(aabb)} parts")
            else:
                print(f"[X-Part] No bounding boxes provided, will auto-detect")
                aabb = None

            # Get X-Part pipeline
            print(f"[X-Part] Initializing pipeline on {self.device}...")
            pipeline = ModelCache.get_xpart_pipeline(
                device=self.device,
                dtype=self.dtype
            )

            # Convert aabb to torch tensor if provided
            if aabb is not None:
                aabb_tensor = torch.from_numpy(aabb).to(self.device).float()
            else:
                aabb_tensor = None

            # Run generation
            print(f"[X-Part] Running part generation...")
            print(f"  - Octree resolution: {octree_resolution}")
            print(f"  - Inference steps: {num_inference_steps}")
            print(f"  - Guidance scale: {guidance_scale}")
            print(f"  - Seed: {seed}")

            obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
                mesh_path=mesh_path,
                aabb=aabb_tensor,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_type="trimesh"
            )

            print(f"[X-Part] Generation complete!")

            # Save outputs to output folder
            output_dir = folder_paths.get_output_directory()

            # Save part meshes
            parts_filename = f"xpart_parts_{seed}.glb"
            parts_path = os.path.join(output_dir, parts_filename)
            save_mesh(obj_mesh, parts_path)
            print(f"[X-Part] Saved part meshes to: {parts_path}")

            # Save exploded view
            exploded_filename = f"xpart_exploded_{seed}.glb"
            exploded_path = os.path.join(output_dir, exploded_filename)
            save_mesh(explode_object, exploded_path)
            print(f"[X-Part] Saved exploded view to: {exploded_path}")

            # Save bbox visualization
            bbox_filename = f"xpart_bbox_{seed}.glb"
            bbox_path = os.path.join(output_dir, bbox_filename)
            save_mesh(out_bbox, bbox_path)
            print(f"[X-Part] Saved bbox visualization to: {bbox_path}")

            # Clean up temp file if we created one
            if mesh_path and mesh_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(mesh_path)
                except:
                    pass

            # Return raw trimesh objects for compatibility with GeometryPack
            return (obj_mesh, explode_object, out_bbox,
                   parts_path, exploded_path, bbox_path)

        except Exception as e:
            print(f"[X-Part] Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class Hunyuan3D_FullPipeline:
    """
    Combined node that runs both P3-SAM and X-Part in sequence.
    Convenience node for users who want the full pipeline in one step.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1
                }),
                "octree_resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "BBOXES_3D", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "bounding_boxes", "parts_path", "segmentation_path")
    FUNCTION = "run_full_pipeline"
    CATEGORY = "Hunyuan3D"

    def run_full_pipeline(self, mesh, seed, octree_resolution, num_inference_steps):
        """
        Run the complete Hunyuan3D-Part pipeline.

        Args:
            mesh: Input mesh
            seed: Random seed
            octree_resolution: Resolution for part generation
            num_inference_steps: Diffusion steps

        Returns:
            Tuple of (part_meshes, bboxes, parts_path, seg_path)
        """
        try:
            print(f"[Full Pipeline] Starting complete Hunyuan3D-Part pipeline...")

            # Step 1: P3-SAM Segmentation
            from .p3sam_node import Hunyuan3D_P3SAM_Segmentation
            p3sam = Hunyuan3D_P3SAM_Segmentation()

            bboxes_output, face_ids, mesh_output, seg_path = p3sam.segment_mesh(
                mesh=mesh,
                seed=seed,
                point_num=100000,
                prompt_num=400,
                threshold=0.95,
                post_process=True
            )

            print(f"[Full Pipeline] P3-SAM complete, found {bboxes_output['num_parts']} parts")

            # Step 2: X-Part Generation
            xpart = Hunyuan3D_XPart_Generation()

            parts_output, exploded, bbox_viz, parts_path, exploded_path, bbox_path = xpart.generate_parts(
                mesh=mesh_output,
                bounding_boxes=bboxes_output,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=-1.0,
                seed=seed
            )

            print(f"[Full Pipeline] X-Part complete, generated parts saved to {parts_path}")
            print(f"[Full Pipeline] Full pipeline completed successfully!")

            return (parts_output, bboxes_output, parts_path, seg_path)

        except Exception as e:
            print(f"[Full Pipeline] Error during pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Hunyuan3D_XPart_Generation": Hunyuan3D_XPart_Generation,
    "Hunyuan3D_FullPipeline": Hunyuan3D_FullPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D_XPart_Generation": "X-Part Generation",
    "Hunyuan3D_FullPipeline": "Hunyuan3D Full Pipeline",
}

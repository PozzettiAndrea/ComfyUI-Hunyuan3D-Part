"""
P3-SAM Segmentation Node for ComfyUI.
Segments 3D meshes into parts using P3-SAM (Native 3D Part Segmentation).
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
import folder_paths
import os

# Import utilities from parent package using relative imports
from ..node_utils.mesh_utils import load_mesh, save_mesh, colorize_segmentation, create_bbox_visualization
from ..node_utils.model_loader import ModelCache


class Hunyuan3D_P3SAM_Segmentation:
    """
    ComfyUI node for P3-SAM 3D part segmentation.

    Takes a 3D mesh and segments it into parts, outputting bounding boxes
    and segmentation masks.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),  # Accept TRIMESH objects from GeometryPack or other nodes
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,  # 2^32 - 1 (numpy random seed limit)
                    "step": 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducibility. Same seed = same results."
                }),
                "point_num": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "display": "number",
                    "tooltip": "Number of points sampled from mesh surface. Higher = better quality but slower. Official default: 100k"
                }),
                "prompt_num": ("INT", {
                    "default": 400,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "display": "number",
                    "tooltip": "Number of prompt points for segmentation. Higher = better part separation, especially for organic shapes. Official default: 400"
                }),
                "prompt_bs": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Prompt batch size. Higher = MORE VRAM usage but faster inference. Lower = less VRAM. Official default: 32"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.7,
                    "max": 0.999,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Post-processing merge threshold. Higher = fewer but larger parts (more merging). Lower = more but smaller parts."
                }),
                "post_process": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable connectivity-based post-processing. Recommended: True for cleaner results."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM after inference. False = move to CPU and clear VRAM (saves ~14GB). True = keep on GPU for faster subsequent runs."
                }),
            },
        }

    RETURN_TYPES = ("BBOXES_3D", "FACE_IDS", "TRIMESH", "STRING")
    RETURN_NAMES = ("bounding_boxes", "face_ids", "mesh", "segmentation_preview_path")
    FUNCTION = "segment_mesh"
    CATEGORY = "Hunyuan3D"

    def segment_mesh(self, mesh, seed, point_num, prompt_num, prompt_bs, threshold, post_process, cache_model):
        """
        Segment a 3D mesh into parts using P3-SAM.

        Args:
            mesh: trimesh.Trimesh object or file path string
            seed: Random seed for reproducibility
            point_num: Number of sampling points
            prompt_num: Number of prompts for segmentation
            prompt_bs: Batch size for prompt processing
            threshold: Segmentation threshold
            post_process: Enable connectivity post-processing

        Returns:
            Tuple of (bboxes, face_ids, mesh, preview_path)
        """
        try:
            # Load mesh - handles file paths, trimesh objects, and custom types
            if isinstance(mesh, dict) and 'trimesh' in mesh:
                # Handle custom MESH_3D dict type
                mesh_obj = mesh['trimesh']
                print(f"[P3-SAM] Using trimesh object from MESH_3D dict")
            elif isinstance(mesh, str):
                # Handle file path string
                print(f"[P3-SAM] Loading mesh from file: {mesh}")
                mesh_obj = load_mesh(mesh)
            elif isinstance(mesh, trimesh.Trimesh):
                # Handle raw trimesh object (from GeometryPack, etc.)
                mesh_obj = mesh
                print(f"[P3-SAM] Using raw trimesh object (from GeometryPack or similar)")
            else:
                # Try to handle other types via load_mesh
                print(f"[P3-SAM] Attempting to load mesh from type: {type(mesh)}")
                mesh_obj = load_mesh(mesh)

            # Get P3-SAM model
            print(f"[P3-SAM] Initializing model...")
            automask = ModelCache.get_p3sam(
                ckpt_path=None,  # Auto-download from HF
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                post_process=post_process
            )

            # Ensure model is on GPU (reload if it was offloaded)
            automask.to_cuda()

            # Run segmentation
            print(f"[P3-SAM] Running segmentation with seed={seed}...")
            aabb, face_ids, processed_mesh = automask.predict_aabb(
                mesh_obj,
                seed=seed,
                post_process=post_process,
                prompt_bs=prompt_bs
            )

            print(f"[P3-SAM] Segmentation complete: found {len(aabb)} parts")

            # VRAM management: offload model if cache_model=False
            if not cache_model:
                print(f"[P3-SAM] Offloading model to CPU and clearing VRAM...")
                automask.to_cpu()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[P3-SAM] Model offloaded. VRAM freed.")
            else:
                print(f"[P3-SAM] Keeping model on GPU for faster subsequent runs.")

            # Add face_ids as a field on the trimesh object
            processed_mesh.metadata['face_part_ids'] = face_ids
            processed_mesh.metadata['part_bboxes'] = aabb
            processed_mesh.metadata['num_parts'] = len(aabb)
            print(f"[P3-SAM] Added segmentation data to mesh metadata")

            # Create colored segmentation visualization
            colored_mesh = colorize_segmentation(processed_mesh, face_ids, seed=seed)

            # Save preview to output folder
            output_dir = folder_paths.get_output_directory()
            preview_path = os.path.join(output_dir, f"p3sam_segmentation_{seed}.glb")
            save_mesh(colored_mesh, preview_path)
            print(f"[P3-SAM] Saved segmentation preview to: {preview_path}")

            # Prepare outputs
            bboxes_output = {
                'bboxes': aabb,
                'num_parts': len(aabb)
            }

            face_ids_output = {
                'face_ids': face_ids,
                'num_parts': len(np.unique(face_ids))
            }

            # Return raw trimesh object for compatibility with GeometryPack
            # Metadata is already stored in processed_mesh.metadata
            return (bboxes_output, face_ids_output, processed_mesh, preview_path)

        except Exception as e:
            print(f"[P3-SAM] Error during segmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class LoadMesh:
    """
    Helper node to load mesh files and convert them to MESH_3D type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get input directory
        input_dir = folder_paths.get_input_directory()

        # Recursively scan for mesh files
        mesh_extensions = {".obj", ".glb", ".gltf", ".ply", ".stl", ".fbx"}
        mesh_files = []

        # Walk through input directory recursively
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in mesh_extensions):
                    # Get relative path from input directory
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, input_dir)
                    mesh_files.append(rel_path)

        # Sort mesh files
        mesh_files.sort()

        # If no files found, provide empty string as default
        if not mesh_files:
            mesh_files = [""]

        return {
            "required": {
                "mesh": (mesh_files, {
                    "default": mesh_files[0] if mesh_files else ""
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D"

    @classmethod
    def VALIDATE_INPUTS(cls, mesh):
        """Validate mesh input - accept both relative and absolute paths for backward compatibility."""
        # Allow empty string
        if not mesh:
            return True

        # If it's an absolute path, check if file exists
        if os.path.isabs(mesh):
            if os.path.exists(mesh):
                return True
            return f"Mesh file not found: {mesh}"

        # If it's a relative path, check in input directory
        input_dir = folder_paths.get_input_directory()
        full_path = os.path.join(input_dir, mesh)
        if os.path.exists(full_path):
            return True

        return f"Mesh file not found: {mesh}"

    def load(self, mesh):
        """Load a mesh file and return as TRIMESH type."""
        try:
            # Handle both absolute and relative paths for backward compatibility
            if os.path.isabs(mesh):
                # Absolute path from old workflows - use directly
                mesh_path = mesh
            else:
                # Relative path - resolve using ComfyUI's annotation system
                mesh_path = folder_paths.get_annotated_filepath(mesh)

            print(f"[LoadMesh] Loading mesh from: {mesh_path}")
            mesh_obj = load_mesh(mesh_path)

            print(f"[LoadMesh] Loaded mesh with {len(mesh_obj.vertices)} vertices, {len(mesh_obj.faces)} faces")
            # Return raw trimesh object for compatibility with GeometryPack
            return (mesh_obj,)

        except Exception as e:
            print(f"[LoadMesh] Error loading mesh: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class SaveMesh:
    """
    Helper node to save TRIMESH objects to files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename": ("STRING", {
                    "default": "output.glb",
                    "multiline": False
                }),
                "format": (["glb", "obj", "ply", "stl"], {
                    "default": "glb"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "Hunyuan3D"
    OUTPUT_NODE = True

    def save(self, mesh, filename, format):
        """Save a mesh to the output folder."""
        try:
            # Ensure correct extension
            if not filename.endswith(f".{format}"):
                filename = f"{filename.rsplit('.', 1)[0]}.{format}"

            # Save to output directory
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, filename)

            save_mesh(mesh, output_path, file_format=format)
            print(f"[SaveMesh] Saved mesh to: {output_path}")

            return (output_path,)

        except Exception as e:
            print(f"[SaveMesh] Error saving mesh: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Hunyuan3D_P3SAM_Segmentation": Hunyuan3D_P3SAM_Segmentation,
    "LoadMesh": LoadMesh,
    "SaveMesh": SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D_P3SAM_Segmentation": "P3-SAM Segmentation",
    "LoadMesh": "Load 3D Mesh",
    "SaveMesh": "Save 3D Mesh",
}

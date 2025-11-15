"""
Processing Nodes for Hunyuan3D-Part.

Refactored P3-SAM and X-Part nodes that accept model inputs
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


class P3SAMSegmentMesh:
    """
    Segment a 3D mesh using P3-SAM.

    Takes mesh and P3-SAM model as inputs, outputs bounding boxes and segmentation.
    Requires Sonata model for feature extraction.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "sonata_model": ("MODEL",),
                "p3sam_model": ("MODEL",),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducibility."
                }),
                "point_num": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points sampled from mesh. Higher = better quality but slower."
                }),
                "prompt_num": ("INT", {
                    "default": 400,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Number of prompt points for segmentation. Higher = better separation."
                }),
                "prompt_bs": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Prompt batch size. Higher = more VRAM but faster."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.7,
                    "max": 0.999,
                    "step": 0.01,
                    "tooltip": "Merge threshold. Higher = fewer but larger parts."
                }),
                "post_process": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable connectivity-based post-processing."
                }),
            },
        }

    RETURN_TYPES = ("BBOXES_3D", "FACE_IDS", "TRIMESH", "STRING")
    RETURN_NAMES = ("bounding_boxes", "face_ids", "mesh", "preview_path")
    FUNCTION = "segment"
    CATEGORY = "Hunyuan3D/Processing"

    def segment(self, mesh, sonata_model, p3sam_model, seed, point_num, prompt_num, prompt_bs, threshold, post_process):
        """Segment mesh into parts."""
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

            # Extract models from MODEL dicts
            sonata = sonata_model["model"]
            p3sam = p3sam_model["model"]

            # Ensure models are on GPU and in eval mode
            sonata = sonata.to(self.device).eval()
            p3sam = p3sam.to(self.device).eval()

            # Import mesh_sam from core
            from .core.p3sam_processing import mesh_sam

            print(f"[P3-SAM Segment] Running segmentation with seed={seed}...")

            # Run segmentation using the mesh_sam function
            # We need to wrap models similar to AutoMask
            p3sam_parallel = torch.nn.DataParallel(p3sam)

            aabb, face_ids, processed_mesh = mesh_sam(
                [p3sam, p3sam_parallel],
                mesh_obj,
                save_path=None,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                post_process=post_process,
                show_info=True,
                save_mid_res=False,
                clean_mesh_flag=True,
                seed=seed,
                prompt_bs=prompt_bs,
            )

            print(f"[P3-SAM Segment] Segmentation complete: found {len(aabb)} parts")

            # Add metadata
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
                'num_parts': len(np.unique(face_ids))
            }

            return (bboxes_output, face_ids_output, processed_mesh, preview_path)

        except Exception as e:
            print(f"[P3-SAM Segment] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class XPartGenerateParts:
    """
    Generate high-quality part meshes using X-Part.

    Takes mesh, bounding boxes, and all X-Part model components as inputs.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "bounding_boxes": ("BBOXES_3D",),
                "dit_model": ("MODEL",),
                "vae_model": ("MODEL",),
                "conditioner_model": ("MODEL",),
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
            },
        }

    RETURN_TYPES = ("TRIMESH", "TRIMESH", "TRIMESH", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "exploded_view", "bbox_viz", "parts_path", "exploded_path", "bbox_path")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D/Processing"

    def generate(self, mesh, bounding_boxes, dit_model, vae_model, conditioner_model,
                octree_resolution, num_inference_steps, guidance_scale, seed, num_chunks):
        """Generate part meshes."""
        try:
            # Load mesh if needed
            mesh_path = None
            if isinstance(mesh, dict) and 'trimesh' in mesh:
                mesh_obj = mesh['trimesh']
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)
            elif isinstance(mesh, str):
                mesh_obj = load_mesh(mesh)
                mesh_path = mesh
            elif isinstance(mesh, trimesh.Trimesh):
                mesh_obj = mesh
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)
            else:
                mesh_obj = load_mesh(mesh)
                mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
                save_mesh(mesh_obj, mesh_path)

            # Extract bounding boxes
            if isinstance(bounding_boxes, dict) and 'bboxes' in bounding_boxes:
                aabb = bounding_boxes['bboxes']
                print(f"[X-Part Generate] Using {len(aabb)} bounding boxes")
            else:
                aabb = None
                print(f"[X-Part Generate] No bounding boxes, will auto-detect")

            # Extract models
            dit = dit_model["model"]
            vae = vae_model["model"]
            conditioner = conditioner_model["model"]
            dtype_str = dit_model.get("dtype", "float32")
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
                dtype=dtype
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

            # Run generation
            obj_mesh, (out_bbox, mesh_gt_bbox, explode_object) = pipeline(
                mesh_path=mesh_path,
                aabb=aabb_tensor,
                octree_resolution=octree_resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_chunks=num_chunks,
                output_type="trimesh"
            )

            print(f"[X-Part Generate] Generation complete!")

            # Save outputs
            output_dir = folder_paths.get_output_directory()

            parts_path = os.path.join(output_dir, f"xpart_parts_{seed}.glb")
            save_mesh(obj_mesh, parts_path)
            print(f"[X-Part Generate] Saved parts to: {parts_path}")

            exploded_path = os.path.join(output_dir, f"xpart_exploded_{seed}.glb")
            save_mesh(explode_object, exploded_path)
            print(f"[X-Part Generate] Saved exploded view to: {exploded_path}")

            bbox_path = os.path.join(output_dir, f"xpart_bbox_{seed}.glb")
            save_mesh(out_bbox, bbox_path)
            print(f"[X-Part Generate] Saved bbox viz to: {bbox_path}")

            # Clean up temp file
            if mesh_path and mesh_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(mesh_path)
                except:
                    pass

            return (obj_mesh, explode_object, out_bbox, parts_path, exploded_path, bbox_path)

        except Exception as e:
            print(f"[X-Part Generate] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "P3SAMSegmentMesh": P3SAMSegmentMesh,
    "XPartGenerateParts": XPartGenerateParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "P3SAMSegmentMesh": "P3-SAM Segment Mesh",
    "XPartGenerateParts": "X-Part Generate Parts",
}

"""
Bounding Box Visualization Node for Hunyuan3D-Part

Preview mesh with bounding boxes overlaid showing segmentation results.
"""

import os
import tempfile
import uuid

import trimesh as trimesh_module

from .mesh_utils import create_bbox_visualization, export_scene_to_vtp

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except ImportError:
    COMFYUI_OUTPUT_FOLDER = None


class PreviewBoundingBoxesNode:
    """
    Preview mesh with bounding boxes using VTK.js viewer.

    Displays the original mesh with wireframe bounding boxes overlaid,
    showing the segmentation results from P3-SAM or other bbox-generating nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "bboxes": ("BBOXES_3D",),
                "line_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Thickness of bounding box wireframe lines in pixels. Rendered by VTK.js."
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_bboxes"
    CATEGORY = "Hunyuan3D/Visualization"

    def preview_bboxes(self, trimesh, bboxes, line_width):
        """
        Export mesh with bounding boxes to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object
            bboxes: BBOXES_3D dict with 'bboxes' (array [N, 2, 3]) and 'num_parts'
            line_width: Thickness of wireframe lines in pixels (for VTK.js rendering)

        Returns:
            dict: UI data for frontend widget
        """
        # Extract bboxes array from BBOXES_3D dictionary
        if isinstance(bboxes, dict):
            bboxes_array = bboxes['bboxes']
            num_parts = bboxes.get('num_parts', len(bboxes_array))
        else:
            # Fallback for raw array
            bboxes_array = bboxes
            num_parts = len(bboxes_array)

        print(f"[PreviewBoundingBoxes] Preparing preview with {num_parts} bounding boxes")
        print(f"[PreviewBoundingBoxes] Mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[PreviewBoundingBoxes] Line width: {line_width}px")

        # Create scene with mesh and bounding boxes
        # Use Path3D wireframes (rendered as lines in VTK.js)
        scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=False)

        # Generate unique filename - use VTP which supports wireframes
        filename = f"preview_bboxes_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export scene to VTP (preserves wireframes)
        try:
            # Use custom VTP exporter that preserves both mesh and wireframe lines
            export_scene_to_vtp(scene, filepath)
            print(f"[PreviewBoundingBoxes] Exported scene with {num_parts} bounding boxes to: {filepath}")
        except Exception as e:
            print(f"[PreviewBoundingBoxes] VTP export failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to STL with tubes
            print(f"[PreviewBoundingBoxes] Falling back to STL with tube meshes")
            scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=True)
            filename = filename.replace('.vtp', '.stl')
            filepath = filepath.replace('.vtp', '.stl')
            scene.export(filepath, file_type='stl')
            print(f"[PreviewBoundingBoxes] Exported to: {filepath}")

        # Calculate bounding box info for camera setup
        # Use the original mesh bounds for camera positioning
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "num_bboxes": [num_parts],
            "line_width": [float(line_width)],
        }

        print(f"[PreviewBoundingBoxes] Created visualization with {num_parts} bounding boxes")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "Hunyuan3DPreviewBoundingBoxes": PreviewBoundingBoxesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DPreviewBoundingBoxes": "Preview Bounding Boxes",
}

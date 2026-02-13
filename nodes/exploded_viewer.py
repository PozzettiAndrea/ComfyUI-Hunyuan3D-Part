"""
Exploded Mesh Viewer Node for Hunyuan3D-Part.

Provides an interactive 3D viewer with explosion slider to separate mesh parts.
"""

import torch
import numpy as np
import trimesh
import folder_paths
import os
import uuid
import tempfile

from .core.mesh_utils import load_mesh


class ExplodedMeshViewer:
    """
    Interactive exploded view of segmented mesh parts.

    Takes a segmented mesh (with face_ids) and creates an interactive
    Three.js viewer with a slider to control part separation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "part_meshes": ("TRIMESH",),
            },
            "optional": {
                "face_ids": ("FACE_IDS",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "create_exploded_view"
    CATEGORY = "Hunyuan3D/Visualization"

    def create_exploded_view(self, part_meshes, face_ids=None):
        """
        Create exploded mesh visualization.

        Args:
            part_meshes: List of trimesh.Trimesh parts, single Trimesh, or Scene
            face_ids: (optional) numpy array mapping each face to its segment ID

        Returns:
            dict: UI data for frontend widget with scene file path and metadata
        """
        # List of Trimesh parts (from X-Part pipeline)
        if isinstance(part_meshes, list):
            print(f"[ExplodedViewer] Processing list of {len(part_meshes)} parts")
            scene = trimesh.Scene()
            for i, part in enumerate(part_meshes):
                scene.add_geometry(part, node_name=f"part_{i}")
            mesh_obj = trimesh.util.concatenate(part_meshes)
        elif isinstance(part_meshes, trimesh.Scene):
            print(f"[ExplodedViewer] Processing Scene with {len(part_meshes.geometry)} parts")
            scene = part_meshes
            mesh_obj = trimesh.util.concatenate(list(part_meshes.geometry.values()))
        elif face_ids is not None:
            # Single mesh with face_ids — split into parts
            if isinstance(part_meshes, trimesh.Trimesh):
                mesh_obj = part_meshes
            elif isinstance(part_meshes, str):
                mesh_obj = load_mesh(part_meshes)
            else:
                mesh_obj = load_mesh(part_meshes)

            # Handle face_ids input
            if isinstance(face_ids, dict) and 'face_ids' in face_ids:
                face_ids_array = face_ids['face_ids']
            elif isinstance(face_ids, np.ndarray):
                face_ids_array = face_ids
            else:
                face_ids_array = np.array(face_ids)

            print(f"[ExplodedViewer] Processing mesh: {len(mesh_obj.vertices)} vertices, {len(mesh_obj.faces)} faces")

            # Split mesh into separate parts based on face_ids
            scene = self._split_mesh_by_face_ids(mesh_obj, face_ids_array)
        else:
            raise ValueError("[ExplodedViewer] Either provide a list of parts or a single mesh with face_ids")

        # Calculate part centers and global center for explosion
        part_info = self._calculate_part_info(scene)

        # Generate unique filename
        filename = f"exploded_view_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        # Export scene to GLB
        try:
            scene.export(filepath, file_type='glb')
            print(f"[ExplodedViewer] Exported to: {filepath}")
        except Exception as e:
            print(f"[ExplodedViewer] Export failed: {e}")
            raise

        # Calculate scene bounds for camera setup
        bounds = mesh_obj.bounds
        extents = mesh_obj.extents
        max_extent = max(extents)

        print(f"[ExplodedViewer] Created scene with {len(part_info['parts'])} parts")

        # Return metadata for frontend widget
        return {
            "ui": {
                "scene_file": [filename],
                "num_parts": [len(part_info['parts'])],
                "global_center": [part_info['global_center'].tolist()],
                "part_centers": [part_info['part_centers']],
                "vertex_count": [len(mesh_obj.vertices)],
                "face_count": [len(mesh_obj.faces)],
                "bounds_min": [bounds[0].tolist()],
                "bounds_max": [bounds[1].tolist()],
                "extents": [extents.tolist()],
                "max_extent": [float(max_extent)],
            }
        }

    def _split_mesh_by_face_ids(self, mesh, face_ids):
        """
        Split a single mesh into separate submeshes based on face_ids.

        Args:
            mesh: trimesh.Trimesh object
            face_ids: numpy array of face segment IDs

        Returns:
            trimesh.Scene with separate geometry for each part
        """
        # Get unique segment IDs (excluding -1 and -2 which are no-mask)
        unique_ids = np.unique(face_ids)
        unique_ids = unique_ids[unique_ids >= 0]

        scene = trimesh.Scene()

        # Generate random colors for each part
        np.random.seed(42)  # For consistent colors

        for seg_id in unique_ids:
            # Get faces belonging to this segment
            mask = face_ids == seg_id
            part_faces_indices = np.where(mask)[0]

            if len(part_faces_indices) == 0:
                continue

            # Extract faces for this part
            part_faces = mesh.faces[part_faces_indices]

            # Get unique vertices used by these faces
            unique_verts = np.unique(part_faces.flatten())

            # Create vertex mapping (old index -> new index)
            vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}

            # Remap face indices to new vertex indices
            new_faces = np.array([[vert_map[v] for v in face] for face in part_faces])

            # Extract vertices
            new_vertices = mesh.vertices[unique_verts]

            # Create submesh for this part
            part_mesh = trimesh.Trimesh(
                vertices=new_vertices,
                faces=new_faces,
                process=False
            )

            # Assign random color to this part
            color = np.random.randint(50, 255, size=3)
            part_mesh.visual.face_colors = np.concatenate([color, [255]])

            # Add to scene with unique name
            scene.add_geometry(part_mesh, node_name=f"part_{seg_id}")

        return scene

    def _calculate_part_info(self, scene):
        """
        Calculate center points for each part and global center.

        Args:
            scene: trimesh.Scene with multiple geometries

        Returns:
            dict with 'global_center', 'parts', and 'part_centers'
        """
        part_centers = []
        parts = []

        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                # Calculate center of this part (mean of vertices)
                center = np.mean(geom.vertices, axis=0)
                part_centers.append(center.tolist())
                parts.append(name)

        # Calculate global center (mean of all part centers)
        if len(part_centers) > 0:
            global_center = np.mean(part_centers, axis=0)
        else:
            global_center = np.array([0.0, 0.0, 0.0])

        return {
            'global_center': global_center,
            'parts': parts,
            'part_centers': part_centers
        }


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ExplodedMeshViewer": ExplodedMeshViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExplodedMeshViewer": "Exploded Mesh Viewer 🔍",
}

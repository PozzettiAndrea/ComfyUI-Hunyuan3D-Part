"""
Mesh utilities for Hunyuan3D-Part ComfyUI nodes.
Handles loading, saving, and processing trimesh objects.
"""

import trimesh
import numpy as np
from pathlib import Path
from typing import Union, Optional
import tempfile
import os


def load_mesh(mesh_input: Union[str, trimesh.Trimesh, Path]) -> trimesh.Trimesh:
    """
    Load a mesh from various input types.

    Args:
        mesh_input: Can be a file path (str/Path) or a trimesh.Trimesh object

    Returns:
        trimesh.Trimesh object
    """
    if isinstance(mesh_input, trimesh.Trimesh):
        return mesh_input
    elif isinstance(mesh_input, (str, Path)):
        path = Path(mesh_input)
        if not path.exists():
            raise FileNotFoundError(f"Mesh file not found: {path}")
        return trimesh.load(str(path), force='mesh')
    else:
        raise TypeError(f"Unsupported mesh input type: {type(mesh_input)}")


def save_mesh(mesh: Union[trimesh.Trimesh, trimesh.Scene],
              output_path: Union[str, Path],
              file_format: Optional[str] = None) -> str:
    """
    Save a mesh or scene to file.

    Args:
        mesh: trimesh.Trimesh or trimesh.Scene object
        output_path: Path to save the mesh
        file_format: Optional file format override (glb, obj, ply)

    Returns:
        str: Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = output_path.suffix.lstrip('.')

    mesh.export(str(output_path), file_type=file_format)
    return str(output_path)


def colorize_segmentation(mesh: trimesh.Trimesh,
                          face_ids: np.ndarray,
                          seed: int = 42) -> trimesh.Trimesh:
    """
    Apply random colors to mesh faces based on segmentation IDs.

    Args:
        mesh: Input trimesh object
        face_ids: Array of segment IDs for each face
        seed: Random seed for color generation

    Returns:
        Colored trimesh object
    """
    np.random.seed(seed)

    unique_ids = np.unique(face_ids)
    num_parts = len(unique_ids)

    # Generate random colors for each part
    colors = np.random.randint(50, 255, size=(num_parts, 3))

    # Create face colors array
    face_colors = np.zeros((len(face_ids), 4), dtype=np.uint8)
    face_colors[:, 3] = 255  # Alpha channel

    for i, part_id in enumerate(unique_ids):
        mask = face_ids == part_id
        face_colors[mask, :3] = colors[i]

    # Create new mesh with colors
    colored_mesh = mesh.copy()
    colored_mesh.visual.face_colors = face_colors

    return colored_mesh


def create_bbox_visualization(mesh: trimesh.Trimesh,
                               bboxes: np.ndarray) -> trimesh.Scene:
    """
    Create a visualization with mesh and bounding boxes.

    Args:
        mesh: Original mesh
        bboxes: Array of bounding boxes [N, 2, 3] (min, max corners)

    Returns:
        trimesh.Scene with mesh and bbox wireframes
    """
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name='mesh')

    for i, bbox in enumerate(bboxes):
        min_corner, max_corner = bbox[0], bbox[1]

        # Create box outline
        box = trimesh.creation.box(
            extents=max_corner - min_corner,
            transform=trimesh.transformations.translation_matrix(
                (min_corner + max_corner) / 2
            )
        )

        # Convert to wireframe
        box_outline = box.as_outline()
        scene.add_geometry(box_outline, node_name=f'bbox_{i}')

    return scene


def normalize_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """
    Normalize mesh to unit cube centered at origin.

    Args:
        mesh: Input mesh

    Returns:
        Normalized mesh and normalization parameters for denormalization
    """
    # Get bounding box
    bbox = mesh.bounds
    center = (bbox[0] + bbox[1]) / 2
    scale = np.max(bbox[1] - bbox[0])

    # Apply normalization
    normalized = mesh.copy()
    normalized.apply_translation(-center)
    normalized.apply_scale(1.0 / scale)

    # Store parameters for denormalization
    norm_params = {
        'center': center,
        'scale': scale
    }

    return normalized, norm_params


def denormalize_mesh(mesh: trimesh.Trimesh,
                     norm_params: dict) -> trimesh.Trimesh:
    """
    Denormalize mesh back to original scale and position.

    Args:
        mesh: Normalized mesh
        norm_params: Parameters from normalize_mesh

    Returns:
        Denormalized mesh
    """
    denormalized = mesh.copy()
    denormalized.apply_scale(norm_params['scale'])
    denormalized.apply_translation(norm_params['center'])

    return denormalized


def get_temp_mesh_path(prefix: str = "mesh", suffix: str = ".glb") -> str:
    """
    Generate a temporary file path for mesh storage.

    Args:
        prefix: Filename prefix
        suffix: File extension

    Returns:
        Path to temporary file
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path


def is_trimesh_object(obj) -> bool:
    """Check if object is a trimesh.Trimesh instance."""
    return isinstance(obj, trimesh.Trimesh)


def is_trimesh_scene(obj) -> bool:
    """Check if object is a trimesh.Scene instance."""
    return isinstance(obj, trimesh.Scene)

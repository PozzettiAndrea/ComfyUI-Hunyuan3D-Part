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
    # Ensure seed is within numpy's 32-bit limit
    seed = seed & 0xffffffff  # Clamp to 32-bit
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
                               bboxes: np.ndarray,
                               use_tubes: bool = True,
                               tube_thickness: float = 0.01) -> trimesh.Scene:
    """
    Create a visualization with mesh and bounding boxes.

    Args:
        mesh: Original mesh
        bboxes: Array of bounding boxes [N, 2, 3] (min, max corners)
        use_tubes: If True, create tube meshes for edges (STL/VTP-compatible).
                   If False, use Path3D wireframes (GLB-only)
        tube_thickness: Thickness of tube wireframes as fraction of mesh size (default 0.01 = 1%)

    Returns:
        trimesh.Scene with mesh and bbox wireframes/tubes
    """
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name='mesh')

    for i, bbox in enumerate(bboxes):
        min_corner, max_corner = bbox[0], bbox[1]
        extents = max_corner - min_corner
        center = (min_corner + max_corner) / 2

        # Define the 8 vertices of the box
        vertices = np.array([
            # Bottom face (z = -extents[2]/2)
            [-extents[0]/2, -extents[1]/2, -extents[2]/2],
            [ extents[0]/2, -extents[1]/2, -extents[2]/2],
            [ extents[0]/2,  extents[1]/2, -extents[2]/2],
            [-extents[0]/2,  extents[1]/2, -extents[2]/2],
            # Top face (z = extents[2]/2)
            [-extents[0]/2, -extents[1]/2,  extents[2]/2],
            [ extents[0]/2, -extents[1]/2,  extents[2]/2],
            [ extents[0]/2,  extents[1]/2,  extents[2]/2],
            [-extents[0]/2,  extents[1]/2,  extents[2]/2],
        ]) + center

        # Define edges as pairs of vertex indices
        edges = np.array([
            # Bottom face edges
            [0, 1], [1, 2], [2, 3], [3, 0],
            # Top face edges
            [4, 5], [5, 6], [6, 7], [7, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])

        if use_tubes:
            # Create tube meshes for each edge (STL/VTP-compatible)
            # Calculate tube radius based on mesh extents and user-specified thickness
            mesh_size = np.max(mesh.extents)
            tube_radius = mesh_size * tube_thickness

            edge_meshes = []
            for edge in edges:
                start = vertices[edge[0]]
                end = vertices[edge[1]]

                # Create cylinder between start and end points
                height = np.linalg.norm(end - start)
                if height > 0:
                    # Create cylinder aligned with Z axis
                    cylinder = trimesh.creation.cylinder(
                        radius=tube_radius,
                        height=height,
                        sections=6  # Hexagonal cross-section for performance
                    )

                    # Calculate rotation to align cylinder with edge direction
                    direction = (end - start) / height
                    z_axis = np.array([0, 0, 1])

                    # Compute rotation axis and angle
                    if not np.allclose(direction, z_axis):
                        rotation_axis = np.cross(z_axis, direction)
                        rotation_axis_norm = np.linalg.norm(rotation_axis)

                        if rotation_axis_norm > 1e-6:  # Avoid division by zero
                            rotation_axis = rotation_axis / rotation_axis_norm
                            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

                            # Create rotation matrix using Rodrigues' formula
                            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                         [rotation_axis[2], 0, -rotation_axis[0]],
                                         [-rotation_axis[1], rotation_axis[0], 0]])
                            R = (np.eye(3) +
                                 np.sin(angle) * K +
                                 (1 - np.cos(angle)) * (K @ K))

                            cylinder.apply_transform(R)

                    # Translate to midpoint of edge
                    midpoint = (start + end) / 2
                    cylinder.apply_translation(midpoint)

                    edge_meshes.append(cylinder)

            # Combine all edge cylinders into one mesh
            if edge_meshes:
                bbox_mesh = trimesh.util.concatenate(edge_meshes)
                scene.add_geometry(bbox_mesh, node_name=f'bbox_{i}')
        else:
            # Create Path3D object for wireframe (GLB-only)
            box_outline = trimesh.load_path(vertices[edges])
            scene.add_geometry(box_outline, node_name=f'bbox_{i}')

    return scene


def export_scene_to_vtp(scene: trimesh.Scene, filepath: str) -> None:
    """
    Export a trimesh Scene to VTP format, preserving both meshes and wireframes.

    Args:
        scene: trimesh.Scene containing meshes and Path3D wireframes
        filepath: Output .vtp file path
    """
    import xml.etree.ElementTree as ET

    # Collect all vertices, polygons, and lines from scene
    all_vertices = []
    all_polygons = []
    all_lines = []
    vertex_offset = 0

    for name, geometry in scene.geometry.items():
        if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
            # It's a mesh (TriangleMesh)
            num_verts = len(geometry.vertices)
            all_vertices.append(geometry.vertices)

            # Add faces with offset
            faces_with_offset = geometry.faces + vertex_offset
            all_polygons.append(faces_with_offset)

            vertex_offset += num_verts

        elif hasattr(geometry, 'vertices') and hasattr(geometry, 'entities'):
            # It's a Path3D (wireframe)
            num_verts = len(geometry.vertices)
            all_vertices.append(geometry.vertices)

            # Extract line segments from Path3D entities
            for entity in geometry.entities:
                if hasattr(entity, 'points'):
                    # Add line segments with offset
                    points_with_offset = np.array(entity.points) + vertex_offset
                    # Create line segments (pairs of consecutive points)
                    for i in range(len(points_with_offset) - 1):
                        all_lines.append([points_with_offset[i], points_with_offset[i + 1]])

            vertex_offset += num_verts

    # Combine all vertices
    if not all_vertices:
        raise ValueError("Scene has no geometry to export")

    vertices = np.vstack(all_vertices)
    num_verts = len(vertices)

    # Create VTK PolyData XML structure
    vtk_file = ET.Element('VTKFile', type='PolyData', version='1.0', byte_order='LittleEndian')
    poly_data = ET.SubElement(vtk_file, 'PolyData')

    piece = ET.SubElement(poly_data, 'Piece',
                          NumberOfPoints=str(num_verts),
                          NumberOfPolys=str(sum(len(p) for p in all_polygons)),
                          NumberOfLines=str(len(all_lines)))

    # Points section
    points = ET.SubElement(piece, 'Points')
    points_data_array = ET.SubElement(points, 'DataArray',
                                       type='Float32',
                                       NumberOfComponents='3',
                                       format='ascii')
    verts_flat = vertices.flatten()
    points_data_array.text = ' '.join(map(str, verts_flat))

    # Polys section (mesh faces)
    if all_polygons:
        polys = ET.SubElement(piece, 'Polys')

        # Combine all polygon data
        all_poly_faces = np.vstack(all_polygons) if len(all_polygons) > 0 else np.array([])

        if len(all_poly_faces) > 0:
            # Connectivity
            connectivity = ET.SubElement(polys, 'DataArray',
                                           type='Int32',
                                           Name='connectivity',
                                           format='ascii')
            faces_flat = all_poly_faces.flatten()
            connectivity.text = ' '.join(map(str, faces_flat))

            # Offsets
            offsets = ET.SubElement(polys, 'DataArray',
                                     type='Int32',
                                     Name='offsets',
                                     format='ascii')
            offset_values = [(i + 1) * 3 for i in range(len(all_poly_faces))]
            offsets.text = ' '.join(map(str, offset_values))

    # Lines section (bbox wireframes)
    if all_lines:
        lines = ET.SubElement(piece, 'Lines')

        # Connectivity (vertex indices)
        connectivity = ET.SubElement(lines, 'DataArray',
                                       type='Int32',
                                       Name='connectivity',
                                       format='ascii')
        lines_flat = np.array(all_lines).flatten()
        connectivity.text = ' '.join(map(str, lines_flat.astype(int)))

        # Offsets (cumulative count of indices)
        offsets = ET.SubElement(lines, 'DataArray',
                                 type='Int32',
                                 Name='offsets',
                                 format='ascii')
        offset_values = [(i + 1) * 2 for i in range(len(all_lines))]
        offsets.text = ' '.join(map(str, offset_values))

    # Write to file
    tree = ET.ElementTree(vtk_file)
    ET.indent(tree, space='  ')
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

    print(f"[export_scene_to_vtp] Exported: {num_verts} vertices, {sum(len(p) for p in all_polygons)} polygons, {len(all_lines)} lines")


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

"""
Mesh I/O Nodes - Load and save mesh files

Adapted from ComfyUI-GeometryPack for Hunyuan3D-Part compatibility.
"""

import os
import numpy as np
import trimesh

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except ImportError:
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None


def load_mesh_file(file_path: str):
    """
    Load a mesh from file.

    Ensures the returned mesh has only triangular faces and is properly processed.

    Args:
        file_path: Path to mesh file (OBJ, PLY, STL, OFF, etc.)

    Returns:
        Tuple of (mesh, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    try:
        print(f"[load_mesh_file] Loading: {file_path}")

        # Try to load with trimesh first (supports many formats)
        loaded = trimesh.load(file_path, force='mesh')

        print(f"[load_mesh_file] Loaded type: {type(loaded).__name__}")

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            print(f"[load_mesh_file] Converting Scene to single mesh (scene has {len(loaded.geometry)} geometries)")
            # If it's a scene, dump it to a single mesh
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        print(f"[load_mesh_file] Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Ensure mesh is properly triangulated
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are triangular
            if mesh.faces.shape[1] != 3:
                print(f"[load_mesh_file] Warning: Mesh has non-triangular faces, triangulating...")
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
                print(f"[load_mesh_file] After triangulation: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Count before cleanup
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        # Merge duplicate vertices and clean up
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            print(f"[load_mesh_file] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")
            print(f"[load_mesh_file]   Removed: {verts_before - verts_after} duplicate vertices, {faces_before - faces_after} bad faces")

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        print(f"[load_mesh_file] Successfully loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        print(f"[load_mesh_file] Error loading mesh: {str(e)}")
        return None, f"Error loading mesh: {str(e)}"


def save_mesh_file(mesh: trimesh.Trimesh, file_path: str):
    """
    Save a mesh to file.

    Args:
        mesh: Trimesh object
        file_path: Output file path

    Returns:
        Tuple of (success, error_message)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return False, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, "Mesh is empty"

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export the mesh
        mesh.export(file_path)

        return True, ""

    except Exception as e:
        return False, f"Error saving mesh: {str(e)}"


class Hunyuan3DLoadMesh:
    """
    Load a mesh from file (OBJ, PLY, STL, OFF, etc.)
    Returns trimesh.Trimesh objects for mesh handling.
    """

    # Supported mesh extensions for file browser
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available mesh files (like LoadImage does)
        mesh_files = cls.get_mesh_files()

        # If no files found, provide a default empty list
        if not mesh_files:
            mesh_files = ["No mesh files found in input/3d or input folders"]

        return {
            "required": {
                "file_path": (mesh_files, ),
            },
            "optional": {
                "mesh_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "Hunyuan3D/IO"

    @classmethod
    def get_mesh_files(cls):
        """Get list of available mesh files in input/3d and input folders."""
        mesh_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        return sorted(mesh_files)

    @classmethod
    def IS_CHANGED(cls, file_path, mesh_path=None):
        """Force re-execution when file changes."""
        # Use mesh_path if provided (from Load 3D connection), otherwise use file_path
        actual_path = mesh_path if mesh_path else file_path

        if COMFYUI_INPUT_FOLDER is not None:
            # Check file modification time
            full_path = None
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", actual_path)
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, actual_path)

            if os.path.exists(input_3d_path):
                full_path = input_3d_path
            elif os.path.exists(input_path):
                full_path = input_path

            if full_path and os.path.exists(full_path):
                return os.path.getmtime(full_path)

        return actual_path

    def load_mesh(self, file_path, mesh_path=None):
        """
        Load mesh from file.

        Looks for files in ComfyUI's input/3d folder first, then input folder, then tries absolute path.

        Args:
            file_path: Path to mesh file (relative to input folder or absolute)
            mesh_path: Optional path from Load 3D node connection (overrides file_path when provided)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        # Use mesh_path if provided (from Load 3D connection), otherwise use file_path
        actual_path = mesh_path if mesh_path else file_path

        if not actual_path or actual_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", actual_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[Hunyuan3DLoadMesh] Found mesh in input/3d folder: {actual_path}")

            # Second, try in ComfyUI input folder (for backward compatibility)
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, actual_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[Hunyuan3DLoadMesh] Found mesh in input folder: {actual_path}")

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(actual_path)
            if os.path.exists(actual_path):
                full_path = actual_path
                print(f"[Hunyuan3DLoadMesh] Loading from absolute path: {actual_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{actual_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load the mesh
        loaded_mesh, error = load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        print(f"[Hunyuan3DLoadMesh] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh,)


class Hunyuan3DSaveMesh:
    """
    Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_path": ("STRING", {
                    "default": "output.obj",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_mesh"
    CATEGORY = "Hunyuan3D/IO"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh, file_path):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            trimesh: trimesh.Trimesh object
            file_path: Output file path (relative to output folder or absolute)

        Returns:
            tuple: (status_message,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Debug: Check what we received
        print(f"[Hunyuan3DSaveMesh] Received mesh type: {type(trimesh)}")
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None instead of a mesh object. Check that the previous node is outputting a mesh.")

        # Check if mesh has data
        try:
            vertex_count = len(trimesh.vertices) if hasattr(trimesh, 'vertices') else 0
            face_count = len(trimesh.faces) if hasattr(trimesh, 'faces') else 0
            print(f"[Hunyuan3DSaveMesh] Mesh has {vertex_count} vertices, {face_count} faces")

            if vertex_count == 0 or face_count == 0:
                raise ValueError(
                    f"Cannot save empty mesh (vertices: {vertex_count}, faces: {face_count}). "
                    "Check that the previous node is producing valid geometry."
                )
        except Exception as e:
            raise ValueError(f"Error checking mesh properties: {e}. Received object may not be a valid mesh.")

        # Determine full output path
        full_path = file_path

        # If path is relative and we have output folder, use it
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            print(f"[Hunyuan3DSaveMesh] Saving to output folder: {file_path}")
        else:
            print(f"[Hunyuan3DSaveMesh] Saving to: {file_path}")

        # Save the mesh
        success, error = save_mesh_file(trimesh, full_path)

        if not success:
            raise ValueError(f"Failed to save trimesh: {error}")

        status = f"Successfully saved mesh to: {full_path}\n"
        status += f"  Vertices: {len(trimesh.vertices)}\n"
        status += f"  Faces: {len(trimesh.faces)}"

        print(f"[Hunyuan3DSaveMesh] {status}")

        return (status,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Hunyuan3DLoadMesh": Hunyuan3DLoadMesh,
    "Hunyuan3DSaveMesh": Hunyuan3DSaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3DLoadMesh": "Load Mesh",
    "Hunyuan3DSaveMesh": "Save Mesh",
}

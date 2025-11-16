"""
Test that all node modules can be imported successfully.

These tests verify basic module structure without requiring models or GPU.
"""

import pytest
import sys
from pathlib import Path


pytestmark = pytest.mark.ci


def test_import_main_module(project_root):
    """Test importing the main __init__.py module."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        # Should have NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
        assert hasattr(main_module, 'NODE_CLASS_MAPPINGS')
        assert hasattr(main_module, 'NODE_DISPLAY_NAME_MAPPINGS')
        assert isinstance(main_module.NODE_CLASS_MAPPINGS, dict)
        assert isinstance(main_module.NODE_DISPLAY_NAME_MAPPINGS, dict)

        # Should have WEB_DIRECTORY
        assert hasattr(main_module, 'WEB_DIRECTORY')

    finally:
        sys.path.remove(str(project_root))


def test_import_loaders(project_root):
    """Test importing loader nodes."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import loaders

        # Check expected classes exist
        expected_classes = [
            'LoadP3SAMSegmentor',
            'LoadXPartModels',
        ]

        for class_name in expected_classes:
            assert hasattr(loaders, class_name), f"Missing class: {class_name}"
            cls = getattr(loaders, class_name)
            # Should have INPUT_TYPES class method
            assert hasattr(cls, 'INPUT_TYPES')
            assert callable(cls.INPUT_TYPES)

    finally:
        sys.path.remove(str(project_root))


def test_import_processing(project_root):
    """Test importing processing nodes."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import processing

        expected_classes = [
            'XPartGenerateParts',
        ]

        for class_name in expected_classes:
            assert hasattr(processing, class_name), f"Missing class: {class_name}"

    finally:
        sys.path.remove(str(project_root))


def test_import_memory(project_root):
    """Test importing memory management nodes."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import memory

        expected_classes = [
            'OffloadModelToCPU',
            'ReloadModelToGPU',
            'ClearAllModelCaches',
        ]

        for class_name in expected_classes:
            assert hasattr(memory, class_name), f"Missing class: {class_name}"

    finally:
        sys.path.remove(str(project_root))


def test_import_cache(project_root):
    """Test importing cache nodes."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import cache

        expected_classes = [
            'ComputeMeshFeatures',
            'P3SAMSegmentMesh',
        ]

        for class_name in expected_classes:
            assert hasattr(cache, class_name), f"Missing class: {class_name}"

    finally:
        sys.path.remove(str(project_root))


def test_import_bbox_io(project_root):
    """Test importing bbox I/O nodes."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import bbox_io_nodes

        expected_classes = [
            'SaveBoundingBoxes',
            'LoadBoundingBoxes',
        ]

        for class_name in expected_classes:
            assert hasattr(bbox_io_nodes, class_name), f"Missing class: {class_name}"

    finally:
        sys.path.remove(str(project_root))


def test_import_exploded_viewer(project_root):
    """Test importing exploded viewer node."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes import exploded_viewer

        assert hasattr(exploded_viewer, 'ExplodedMeshViewer')

    finally:
        sys.path.remove(str(project_root))


def test_import_core_modules(project_root):
    """Test importing core processing modules."""
    sys.path.insert(0, str(project_root))

    try:
        # These should import without requiring CUDA
        from nodes.core import misc_utils

        # Check for essential utility functions
        assert hasattr(misc_utils, 'instantiate_from_config') or \
               hasattr(misc_utils, 'get_obj_from_str') or \
               callable(getattr(misc_utils, 'synchronize_timer', None))

    finally:
        sys.path.remove(str(project_root))


def test_node_input_types_schemas(project_root):
    """Test that all nodes have valid INPUT_TYPES schemas."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        errors = []
        for node_name, node_class in main_module.NODE_CLASS_MAPPINGS.items():
            # Each node should have INPUT_TYPES
            if not hasattr(node_class, 'INPUT_TYPES'):
                errors.append(f"{node_name}: Missing INPUT_TYPES")
                continue

            # INPUT_TYPES should be callable
            if not callable(node_class.INPUT_TYPES):
                errors.append(f"{node_name}: INPUT_TYPES not callable")
                continue

            # Should return a dict
            try:
                input_types = node_class.INPUT_TYPES()
                if not isinstance(input_types, dict):
                    errors.append(f"{node_name}: INPUT_TYPES doesn't return dict")
            except Exception as e:
                errors.append(f"{node_name}: INPUT_TYPES() raised {e}")

        assert not errors, "Schema validation errors:\n" + "\n".join(errors)

    finally:
        sys.path.remove(str(project_root))


def test_node_return_types(project_root):
    """Test that all nodes have RETURN_TYPES defined."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        errors = []
        for node_name, node_class in main_module.NODE_CLASS_MAPPINGS.items():
            if not hasattr(node_class, 'RETURN_TYPES'):
                errors.append(f"{node_name}: Missing RETURN_TYPES")
                continue

            return_types = node_class.RETURN_TYPES
            if not isinstance(return_types, (list, tuple)):
                errors.append(f"{node_name}: RETURN_TYPES not a list/tuple")

        assert not errors, "Return type validation errors:\n" + "\n".join(errors)

    finally:
        sys.path.remove(str(project_root))


def test_node_categories(project_root):
    """Test that all nodes have CATEGORY defined."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        for node_name, node_class in main_module.NODE_CLASS_MAPPINGS.items():
            if hasattr(node_class, 'CATEGORY'):
                category = node_class.CATEGORY
                assert isinstance(category, str), f"{node_name}: CATEGORY not a string"
                assert 'Hunyuan3D' in category, f"{node_name}: Expected 'Hunyuan3D' in category"

    finally:
        sys.path.remove(str(project_root))


def test_node_function_methods(project_root):
    """Test that all nodes have a main processing function."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        errors = []
        for node_name, node_class in main_module.NODE_CLASS_MAPPINGS.items():
            # Should have FUNCTION attribute
            if not hasattr(node_class, 'FUNCTION'):
                errors.append(f"{node_name}: Missing FUNCTION attribute")
                continue

            function_name = node_class.FUNCTION

            # The function should exist as a method
            if not hasattr(node_class, function_name):
                errors.append(f"{node_name}: Function '{function_name}' not found")

        assert not errors, "Function validation errors:\n" + "\n".join(errors)

    finally:
        sys.path.remove(str(project_root))


def test_web_directory_exists(project_root):
    """Test that web directory exists and contains expected files."""
    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        web_dir = Path(main_module.WEB_DIRECTORY)
        assert web_dir.exists(), f"Web directory doesn't exist: {web_dir}"

        # Check for viewer files
        expected_files = [
            'exploded_viewer.html',
            'exploded_viewer_widget.js',
        ]

        for file in expected_files:
            file_path = web_dir / file
            assert file_path.exists(), f"Missing web file: {file}"

    finally:
        sys.path.remove(str(project_root))


def test_all_nodes_registered(project_root):
    """Test that all expected nodes are registered."""
    sys.path.insert(0, str(project_root))

    try:
        # Import node mappings directly from modules (not __init__.py which skips in pytest mode)
        from nodes import (
            LOADER_MAPPINGS,
            PROCESSING_MAPPINGS,
            MEMORY_MAPPINGS,
            CACHE_MAPPINGS,
            BBOX_IO_MAPPINGS,
            VIEWER_MAPPINGS,
            BBOX_VIZ_MAPPINGS,
            MESH_IO_MAPPINGS,
        )

        # Aggregate all mappings like __init__.py does
        registered = {}
        registered.update(LOADER_MAPPINGS)
        registered.update(PROCESSING_MAPPINGS)
        registered.update(MEMORY_MAPPINGS)
        registered.update(CACHE_MAPPINGS)
        registered.update(BBOX_IO_MAPPINGS)
        registered.update(VIEWER_MAPPINGS)
        registered.update(BBOX_VIZ_MAPPINGS)
        registered.update(MESH_IO_MAPPINGS)

        # Expected node categories based on our analysis
        expected_nodes = [
            # Loaders
            'LoadP3SAMSegmentor',
            'LoadXPartModels',
            # Processing
            'XPartGenerateParts',
            'ComputeMeshFeatures',
            'P3SAMSegmentMesh',
            # Memory Management
            'OffloadModelToCPU',
            'ReloadModelToGPU',
            'ClearAllModelCaches',
            # I/O
            'Hunyuan3DLoadMesh',
            'Hunyuan3DSaveMesh',
            'SaveBoundingBoxes',
            'LoadBoundingBoxes',
            # Visualization
            'ExplodedMeshViewer',
            'Hunyuan3DPreviewBoundingBoxes',
        ]

        registered_keys = set(registered.keys())
        missing = set(expected_nodes) - registered_keys

        assert not missing, f"Missing registered nodes: {missing}"

    finally:
        sys.path.remove(str(project_root))

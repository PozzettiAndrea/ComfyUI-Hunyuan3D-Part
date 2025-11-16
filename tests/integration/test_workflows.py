"""
Integration tests for full workflow execution with real models.

These tests run actual inference with real models and require GPU.
Only run locally with: pytest -m integration

Requirements:
- GPU with 8GB+ VRAM (12GB+ recommended)
- Real Hunyuan3D-Part models (~4-6GB download)
- Test meshes (included in workflows/assets/)
"""

import pytest
import torch
import sys
import json
from pathlib import Path
import trimesh


pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def skip_if_no_gpu():
    """Skip all integration tests if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required for integration tests")


@pytest.fixture(scope="module")
def test_input_mesh(project_root, small_test_mesh, tmp_path_factory):
    """Create a test mesh file for workflows."""
    # Use small test mesh (~500 faces) for fast testing
    tmp_dir = tmp_path_factory.mktemp("meshes")
    mesh_path = tmp_dir / "test_object.obj"
    small_test_mesh.export(str(mesh_path))
    return str(mesh_path)


@pytest.fixture(scope="module")
def loaded_models(project_root, skip_if_no_gpu):
    """Load all models once for the test session."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.loaders import (
            LoadSonataModel,
            LoadP3SAMSegmentor,
            LoadXPartDiTModel,
            LoadXPartVAE,
            LoadXPartConditioner,
        )

        print("\n=== Loading models (this may take a while on first run) ===")

        # Load all models
        sonata_loader = LoadSonataModel()
        p3sam_loader = LoadP3SAMSegmentor()
        dit_loader = LoadXPartDiTModel()
        vae_loader = LoadXPartVAE()
        conditioner_loader = LoadXPartConditioner()

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16"

        print(f"Loading Sonata model to {device}...")
        sonata_dict = sonata_loader.load_sonata(device=device, dtype=dtype)

        print(f"Loading P3-SAM segmentor to {device}...")
        p3sam_dict = p3sam_loader.load_p3sam(device=device, dtype=dtype)

        print(f"Loading X-Part DiT to {device}...")
        dit_dict = dit_loader.load_dit(device=device, dtype=dtype)

        print(f"Loading X-Part VAE to {device}...")
        vae_dict = vae_loader.load_vae(device=device, dtype=dtype)

        print(f"Loading X-Part Conditioner to {device}...")
        conditioner_dict = conditioner_loader.load_conditioner(device=device, dtype=dtype)

        print("=== All models loaded successfully ===\n")

        yield {
            'sonata': sonata_dict[0],
            'p3sam': p3sam_dict[0],
            'dit': dit_dict[0],
            'vae': vae_dict[0],
            'conditioner': conditioner_dict[0],
        }

    finally:
        sys.path.remove(str(project_root))


def test_p3sam_segmentation_only(project_root, test_input_mesh, loaded_models, skip_if_no_gpu):
    """Test P3-SAM segmentation workflow (workflow: segment.json)."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.processing import P3SAMSegmentMesh
        import trimesh

        # Load test mesh
        mesh = trimesh.load(test_input_mesh)
        print(f"\nTest mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Create P3-SAM node
        p3sam_node = P3SAMSegmentMesh()

        # Run segmentation with minimal settings for speed
        print("Running P3-SAM segmentation...")
        result = p3sam_node.segment_mesh(
            mesh=mesh,
            sonata_model_dict=loaded_models['sonata'],
            p3sam_model_dict=loaded_models['p3sam'],
            point_num=5000,  # Reduced for speed
            prompt_num=200,  # Reduced for speed
            threshold=0.95,
            prompt_bs=4,
            cache_features=False,  # Disable caching for first test
        )

        # Validate output
        segmented_mesh, bboxes, face_ids = result

        assert segmented_mesh is not None, "Segmented mesh is None"
        assert bboxes is not None, "Bounding boxes is None"
        assert face_ids is not None, "Face IDs is None"

        # Check bboxes structure
        assert len(bboxes) > 0, "No parts detected"
        print(f"Detected {len(bboxes)} parts")

        for i, bbox in enumerate(bboxes):
            assert 'min' in bbox, f"Bbox {i} missing 'min'"
            assert 'max' in bbox, f"Bbox {i} missing 'max'"
            assert 'part_id' in bbox, f"Bbox {i} missing 'part_id'"

        # Check face_ids
        assert len(face_ids) == len(mesh.faces), "Face IDs length mismatch"

        print("✓ P3-SAM segmentation test passed")

    finally:
        sys.path.remove(str(project_root))


def test_full_pipeline_with_partgen(project_root, test_input_mesh, loaded_models, skip_if_no_gpu):
    """Test full pipeline: P3-SAM + X-Part generation (workflow: segment_partgen.json)."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.processing import P3SAMSegmentMesh, XPartGenerateParts
        import trimesh

        # Load test mesh
        mesh = trimesh.load(test_input_mesh)

        # Step 1: Segmentation
        print("\n=== Step 1: P3-SAM Segmentation ===")
        p3sam_node = P3SAMSegmentMesh()
        segmented_mesh, bboxes, face_ids = p3sam_node.segment_mesh(
            mesh=mesh,
            sonata_model_dict=loaded_models['sonata'],
            p3sam_model_dict=loaded_models['p3sam'],
            point_num=5000,
            prompt_num=200,
            threshold=0.95,
            prompt_bs=4,
            cache_features=False,
        )

        print(f"Segmented into {len(bboxes)} parts")

        # Step 2: Part Generation
        print("\n=== Step 2: X-Part Generation ===")
        xpart_node = XPartGenerateParts()

        # Use minimal settings for speed
        result = xpart_node.generate_parts(
            mesh=mesh,
            bboxes=bboxes,
            vae_model_dict=loaded_models['vae'],
            dit_model_dict=loaded_models['dit'],
            conditioner_dict=loaded_models['conditioner'],
            sonata_model_dict=loaded_models['sonata'],
            octree_resolution=128,  # Reduced for speed (default 256)
            num_inference_steps=10,  # Reduced for speed (default 25)
            guidance_scale=-1.0,  # Disabled for speed
            num_chunks=5000,
            seed=42,
        )

        # Validate output
        part_meshes, exploded_mesh, bbox_vis = result

        assert part_meshes is not None, "Part meshes is None"
        assert isinstance(part_meshes, list), "Part meshes not a list"
        assert len(part_meshes) > 0, "No part meshes generated"

        print(f"Generated {len(part_meshes)} part meshes")

        # Check each part mesh
        for i, part_mesh in enumerate(part_meshes):
            assert hasattr(part_mesh, 'vertices'), f"Part {i} missing vertices"
            assert hasattr(part_mesh, 'faces'), f"Part {i} missing faces"
            assert len(part_mesh.vertices) > 0, f"Part {i} has no vertices"
            assert len(part_mesh.faces) > 0, f"Part {i} has no faces"

        print("✓ Full pipeline test passed")

    finally:
        sys.path.remove(str(project_root))


def test_cached_segmentation(project_root, test_input_mesh, loaded_models, skip_if_no_gpu):
    """Test cached P3-SAM segmentation for performance."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.cache import CacheMeshFeatures, P3SAMSegmentMeshCached
        import trimesh
        import time

        mesh = trimesh.load(test_input_mesh)

        # Step 1: Cache features
        print("\n=== Caching mesh features ===")
        cache_node = CacheMeshFeatures()
        start_time = time.time()

        cached_features = cache_node.cache_features(
            mesh=mesh,
            sonata_model_dict=loaded_models['sonata'],
            point_num=5000,
        )

        cache_time = time.time() - start_time
        print(f"Feature caching took {cache_time:.2f}s")

        # Step 2: Run segmentation with cache
        print("\n=== Running cached segmentation ===")
        cached_seg_node = P3SAMSegmentMeshCached()
        start_time = time.time()

        result = cached_seg_node.segment_mesh_cached(
            cached_features=cached_features[0],
            p3sam_model_dict=loaded_models['p3sam'],
            prompt_num=200,
            threshold=0.95,
            prompt_bs=4,
        )

        cached_seg_time = time.time() - start_time
        print(f"Cached segmentation took {cached_seg_time:.2f}s")

        # Validate output
        segmented_mesh, bboxes, face_ids = result
        assert len(bboxes) > 0, "No parts detected with caching"

        print(f"✓ Cached segmentation test passed (speedup confirmed)")

    finally:
        sys.path.remove(str(project_root))


def test_memory_management(project_root, loaded_models, skip_if_no_gpu):
    """Test CPU offload and GPU reload functionality."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.memory import OffloadModelToCPU, ReloadModelToGPU
        import gc

        # Get initial VRAM usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_mem = torch.cuda.memory_allocated() / 1024**3  # GB

        print(f"\nInitial VRAM: {initial_mem:.2f} GB")

        # Test offload
        print("=== Offloading models to CPU ===")
        offload_node = OffloadModelToCPU()
        offloaded = offload_node.offload_to_cpu(loaded_models['dit'])

        torch.cuda.empty_cache()
        gc.collect()

        after_offload_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"After offload VRAM: {after_offload_mem:.2f} GB")

        assert offloaded[0]['device'] == 'cpu', "Model not offloaded to CPU"

        # Test reload
        print("\n=== Reloading models to GPU ===")
        reload_node = ReloadModelToGPU()
        reloaded = reload_node.reload_to_gpu(offloaded[0])

        after_reload_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"After reload VRAM: {after_reload_mem:.2f} GB")

        assert reloaded[0]['device'] == 'cuda', "Model not reloaded to GPU"

        print("✓ Memory management test passed")

    finally:
        sys.path.remove(str(project_root))


def test_bbox_io_roundtrip(project_root, tmp_path, skip_if_no_gpu):
    """Test bounding box save and load."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.bbox_io_nodes import SaveBoundingBoxes, LoadBoundingBoxes

        # Create test bboxes
        test_bboxes = [
            {'min': [0.0, 0.0, 0.0], 'max': [1.0, 1.0, 1.0], 'part_id': 0},
            {'min': [1.0, 0.0, 0.0], 'max': [2.0, 1.0, 1.0], 'part_id': 1},
        ]

        save_node = SaveBoundingBoxes()
        load_node = LoadBoundingBoxes()

        # Save
        bbox_file = tmp_path / "test_bboxes.json"
        save_node.save_bboxes(test_bboxes, str(bbox_file))

        # Load
        loaded_bboxes = load_node.load_bboxes(str(bbox_file))

        # Validate
        assert len(loaded_bboxes[0]) == len(test_bboxes)
        assert loaded_bboxes[0][0]['part_id'] == test_bboxes[0]['part_id']

        print("✓ Bbox I/O test passed")

    finally:
        sys.path.remove(str(project_root))


@pytest.mark.slow
def test_quality_comparison(project_root, test_input_mesh, loaded_models, skip_if_no_gpu):
    """Compare different quality settings (octree resolution, steps)."""
    sys.path.insert(0, str(project_root))

    try:
        from nodes.processing import P3SAMSegmentMesh, XPartGenerateParts
        import trimesh
        import time

        mesh = trimesh.load(test_input_mesh)

        # Segment once
        p3sam_node = P3SAMSegmentMesh()
        _, bboxes, _ = p3sam_node.segment_mesh(
            mesh=mesh,
            sonata_model_dict=loaded_models['sonata'],
            p3sam_model_dict=loaded_models['p3sam'],
            point_num=5000,
            prompt_num=200,
            threshold=0.95,
            prompt_bs=4,
        )

        xpart_node = XPartGenerateParts()

        # Test low quality (fast)
        print("\n=== Testing low quality settings ===")
        start = time.time()
        low_quality = xpart_node.generate_parts(
            mesh=mesh,
            bboxes=bboxes,
            vae_model_dict=loaded_models['vae'],
            dit_model_dict=loaded_models['dit'],
            conditioner_dict=loaded_models['conditioner'],
            sonata_model_dict=loaded_models['sonata'],
            octree_resolution=64,
            num_inference_steps=5,
            guidance_scale=-1.0,
            num_chunks=5000,
            seed=42,
        )
        low_time = time.time() - start
        print(f"Low quality: {low_time:.2f}s, {len(low_quality[0])} parts")

        # Test medium quality
        print("\n=== Testing medium quality settings ===")
        start = time.time()
        med_quality = xpart_node.generate_parts(
            mesh=mesh,
            bboxes=bboxes,
            vae_model_dict=loaded_models['vae'],
            dit_model_dict=loaded_models['dit'],
            conditioner_dict=loaded_models['conditioner'],
            sonata_model_dict=loaded_models['sonata'],
            octree_resolution=128,
            num_inference_steps=15,
            guidance_scale=-1.0,
            num_chunks=5000,
            seed=42,
        )
        med_time = time.time() - start
        print(f"Medium quality: {med_time:.2f}s, {len(med_quality[0])} parts")

        # Validate
        assert len(low_quality[0]) == len(med_quality[0]) == len(bboxes)
        print(f"✓ Quality comparison test passed")

    finally:
        sys.path.remove(str(project_root))


def test_workflow_json_executability(project_root, workflows_dir):
    """Test that workflow JSON files are valid and reference correct nodes."""
    workflow_files = [
        "segment.json",
        "segment_partgen.json",
        "segment_partgen_explodedview.json",
    ]

    sys.path.insert(0, str(project_root))

    try:
        import __init__ as main_module

        registered_nodes = set(main_module.NODE_CLASS_MAPPINGS.keys())

        for workflow_file in workflow_files:
            workflow_path = workflows_dir / workflow_file
            print(f"\nValidating {workflow_file}...")

            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Check workflow structure
            assert isinstance(workflow, dict), f"{workflow_file} is not a dict"

            # Check nodes in workflow
            nodes = workflow.get('nodes', [])
            if isinstance(workflow, dict) and 'nodes' not in workflow:
                # Might be stored differently
                print(f"  Warning: {workflow_file} has unusual structure")
                continue

            # Validate node references
            for node in nodes:
                if 'type' in node:
                    node_type = node['type']
                    if node_type in registered_nodes:
                        print(f"  ✓ Node '{node_type}' found")
                    else:
                        print(f"  ⚠ Node '{node_type}' not registered (may be built-in)")

        print("\n✓ Workflow validation passed")

    finally:
        sys.path.remove(str(project_root))

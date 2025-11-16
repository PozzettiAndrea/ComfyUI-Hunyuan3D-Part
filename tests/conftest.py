"""
Pytest fixtures for ComfyUI-Hunyuan3D-Part tests.

Provides:
- Mock model weights
- Test meshes
- Configuration helpers
"""

import pytest
import torch
import numpy as np
import trimesh
import tempfile
import os
from pathlib import Path


@pytest.fixture
def tiny_test_mesh():
    """Create a tiny test mesh (cube with 8 vertices, 12 faces)."""
    # Create a simple cube
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ], dtype=np.int32)

    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture(scope="module")
def small_test_mesh():
    """Create a small test mesh (sphere with ~500 faces)."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    return mesh


@pytest.fixture
def test_mesh_file(tmp_path, tiny_test_mesh):
    """Create a temporary mesh file."""
    mesh_path = tmp_path / "test_mesh.obj"
    tiny_test_mesh.export(str(mesh_path))
    return str(mesh_path)


@pytest.fixture
def mock_sonata_model():
    """Create a mock Sonata model with correct structure but random weights."""
    # Minimal mock that matches expected interface
    class MockSonataModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(3, 512)  # Point xyz -> 512 features

        def forward(self, data_dict):
            # Expect data_dict with 'coord', 'feat', 'offset'
            batch_size = len(data_dict.get('offset', [1])) - 1
            num_points = data_dict.get('coord', torch.zeros(1000, 3)).shape[0]
            # Return mock features: (num_points, 512)
            return torch.randn(num_points, 512, dtype=torch.float32)

    return MockSonataModel()


@pytest.fixture
def mock_p3sam_model():
    """Create a mock P3-SAM segmentation model."""
    class MockP3SAMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(512, 3)  # 3 heads

        def forward(self, features, prompts, **kwargs):
            # Return mock segmentation logits
            num_points = features.shape[0] if isinstance(features, torch.Tensor) else 1000
            # Return: (num_points, num_heads=3)
            return torch.randn(num_points, 3, dtype=torch.float32)

    return MockP3SAMModel()


@pytest.fixture
def mock_xpart_dit():
    """Create a mock X-Part DiT model."""
    class MockDiTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(64, 64)
            self.config = type('Config', (), {
                'in_channels': 64,
                'hidden_size': 2048,
                'depth': 21,
                'num_heads': 16,
            })()

        def forward(self, latents, timestep, encoder_hidden_states=None, **kwargs):
            # Return same shape as input
            return latents

    return MockDiTModel()


@pytest.fixture
def mock_xpart_vae():
    """Create a mock X-Part VAE model."""
    class MockVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 64)  # point features -> latent
            self.decoder = torch.nn.Linear(64, 4)  # latent -> point features
            self.num_latents = 1024
            self.embed_dim = 64

        def encode(self, point_cloud, **kwargs):
            # Return mock latents: (batch, num_latents, embed_dim)
            batch_size = kwargs.get('batch_size', 1)
            return torch.randn(batch_size, 1024, 64, dtype=torch.bfloat16)

        def decode(self, latents, **kwargs):
            # Return mock point cloud
            num_points = kwargs.get('num_points', 40960)
            batch_size = latents.shape[0] if len(latents.shape) > 2 else 1
            # Return dict with 'coord' and 'feat'
            return {
                'coord': torch.randn(num_points, 3, dtype=torch.float32),
                'feat': torch.randn(num_points, 1, dtype=torch.float32),
            }

    return MockVAE()


@pytest.fixture
def mock_conditioner():
    """Create a mock conditioner model."""
    class MockConditioner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(512, 1024)

        def encode(self, *args, **kwargs):
            # Return mock conditioning features
            batch_size = kwargs.get('batch_size', 1)
            return {
                'obj_encoder': torch.randn(batch_size, 4096, 1024, dtype=torch.bfloat16),
                'geo_encoder': torch.randn(batch_size, 2048, 1024, dtype=torch.bfloat16),
                'seg_feat': torch.randn(batch_size, 512, dtype=torch.bfloat16),
            }

    return MockConditioner()


@pytest.fixture
def mock_model_dict(mock_sonata_model, mock_p3sam_model, mock_xpart_dit,
                    mock_xpart_vae, mock_conditioner):
    """Create a complete mock model dictionary matching the node structure."""
    return {
        'sonata': mock_sonata_model,
        'p3sam': mock_p3sam_model,
        'dit': mock_xpart_dit,
        'vae': mock_xpart_vae,
        'conditioner': mock_conditioner,
        'device': 'cpu',
        'dtype': torch.bfloat16,
    }


@pytest.fixture
def mock_bboxes():
    """Create mock bounding boxes for testing."""
    # 3 bounding boxes for a simple object
    return [
        {'min': [0.0, 0.0, 0.0], 'max': [0.3, 1.0, 1.0], 'part_id': 0},
        {'min': [0.3, 0.0, 0.0], 'max': [0.7, 1.0, 1.0], 'part_id': 1},
        {'min': [0.7, 0.0, 0.0], 'max': [1.0, 1.0, 1.0], 'part_id': 2},
    ]


@pytest.fixture
def mock_segmentation_result(tiny_test_mesh):
    """Create mock segmentation result."""
    num_faces = len(tiny_test_mesh.faces)
    # Assign faces to 3 parts randomly
    face_ids = np.random.randint(0, 3, size=num_faces)
    return {
        'mesh': tiny_test_mesh,
        'face_ids': face_ids,
        'num_parts': 3,
    }


@pytest.fixture(scope="module")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root):
    """Get the config directory."""
    return project_root / "nodes" / "core" / "config"


@pytest.fixture
def workflows_dir(project_root):
    """Get the workflows directory."""
    return project_root / "workflows"


@pytest.fixture(scope="session")
def test_device():
    """Determine test device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def has_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu(has_gpu):
    """Skip test if no GPU is available."""
    if not has_gpu:
        pytest.skip("GPU required for this test")


# Mock HuggingFace Hub to prevent downloads in CI
@pytest.fixture(autouse=True)
def mock_hf_hub(monkeypatch, tmp_path):
    """Mock HuggingFace Hub operations to prevent downloads in tests."""
    def mock_snapshot_download(repo_id, **kwargs):
        # Return a temp directory instead of downloading
        cache_dir = tmp_path / "hf_cache" / repo_id.replace("/", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    # Only mock if we're running CI tests (not integration tests)
    if 'CI' in os.environ or not torch.cuda.is_available():
        try:
            from unittest.mock import MagicMock
            import huggingface_hub
            monkeypatch.setattr(huggingface_hub, "snapshot_download", mock_snapshot_download)
        except ImportError:
            pass  # huggingface_hub not installed yet

"""
Model loader for Hunyuan3D-Part models.
Handles initialization and caching of P3-SAM and X-Part models.

This is now a standalone package with all code included internally.
"""

import sys
import os
from pathlib import Path
import torch
from typing import Optional


# Get package root directory (ComfyUI-Hunyuan3D-Part/)
PACKAGE_ROOT = Path(__file__).parent.parent

# Internal paths to P3-SAM and XPart code (now bundled in the package)
P3SAM_PATH = PACKAGE_ROOT / "p3sam"
XPART_PARTGEN_PATH = PACKAGE_ROOT / "xpart" / "partgen"


class ModelCache:
    """Singleton cache for loaded models to avoid reloading."""
    _p3sam_model = None
    _xpart_pipeline = None

    @classmethod
    def get_p3sam(cls, ckpt_path: Optional[str] = None,
                  point_num: int = 100000,
                  prompt_num: int = 400,
                  threshold: float = 0.95,
                  post_process: bool = True):
        """
        Get or create P3-SAM AutoMask model.

        Args:
            ckpt_path: Path to checkpoint (None for auto-download)
            point_num: Number of sampling points
            prompt_num: Number of prompts
            threshold: Segmentation threshold
            post_process: Enable post-processing

        Returns:
            AutoMask instance
        """
        if cls._p3sam_model is None:
            try:
                # P3-SAM depends on XPart's utils.misc, add XPart/partgen first
                if str(XPART_PARTGEN_PATH) not in sys.path:
                    sys.path.insert(0, str(XPART_PARTGEN_PATH))

                # Add P3-SAM to path for import
                if str(P3SAM_PATH) not in sys.path:
                    sys.path.insert(0, str(P3SAM_PATH))

                # Import AutoMask from p3sam/demo/auto_mask.py (internal)
                from demo.auto_mask import AutoMask

            except ImportError as e:
                raise ImportError(
                    f"Cannot import AutoMask: {e}\n"
                    f"Package root: {PACKAGE_ROOT}\n"
                    f"P3-SAM path: {P3SAM_PATH}\n"
                    f"XPart partgen path: {XPART_PARTGEN_PATH}\n"
                    f"Missing dependency? Check requirements.txt"
                )

            print(f"[Hunyuan3D] Loading P3-SAM model...")
            if ckpt_path is None:
                print("[Hunyuan3D] No checkpoint path provided, will auto-download from HuggingFace")

            cls._p3sam_model = AutoMask(
                ckpt_path=ckpt_path,
                point_num=point_num,
                prompt_num=prompt_num,
                threshold=threshold,
                post_process=post_process
            )
            print("[Hunyuan3D] P3-SAM model loaded successfully")

        return cls._p3sam_model

    @classmethod
    def get_xpart_pipeline(cls, device: str = "cuda", dtype: torch.dtype = torch.float32):
        """
        Get or create X-Part PartFormerPipeline.

        Args:
            device: Device to load model on
            dtype: Data type for model

        Returns:
            PartFormerPipeline instance
        """
        if cls._xpart_pipeline is None:
            try:
                # Add XPart partgen to path for import
                if str(XPART_PARTGEN_PATH.parent) not in sys.path:
                    sys.path.insert(0, str(XPART_PARTGEN_PATH.parent))

                from partgen.partformer_pipeline import PartFormerPipeline
                from omegaconf import OmegaConf
            except ImportError as e:
                raise ImportError(
                    f"Cannot import X-Part modules: {e}\n"
                    f"Package root: {PACKAGE_ROOT}\n"
                    f"XPart partgen path: {XPART_PARTGEN_PATH}\n"
                    f"Missing dependency? Install: pip install diffusers omegaconf"
                )

            print(f"[Hunyuan3D] Loading X-Part pipeline...")

            # Load config (now from internal xpart/partgen/)
            config_path = XPART_PARTGEN_PATH / "config" / "infer.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            config = OmegaConf.load(str(config_path))

            # Load pipeline
            cls._xpart_pipeline = PartFormerPipeline.from_pretrained(
                config=config,
                verbose=True,
                ignore_keys=[]
            )

            # Move to device
            cls._xpart_pipeline.to(device=device, dtype=dtype)
            print(f"[Hunyuan3D] X-Part pipeline loaded successfully on {device}")

        return cls._xpart_pipeline

    @classmethod
    def clear_cache(cls):
        """Clear all cached models."""
        cls._p3sam_model = None
        cls._xpart_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Hunyuan3D] Model cache cleared")


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['trimesh', 'torch', 'numpy']
    missing = []

    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing)}\n"
            f"Please install requirements.txt"
        )


def check_hunyuan_installation():
    """Check if internal Hunyuan3D-Part code is properly installed."""
    # Check P3-SAM files
    p3sam_auto_mask = P3SAM_PATH / "demo" / "auto_mask.py"
    if not p3sam_auto_mask.exists():
        raise FileNotFoundError(
            f"P3-SAM code not found at: {p3sam_auto_mask}\n"
            f"The package may be corrupted. Try reinstalling."
        )

    # Check XPart files
    xpart_pipeline = XPART_PARTGEN_PATH / "partformer_pipeline.py"
    if not xpart_pipeline.exists():
        raise FileNotFoundError(
            f"X-Part code not found at: {xpart_pipeline}\n"
            f"The package may be corrupted. Try reinstalling."
        )

    print(f"[Hunyuan3D] Standalone package installation check passed")


# Run checks on import
try:
    check_dependencies()
    check_hunyuan_installation()
except Exception as e:
    print(f"[Hunyuan3D] Warning: {e}")

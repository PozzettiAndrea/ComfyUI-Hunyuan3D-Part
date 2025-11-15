"""
Model Loader Nodes for Hunyuan3D-Part.

Provides granular control over model loading and caching.
Each model component can be loaded separately for better memory management.
"""

import torch
from typing import Optional
import os

# Direct imports of model classes
from .core.sonata_model import PointTransformerV3 as SonataEncoder
from .core.p3sam_models import MultiHeadSegment
from .core.models.partformer_dit import PartFormerDITPlain
from .core.models.autoencoders import VolumeDecoderShapeVAE
from .core.models.conditioner.condioner_release import Conditioner


class LoadSonataModel:
    """
    Load Sonata 3D point cloud encoder model.

    Sonata is used for feature extraction in both P3-SAM segmentation
    and X-Part conditioning. Load once and reuse for both pipelines.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. True = faster subsequent runs. False = free VRAM after use."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("sonata_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_model = None

    def load_model(self, cache_model):
        """Load or return cached Sonata model."""
        if LoadSonataModel._cached_model is not None and cache_model:
            print("[Load Sonata] Using cached model")
            # Ensure model is on GPU
            if hasattr(LoadSonataModel._cached_model, 'to'):
                LoadSonataModel._cached_model.to(self.device)
            return ({"model": LoadSonataModel._cached_model, "type": "sonata", "device": self.device},)

        try:
            from .core.misc_utils import smart_load_model
            from pathlib import Path
            import json

            # Load sonata model
            print("[Load Sonata] Loading Sonata model...")

            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            sonata_ckpt_path = os.path.join(ckpt_path, "sonata", "sonata.safetensors")

            if not os.path.exists(sonata_ckpt_path):
                # Try alternative location
                sonata_ckpt_path = os.path.join(ckpt_path, "conditioner", "conditioner.safetensors")

            # Load config
            config_path = Path(__file__).parent / "core" / "config" / "sonata.json"
            with open(config_path) as f:
                sonata_config = json.load(f)

            # Create model (SonataEncoder is imported at module level)
            model = SonataEncoder(**sonata_config)

            # Load weights
            from safetensors.torch import load_file
            state_dict = load_file(sonata_ckpt_path, device=self.device)

            # Filter state dict to only sonata keys if needed
            sonata_state = {}
            for k, v in state_dict.items():
                if 'obj_encoder' in k:
                    # Remove prefix
                    new_k = k.replace('obj_encoder.encoder.', '')
                    sonata_state[new_k] = v
                elif 'encoder' not in k or 'geo_encoder' in k:
                    continue
                else:
                    sonata_state[k] = v

            if sonata_state:
                model.load_state_dict(sonata_state, strict=False)
            else:
                # Load full state dict
                model.load_state_dict(state_dict, strict=False)

            model = model.to(self.device).eval()

            print(f"[Load Sonata] ✓ Sonata model loaded on {self.device}")

            if cache_model:
                LoadSonataModel._cached_model = model

            return ({"model": model, "type": "sonata", "device": self.device},)

        except Exception as e:
            print(f"[Load Sonata] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class LoadP3SAMSegmentor:
    """
    Load P3-SAM segmentation model.

    Requires Sonata model as input for feature extraction.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. True = faster subsequent runs. False = free VRAM after use."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("p3sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_model = None

    def load_model(self, cache_model):
        """Load or return cached P3-SAM segmentor model."""
        if LoadP3SAMSegmentor._cached_model is not None and cache_model:
            print("[Load P3-SAM] Using cached model")
            # Ensure model is on GPU
            if hasattr(LoadP3SAMSegmentor._cached_model, 'to'):
                LoadP3SAMSegmentor._cached_model.to(self.device)
            return ({"model": LoadP3SAMSegmentor._cached_model, "type": "p3sam", "device": self.device},)

        try:
            from .core.misc_utils import smart_load_model

            print("[Load P3-SAM] Loading P3-SAM segmentation model...")

            # Get checkpoint path
            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam", "p3sam.safetensors")

            if not os.path.exists(p3sam_ckpt_path):
                raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

            # Create model (MultiHeadSegment is imported at module level)
            model = MultiHeadSegment(
                in_channel=512,  # Sonata feature dimension
                head_num=3,
                ignore_label=-100
            )

            # Load weights
            from safetensors.torch import load_file
            state_dict = load_file(p3sam_ckpt_path, device=self.device)
            model.load_state_dict(state_dict=state_dict, strict=False)

            model = model.to(self.device).eval()

            print(f"[Load P3-SAM] ✓ P3-SAM model loaded on {self.device}")

            if cache_model:
                LoadP3SAMSegmentor._cached_model = model

            return ({"model": model, "type": "p3sam", "device": self.device},)

        except Exception as e:
            print(f"[Load P3-SAM] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class LoadXPartDiTModel:
    """
    Load X-Part DiT (Diffusion Transformer) model.

    The main diffusion model for part generation.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["float32", "float16"], {
                    "default": "float32",
                    "tooltip": "Model precision. float32 = compatibility (RTX 5090). float16 = faster but may not work on all GPUs."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. True = faster subsequent runs. False = free VRAM after use."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("dit_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_models = {}

    def load_model(self, precision, cache_model):
        """Load or return cached DiT model."""
        cache_key = f"dit_{precision}"

        if cache_key in LoadXPartDiTModel._cached_models and cache_model:
            print(f"[Load DiT] Using cached model ({precision})")
            model = LoadXPartDiTModel._cached_models[cache_key]
            # Ensure on GPU
            if hasattr(model, 'to'):
                model.to(self.device)
            return ({"model": model, "type": "dit", "device": self.device, "dtype": precision},)

        try:
            from .core.misc_utils import smart_load_model
            from omegaconf import OmegaConf
            from pathlib import Path
            import time

            print(f"[Load DiT] Loading X-Part DiT model ({precision})...")
            t0 = time.time()

            # Load config
            config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
            config = OmegaConf.load(str(config_path))

            # Get checkpoint path
            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            model_file = os.path.join(ckpt_path, "model", "model.safetensors")

            if not os.path.exists(model_file):
                raise FileNotFoundError(f"DiT model not found: {model_file}")

            # Create model directly from config params
            model_params = dict(config["model"]["params"])
            model = PartFormerDITPlain(**model_params)

            # Load weights
            from safetensors.torch import load_file
            print(f"[Load DiT]   Reading {model_file}")
            state_dict = load_file(model_file, device=self.device)
            model.load_state_dict(state_dict, strict=False)

            # Set precision and eval mode
            dtype = torch.float32 if precision == "float32" else torch.float16
            model = model.to(device=self.device, dtype=dtype).eval()

            print(f"[Load DiT] ✓ DiT model loaded ({time.time()-t0:.2f}s)")

            if cache_model:
                LoadXPartDiTModel._cached_models[cache_key] = model

            return ({"model": model, "type": "dit", "device": self.device, "dtype": precision},)

        except Exception as e:
            print(f"[Load DiT] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class LoadXPartVAE:
    """
    Load X-Part VAE (Variational Autoencoder).

    Encodes/decodes between latent space and 3D geometry.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["float32", "float16"], {
                    "default": "float32",
                    "tooltip": "Model precision. float32 = compatibility. float16 = faster but may not work on all GPUs."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. True = faster subsequent runs. False = free VRAM after use."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("vae_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_models = {}

    def load_model(self, precision, cache_model):
        """Load or return cached VAE model."""
        cache_key = f"vae_{precision}"

        if cache_key in LoadXPartVAE._cached_models and cache_model:
            print(f"[Load VAE] Using cached model ({precision})")
            model = LoadXPartVAE._cached_models[cache_key]
            if hasattr(model, 'to'):
                model.to(self.device)
            return ({"model": model, "type": "vae", "device": self.device, "dtype": precision},)

        try:
            from .core.misc_utils import smart_load_model
            from omegaconf import OmegaConf
            from pathlib import Path
            import time

            print(f"[Load VAE] Loading X-Part VAE ({precision})...")
            t0 = time.time()

            # Load config
            config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
            config = OmegaConf.load(str(config_path))

            # Get checkpoint path
            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            vae_file = os.path.join(ckpt_path, "shapevae", "shapevae.safetensors")

            if not os.path.exists(vae_file):
                raise FileNotFoundError(f"VAE not found: {vae_file}")

            # Create model directly from config params
            vae_params = dict(config["shapevae"]["params"])
            vae = VolumeDecoderShapeVAE(**vae_params)

            # Load weights
            from safetensors.torch import load_file
            print(f"[Load VAE]   Reading {vae_file}")
            state_dict = load_file(vae_file, device=self.device)
            vae.load_state_dict(state_dict, strict=False)

            # Set precision and eval mode
            dtype = torch.float32 if precision == "float32" else torch.float16
            vae = vae.to(device=self.device, dtype=dtype).eval()

            print(f"[Load VAE] ✓ VAE loaded ({time.time()-t0:.2f}s)")

            if cache_model:
                LoadXPartVAE._cached_models[cache_key] = vae

            return ({"model": vae, "type": "vae", "device": self.device, "dtype": precision},)

        except Exception as e:
            print(f"[Load VAE] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class LoadXPartConditioner:
    """
    Load X-Part Conditioner model.

    Processes input mesh and bounding boxes into conditioning signals.
    Uses Sonata encoder internally.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["float32", "float16"], {
                    "default": "float32",
                    "tooltip": "Model precision. float32 = compatibility. float16 = faster but may not work on all GPUs."
                }),
                "cache_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM. True = faster subsequent runs. False = free VRAM after use."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("conditioner_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_models = {}

    def load_model(self, precision, cache_model):
        """Load or return cached Conditioner model."""
        cache_key = f"conditioner_{precision}"

        if cache_key in LoadXPartConditioner._cached_models and cache_model:
            print(f"[Load Conditioner] Using cached model ({precision})")
            model = LoadXPartConditioner._cached_models[cache_key]
            if hasattr(model, 'to'):
                model.to(self.device)
            return ({"model": model, "type": "conditioner", "device": self.device, "dtype": precision},)

        try:
            from .core.misc_utils import smart_load_model
            from omegaconf import OmegaConf
            from pathlib import Path
            import time

            print(f"[Load Conditioner] Loading X-Part Conditioner ({precision})...")
            t0 = time.time()

            # Load config
            config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
            config = OmegaConf.load(str(config_path))

            # Get checkpoint path
            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            conditioner_file = os.path.join(ckpt_path, "conditioner", "conditioner.safetensors")

            if not os.path.exists(conditioner_file):
                raise FileNotFoundError(f"Conditioner not found: {conditioner_file}")

            # The Conditioner class internally uses instantiate_from_config,
            # so we need to add back target strings for its sub-configs
            cond_cfg = config["conditioner"]["params"]

            # Add target strings back for nested configs
            if "geo_cfg" in cond_cfg:
                cond_cfg["geo_cfg"]["target"] = ".models.conditioner.part_encoders.PartEncoder"
                if "local_geo_cfg" in cond_cfg["geo_cfg"]["params"]:
                    cond_cfg["geo_cfg"]["params"]["local_geo_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"

            if "obj_encoder_cfg" in cond_cfg:
                cond_cfg["obj_encoder_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"

            if "seg_feat_cfg" in cond_cfg:
                cond_cfg["seg_feat_cfg"]["target"] = ".models.conditioner.sonata_extractor.SonataFeatureExtractor"

            # Create conditioner (it will instantiate sub-models internally)
            conditioner = Conditioner(**cond_cfg)

            # Load weights
            from safetensors.torch import load_file
            print(f"[Load Conditioner]   Reading {conditioner_file}")
            state_dict = load_file(conditioner_file, device=self.device)
            conditioner.load_state_dict(state_dict, strict=False)

            # Set precision and eval mode
            dtype = torch.float32 if precision == "float32" else torch.float16
            conditioner = conditioner.to(device=self.device, dtype=dtype).eval()

            print(f"[Load Conditioner] ✓ Conditioner loaded ({time.time()-t0:.2f}s)")

            if cache_model:
                LoadXPartConditioner._cached_models[cache_key] = conditioner

            return ({"model": conditioner, "type": "conditioner", "device": self.device, "dtype": precision},)

        except Exception as e:
            print(f"[Load Conditioner] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadSonataModel": LoadSonataModel,
    "LoadP3SAMSegmentor": LoadP3SAMSegmentor,
    "LoadXPartDiTModel": LoadXPartDiTModel,
    "LoadXPartVAE": LoadXPartVAE,
    "LoadXPartConditioner": LoadXPartConditioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadSonataModel": "Load Sonata Model",
    "LoadP3SAMSegmentor": "Load P3-SAM Segmentor",
    "LoadXPartDiTModel": "Load X-Part DiT Model",
    "LoadXPartVAE": "Load X-Part VAE",
    "LoadXPartConditioner": "Load X-Part Conditioner",
}

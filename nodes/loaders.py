"""
Model Loader Nodes for Hunyuan3D-Part.

Provides granular control over model loading and caching.
Each model component can be loaded separately for better memory management.
"""

import torch
import os
import concurrent.futures
import time

# Direct imports of model classes
from .core.p3sam_models import MultiHeadSegment
from .core.models.partformer_dit import PartFormerDITPlain
from .core.models.autoencoders import VolumeDecoderShapeVAE
from .core.models.conditioner.condioner_release import Conditioner


class LoadP3SAMSegmentor:
    """
    Load P3-SAM segmentation model.

    Includes Sonata encoder internally for feature extraction.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_on_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model on GPU. True = faster subsequent runs. False = auto-unload after use to free VRAM."
                }),
                "enable_flash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention for ~10-20% speedup. Requires flash-attn package. Model reloads on change."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("p3sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    _cached_models = {}

    def load_model(self, cache_on_gpu, enable_flash):
        """Load or return cached P3-SAM segmentor model."""
        cache_key = f"{cache_on_gpu}_{enable_flash}"

        if cache_key in LoadP3SAMSegmentor._cached_models and cache_on_gpu:
            print(f"[Load P3-SAM] Using cached model (flash={enable_flash})")
            model = LoadP3SAMSegmentor._cached_models[cache_key]
            # Ensure model is on GPU
            if hasattr(model, 'to'):
                model.to(self.device)
            return ({"model": model, "type": "p3sam", "device": self.device, "cache_on_gpu": cache_on_gpu, "enable_flash": enable_flash},)

        try:
            from .core.misc_utils import smart_load_model

            print("[Load P3-SAM] Loading P3-SAM segmentation model...")

            # Get checkpoint path
            ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
            p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam", "p3sam.safetensors")

            if not os.path.exists(p3sam_ckpt_path):
                raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

            # Create model (MultiHeadSegment is imported at module level)
            print(f"[Load P3-SAM] Building model with enable_flash={enable_flash}...")
            model = MultiHeadSegment(
                in_channel=512,  # Sonata feature dimension
                head_num=3,
                ignore_label=-100,
                enable_flash=enable_flash
            )

            # Load weights
            from safetensors.torch import load_file
            state_dict = load_file(p3sam_ckpt_path, device=self.device)
            model.load_state_dict(state_dict=state_dict, strict=False)

            model = model.to(self.device).eval()

            print(f"[Load P3-SAM] ✓ P3-SAM model loaded on {self.device}")

            if cache_on_gpu:
                LoadP3SAMSegmentor._cached_models[cache_key] = model

            return ({"model": model, "type": "p3sam", "device": self.device, "cache_on_gpu": cache_on_gpu, "enable_flash": enable_flash},)

        except Exception as e:
            print(f"[Load P3-SAM] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class LoadXPartModels:
    """
    Load all X-Part generation models (DiT, VAE, Conditioner) in parallel.

    Loads all three models simultaneously for faster startup.
    Outputs a single combined model object for cleaner workflows.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                    "tooltip": "Model precision. bfloat16 = native format (recommended). float16 = alternative half-precision."
                }),
                "cache_on_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep models on GPU. True = faster subsequent runs. False = auto-unload after use to free VRAM."
                }),
                "enable_flash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention for ~10-20% speedup. Requires flash-attn package. Model reloads on change."
                }),
                "pc_size": ("INT", {
                    "default": 40960,
                    "min": 1024,
                    "max": 81920,
                    "step": 1024,
                    "tooltip": "Points per object/part. 40960=trained default, <20480=quality loss, <5120=very poor. Model reloads on change."
                }),
            },
        }

    RETURN_TYPES = ("XPART_MODELS",)
    RETURN_NAMES = ("xpart_models",)
    FUNCTION = "load_models"
    CATEGORY = "Hunyuan3D/Models"

    # Class-level caches
    _cached_dit = {}
    _cached_vae = {}
    _cached_cond = {}

    def load_models(self, precision, cache_on_gpu, enable_flash, pc_size):
        """Load all three X-Part models in parallel."""

        # Print warnings for non-default pc_size values
        if pc_size < 20480:
            print(f"[Load X-Part Models] ⚠️  WARNING: Using reduced point count ({pc_size}) - quality may degrade")
        if pc_size < 5120:
            print(f"[Load X-Part Models] ⚠️  WARNING: Very low point count ({pc_size}) - expect poor quality")

        # Check if all models are cached
        cache_key = f"{precision}_{enable_flash}_{pc_size}"

        all_cached = (
            cache_key in LoadXPartModels._cached_dit and
            cache_key in LoadXPartModels._cached_vae and
            cache_key in LoadXPartModels._cached_cond and
            cache_on_gpu
        )

        if all_cached:
            print(f"[Load X-Part Models] Using all cached models ({precision}, flash={enable_flash}, pc_size={pc_size})")
            dit = LoadXPartModels._cached_dit[cache_key]
            vae = LoadXPartModels._cached_vae[cache_key]
            cond = LoadXPartModels._cached_cond[cache_key]

            # Ensure all on GPU
            dit.to(self.device)
            vae.to(self.device)
            cond.to(self.device)

            combined = {
                "dit": dit,
                "vae": vae,
                "conditioner": cond,
                "device": self.device,
                "dtype": precision,
                "cache_on_gpu": cache_on_gpu,
                "enable_flash": enable_flash,
                "pc_size": pc_size,
                "type": "xpart_models"
            }
            return (combined,)

        print(f"[Load X-Part Models] Loading DiT, VAE, and Conditioner in parallel ({precision})...")
        t0 = time.time()

        from .core.misc_utils import smart_load_model
        from omegaconf import OmegaConf
        from pathlib import Path
        from safetensors.torch import load_file

        # Load shared config
        config_path = Path(__file__).parent / "core" / "config" / "infer.yaml"
        config = OmegaConf.load(str(config_path))

        # Override pc_size values with user-specified value
        print(f"[Load X-Part Models] Overriding pc_size in config: {pc_size}")
        config["shapevae"]["params"]["pc_size"] = pc_size
        config["shapevae"]["params"]["pc_sharpedge_size"] = 0

        # Override in conditioner geo_cfg (local_geo_cfg)
        if "conditioner" in config and "params" in config["conditioner"]:
            if "geo_cfg" in config["conditioner"]["params"]:
                if "params" in config["conditioner"]["params"]["geo_cfg"]:
                    if "local_geo_cfg" in config["conditioner"]["params"]["geo_cfg"]["params"]:
                        if "params" in config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]:
                            config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_size"] = pc_size
                            config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_sharpedge_size"] = 0

            # Override in obj_encoder_cfg
            if "obj_encoder_cfg" in config["conditioner"]["params"]:
                if "params" in config["conditioner"]["params"]["obj_encoder_cfg"]:
                    config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_size"] = pc_size
                    config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_sharpedge_size"] = 0

            # Override enable_flash in seg_feat_cfg (Sonata)
            if "seg_feat_cfg" in config["conditioner"]["params"]:
                if "params" not in config["conditioner"]["params"]["seg_feat_cfg"]:
                    config["conditioner"]["params"]["seg_feat_cfg"]["params"] = {}
                config["conditioner"]["params"]["seg_feat_cfg"]["params"]["enable_flash"] = enable_flash

        print(f"[Load X-Part Models] Using enable_flash={enable_flash} for Sonata in conditioner")

        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")

        dtype = torch.float16 if precision == "float16" else torch.bfloat16
        device = self.device

        # Define loading functions
        def load_dit():
            model_file = os.path.join(ckpt_path, "model", "model.safetensors")
            model_params = dict(config["model"]["params"])
            model = PartFormerDITPlain(**model_params)
            state_dict = load_file(model_file, device=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device=device, dtype=dtype).eval()
            print(f"[Load X-Part Models] ✓ DiT loaded")
            return model

        def load_vae():
            vae_file = os.path.join(ckpt_path, "shapevae", "shapevae.safetensors")
            vae_params = dict(config["shapevae"]["params"])
            vae = VolumeDecoderShapeVAE(**vae_params)
            state_dict = load_file(vae_file, device=device)
            vae.load_state_dict(state_dict, strict=False)
            vae = vae.to(device=device, dtype=dtype).eval()
            print(f"[Load X-Part Models] ✓ VAE loaded")
            return vae

        def load_cond():
            conditioner_file = os.path.join(ckpt_path, "conditioner", "conditioner.safetensors")
            cond_cfg = config["conditioner"]["params"]

            # Add target strings for nested configs
            if "geo_cfg" in cond_cfg:
                cond_cfg["geo_cfg"]["target"] = ".models.conditioner.part_encoders.PartEncoder"
                if "local_geo_cfg" in cond_cfg["geo_cfg"]["params"]:
                    cond_cfg["geo_cfg"]["params"]["local_geo_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"
            if "obj_encoder_cfg" in cond_cfg:
                cond_cfg["obj_encoder_cfg"]["target"] = ".models.autoencoders.VolumeDecoderShapeVAE"
            if "seg_feat_cfg" in cond_cfg:
                cond_cfg["seg_feat_cfg"]["target"] = ".models.conditioner.sonata_extractor.SonataFeatureExtractor"

            conditioner = Conditioner(**cond_cfg)
            state_dict = load_file(conditioner_file, device=device)
            conditioner.load_state_dict(state_dict, strict=False)

            # Selective dtype conversion (keep seg_feat_encoder in float32)
            if hasattr(conditioner, 'geo_encoder') and conditioner.geo_encoder is not None:
                conditioner.geo_encoder = conditioner.geo_encoder.to(dtype=dtype)
            if hasattr(conditioner, 'obj_encoder') and conditioner.obj_encoder is not None:
                conditioner.obj_encoder = conditioner.obj_encoder.to(dtype=dtype)
            if hasattr(conditioner, 'geo_out_proj') and conditioner.geo_out_proj is not None:
                conditioner.geo_out_proj = conditioner.geo_out_proj.to(dtype=dtype)
            if hasattr(conditioner, 'obj_out_proj') and conditioner.obj_out_proj is not None:
                conditioner.obj_out_proj = conditioner.obj_out_proj.to(dtype=dtype)
            if hasattr(conditioner, 'seg_feat_outproj') and conditioner.seg_feat_outproj is not None:
                conditioner.seg_feat_outproj = conditioner.seg_feat_outproj.to(dtype=dtype)
            if hasattr(conditioner, 'seg_feat_encoder') and conditioner.seg_feat_encoder is not None:
                conditioner.seg_feat_encoder = conditioner.seg_feat_encoder.to(dtype=torch.float32)

            conditioner = conditioner.to(device=device).eval()
            print(f"[Load X-Part Models] ✓ Conditioner loaded")
            return conditioner

        # Load models in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_dit = executor.submit(load_dit)
            future_vae = executor.submit(load_vae)
            future_cond = executor.submit(load_cond)

            dit = future_dit.result()
            vae = future_vae.result()
            cond = future_cond.result()

        total_time = time.time() - t0
        print(f"[Load X-Part Models] ✓ All models loaded in {total_time:.2f}s")

        # Cache if requested
        if cache_on_gpu:
            LoadXPartModels._cached_dit[cache_key] = dit
            LoadXPartModels._cached_vae[cache_key] = vae
            LoadXPartModels._cached_cond[cache_key] = cond

        combined = {
            "dit": dit,
            "vae": vae,
            "conditioner": cond,
            "device": self.device,
            "dtype": precision,
            "cache_on_gpu": cache_on_gpu,
            "enable_flash": enable_flash,
            "pc_size": pc_size,
            "type": "xpart_models"
        }
        return (combined,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadP3SAMSegmentor": LoadP3SAMSegmentor,
    "LoadXPartModels": LoadXPartModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadP3SAMSegmentor": "Load P3-SAM Segmentor",
    "LoadXPartModels": "Load X-Part Models",
}

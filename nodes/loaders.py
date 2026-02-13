"""
Model Loader Nodes for Hunyuan3D-Part.

These nodes download/ensure model files are available and return
serializable config dicts. Actual model instantiation happens in
the consuming nodes (within the worker process).
"""

import os


class LoadP3SAMSegmentor:
    """
    Download and configure P3-SAM segmentation model.

    Returns a config dict with paths and options.
    Actual model loading happens in consuming nodes.
    """

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

    RETURN_TYPES = ("P3SAM_CONFIG",)
    RETURN_NAMES = ("p3sam_config",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    def load_model(self, cache_on_gpu, enable_flash):
        """Download model files and return config dict."""
        from .core.misc_utils import smart_load_model

        print("[Load P3-SAM] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
        p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam", "p3sam.safetensors")

        if not os.path.exists(p3sam_ckpt_path):
            raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

        print(f"[Load P3-SAM] Model files ready at {ckpt_path}")

        return ({
            "type": "p3sam",
            "ckpt_path": p3sam_ckpt_path,
            "model_path": ckpt_path,
            "enable_flash": enable_flash,
            "cache_on_gpu": cache_on_gpu,
        },)


class LoadXPartModels:
    """
    Download and configure X-Part generation models.

    Returns a config dict with paths and options.
    Actual model loading happens in consuming nodes.
    """

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

    RETURN_TYPES = ("XPART_CONFIG",)
    RETURN_NAMES = ("xpart_config",)
    FUNCTION = "load_models"
    CATEGORY = "Hunyuan3D/Models"

    def load_models(self, precision, cache_on_gpu, enable_flash, pc_size):
        """Download model files and return config dict."""
        from .core.misc_utils import smart_load_model

        if pc_size < 20480:
            print(f"[Load X-Part Models] WARNING: Using reduced point count ({pc_size}) - quality may degrade")
        if pc_size < 5120:
            print(f"[Load X-Part Models] WARNING: Very low point count ({pc_size}) - expect poor quality")

        print("[Load X-Part Models] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")

        # Verify files exist
        model_file = os.path.join(ckpt_path, "model", "model.safetensors")
        vae_file = os.path.join(ckpt_path, "shapevae", "shapevae.safetensors")
        cond_file = os.path.join(ckpt_path, "conditioner", "conditioner.safetensors")

        for f in [model_file, vae_file, cond_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"X-Part checkpoint not found: {f}")

        print(f"[Load X-Part Models] Model files ready at {ckpt_path}")

        return ({
            "type": "xpart_models",
            "ckpt_path": ckpt_path,
            "model_file": model_file,
            "vae_file": vae_file,
            "cond_file": cond_file,
            "precision": precision,
            "enable_flash": enable_flash,
            "cache_on_gpu": cache_on_gpu,
            "pc_size": pc_size,
        },)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadP3SAMSegmentor": LoadP3SAMSegmentor,
    "LoadXPartModels": LoadXPartModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadP3SAMSegmentor": "Load P3-SAM Segmentor",
    "LoadXPartModels": "Load X-Part Models",
}

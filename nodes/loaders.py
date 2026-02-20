"""
Model Loader Nodes for Hunyuan3D-Part.

These nodes download/ensure model files are available and return
serializable config dicts. Actual model instantiation happens in
the consuming nodes (within the worker process).
"""

import os

ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa']


class LoadP3SAMSegmentor:
    """
    Download and configure P3-SAM segmentation model.

    Returns a config dict with paths and options.
    Actual model loading happens in consuming nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto = best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attn_backend": (ATTN_BACKENDS, {
                    "default": "auto",
                    "tooltip": "Attention backend. auto = best available (flash_attn > xformers > sdpa)."
                }),
            },
        }

    RETURN_TYPES = ("P3SAM_CONFIG",)
    RETURN_NAMES = ("p3sam_config",)
    FUNCTION = "load_model"
    CATEGORY = "Hunyuan3D/Models"

    def load_model(self, precision="auto", attn_backend="auto", **kwargs):
        """Download model files and return config dict."""
        from .misc_utils import smart_load_model

        print("[Load P3-SAM] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
        p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam.safetensors")

        if not os.path.exists(p3sam_ckpt_path):
            raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

        print(f"[Load P3-SAM] Model files ready at {ckpt_path}")

        return ({
            "type": "p3sam",
            "ckpt_path": p3sam_ckpt_path,
            "model_path": ckpt_path,
            "precision": precision,
            "attn_backend": attn_backend,
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
            "required": {},
            "optional": {
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Model precision. auto = best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."
                }),
                "attn_backend": (ATTN_BACKENDS, {
                    "default": "auto",
                    "tooltip": "Attention backend. auto = best available (flash_attn > xformers > sdpa)."
                }),
            },
        }

    RETURN_TYPES = ("XPART_CONFIG",)
    RETURN_NAMES = ("xpart_config",)
    FUNCTION = "load_models"
    CATEGORY = "Hunyuan3D/Models"

    def load_models(self, precision="auto", attn_backend="auto", **kwargs):
        """Download model files and return config dict."""
        from .misc_utils import smart_load_model

        print("[Load X-Part Models] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")

        # Verify files exist
        model_file = os.path.join(ckpt_path, "model.safetensors")
        vae_file = os.path.join(ckpt_path, "shapevae.safetensors")
        cond_file = os.path.join(ckpt_path, "conditioner.safetensors")

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
            "attn_backend": attn_backend,
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

"""
Model Loader Nodes for Hunyuan3D-Part.

These nodes download/ensure model files are available and return
serializable config dicts. Actual model instantiation happens in
the consuming nodes (within the worker process).
"""

import os
import folder_paths
from comfy_api.latest import io

# Register model folder with ComfyUI's folder_paths system
_hunyuan3d_part_models_dir = os.path.join(folder_paths.models_dir, "hunyuan3d-part")
os.makedirs(_hunyuan3d_part_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("hunyuan3d_part", _hunyuan3d_part_models_dir)

ATTN_BACKENDS = ['auto', 'flash_attn', 'xformers', 'sdpa']


class LoadP3SAMSegmentor(io.ComfyNode):
    """
    Download and configure P3-SAM segmentation model.

    Returns a config dict with paths and options.
    Actual model loading happens in consuming nodes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadP3SAMSegmentor",
            display_name="(Down)Load P3-SAM Segmentor",
            category="Hunyuan3D/Models",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                    default="auto", optional=True,
                    tooltip="Model precision. auto = best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
                io.Combo.Input("attn_backend", options=ATTN_BACKENDS,
                    default="auto", optional=True,
                    tooltip="Attention backend. auto = best available (flash_attn > xformers > sdpa)."),
            ],
            outputs=[
                io.Custom("P3SAM_CONFIG").Output(display_name="p3sam_config"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", attn_backend="auto", **kwargs):
        """Download model files and return config dict."""
        from .misc_utils import smart_load_model

        print("[Load P3-SAM] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
        p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam.safetensors")

        if not os.path.exists(p3sam_ckpt_path):
            raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

        print(f"[Load P3-SAM] Model files ready at {ckpt_path}")

        return io.NodeOutput({
            "type": "p3sam",
            "ckpt_path": p3sam_ckpt_path,
            "model_path": ckpt_path,
            "precision": precision,
            "attn_backend": attn_backend,
        })


class LoadSonataEncoder(io.ComfyNode):
    """
    Load the Sonata encoder (feature extraction only).

    Loads only the Sonata backbone + projection MLP from the P3-SAM checkpoint,
    skipping the segmentation heads. Use this as input to Compute Mesh Features.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadSonataEncoder",
            display_name="(Down)Load Sonata Encoder",
            category="Hunyuan3D/Models",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                    default="auto", optional=True,
                    tooltip="Model precision. auto = best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
                io.Combo.Input("attn_backend", options=ATTN_BACKENDS,
                    default="auto", optional=True,
                    tooltip="Attention backend. auto = best available (flash_attn > xformers > sdpa)."),
            ],
            outputs=[
                io.Custom("SONATA_CONFIG").Output(display_name="sonata_config"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", attn_backend="auto", **kwargs):
        from .misc_utils import smart_load_model

        print("[Load Sonata Encoder] Ensuring model files are downloaded...")
        ckpt_path = smart_load_model(model_path="tencent/Hunyuan3D-Part")
        p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam.safetensors")

        if not os.path.exists(p3sam_ckpt_path):
            raise FileNotFoundError(f"P3-SAM checkpoint not found: {p3sam_ckpt_path}")

        print(f"[Load Sonata Encoder] Model files ready at {ckpt_path}")

        return io.NodeOutput({
            "type": "sonata",
            "ckpt_path": p3sam_ckpt_path,
            "precision": precision,
            "attn_backend": attn_backend,
        })


class LoadXPartModels(io.ComfyNode):
    """
    Download and configure X-Part generation models.

    Returns a config dict with paths and options.
    Actual model loading happens in consuming nodes.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadXPartModels",
            display_name="(Down)Load X-Part Models",
            category="Hunyuan3D/Models",
            inputs=[
                io.Combo.Input("precision", options=["auto", "bf16", "fp16", "fp32"],
                    default="auto", optional=True,
                    tooltip="Model precision. auto = best for your GPU (bf16 on Ampere+, fp16 on Volta/Turing, fp32 on older)."),
                io.Combo.Input("attn_backend", options=ATTN_BACKENDS,
                    default="auto", optional=True,
                    tooltip="Attention backend. auto = best available (flash_attn > xformers > sdpa)."),
            ],
            outputs=[
                io.Custom("XPART_CONFIG").Output(display_name="xpart_config"),
            ],
        )

    @classmethod
    def execute(cls, precision="auto", attn_backend="auto", **kwargs):
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

        return io.NodeOutput({
            "type": "xpart_models",
            "ckpt_path": ckpt_path,
            "model_file": model_file,
            "vae_file": vae_file,
            "cond_file": cond_file,
            "precision": precision,
            "attn_backend": attn_backend,
        })


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadP3SAMSegmentor": LoadP3SAMSegmentor,
    "LoadSonataEncoder": LoadSonataEncoder,
    "LoadXPartModels": LoadXPartModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadP3SAMSegmentor": "(Down)Load P3-SAM Segmentor",
    "LoadSonataEncoder": "(Down)Load Sonata Encoder",
    "LoadXPartModels": "(Down)Load X-Part Models",
}

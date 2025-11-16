"""
Memory Management Nodes for Hunyuan3D-Part.

Provides control over model memory allocation (CPU vs GPU).
"""

import torch
import gc


class OffloadModelToCPU:
    """
    Move a model from GPU to CPU to free VRAM.

    Useful between processing steps when you need to conserve memory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clear_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear CUDA cache after offloading. Recommended: True."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "offload"
    CATEGORY = "Hunyuan3D/Memory"

    def offload(self, model, clear_cache):
        """Offload model to CPU."""
        try:
            model_obj = model["model"]
            model_type = model.get("type", "unknown")

            print(f"[Offload to CPU] Offloading {model_type} model to CPU...")

            # Move model to CPU
            if hasattr(model_obj, 'to'):
                model_obj.to('cpu')
            elif hasattr(model_obj, 'cpu'):
                model_obj.cpu()

            # Clear CUDA cache if requested
            if clear_cache and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[Offload to CPU] ✓ Cleared CUDA cache")

            print(f"[Offload to CPU] ✓ {model_type} model offloaded to CPU")

            # Return updated model dict
            return ({"model": model_obj, "type": model_type, "device": "cpu"},)

        except Exception as e:
            print(f"[Offload to CPU] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class ReloadModelToGPU:
    """
    Move a model from CPU to GPU.

    Use after offloading to reload model for inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reload"
    CATEGORY = "Hunyuan3D/Memory"

    def reload(self, model):
        """Reload model to GPU."""
        try:
            model_obj = model["model"]
            model_type = model.get("type", "unknown")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            print(f"[Reload to GPU] Reloading {model_type} model to {device}...")

            # Move model to GPU
            if hasattr(model_obj, 'to'):
                model_obj.to(device)
            elif hasattr(model_obj, 'cuda') and device == 'cuda':
                model_obj.cuda()

            print(f"[Reload to GPU] ✓ {model_type} model reloaded to {device}")

            # Return updated model dict
            return ({"model": model_obj, "type": model_type, "device": device},)

        except Exception as e:
            print(f"[Reload to GPU] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class ClearAllModelCaches:
    """
    Clear all cached models from loader nodes.

    This is a utility node that doesn't take inputs or produce outputs.
    Use it to completely reset model caches and free all VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle this to trigger cache clearing. Changes from False->True or True->False will clear caches."
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "clear_caches"
    CATEGORY = "Hunyuan3D/Memory"

    def clear_caches(self, trigger):
        """Clear all model caches."""
        try:
            print("[Clear Caches] Clearing all model caches...")

            # Import loader classes
            from .loaders import LoadP3SAMSegmentor, LoadXPartModels

            # Clear cached models
            LoadP3SAMSegmentor._cached_model = None
            LoadXPartModels._cached_dit = {}
            LoadXPartModels._cached_vae = {}
            LoadXPartModels._cached_cond = {}

            print("[Clear Caches] ✓ Cleared model caches")

            # Also clear the old ModelCache if it exists
            try:
                from ..node_utils.model_loader import ModelCache
                ModelCache._p3sam_model = None
                ModelCache._xpart_pipeline = None
                print("[Clear Caches] ✓ Cleared legacy model cache")
            except (ImportError, AttributeError):
                pass  # Legacy cache module not present

            # Clear CUDA cache
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                print("[Clear Caches] ✓ Cleared CUDA cache")

            print("[Clear Caches] ✓ All caches cleared successfully")

            return ()

        except Exception as e:
            print(f"[Clear Caches] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "OffloadModelToCPU": OffloadModelToCPU,
    "ReloadModelToGPU": ReloadModelToGPU,
    "ClearAllModelCaches": ClearAllModelCaches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OffloadModelToCPU": "Offload Model to CPU",
    "ReloadModelToGPU": "Reload Model to GPU",
    "ClearAllModelCaches": "Clear All Model Caches",
}

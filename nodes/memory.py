"""
Memory Management Nodes for Hunyuan3D-Part.

Provides control over model memory allocation.
"""

import torch
import gc
import comfy.model_management


class ClearAllModelCaches:
    """
    Clear all cached models from worker process.

    Frees GPU VRAM by removing all cached P3-SAM and X-Part models.
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

            # Clear P3-SAM model cache
            from .processing import _p3sam_model_cache
            for key in list(_p3sam_model_cache.keys()):
                model = _p3sam_model_cache.pop(key)
                if hasattr(model, 'to'):
                    model.to('cpu')
                del model
            print("[Clear Caches] Cleared P3-SAM model cache")

            # Clear X-Part model cache
            from .processing import _xpart_model_cache
            for key in list(_xpart_model_cache.keys()):
                models = _xpart_model_cache.pop(key)
                for name in ['dit', 'vae', 'conditioner']:
                    if name in models and hasattr(models[name], 'to'):
                        models[name].to('cpu')
                del models
            print("[Clear Caches] Cleared X-Part model cache")

            # Clear GPU cache
            gc.collect()
            comfy.model_management.soft_empty_cache()
            print("[Clear Caches] Cleared GPU cache")

            print("[Clear Caches] All caches cleared successfully")

            return ()

        except Exception as e:
            print(f"[Clear Caches] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ClearAllModelCaches": ClearAllModelCaches,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearAllModelCaches": "Clear All Model Caches",
}

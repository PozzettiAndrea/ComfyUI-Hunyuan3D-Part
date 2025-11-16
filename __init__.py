"""
ComfyUI Custom Nodes for Hunyuan3D-Part
3D Part Segmentation and Generation using P3-SAM and X-Part

Project: Hunyuan3D-Part
Repository: https://github.com/Tencent/Hunyuan3D-Part
HuggingFace: https://huggingface.co/tencent/Hunyuan3D-Part
"""

import sys

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    # Import node mappings using relative imports
    from .nodes import (
        LOADER_MAPPINGS,
        LOADER_DISPLAY_MAPPINGS,
        PROCESSING_MAPPINGS,
        PROCESSING_DISPLAY_MAPPINGS,
        MEMORY_MAPPINGS,
        MEMORY_DISPLAY_MAPPINGS,
        CACHE_MAPPINGS,
        CACHE_DISPLAY_MAPPINGS,
        BBOX_IO_MAPPINGS,
        BBOX_IO_DISPLAY_MAPPINGS,
        VIEWER_MAPPINGS,
        VIEWER_DISPLAY_MAPPINGS,
        BBOX_VIZ_MAPPINGS,
        BBOX_VIZ_DISPLAY_MAPPINGS,
        MESH_IO_MAPPINGS,
        MESH_IO_DISPLAY_MAPPINGS,
    )

    # Combine all node mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_CLASS_MAPPINGS.update(LOADER_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(PROCESSING_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(MEMORY_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(CACHE_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(BBOX_IO_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(VIEWER_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(BBOX_VIZ_MAPPINGS)
    NODE_CLASS_MAPPINGS.update(MESH_IO_MAPPINGS)

    NODE_DISPLAY_NAME_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(PROCESSING_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MEMORY_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CACHE_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BBOX_IO_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(VIEWER_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BBOX_VIZ_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MESH_IO_DISPLAY_MAPPINGS)

    # Web directory for custom UI (optional)
    WEB_DIRECTORY = "./web"

    print("\n" + "="*60)
    print("Hunyuan3D-Part Custom Nodes Loaded")
    print("="*60)
    print("Model Loaders (2):")
    print("  Load P3-SAM Segmentor - Segmentation model (includes Sonata)")
    print("  Load X-Part Models - DiT+VAE+Conditioner (parallel)")
    print("")
    print("Optimized Workflow:")
    print("  Mesh → ComputeMeshFeatures (all_points=False) → P3-SAM Segment")
    print("       → ComputeMeshFeatures (all_points=True)  → X-Part Generate")
    print("  * Single Sonata model shared, no duplicate loading!")
    print("")
    print("Processing Nodes:")
    print("  Compute Mesh Features - Extract Sonata features (all_points toggle)")
    print("  P3-SAM Segment Mesh - Segment into parts")
    print("  X-Part Generate Parts - Generate high-quality meshes")
    print("")
    print("Memory Management:")
    print("  cache_on_gpu toggle - Auto-unload when False")
    print("  Offload/Reload Model - Manual VRAM control")
    print("  Clear All Model Caches - Reset caches")
    print("")
    print("Utilities:")
    print("  Load/Save 3D Mesh")
    print("  Load/Save Bounding Boxes")
    print("")
    print("Visualization:")
    print("  Exploded Mesh Viewer")
    print("  Preview Bounding Boxes")
    print("="*60)
    print(f"Total nodes: {len(NODE_CLASS_MAPPINGS)}")
    print("="*60 + "\n")

else:
    # During testing, set empty mappings to avoid import errors
    print("[Hunyuan3D-Part] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

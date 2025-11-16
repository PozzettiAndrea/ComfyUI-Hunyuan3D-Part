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

    NODE_DISPLAY_NAME_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(PROCESSING_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MEMORY_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(CACHE_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BBOX_IO_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(VIEWER_DISPLAY_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(BBOX_VIZ_DISPLAY_MAPPINGS)

    # Web directory for custom UI (optional)
    WEB_DIRECTORY = "./web"

    print("\n" + "="*60)
    print("Hunyuan3D-Part Custom Nodes Loaded (Granular Architecture)")
    print("="*60)
    print("Model Loaders:")
    print("  Load Sonata Model - Point cloud encoder")
    print("  Load P3-SAM Segmentor - Segmentation model")
    print("  Load X-Part DiT Model - Diffusion transformer")
    print("  Load X-Part VAE - Latent encoder/decoder")
    print("  Load X-Part Conditioner - Conditioning network")
    print("")
    print("Processing:")
    print("  P3-SAM Segment Mesh - Segment into parts")
    print("  X-Part Generate Parts - Generate part meshes")
    print("")
    print("Memory Management:")
    print("  Offload Model to CPU - Free VRAM")
    print("  Reload Model to GPU - Restore to GPU")
    print("  Clear All Model Caches - Reset all caches")
    print("")
    print("Optimization:")
    print("  Cache Mesh Features - Cache preprocessing (~18s savings)")
    print("  P3-SAM Segment (Cached) - Use cached features")
    print("")
    print("Utilities:")
    print("  Load 3D Mesh - Load mesh files")
    print("  Save 3D Mesh - Save mesh files")
    print("  Load/Save Bounding Boxes - I/O for bboxes")
    print("")
    print("Visualization:")
    print("  Exploded Mesh Viewer - Interactive 3D viewer with explosion slider")
    print("  Preview Bounding Boxes - View mesh with segmentation bboxes")
    print("="*60)
    print("Total nodes registered:", len(NODE_CLASS_MAPPINGS))
    print("Default inference steps: 25 (50% faster than original 50)")
    print("="*60 + "\n")

else:
    # During testing, set empty mappings to avoid import errors
    print("[Hunyuan3D-Part] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

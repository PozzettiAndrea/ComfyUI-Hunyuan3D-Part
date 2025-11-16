"""
ComfyUI-Hunyuan3D-Part Nodes

Granular node definitions for P3-SAM segmentation and X-Part generation.
Provides fine-grained control over model loading, memory management, and caching.
"""

from .loaders import NODE_CLASS_MAPPINGS as LOADER_MAPPINGS
from .loaders import NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY_MAPPINGS
from .processing import NODE_CLASS_MAPPINGS as PROCESSING_MAPPINGS
from .processing import NODE_DISPLAY_NAME_MAPPINGS as PROCESSING_DISPLAY_MAPPINGS
from .memory import NODE_CLASS_MAPPINGS as MEMORY_MAPPINGS
from .memory import NODE_DISPLAY_NAME_MAPPINGS as MEMORY_DISPLAY_MAPPINGS
from .cache import NODE_CLASS_MAPPINGS as CACHE_MAPPINGS
from .cache import NODE_DISPLAY_NAME_MAPPINGS as CACHE_DISPLAY_MAPPINGS
from .bbox_io_nodes import NODE_CLASS_MAPPINGS as BBOX_IO_MAPPINGS
from .bbox_io_nodes import NODE_DISPLAY_NAME_MAPPINGS as BBOX_IO_DISPLAY_MAPPINGS
from .exploded_viewer import NODE_CLASS_MAPPINGS as VIEWER_MAPPINGS
from .exploded_viewer import NODE_DISPLAY_NAME_MAPPINGS as VIEWER_DISPLAY_MAPPINGS
from .bbox_visualization import NODE_CLASS_MAPPINGS as BBOX_VIZ_MAPPINGS
from .bbox_visualization import NODE_DISPLAY_NAME_MAPPINGS as BBOX_VIZ_DISPLAY_MAPPINGS

__all__ = [
    "LOADER_MAPPINGS",
    "LOADER_DISPLAY_MAPPINGS",
    "PROCESSING_MAPPINGS",
    "PROCESSING_DISPLAY_MAPPINGS",
    "MEMORY_MAPPINGS",
    "MEMORY_DISPLAY_MAPPINGS",
    "CACHE_MAPPINGS",
    "CACHE_DISPLAY_MAPPINGS",
    "BBOX_IO_MAPPINGS",
    "BBOX_IO_DISPLAY_MAPPINGS",
    "VIEWER_MAPPINGS",
    "VIEWER_DISPLAY_MAPPINGS",
    "BBOX_VIZ_MAPPINGS",
    "BBOX_VIZ_DISPLAY_MAPPINGS",
]

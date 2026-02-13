"""ComfyUI-Hunyuan3D-Part Nodes."""

from .loaders import NODE_CLASS_MAPPINGS as _loaders, NODE_DISPLAY_NAME_MAPPINGS as _loaders_d
from .processing import NODE_CLASS_MAPPINGS as _processing, NODE_DISPLAY_NAME_MAPPINGS as _processing_d
from .memory import NODE_CLASS_MAPPINGS as _memory, NODE_DISPLAY_NAME_MAPPINGS as _memory_d
from .cache import NODE_CLASS_MAPPINGS as _cache, NODE_DISPLAY_NAME_MAPPINGS as _cache_d
from .bbox_io_nodes import NODE_CLASS_MAPPINGS as _bbox_io, NODE_DISPLAY_NAME_MAPPINGS as _bbox_io_d
from .exploded_viewer import NODE_CLASS_MAPPINGS as _viewer, NODE_DISPLAY_NAME_MAPPINGS as _viewer_d
from .bbox_visualization import NODE_CLASS_MAPPINGS as _bbox_viz, NODE_DISPLAY_NAME_MAPPINGS as _bbox_viz_d
from .mesh_io import NODE_CLASS_MAPPINGS as _mesh_io, NODE_DISPLAY_NAME_MAPPINGS as _mesh_io_d

NODE_CLASS_MAPPINGS = {
    **_loaders,
    **_processing,
    **_memory,
    **_cache,
    **_bbox_io,
    **_viewer,
    **_bbox_viz,
    **_mesh_io,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_loaders_d,
    **_processing_d,
    **_memory_d,
    **_cache_d,
    **_bbox_io_d,
    **_viewer_d,
    **_bbox_viz_d,
    **_mesh_io_d,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

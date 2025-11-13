"""
ComfyUI-Hunyuan3D-Part Nodes

Node definitions for P3-SAM segmentation and X-Part generation.
"""

from .p3sam_node import NODE_CLASS_MAPPINGS as P3SAM_MAPPINGS
from .p3sam_node import NODE_DISPLAY_NAME_MAPPINGS as P3SAM_DISPLAY_MAPPINGS
from .xpart_node import NODE_CLASS_MAPPINGS as XPART_MAPPINGS
from .xpart_node import NODE_DISPLAY_NAME_MAPPINGS as XPART_DISPLAY_MAPPINGS
from .bbox_io_nodes import NODE_CLASS_MAPPINGS as BBOX_IO_MAPPINGS
from .bbox_io_nodes import NODE_DISPLAY_NAME_MAPPINGS as BBOX_IO_DISPLAY_MAPPINGS

__all__ = [
    "P3SAM_MAPPINGS",
    "P3SAM_DISPLAY_MAPPINGS",
    "XPART_MAPPINGS",
    "XPART_DISPLAY_MAPPINGS",
    "BBOX_IO_MAPPINGS",
    "BBOX_IO_DISPLAY_MAPPINGS",
]

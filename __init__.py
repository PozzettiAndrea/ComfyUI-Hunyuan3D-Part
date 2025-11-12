"""
ComfyUI Custom Nodes for Hunyuan3D-Part
3D Part Segmentation and Generation using P3-SAM and X-Part

Project: Hunyuan3D-Part
Repository: https://github.com/Tencent/Hunyuan3D-Part
HuggingFace: https://huggingface.co/tencent/Hunyuan3D-Part
"""

# Import node mappings using relative imports
from .nodes import (
    P3SAM_MAPPINGS,
    P3SAM_DISPLAY_MAPPINGS,
    XPART_MAPPINGS,
    XPART_DISPLAY_MAPPINGS,
)

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(P3SAM_MAPPINGS)
NODE_CLASS_MAPPINGS.update(XPART_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(P3SAM_DISPLAY_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(XPART_DISPLAY_MAPPINGS)

# Web directory for custom UI (optional)
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("\n" + "="*60)
print("🚀 Hunyuan3D-Part Custom Nodes Loaded")
print("="*60)
print("Available nodes:")
print("  📦 Load 3D Mesh - Load mesh files (.glb, .obj, .ply)")
print("  💾 Save 3D Mesh - Save mesh objects to files")
print("  🔍 P3-SAM Segmentation - Segment meshes into parts")
print("  ✨ X-Part Generation - Generate high-quality part meshes")
print("  🔄 Hunyuan3D Full Pipeline - Complete pipeline in one node")
print("="*60)
print("Total nodes registered:", len(NODE_CLASS_MAPPINGS))
print("="*60 + "\n")

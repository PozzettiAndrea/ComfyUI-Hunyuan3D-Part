"""Pre-startup script for Hunyuan3D-Part."""

from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input" / "3d")

# Copy example workflows to user directory
copy_files(SCRIPT_DIR / "workflows", COMFYUI_DIR / "user" / "default" / "workflows", "*.json")

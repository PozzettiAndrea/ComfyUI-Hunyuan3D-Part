"""
Pre-startup script for Hunyuan3D-Part.

This runs on EVERY ComfyUI startup BEFORE the main UI loads.
Must be FAST (<2 seconds) - no heavy imports!

Purpose:
- Setup LD_LIBRARY_PATH for CUDA extensions
- Quick validation checks
- Copy example workflows to user directory with H3DPART- prefix
- One-time first-run setup

Based on ComfyUI-HunyuanX pre-startup pattern.
"""

import os
import sys
import site
from pathlib import Path
import importlib.util
import shutil
import json


def get_node_dir():
    """Get the directory of this node package."""
    return Path(__file__).parent


def setup_torch_library_path():
    """
    Add PyTorch library path to LD_LIBRARY_PATH.

    CUDA extensions need to find libc10.so, libtorch.so, etc.
    This ensures they can find these libraries without errors.
    """
    try:
        # Get site-packages directory
        site_packages = site.getsitepackages()
        if not site_packages:
            return

        # Find torch lib directory
        torch_lib_path = None
        for sp in site_packages:
            potential_path = os.path.join(sp, 'torch', 'lib')
            if os.path.exists(potential_path):
                torch_lib_path = potential_path
                break

        if not torch_lib_path:
            return

        # Add to LD_LIBRARY_PATH if not already present
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')

        if torch_lib_path not in current_ld_path:
            if current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}"
            else:
                os.environ['LD_LIBRARY_PATH'] = torch_lib_path

            print(f"[Hunyuan3D] Added PyTorch libs to LD_LIBRARY_PATH: {torch_lib_path}")

    except Exception as e:
        # Silent fail - not critical
        pass


def check_hunyuan_installation():
    """
    Quick check if internal Hunyuan3D-Part code exists.

    This is now a standalone package with P3-SAM and XPart included internally.
    Does NOT import any heavy modules - just checks paths.
    """
    node_dir = get_node_dir()

    # Check internal p3sam directory
    p3sam_path = node_dir / "p3sam"
    xpart_path = node_dir / "xpart" / "partgen"

    if not p3sam_path.exists():
        print(f"\n⚠️  WARNING: P3-SAM code not found at {p3sam_path}")
        print(f"   The package may be corrupted. Try reinstalling.")
        return False

    if not xpart_path.exists():
        print(f"\n⚠️  WARNING: X-Part code not found at {xpart_path}")
        print(f"   The package may be corrupted. Try reinstalling.")
        return False

    return True


def check_dependencies():
    """
    Quick check for critical dependencies.

    Only checks if packages are importable, doesn't actually import them.
    """
    critical_packages = ['spconv', 'torch_scatter', 'torch_cluster', 'diffusers', 'timm']
    missing = []

    for package in critical_packages:
        # Use find_spec to check existence without importing (no torch initialization!)
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)

    if missing:
        print(f"\n⚠️  WARNING: Missing required dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print(f"\n   Run installation script:")
        print(f"   cd {get_node_dir()}")
        print(f"   python install.py")
        print()
        return False

    return True


def copy_workflows_to_user():
    """
    Copy example workflows to user workflows directory with H3DPART- prefix.

    Copies all .json files from workflows/ to ComfyUI/user/default/workflows/
    with the prefix "H3DPART-". Only copies if file doesn't exist or has changed.
    """
    try:
        node_dir = get_node_dir()
        workflows_source = node_dir / "workflows"

        # Find ComfyUI root (3 levels up from custom_nodes/ComfyUI-Hunyuan3D-Part)
        comfyui_root = node_dir.parent.parent
        workflows_target = comfyui_root / "user" / "default" / "workflows"

        # Check if source workflows directory exists
        if not workflows_source.exists():
            return

        # Create target directory if it doesn't exist
        workflows_target.mkdir(parents=True, exist_ok=True)

        # Copy all .json workflow files
        workflow_files = list(workflows_source.glob("*.json"))

        if not workflow_files:
            return

        copied_count = 0
        for src_file in workflow_files:
            # Create target filename with H3DPART- prefix
            target_file = workflows_target / f"H3DPART-{src_file.name}"

            # Check if we should copy (file doesn't exist or content differs)
            should_copy = False

            if not target_file.exists():
                should_copy = True
            else:
                # Compare file contents to avoid overwriting user modifications
                try:
                    with open(src_file, 'r') as f1, open(target_file, 'r') as f2:
                        # Parse as JSON to compare (handles formatting differences)
                        src_json = json.load(f1)
                        target_json = json.load(f2)
                        if src_json != target_json:
                            should_copy = True
                except:
                    # If JSON parse fails, compare raw content
                    should_copy = (src_file.read_bytes() != target_file.read_bytes())

            if should_copy:
                shutil.copy2(src_file, target_file)
                copied_count += 1

        if copied_count > 0:
            print(f"[Hunyuan3D] Copied {copied_count} workflow(s) to user directory")

    except Exception as e:
        # Silent fail - not critical for node functionality
        print(f"[Hunyuan3D] Warning: Could not copy workflows: {e}")
        pass


def copy_assets_to_input():
    """
    Copy asset files to ComfyUI input/3d directory.

    Copies all files from assets/ to ComfyUI/input/3d/
    Only copies if file doesn't exist or has changed.
    """
    try:
        node_dir = get_node_dir()
        assets_source = node_dir / "assets"

        # Find ComfyUI root (2 levels up from custom_nodes/ComfyUI-Hunyuan3D-Part)
        comfyui_root = node_dir.parent.parent
        assets_target = comfyui_root / "input" / "3d"

        # Check if source assets directory exists
        if not assets_source.exists():
            return

        # Create target directory if it doesn't exist
        assets_target.mkdir(parents=True, exist_ok=True)

        # Get all files from assets directory
        asset_files = [f for f in assets_source.iterdir() if f.is_file()]

        if not asset_files:
            return

        copied_count = 0
        for src_file in asset_files:
            # Create target filename (no prefix for assets)
            target_file = assets_target / src_file.name

            # Check if we should copy (file doesn't exist or content differs)
            should_copy = False

            if not target_file.exists():
                should_copy = True
            else:
                # Compare file contents (byte comparison for binary files)
                should_copy = (src_file.read_bytes() != target_file.read_bytes())

            if should_copy:
                shutil.copy2(src_file, target_file)
                copied_count += 1

        if copied_count > 0:
            print(f"[Hunyuan3D] Copied {copied_count} asset(s) to input/3d directory")

    except Exception as e:
        # Silent fail - not critical for node functionality
        print(f"[Hunyuan3D] Warning: Could not copy assets: {e}")
        pass


def main():
    """
    Main pre-startup routine.

    Runs before ComfyUI loads nodes.
    Must be fast - no heavy imports!
    """
    # Setup environment
    setup_torch_library_path()

    # Quick validation (don't block startup if validation fails)
    repo_ok = check_hunyuan_installation()
    deps_ok = check_dependencies()

    # Copy example workflows to user directory
    copy_workflows_to_user()

    # Copy asset files to input/3d directory
    copy_assets_to_input()

    # Only print if everything is OK (reduce console spam)
    if repo_ok and deps_ok:
        print("[Hunyuan3D] Pre-startup checks passed")


# Run pre-startup checks
try:
    main()
except Exception as e:
    print(f"[Hunyuan3D] Pre-startup error: {e}")
    # Don't crash ComfyUI startup
    pass

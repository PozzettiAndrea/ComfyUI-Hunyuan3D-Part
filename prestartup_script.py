"""
Pre-startup script for Hunyuan3D-Part.

This runs on EVERY ComfyUI startup BEFORE the main UI loads.
Must be FAST (<2 seconds) - no heavy imports!

Purpose:
- Setup LD_LIBRARY_PATH for CUDA extensions
- Quick validation checks
- One-time first-run setup

Based on ComfyUI-HunyuanX pre-startup pattern.
"""

import os
import sys
import site
from pathlib import Path
import importlib.util


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

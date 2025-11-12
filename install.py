"""
Installation script for Hunyuan3D-Part dependencies.

This script is called by ComfyUI Manager during install/update.
Handles complex CUDA-specific dependencies that can't be installed via simple pip.

Based on ComfyUI-HunyuanX dependency management strategy.
"""

import subprocess
import sys
import os
from pathlib import Path


def get_torch_cuda_version():
    """
    Detect PyTorch and CUDA versions.

    Returns:
        tuple: (torch_version, cuda_version) e.g., ("2.9", "128")
               or (None, None) if detection fails
    """
    try:
        import torch

        # Get torch version: "2.9.0+cu128" -> "2.9"
        torch_version = torch.__version__.split('+')[0]
        torch_major_minor = '.'.join(torch_version.split('.')[:2])

        # Get CUDA version: "12.8" -> "128"
        if torch.version.cuda:
            cuda_version = torch.version.cuda.replace('.', '')
        else:
            cuda_version = None

        return torch_major_minor, cuda_version

    except Exception as e:
        print(f"⚠️  Could not detect torch/CUDA version: {e}")
        return None, None


def get_python_version():
    """
    Get Python version string for wheel filenames.

    Returns:
        str: e.g., "cp310" for Python 3.10
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def run_pip_install(package, timeout=300, extra_args=None):
    """
    Run pip install with timeout and error handling.

    Args:
        package: Package name or URL
        timeout: Timeout in seconds
        extra_args: List of additional arguments

    Returns:
        bool: True if successful, False otherwise
    """
    cmd = [sys.executable, "-m", "pip", "install", package]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏱️  Installation timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def install_spconv():
    """
    Install spconv (sparse convolution library).

    Required for P3-SAM's Sonata 3D feature extractor.
    Must match CUDA version exactly.

    Returns:
        bool: True if installed successfully
    """
    print("\n📦 Installing spconv (sparse convolution library)...")

    # Check if already installed
    try:
        import spconv
        print("  ✅ spconv already installed")
        return True
    except ImportError:
        pass

    # Detect CUDA version
    torch_ver, cuda_ver = get_torch_cuda_version()

    if not cuda_ver:
        print("  ⚠️  Could not detect CUDA version, trying default spconv")
        return run_pip_install("spconv")

    # Install appropriate spconv version
    package_name = f"spconv-cu{cuda_ver}"
    print(f"  📥 Installing {package_name} for CUDA {cuda_ver}...")

    success = run_pip_install(package_name, timeout=300)

    if success:
        print(f"  ✅ {package_name} installed successfully")
    else:
        print(f"  ⚠️  {package_name} installation failed, trying generic spconv")
        success = run_pip_install("spconv")

    return success


def install_torch_geometric_packages():
    """
    Install torch-scatter and torch-cluster from PyTorch Geometric wheel repo.

    These packages are required for point cloud processing but only have
    source distributions on PyPI. PyG hosts prebuilt wheels.

    Returns:
        tuple: (scatter_ok, cluster_ok)
    """
    print("\n📦 Installing PyTorch Geometric packages...")

    scatter_ok = False
    cluster_ok = False

    # Check existing installations
    try:
        import torch_scatter
        print("  ✅ torch-scatter already installed")
        scatter_ok = True
    except ImportError:
        pass

    try:
        import torch_cluster
        print("  ✅ torch-cluster already installed")
        cluster_ok = True
    except ImportError:
        pass

    if scatter_ok and cluster_ok:
        return scatter_ok, cluster_ok

    # Detect versions
    torch_ver, cuda_ver = get_torch_cuda_version()

    if not torch_ver or not cuda_ver:
        print("  ⚠️  Could not detect torch/CUDA version")
        print("  ⚠️  PyTorch Geometric packages require version matching")
        return scatter_ok, cluster_ok

    # PyTorch Geometric wheel repository
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_ver}+cu{cuda_ver}.html"
    print(f"  🌐 Using PyG wheel repo: torch-{torch_ver}+cu{cuda_ver}")

    # Install torch-scatter
    if not scatter_ok:
        print("  📥 Installing torch-scatter...")
        scatter_ok = run_pip_install(
            "torch-scatter",
            extra_args=["-f", wheel_url],
            timeout=300
        )
        if scatter_ok:
            print("  ✅ torch-scatter installed")
        else:
            print("  ⚠️  torch-scatter installation failed")

    # Install torch-cluster
    if not cluster_ok:
        print("  📥 Installing torch-cluster...")
        cluster_ok = run_pip_install(
            "torch-cluster",
            extra_args=["-f", wheel_url],
            timeout=300
        )
        if cluster_ok:
            print("  ✅ torch-cluster installed")
        else:
            print("  ⚠️  torch-cluster installation failed")

    return scatter_ok, cluster_ok


def install_requirements():
    """
    Install standard requirements from requirements.txt.

    Returns:
        bool: True if successful
    """
    print("\n📦 Installing standard requirements...")

    script_dir = Path(__file__).parent
    req_file = script_dir / "requirements.txt"

    if not req_file.exists():
        print(f"  ⚠️  requirements.txt not found at {req_file}")
        return False

    print(f"  📥 Installing from {req_file}...")
    success = run_pip_install(f"-r {req_file}", timeout=600)

    if success:
        print("  ✅ Requirements installed successfully")
    else:
        print("  ⚠️  Some requirements failed to install")

    return success


def print_installation_summary(req_ok, spconv_ok, scatter_ok, cluster_ok):
    """Print a summary of what was installed."""
    print("\n" + "="*60)
    print("🎯 Hunyuan3D-Part Installation Summary")
    print("="*60)

    status = lambda x: "✅" if x else "⚠️ "

    print(f"{status(req_ok)} Standard requirements (requirements.txt)")
    print(f"{status(spconv_ok)} spconv (REQUIRED - sparse convolution for P3-SAM)")
    print(f"{status(scatter_ok)} torch-scatter (REQUIRED - point cloud ops)")
    print(f"{status(cluster_ok)} torch-cluster (REQUIRED - FPS sampling)")

    print("\n" + "-"*60)

    if all([req_ok, spconv_ok, scatter_ok, cluster_ok]):
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("\nYou can now use Hunyuan3D-Part nodes.")
    else:
        print("⚠️  SOME DEPENDENCIES FAILED TO INSTALL")
        print("\nMissing dependencies:")
        if not req_ok:
            print("  - Standard requirements (check requirements.txt)")
        if not spconv_ok:
            print("  - spconv (try: pip install spconv-cu<your_cuda_version>)")
        if not scatter_ok:
            print("  - torch-scatter (see: https://pytorch-geometric.readthedocs.io)")
        if not cluster_ok:
            print("  - torch-cluster (see: https://pytorch-geometric.readthedocs.io)")
        print("\nThe nodes may not work until these are installed.")

    print("="*60 + "\n")


def main():
    """Main installation routine."""
    print("\n" + "="*60)
    print("🚀 Installing Hunyuan3D-Part Dependencies")
    print("="*60)

    # Detect environment
    torch_ver, cuda_ver = get_torch_cuda_version()
    py_ver = get_python_version()

    print(f"\n🔍 Detected environment:")
    print(f"  Python: {sys.version.split()[0]} ({py_ver})")
    print(f"  PyTorch: {torch_ver or 'Not detected'}")
    print(f"  CUDA: {cuda_ver or 'Not detected'}")

    # Install in order
    req_ok = install_requirements()
    spconv_ok = install_spconv()
    scatter_ok, cluster_ok = install_torch_geometric_packages()

    # Summary
    print_installation_summary(req_ok, spconv_ok, scatter_ok, cluster_ok)

    return all([req_ok, spconv_ok, scatter_ok, cluster_ok])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

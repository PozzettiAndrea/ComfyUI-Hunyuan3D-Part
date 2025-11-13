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
        tuple: (torch_version, cuda_version) e.g., ("2.8.0", "128")
               or (None, None) if detection fails
    """
    try:
        import torch

        # Get torch version: "2.8.0+cu128" -> "2.8.0"
        # PyG wheel repo requires full version (major.minor.patch)
        torch_version = torch.__version__.split('+')[0]

        # Get CUDA version: "12.8" -> "128"
        if torch.version.cuda:
            cuda_version = torch.version.cuda.replace('.', '')
        else:
            cuda_version = None

        return torch_version, cuda_version

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


def run_pip_install(package, timeout=300, extra_args=None, use_uv=False, only_binary=False):
    """
    Run pip install with timeout and error handling.

    Args:
        package: Package name or URL
        timeout: Timeout in seconds
        extra_args: List of additional arguments
        use_uv: If True, try uv pip install first (faster)
        only_binary: If True, add --only-binary=:all: to prevent source builds

    Returns:
        tuple: (success: bool, building_from_source: bool)
    """
    # Build extra args list
    all_extra_args = extra_args or []
    if only_binary:
        all_extra_args.extend(["--only-binary", ":all:"])

    # Try uv first if requested
    if use_uv:
        try:
            cmd = ["uv", "pip", "install", package] + all_extra_args
            uv_result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True
            )

            # Check if uv tried to build from source
            output = uv_result.stdout + uv_result.stderr
            building_from_source = any(indicator in output.lower() for indicator in [
                "building wheel",
                "running setup.py",
                "building extension",
                "compiling"
            ])

            if uv_result.returncode == 0:
                return True, building_from_source

            # If uv failed but was trying to build from source, return info
            if building_from_source:
                return False, True

        except (FileNotFoundError, subprocess.TimeoutExpired):
            # uv not available or timed out, fall back to pip
            pass

    # Standard pip install
    cmd = [sys.executable, "-m", "pip", "install", package] + all_extra_args

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True
        )

        # Check if pip is building from source
        output = result.stdout + result.stderr
        building_from_source = any(indicator in output.lower() for indicator in [
            "building wheel",
            "running setup.py",
            "building extension",
            "compiling"
        ])

        return result.returncode == 0, building_from_source

    except subprocess.TimeoutExpired:
        print(f"⏱️  Installation timed out after {timeout}s")
        return False, True  # Timeout usually means building from source
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False, False


def get_spconv_fallback_versions(cuda_ver):
    """
    Get list of spconv CUDA versions to try, in order of preference.

    Args:
        cuda_ver: CUDA version string (e.g., "128")

    Returns:
        list: Package names to try in order (e.g., ["spconv-cu128", "spconv-cu126", ...])
    """
    # Available spconv CUDA versions on PyPI (as of Jan 2025)
    # Ordered from newest to oldest
    available_versions = ["126", "124", "121", "120", "118", "117", "114", "113", "102"]

    # Start with exact match
    fallback_list = [f"spconv-cu{cuda_ver}"]

    # Separate into same major version and different major version
    cuda_major = cuda_ver[:2] if len(cuda_ver) >= 2 else cuda_ver[:1]
    cuda_num = int(cuda_ver) if cuda_ver.isdigit() else 0

    same_major = []
    other_versions = []

    for ver in available_versions:
        ver_major = ver[:2] if len(ver) >= 2 else ver[:1]
        ver_num = int(ver) if ver.isdigit() else 0

        if ver_major == cuda_major:
            # Same major version - prefer closest version (descending from target)
            same_major.append((ver_num, f"spconv-cu{ver}"))
        else:
            other_versions.append(f"spconv-cu{ver}")

    # Sort same_major by version number descending (closest to target first)
    same_major.sort(reverse=True, key=lambda x: x[0])

    # Build final list: exact match, then same major (descending), then others
    fallback_list.extend([pkg for _, pkg in same_major])
    fallback_list.extend(other_versions)

    return fallback_list


def install_spconv():
    """
    Install spconv (sparse convolution library) with CUDA support.

    Required for P3-SAM's Sonata 3D feature extractor.
    Tries exact CUDA version match first with uv (fast), then falls back to
    closest available version if building from source.
    CUDA minor versions are forward compatible (e.g., cu126 works with CUDA 12.8).

    Returns:
        bool: True if installed successfully
    """
    print("\n📦 Installing spconv (sparse convolution library)...")

    # Check if already installed and has CUDA support
    try:
        import spconv
        import torch

        # Test if spconv can use CUDA (not just CPU-only build)
        if torch.cuda.is_available():
            try:
                # Quick test: create a sparse tensor on GPU
                import spconv.pytorch as spconv_torch
                print("  ✅ spconv already installed with CUDA support")
                return True
            except Exception:
                print("  ⚠️  spconv installed but may be CPU-only, reinstalling...")
                # Uninstall and reinstall with CUDA support
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "spconv", "-y"],
                             capture_output=True)
        else:
            print("  ✅ spconv already installed (CUDA not available)")
            return True
    except ImportError:
        pass

    # Detect CUDA version
    torch_ver, cuda_ver = get_torch_cuda_version()

    if not cuda_ver:
        print("  ⚠️  Could not detect CUDA version, trying default spconv")
        success, _ = run_pip_install("spconv")
        return success

    print(f"  🔍 Target CUDA version: {cuda_ver}")

    # Step 1: Try exact match with uv (wheels only, no compilation)
    exact_match = f"spconv-cu{cuda_ver}"
    print(f"  📥 Step 1: Trying {exact_match} from PyPI (wheels only)...")

    success, _ = run_pip_install(
        exact_match,
        timeout=60,
        use_uv=True,
        only_binary=True
    )

    if success:
        print(f"  ✅ {exact_match} installed successfully (prebuilt wheel)")
        return True

    print(f"  ⚠️  {exact_match} not available as prebuilt wheel")

    # Step 2: Try fallback versions - look for prebuilt wheels only
    fallback_versions = get_spconv_fallback_versions(cuda_ver)[1:]
    print(f"  📥 Step 2: Trying fallback versions (wheels only)...")

    for package_name in fallback_versions:
        print(f"    ├─ Trying {package_name}...")

        success, _ = run_pip_install(
            package_name,
            timeout=60,
            use_uv=True,
            only_binary=True
        )

        if success:
            print(f"  ✅ {package_name} installed successfully (prebuilt wheel)")
            return True

    print(f"  ⚠️  No prebuilt CUDA wheels found for CUDA {cuda_ver}")

    # Last resort: generic spconv (CPU-only)
    print("  ⚠️  No prebuilt CUDA wheels found, trying generic spconv (CPU-only)")
    success, _ = run_pip_install("spconv")
    if success:
        print("  ⚠️  Installed CPU-only spconv (GPU acceleration not available)")
        print("  💡 Consider building spconv from source for CUDA support")

    return success


def install_flash_attn():
    """
    Install flash-attn (optional performance optimization for Sonata model).

    Uses 3-step approach:
    1. Try PyPI with uv (wheels only, no compilation)
    2. Try mjun0812's prebuilt wheels for PyTorch/CUDA/Python combo
    3. Skip gracefully if no wheels found (flash-attn is optional)

    Returns:
        bool: True if installed successfully, False if skipped
    """
    print("\n📦 Installing flash-attn (optional performance optimization)...")

    # Check if already installed
    try:
        import flash_attn
        print("  ✅ flash-attn already installed")
        return True
    except ImportError:
        pass

    # Detect versions
    torch_ver, cuda_ver = get_torch_cuda_version()
    py_ver = get_python_version()

    if not torch_ver or not cuda_ver:
        print("  ⚠️  Could not detect torch/CUDA version, skipping flash-attn")
        return False

    print(f"  🔍 Environment: PyTorch {torch_ver}, CUDA {cuda_ver}, Python {py_ver}")

    # Step 1: Try PyPI with uv (wheels only, no compilation)
    print(f"  📥 Step 1: Trying PyPI (wheels only)...")
    success, _ = run_pip_install(
        "flash-attn>=2.6.0",
        timeout=60,
        use_uv=True,
        only_binary=True
    )

    if success:
        print("  ✅ flash-attn installed from PyPI")
        return True

    print("  ⚠️  No PyPI wheel found for your environment")

    # Step 2: Try mjun0812's prebuilt wheels from GitHub
    print(f"  📥 Step 2: Trying prebuilt wheels from GitHub...")

    # GitHub releases URL pattern
    base_url = "https://github.com/mjun0812/flash-attn-wheels/releases/download"

    # Try multiple flash-attn versions
    flash_versions = ["2.8.1", "2.7.0", "2.6.3"]

    for flash_ver in flash_versions:
        # Construct wheel URL
        # Format: flash_attn-{version}+torch{torch_ver}cu{cuda_ver}-{py_ver}-{py_ver}-linux_x86_64.whl
        wheel_filename = f"flash_attn-{flash_ver}+torch{torch_ver}cu{cuda_ver}-{py_ver}-{py_ver}-linux_x86_64.whl"
        wheel_url = f"{base_url}/torch{torch_ver}-cu{cuda_ver}/{wheel_filename}"

        print(f"    ├─ Trying flash-attn {flash_ver}...")

        success, _ = run_pip_install(
            wheel_url,
            timeout=60,
            use_uv=True
        )

        if success:
            print(f"  ✅ flash-attn {flash_ver} installed from GitHub")
            return True

    print("  ⚠️  No prebuilt wheels found on GitHub")

    # Step 3: Skip gracefully
    print("  💡 Skipping flash-attn (optional - provides 10-20% speedup)")
    print("  💡 Nodes will work without it, just slightly slower")
    return False


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
        scatter_ok, _ = run_pip_install(
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
        cluster_ok, _ = run_pip_install(
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

    # Use direct subprocess call with correct argument format
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    try:
        result = subprocess.run(
            cmd,
            timeout=600,
            capture_output=True,
            text=True
        )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Installation timed out after 600s")
        success = False
    except Exception as e:
        print(f"  ❌ Installation error: {e}")
        success = False

    if success:
        print("  ✅ Requirements installed successfully")
    else:
        print("  ⚠️  Some requirements failed to install")

    return success


def print_installation_summary(req_ok, spconv_ok, flash_attn_ok, scatter_ok, cluster_ok):
    """Print a summary of what was installed."""
    print("\n" + "="*60)
    print("🎯 Hunyuan3D-Part Installation Summary")
    print("="*60)

    status = lambda x: "✅" if x else "⚠️ "
    optional_status = lambda x: "✅" if x else "⊘ "

    print(f"{status(req_ok)} Standard requirements (requirements.txt)")
    print(f"{status(spconv_ok)} spconv (REQUIRED - sparse convolution for P3-SAM)")
    print(f"{optional_status(flash_attn_ok)} flash-attn (OPTIONAL - 10-20% speedup for Sonata)")
    print(f"{status(scatter_ok)} torch-scatter (REQUIRED - point cloud ops)")
    print(f"{status(cluster_ok)} torch-cluster (REQUIRED - FPS sampling)")

    print("\n" + "-"*60)

    required_ok = all([req_ok, spconv_ok, scatter_ok, cluster_ok])

    if required_ok:
        print("✅ ALL REQUIRED DEPENDENCIES INSTALLED SUCCESSFULLY!")
        if not flash_attn_ok:
            print("⊘  flash-attn not installed (optional - provides modest speedup)")
        print("\nYou can now use Hunyuan3D-Part nodes.")
    else:
        print("⚠️  SOME REQUIRED DEPENDENCIES FAILED TO INSTALL")
        print("\nMissing dependencies:")
        if not req_ok:
            print("  - Standard requirements (check requirements.txt)")
        if not spconv_ok:
            print("  - spconv (REQUIRED - try: pip install spconv-cu<your_cuda_version>)")
        if not scatter_ok:
            print("  - torch-scatter (REQUIRED - see: https://pytorch-geometric.readthedocs.io)")
        if not cluster_ok:
            print("  - torch-cluster (REQUIRED - see: https://pytorch-geometric.readthedocs.io)")
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
    flash_attn_ok = install_flash_attn()  # Optional
    scatter_ok, cluster_ok = install_torch_geometric_packages()

    # Summary
    print_installation_summary(req_ok, spconv_ok, flash_attn_ok, scatter_ok, cluster_ok)

    # Only require critical dependencies (flash-attn is optional)
    return all([req_ok, spconv_ok, scatter_ok, cluster_ok])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

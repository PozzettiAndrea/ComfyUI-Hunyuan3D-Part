"""
Test script to verify the node structure without requiring all dependencies.
This validates that the package structure is correct for ComfyUI loading.
"""

import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_structure():
    """Test that all required files exist."""
    print("Testing ComfyUI-Hunyuan3D-Part structure...\n")

    required_files = [
        "__init__.py",
        "README.md",
        "requirements.txt",
        "nodes/p3sam_node.py",
        "nodes/xpart_node.py",
        "utils/mesh_utils.py",
        "utils/model_loader.py",
    ]

    all_exist = True
    for file in required_files:
        path = os.path.join(current_dir, file)
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {file}")
        if not exists:
            all_exist = False

    print()

    if all_exist:
        print("✅ All required files present")
    else:
        print("❌ Some files are missing")
        return False

    # Check that files can be parsed (syntax check)
    print("\nChecking Python syntax...")
    py_files = ["__init__.py", "nodes/p3sam_node.py", "nodes/xpart_node.py",
                "utils/mesh_utils.py", "utils/model_loader.py"]

    for file in py_files:
        path = os.path.join(current_dir, file)
        try:
            with open(path, 'r') as f:
                compile(f.read(), path, 'exec')
            print(f"✅ {file}")
        except SyntaxError as e:
            print(f"❌ {file}: {e}")
            return False

    print("\n✅ All files have valid Python syntax")
    print("\n" + "="*60)
    print("Package structure validation: SUCCESS")
    print("="*60)
    print("\nNote: Actual functionality requires dependencies:")
    print("  - PyTorch with CUDA support")
    print("  - trimesh, numpy, spconv, torch-scatter, etc.")
    print("  - Hunyuan3D-Part repository cloned in parent directory")
    print("\nTo install dependencies:")
    print("  pip install -r requirements.txt")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_structure()
    sys.exit(0 if success else 1)

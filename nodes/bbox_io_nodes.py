"""
Bounding Box I/O Nodes for ComfyUI.
Provides nodes for saving and loading bounding boxes to/from JSON files.
"""

import json
import numpy as np
import folder_paths
import os
from pathlib import Path


class SaveBoundingBoxes:
    """
    Save BBOXES_3D to JSON file in ComfyUI output directory.
    Useful for caching P3-SAM segmentation results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bounding_boxes": ("BBOXES_3D",),
                "filename": ("STRING", {
                    "default": "bboxes.json",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    CATEGORY = "Hunyuan3D"
    OUTPUT_NODE = True

    def save(self, bounding_boxes, filename):
        """Save bounding boxes to JSON file."""
        try:
            # Ensure correct extension
            if not filename.endswith(".json"):
                filename = f"{filename.rsplit('.', 1)[0]}.json"

            # Save to output directory
            output_dir = folder_paths.get_output_directory()
            output_path = os.path.join(output_dir, filename)

            # Extract data from BBOXES_3D dict
            bboxes_array = bounding_boxes['bboxes']
            num_parts = bounding_boxes['num_parts']

            # Convert numpy array to list for JSON serialization
            bboxes_list = bboxes_array.tolist()

            # Create JSON structure
            data = {
                "num_parts": int(num_parts),
                "bboxes": bboxes_list
            }

            # Save to file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[SaveBoundingBoxes] Saved {num_parts} bounding boxes to: {output_path}")

            return (output_path,)

        except Exception as e:
            print(f"[SaveBoundingBoxes] Error saving bounding boxes: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


class LoadBoundingBoxes:
    """
    Load BBOXES_3D from JSON file.
    Useful for restoring cached P3-SAM segmentation results.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("BBOXES_3D",)
    RETURN_NAMES = ("bounding_boxes",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D"

    def load(self, file_path):
        """Load bounding boxes from JSON file."""
        try:
            # Handle empty path
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"Bounding boxes file not found: {file_path}")

            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Validate structure
            if 'num_parts' not in data or 'bboxes' not in data:
                raise ValueError("Invalid bounding boxes JSON format. Must contain 'num_parts' and 'bboxes' keys.")

            num_parts = data['num_parts']
            bboxes_list = data['bboxes']

            # Validate bboxes structure
            if len(bboxes_list) != num_parts:
                raise ValueError(f"Mismatch: num_parts={num_parts} but found {len(bboxes_list)} bounding boxes")

            # Convert to numpy array
            bboxes_array = np.array(bboxes_list, dtype=np.float32)

            # Validate shape [N, 2, 3]
            if bboxes_array.ndim != 3 or bboxes_array.shape[1] != 2 or bboxes_array.shape[2] != 3:
                raise ValueError(f"Invalid bboxes shape: {bboxes_array.shape}. Expected [N, 2, 3]")

            # Create BBOXES_3D dict
            bboxes_output = {
                'bboxes': bboxes_array,
                'num_parts': num_parts
            }

            print(f"[LoadBoundingBoxes] Loaded {num_parts} bounding boxes from: {file_path}")

            return (bboxes_output,)

        except Exception as e:
            print(f"[LoadBoundingBoxes] Error loading bounding boxes: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SaveBoundingBoxes": SaveBoundingBoxes,
    "LoadBoundingBoxes": LoadBoundingBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveBoundingBoxes": "Save Bounding Boxes",
    "LoadBoundingBoxes": "Load Bounding Boxes",
}

"""
Bounding Box I/O Nodes for ComfyUI.
Provides nodes for saving and loading bounding boxes to/from JSON files.
"""

import json
import numpy as np
import folder_paths
import os
from pathlib import Path
from comfy_api.latest import io


class SaveBoundingBoxes(io.ComfyNode):
    """
    Save BBOXES_3D to JSON file in ComfyUI output directory.
    Useful for caching P3-SAM segmentation results.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SaveBoundingBoxes",
            display_name="Save Bounding Boxes",
            category="Hunyuan3D/IO",
            is_output_node=True,
            inputs=[
                io.Custom("BBOXES_3D").Input("bounding_boxes"),
                io.String.Input("filename", default="bboxes.json", multiline=False),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(cls, bounding_boxes, filename):
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

            return io.NodeOutput(output_path)

        except Exception as e:
            print(f"[SaveBoundingBoxes] Error saving bounding boxes: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class LoadBoundingBoxes(io.ComfyNode):
    """
    Load BBOXES_3D from JSON file.
    Useful for restoring cached P3-SAM segmentation results.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadBoundingBoxes",
            display_name="Load Bounding Boxes",
            category="Hunyuan3D/IO",
            inputs=[
                io.String.Input("file_path", default="", multiline=False),
            ],
            outputs=[
                io.Custom("BBOXES_3D").Output(display_name="bounding_boxes"),
            ],
        )

    @classmethod
    def execute(cls, file_path):
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

            return io.NodeOutput(bboxes_output)

        except Exception as e:
            print(f"[LoadBoundingBoxes] Error loading bounding boxes: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SaveBoundingBoxes": SaveBoundingBoxes,
    "LoadBoundingBoxes": LoadBoundingBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveBoundingBoxes": "Save Bounding Boxes",
    "LoadBoundingBoxes": "Load Bounding Boxes",
}

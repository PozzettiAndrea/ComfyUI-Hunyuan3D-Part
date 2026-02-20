import torch
import torch.nn as nn

from typing import Dict, Optional

from . import load_by_config
from . import transform as sonata_transform


class SonataFeatureExtractor(nn.Module):
    """
    Feature extractor using Sonata backbone with MLP projection.
    Supports batch processing and gradient computation.
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = "",
        enable_flash: bool = True,
    ):
        super().__init__()

        # Load Sonata model with enable_flash override
        custom_config = {"enable_flash": enable_flash}
        self.sonata = load_by_config(custom_config=custom_config)

        # Define MLP projection head (same as in train-sonata.py)
        self.mlp = nn.Sequential(
            nn.Linear(1232, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

        # Define transform
        self.transform = sonata_transform.default()

        # Load checkpoint if provided
        if ckpt_path:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(checkpoint_path, device="cuda")
            state_dict = checkpoint
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            else:
                state_dict = checkpoint

        print("\n=== Checkpoint Keys ===")
        print(f"Total keys in checkpoint: {len(state_dict)}")
        print("\nSample keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more keys")

        sonata_dict = {
            k.replace("sonata.", ""): v
            for k, v in state_dict.items()
            if k.startswith("sonata.")
        }
        mlp_dict = {
            k.replace("mlp.", ""): v
            for k, v in state_dict.items()
            if k.startswith("mlp.")
        }

        print(f"\nFound {len(sonata_dict)} Sonata keys")
        print(f"Found {len(mlp_dict)} MLP keys")

        if sonata_dict:
            print("\n=== Loading Sonata Weights ===")
            result = self.sonata.load_state_dict(sonata_dict, strict=False)
            if result.missing_keys:
                print(f"\nMissing keys ({len(result.missing_keys)}):")
                for key in result.missing_keys[:20]:
                    print(f"  - {key}")
                if len(result.missing_keys) > 20:
                    print(f"  ... and {len(result.missing_keys) - 20} more")
            else:
                print("No missing keys!")

            if result.unexpected_keys:
                print(f"\nUnexpected keys ({len(result.unexpected_keys)}):")
                for key in result.unexpected_keys[:20]:
                    print(f"  - {key}")
                if len(result.unexpected_keys) > 20:
                    print(f"  ... and {len(result.unexpected_keys) - 20} more")
            else:
                print("No unexpected keys!")

        if mlp_dict:
            print("\n=== Loading MLP Weights ===")
            result = self.mlp.load_state_dict(mlp_dict, strict=False)
            if result.missing_keys:
                print(f"\nMissing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"Unexpected keys: {result.unexpected_keys}")
            print("MLP weights loaded successfully!")

        print(f"\n[OK] Loaded checkpoint from {checkpoint_path}")

    def prepare_batch_data(
        self, points: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Prepare batch data for Sonata model.

        Args:
            points: [B, N, 3] or [N, 3] tensor of point coordinates
            normals: [B, N, 3] or [N, 3] tensor of normals (optional)

        Returns:
            Dictionary formatted for Sonata input
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)
            if normals is not None:
                normals = normals.unsqueeze(0)
        B, N, _ = points.shape

        batch_idx = torch.arange(B).view(-1, 1).repeat(1, N).reshape(-1)

        coord = points.reshape(B * N, 3)

        if normals is not None:
            normal = normals.reshape(B * N, 3)
        else:
            normal = torch.ones_like(coord)

        color = torch.ones_like(coord)

        def to_numpy(tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            if tensor.dtype not in [
                torch.float32,
                torch.float64,
                torch.int32,
                torch.int64,
                torch.uint8,
                torch.int8,
                torch.int16,
            ]:
                tensor = tensor.to(torch.float32)
            return tensor.numpy()

        data_dict = {
            "coord": to_numpy(coord),
            "normal": to_numpy(normal),
            "color": to_numpy(color),
            "batch": to_numpy(batch_idx),
        }

        data_dict = self.transform(data_dict)

        return data_dict, B, N

    def forward(
        self, points: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from point clouds.

        Args:
            points: [B, N, 3] or [N, 3] tensor of point coordinates
            normals: [B, N, 3] or [N, 3] tensor of normals (optional)

        Returns:
            features: [B, N, 512] or [N, 512] tensor of features
        """
        original_shape = points.shape
        single_batch = points.dim() == 2

        data_dict, B, N = self.prepare_batch_data(points, normals)

        device = points.device
        dtype = points.dtype

        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                if data_dict[key].is_floating_point():
                    data_dict[key] = data_dict[key].to(device=device, dtype=dtype)
                else:
                    data_dict[key] = data_dict[key].to(device)

        point = self.sonata(data_dict)

        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        feat = point.feat  # [M, 1232]
        feat = self.mlp(feat)  # [M, 512]

        feat = feat[point.inverse]  # [B*N, 512]

        feat = feat.reshape(B, -1, feat.shape[-1])  # [B, N, 512]

        if single_batch:
            feat = feat.squeeze(0)  # [N, 512]

        return feat

    def extract_features_batch(
        self,
        points_list: list,
        normals_list: Optional[list] = None,
        batch_size: int = 8,
    ) -> list:
        """
        Extract features for multiple point clouds in batches.

        Args:
            points_list: List of [N_i, 3] tensors
            normals_list: List of [N_i, 3] tensors (optional)
            batch_size: Batch size for processing

        Returns:
            List of [N_i, 512] feature tensors
        """
        features_list = []

        for i in range(0, len(points_list), batch_size):
            batch_points = points_list[i : i + batch_size]
            batch_normals = normals_list[i : i + batch_size] if normals_list else None

            max_n = max(p.shape[0] for p in batch_points)

            padded_points = []
            masks = []
            for points in batch_points:
                n = points.shape[0]
                if n < max_n:
                    padding = torch.zeros(max_n - n, 3, device=points.device)
                    points = torch.cat([points, padding], dim=0)
                padded_points.append(points)
                mask = torch.zeros(max_n, dtype=torch.bool, device=points.device)
                mask[:n] = True
                masks.append(mask)

            batch_tensor = torch.stack(padded_points)  # [B, max_n, 3]

            if batch_normals:
                padded_normals = []
                for j, normals in enumerate(batch_normals):
                    n = normals.shape[0]
                    if n < max_n:
                        padding = torch.ones(max_n - n, 3, device=normals.device)
                        normals = torch.cat([normals, padding], dim=0)
                    padded_normals.append(normals)
                normals_tensor = torch.stack(padded_normals)
            else:
                normals_tensor = None

            with torch.cuda.amp.autocast(enabled=True):
                batch_features = self.forward(
                    batch_tensor, normals_tensor
                )  # [B, max_n, 512]

            for j, (feat, mask) in enumerate(zip(batch_features, masks)):
                features_list.append(feat[mask])

        return features_list

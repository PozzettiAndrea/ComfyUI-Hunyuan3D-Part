import torch
import torch.nn as nn
import os

from .. import sonata


class MultiHeadSegment(nn.Module):
    '''
    P3-SAM model: Sonata backbone + two-stage multi-head segmentor + IoU predictor.
    '''

    def __init__(self, in_channel=512, head_num=3, ignore_label=-100, enable_flash=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channel = in_channel
        self.head_num = head_num
        self.ignore_label = ignore_label
        self.enable_flash = enable_flash
        self._build(dtype=dtype, device=device, operations=operations)

    def _build(self, dtype=None, device=None, operations=None):
        if operations is None:
            operations = nn

        # Sonata backbone (spconv — no operations=)
        try:
            import folder_paths
            sonata_cache = os.path.join(folder_paths.models_dir, "sonata")
        except Exception:
            sonata_cache = os.path.expanduser("~/.cache/sonata/ckpt")

        custom_config = {"enable_flash": self.enable_flash}
        self.sonata = sonata.load("sonata", repo_id="facebook/sonata", download_root=sonata_cache, custom_config=custom_config)
        self.mlp = nn.Sequential(
            operations.Linear(1232, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 512, dtype=dtype, device=device),
        )
        self.transform = sonata.transform.default()

        # SEG1: three segmentation heads
        self.seg_mlp_1 = nn.Sequential(
            operations.Linear(512 + 3 + 3, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 1, dtype=dtype, device=device),
        )
        self.seg_mlp_2 = nn.Sequential(
            operations.Linear(512 + 3 + 3, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 1, dtype=dtype, device=device),
        )
        self.seg_mlp_3 = nn.Sequential(
            operations.Linear(512 + 3 + 3, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 512, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(512, 1, dtype=dtype, device=device),
        )

        # SEG2: global + per-head refinement
        self.seg_s2_mlp_g = nn.Sequential(
            operations.Linear(512 + 3 + 3 + 3, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
        )
        self.seg_s2_mlp_1 = nn.Sequential(
            operations.Linear(512 + 3 + 3 + 3 + 256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 1, dtype=dtype, device=device),
        )
        self.seg_s2_mlp_2 = nn.Sequential(
            operations.Linear(512 + 3 + 3 + 3 + 256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 1, dtype=dtype, device=device),
        )
        self.seg_s2_mlp_3 = nn.Sequential(
            operations.Linear(512 + 3 + 3 + 3 + 256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 1, dtype=dtype, device=device),
        )

        # IoU predictor
        self.iou_mlp = nn.Sequential(
            operations.Linear(512 + 3 + 3 + 3 + 256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
        )
        self.iou_mlp_out = nn.Sequential(
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 256, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(256, 3, dtype=dtype, device=device),
        )
        self.iou_criterion = torch.nn.MSELoss()

    def forward(self, feats, points, point_prompt, iter=1):
        '''
        feats: [K, N, 512]
        points: [K, N, 3]
        point_prompt: [K, N, 3]
        '''
        point_num = points.shape[1]
        feats = feats.transpose(0, 1)  # [N, K, 512]
        points = points.transpose(0, 1)  # [N, K, 3]
        point_prompt = point_prompt.transpose(0, 1)  # [N, K, 3]
        feats_seg = torch.cat([feats, points, point_prompt], dim=-1)  # [N, K, 512+3+3]

        # Predict mask stage-1
        pred_mask_1 = self.seg_mlp_1(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_2 = self.seg_mlp_2(feats_seg).squeeze(-1)  # [N, K]
        pred_mask_3 = self.seg_mlp_3(feats_seg).squeeze(-1)  # [N, K]
        pred_mask = torch.stack(
            [pred_mask_1, pred_mask_2, pred_mask_3], dim=-1
        )  # [N, K, 3]

        for _ in range(iter):
            # Predict mask stage-2
            feats_seg_2 = torch.cat([feats_seg, pred_mask], dim=-1)  # [N, K, 512+3+3+3]
            feats_seg_global = self.seg_s2_mlp_g(feats_seg_2)  # [N, K, 256]
            feats_seg_global = torch.max(feats_seg_global, dim=0).values  # [K, 256]
            feats_seg_global = feats_seg_global.unsqueeze(0).repeat(
                point_num, 1, 1
            )  # [N, K, 256]
            feats_seg_3 = torch.cat(
                [feats_seg_global, feats_seg_2], dim=-1
            )  # [N, K, 256+512+3+3+3]
            pred_mask_s2_1 = self.seg_s2_mlp_1(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_2 = self.seg_s2_mlp_2(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2_3 = self.seg_s2_mlp_3(feats_seg_3).squeeze(-1)  # [N, K]
            pred_mask_s2 = torch.stack(
                [pred_mask_s2_1, pred_mask_s2_2, pred_mask_s2_3], dim=-1
            )  # [N, K, 3]
            pred_mask = pred_mask_s2

        mask_1 = torch.sigmoid(pred_mask_s2_1).to(dtype=torch.float32)  # [N, K]
        mask_2 = torch.sigmoid(pred_mask_s2_2).to(dtype=torch.float32)  # [N, K]
        mask_3 = torch.sigmoid(pred_mask_s2_3).to(dtype=torch.float32)  # [N, K]

        feats_iou = torch.cat(
            [feats_seg_global, feats_seg, pred_mask_s2], dim=-1
        )  # [N, K, 256+512+3+3+3]
        feats_iou = self.iou_mlp(feats_iou)  # [N, K, 256]
        feats_iou = torch.max(feats_iou, dim=0).values  # [K, 256]
        pred_iou = self.iou_mlp_out(feats_iou)  # [K, 3]
        pred_iou = torch.sigmoid(pred_iou).to(dtype=torch.float32)  # [K, 3]

        mask_1 = mask_1.transpose(0, 1)  # [K, N]
        mask_2 = mask_2.transpose(0, 1)  # [K, N]
        mask_3 = mask_3.transpose(0, 1)  # [K, N]

        return mask_1, mask_2, mask_3, pred_iou

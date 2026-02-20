"""ComfyUI-native VAE for Hunyuan3D-Part (VolumeDecoderShapeVAE).

Consolidates autoencoders/model.py + autoencoders/attention_blocks.py +
autoencoders/attention_processors.py + volume_decoders.py + surface_extractors.py
into a single self-contained file with operations= parameter and ComfyUI
attention dispatch.
"""

import copy
from functools import partial
from typing import Optional, Union, List, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from skimage import measure
from torch import Tensor
from tqdm import tqdm

from comfy.ldm.modules.attention import optimized_attention

from ..misc_utils import synchronize_timer
from ..geometry_utils import extract_geometry_fast, generate_dense_grid_points


def _attention(q, k, v, heads):
    """q/k/v in [B, N, heads*head_dim] format."""
    return optimized_attention(q, k, v, heads=heads)


# --------------------------------------------------------------------------- #
#  Latent2MeshOutput (data class)
# --------------------------------------------------------------------------- #

class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


# --------------------------------------------------------------------------- #
#  Surface extractors
# --------------------------------------------------------------------------- #

def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds, octree_resolution):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))
            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)
        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logit.cpu().numpy(), mc_level, method="lewiner"
        )
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except Exception:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}


# --------------------------------------------------------------------------- #
#  VanillaVolumeDecoder (no learnable weights)
# --------------------------------------------------------------------------- #

class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij",
        )
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Volume Decoding",
            disable=not enable_pbar,
        ):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits


# --------------------------------------------------------------------------- #
#  FourierEmbedder (no learnable weights)
# --------------------------------------------------------------------------- #

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=6, logspace=True, input_dim=3,
                 include_input=True, include_pi=True):
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x):
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


# --------------------------------------------------------------------------- #
#  DropPath
# --------------------------------------------------------------------------- #

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# --------------------------------------------------------------------------- #
#  MLP (VAE FFN)
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, *, width, expand_ratio=4, output_width=None,
                 drop_path_rate=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.width = width
        self.c_fc = operations.Linear(width, width * expand_ratio, dtype=dtype, device=device)
        self.c_proj = operations.Linear(
            width * expand_ratio, output_width if output_width is not None else width,
            dtype=dtype, device=device,
        )
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


# --------------------------------------------------------------------------- #
#  Attention blocks (VAE)
# --------------------------------------------------------------------------- #

class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads, width=None, qk_norm=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q_norm = (
            operations.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            operations.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, "b n h d -> b n (h d)")
        k = rearrange(k, "b n h d -> b n (h d)")
        v = rearrange(v, "b n h d -> b n (h d)")
        out = _attention(q, k, v, heads=self.heads)
        return out.view(bs, n_ctx, -1)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, *, width, heads, qkv_bias=True, data_width=None,
                 qk_norm=False, kv_cache=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = operations.Linear(width, width, bias=qkv_bias, dtype=dtype, device=device)
        self.c_kv = operations.Linear(self.data_width, width * 2, bias=qkv_bias, dtype=dtype, device=device)
        self.c_proj = operations.Linear(width, width, dtype=dtype, device=device)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, width=width, qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, *, width, heads, mlp_expand_ratio=4, data_width=None,
                 qkv_bias=True, qk_norm=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(
            width=width, heads=heads, data_width=data_width,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.ln_1 = operations.LayerNorm(width, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.ln_2 = operations.LayerNorm(data_width, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.ln_3 = operations.LayerNorm(width, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio,
                       dtype=dtype, device=device, operations=operations)

    def forward(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads, width=None, qk_norm=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q_norm = (
            operations.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            operations.LayerNorm(width // heads, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, "b n h d -> b n (h d)")
        k = rearrange(k, "b n h d -> b n (h d)")
        v = rearrange(v, "b n h d -> b n (h d)")
        out = _attention(q, k, v, heads=self.heads)
        return out


class MultiheadAttention(nn.Module):
    def __init__(self, *, width, heads, qkv_bias, qk_norm=False,
                 drop_path_rate=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.width = width
        self.heads = heads
        self.c_qkv = operations.Linear(width, width * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.c_proj = operations.Linear(width, width, dtype=dtype, device=device)
        self.attention = QKVMultiheadAttention(
            heads=heads, width=width, qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, *, width, heads, qkv_bias=True, qk_norm=False,
                 drop_path_rate=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = MultiheadAttention(
            width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
            dtype=dtype, device=device, operations=operations,
        )
        self.ln_1 = operations.LayerNorm(width, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate,
                       dtype=dtype, device=device, operations=operations)
        self.ln_2 = operations.LayerNorm(width, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, *, width, layers, heads, qkv_bias=True, qk_norm=False,
                 drop_path_rate=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                drop_path_rate=drop_path_rate,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


# --------------------------------------------------------------------------- #
#  CrossAttentionDecoder
# --------------------------------------------------------------------------- #

class CrossAttentionDecoder(nn.Module):
    def __init__(self, *, out_channels, fourier_embedder, width, heads,
                 mlp_expand_ratio=4, downsample_ratio=1, enable_ln_post=True,
                 qkv_bias=True, qk_norm=False, label_type="binary",
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = operations.Linear(self.fourier_embedder.out_dim, width, dtype=dtype, device=device)
        if self.downsample_ratio != 1:
            self.latents_proj = operations.Linear(width * downsample_ratio, width, dtype=dtype, device=device)
        if not self.enable_ln_post:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width, mlp_expand_ratio=mlp_expand_ratio, heads=heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )
        if self.enable_ln_post:
            self.ln_post = operations.LayerNorm(width, dtype=dtype, device=device)
        self.output_proj = operations.Linear(width, out_channels, dtype=dtype, device=device)
        self.label_type = label_type
        self.count = 0

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            query_embeddings = self.query_proj(
                self.fourier_embedder(queries).to(latents.dtype)
            )
        self.count += query_embeddings.shape[1]
        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)
        x = self.cross_attn_decoder(query_embeddings, latents)
        if self.enable_ln_post:
            x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


# --------------------------------------------------------------------------- #
#  FPS (torch_cluster - external, no change)
# --------------------------------------------------------------------------- #

def fps(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None):
    src = src.float()
    from torch_cluster import fps as fps_fn
    return fps_fn(src, batch, ratio, random_start, batch_size, ptr)


# --------------------------------------------------------------------------- #
#  PointCrossAttentionEncoder
# --------------------------------------------------------------------------- #

class PointCrossAttentionEncoder(nn.Module):
    def __init__(self, *, num_latents, downsample_ratio, pc_size,
                 pc_sharpedge_size, fourier_embedder, point_feats,
                 width, heads, layers, normal_pe=False, qkv_bias=True,
                 use_ln_post=False, use_checkpoint=False, qk_norm=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        if pc_sharpedge_size == 0:
            print(f"PointCrossAttentionEncoder INFO: pc_sharpedge_size is not given,"
                  f" using pc_size as pc_sharpedge_size")
        else:
            print(f"PointCrossAttentionEncoder INFO: pc_sharpedge_size is given, using"
                  f" pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}")

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.fourier_embedder = fourier_embedder

        self.input_proj = operations.Linear(self.fourier_embedder.out_dim + point_feats, width, dtype=dtype, device=device)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width, heads=heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )

        self.self_attn = None
        if layers > 0:
            self.self_attn = Transformer(
                width=width, layers=layers, heads=heads,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                dtype=dtype, device=device, operations=operations,
            )

        if use_ln_post:
            self.ln_post = operations.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def sample_points_and_latents(self, pc, feats=None):
        B, N, D = pc.shape
        num_pts = self.num_latents * self.downsample_ratio
        num_latents = int(num_pts / self.downsample_ratio)

        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size
        assert sharpedge_pc.shape[1] <= self.pc_sharpedge_size

        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        original_size = input_random_pc_size
        input_random_pc_size = min(input_random_pc_size, random_pc.shape[1])
        if original_size != input_random_pc_size:
            print(f"[X-Part] DEBUG: Clamped input_random_pc_size from {original_size} to {input_random_pc_size} (available: {random_pc.shape[1]})")
        random_query_ratio = num_random_query / input_random_pc_size
        idx_random_pc = torch.randperm(random_pc.shape[1], device=random_pc.device)[:input_random_pc_size]
        input_random_pc = random_pc[:, idx_random_pc, :]
        flatten_input_random_pc = input_random_pc.view(B * input_random_pc_size, D)
        N_down = int(flatten_input_random_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps(flatten_input_random_pc, batch_down, ratio=random_query_ratio)
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        original_sharpedge_size = input_sharpedge_pc_size
        input_sharpedge_pc_size = min(input_sharpedge_pc_size, sharpedge_pc.shape[1])
        if original_sharpedge_size != input_sharpedge_pc_size and input_sharpedge_pc_size > 0:
            print(f"[X-Part] DEBUG: Clamped input_sharpedge_pc_size from {original_sharpedge_size} to {input_sharpedge_pc_size} (available: {sharpedge_pc.shape[1]})")
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype).to(pc.device)
            query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_random_pc.dtype).to(pc.device)
        else:
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1], device=sharpedge_pc.device)[:input_sharpedge_pc_size]
            input_sharpedge_pc = sharpedge_pc[:, idx_sharpedge_pc, :]
            flatten_input_sharpedge_surface_points = input_sharpedge_pc.view(B * input_sharpedge_pc_size, D)
            N_down = int(flatten_input_sharpedge_surface_points.shape[0] / B)
            batch_down = torch.arange(B).to(pc.device)
            batch_down = torch.repeat_interleave(batch_down, N_down)
            idx_query_sharpedge = fps(flatten_input_sharpedge_surface_points, batch_down, ratio=sharpedge_query_ratio)
            query_sharpedge_pc = flatten_input_sharpedge_surface_points[idx_query_sharpedge].view(B, -1, D)

        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        if self.point_feats != 0:
            random_surface_feats, sharpedge_surface_feats = torch.split(
                feats, [self.pc_size, self.pc_sharpedge_size], dim=1
            )
            input_random_surface_feats = random_surface_feats[:, idx_random_pc, :]
            flatten_input_random_surface_feats = input_random_surface_feats.view(B * input_random_pc_size, -1)
            query_random_feats = flatten_input_random_surface_feats[idx_query_random].view(
                B, -1, flatten_input_random_surface_feats.shape[-1]
            )

            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_feats = torch.zeros(
                    B, 0, self.point_feats, dtype=input_random_surface_feats.dtype
                ).to(pc.device)
                query_sharpedge_feats = torch.zeros(
                    B, 0, self.point_feats, dtype=query_random_feats.dtype
                ).to(pc.device)
            else:
                input_sharpedge_surface_feats = sharpedge_surface_feats[:, idx_sharpedge_pc, :]
                flatten_input_sharpedge_surface_feats = input_sharpedge_surface_feats.view(
                    B * input_sharpedge_pc_size, -1
                )
                query_sharpedge_feats = flatten_input_sharpedge_surface_feats[idx_query_sharpedge].view(
                    B, -1, flatten_input_sharpedge_surface_feats.shape[-1]
                )

            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)
            input_feats = torch.cat([input_random_surface_feats, input_sharpedge_surface_feats], dim=1)

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)

        return (
            query.view(B, -1, query.shape[-1]),
            data.view(B, -1, data.shape[-1]),
            [query_pc, input_pc, query_random_pc, input_random_pc, query_sharpedge_pc, input_sharpedge_pc],
        )

    def forward(self, pc, feats):
        query, data, pc_infos = self.sample_points_and_latents(pc, feats)
        query = self.input_proj(query)
        data = self.input_proj(data)
        latents = self.cross_attn(query, data)
        if self.self_attn is not None:
            latents = self.self_attn(latents)
        if self.ln_post is not None:
            latents = self.ln_post(latents)
        return latents, pc_infos


# --------------------------------------------------------------------------- #
#  DiagonalGaussianDistribution
# --------------------------------------------------------------------------- #

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters
        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        if other is None:
            return 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims
            )
        return 0.5 * torch.mean(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=dims,
        )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


# --------------------------------------------------------------------------- #
#  VectsetVAE (base)
# --------------------------------------------------------------------------- #

class VectsetVAE(nn.Module):
    def __init__(self, volume_decoder=None, surface_extractor=None):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents, **kwargs):
        with synchronize_timer("Volume decoding"):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        with synchronize_timer("Surface extraction"):
            outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs


# --------------------------------------------------------------------------- #
#  VolumeDecoderShapeVAE
# --------------------------------------------------------------------------- #

class VolumeDecoderShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents,
        embed_dim,
        width,
        heads,
        num_decoder_layers,
        num_encoder_layers=8,
        pc_size=5120,
        pc_sharpedge_size=5120,
        point_feats=3,
        downsample_ratio=20,
        geo_decoder_downsample_ratio=1,
        geo_decoder_mlp_expand_ratio=4,
        geo_decoder_ln_post=True,
        num_freqs=8,
        include_pi=True,
        qkv_bias=True,
        qk_norm=False,
        label_type="binary",
        drop_path_rate=0.0,
        scale_factor=1.0,
        use_ln_post=True,
        ckpt_path=None,
        volume_decoder=None,
        surface_extractor=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__(volume_decoder, surface_extractor)
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=self.downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width, heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias, use_ln_post=use_ln_post,
            qk_norm=qk_norm,
            dtype=dtype, device=device, operations=operations,
        )

        self.pre_kl = operations.Linear(width, embed_dim * 2, dtype=dtype, device=device)
        self.post_kl = operations.Linear(embed_dim, width, dtype=dtype, device=device)

        self.transformer = Transformer(
            width=width, layers=num_decoder_layers, heads=heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
            dtype=dtype, device=device, operations=operations,
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            label_type=label_type,
            dtype=dtype, device=device, operations=operations,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def encode(self, surface, sample_posterior=True, return_pc_info=False):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, pc_infos = self.encoder(pc, feats)
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        if sample_posterior:
            latents = posterior.sample()
        else:
            latents = posterior.mode()
        if return_pc_info:
            return latents, pc_infos
        return latents

    def encode_shape(self, surface, return_pc_info=False):
        pc, feats = surface[:, :, :3], surface[:, :, 3:]
        latents, pc_infos = self.encoder(pc, feats)
        if return_pc_info:
            return latents, pc_infos
        return latents

    def decode(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def query_geometry(self, queries, latents):
        logits = self.geo_decoder(queries=queries, latents=latents).squeeze(-1)
        return logits

    def latents2mesh(self, latents, **kwargs):
        coarse_kwargs = copy.deepcopy(kwargs)
        coarse_kwargs["octree_resolution"] = 256

        with synchronize_timer("Coarse Volume decoding"):
            coarse_grid_logits = self.volume_decoder(latents, self.geo_decoder, **coarse_kwargs)
        with synchronize_timer("Coarse Surface extraction"):
            coarse_mesh = self.surface_extractor(coarse_grid_logits, **coarse_kwargs)

        assert len(coarse_mesh) == 1
        bbox_gen = np.stack([coarse_mesh[0].mesh_v.max(0), coarse_mesh[0].mesh_v.min(0)])
        bbox_range = bbox_gen[0] - bbox_gen[1]
        bbox_gen[0] += bbox_range * 0.1
        bbox_gen[1] -= bbox_range * 0.1

        with synchronize_timer("Fine-grained Volume decoding"):
            grid_logits = self.volume_decoder(
                latents, self.geo_decoder, bbox_corner=bbox_gen[None], **kwargs
            )
        with synchronize_timer("Fine-grained Surface extraction"):
            outputs = self.surface_extractor(grid_logits, bbox_corner=bbox_gen[None], **kwargs)

        return outputs

    def latent2mesh_2(self, latents, bounds=1.1, octree_depth=7,
                      num_chunks=10000, mc_level=-1/512,
                      octree_resolution=None, mc_mode="mc"):
        outputs = []
        geometric_func = partial(self.query_geometry, latents=latents)
        device = latents.device
        if mc_mode == "dmc" and not hasattr(self, "diffdmc"):
            from diso import DiffDMC
            self.diffdmc = DiffDMC(dtype=torch.float32).to(device)
        mesh_v_f, has_surface = extract_geometry_fast(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=False,
            mc_level=mc_level,
            octree_resolution=octree_resolution,
            diffdmc=self.diffdmc if mc_mode == "dmc" else None,
            mc_mode=mc_mode,
        )
        for (mesh_v, mesh_f), is_surface in zip(mesh_v_f, has_surface):
            if not is_surface:
                outputs.append(None)
                continue
            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            outputs.append(out)
        return outputs

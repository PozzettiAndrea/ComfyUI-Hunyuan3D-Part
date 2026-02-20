"""ComfyUI-native DiT model for Hunyuan3D-Part (PartFormerDITPlain).

Consolidates partformer_dit.py + moe_layers.py into a single flat file
with operations= parameter, transformer_options threading, WrapperExecutor,
and block replacement support.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.patcher_extension
from comfy.ldm.modules.attention import optimized_attention


def _attention(q, k, v, heads):
    """q/k/v in [B, N, heads*head_dim] format."""
    return optimized_attention(q, k, v, heads=heads)


# --------------------------------------------------------------------------- #
#  Positional / timestep embeddings (no learnable weights)
# --------------------------------------------------------------------------- #

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class Timesteps(nn.Module):
    def __init__(self, num_channels, downscale_freq_shift=0.0, scale=1, max_period=10000):
        super().__init__()
        self.num_channels = num_channels
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps):
        assert len(timesteps.shape) == 1
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = self.scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


# --------------------------------------------------------------------------- #
#  TimestepEmbedder
# --------------------------------------------------------------------------- #

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256,
                 cond_proj_dim=None, out_size=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, frequency_embedding_size, bias=True, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(frequency_embedding_size, out_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

        if cond_proj_dim is not None:
            self.cond_proj = operations.Linear(cond_proj_dim, frequency_embedding_size, bias=False, dtype=dtype, device=device)

        self.time_embed = Timesteps(hidden_size)

    def forward(self, t, condition):
        t_freq = self.time_embed(t).type(self.mlp[0].weight.dtype)
        if condition is not None:
            t_freq = t_freq + self.cond_proj(condition)
        t = self.mlp(t_freq)
        t = t.unsqueeze(dim=1)
        return t


# --------------------------------------------------------------------------- #
#  MLP (DiT FFN)
# --------------------------------------------------------------------------- #

class MLP(nn.Module):
    def __init__(self, *, width, dtype=None, device=None, operations=None):
        super().__init__()
        self.width = width
        self.fc1 = operations.Linear(width, width * 4, dtype=dtype, device=device)
        self.fc2 = operations.Linear(width * 4, width, dtype=dtype, device=device)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# --------------------------------------------------------------------------- #
#  FeedForward (replaces diffusers.models.attention.FeedForward)
#  State dict keys: net.0.proj.{weight,bias}, net.2.{weight,bias}
# --------------------------------------------------------------------------- #

class FeedForwardGELU(nn.Module):
    def __init__(self, dim, inner_dim, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim, inner_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, x):
        return F.gelu(self.proj(x))


class FeedForward(nn.Module):
    def __init__(self, dim, inner_dim=None, bias=True, dropout=0.0,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        self.net = nn.ModuleList([
            FeedForwardGELU(dim, inner_dim, bias=bias, dtype=dtype, device=device, operations=operations),
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim, bias=bias, dtype=dtype, device=device),
        ])

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


# --------------------------------------------------------------------------- #
#  MoE layers
# --------------------------------------------------------------------------- #

class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean()
                aux_loss = aux_loss * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts=8, moe_top_k=2, activation_fn="gelu",
                 dropout=0.0, final_dropout=False, ff_inner_dim=None, ff_bias=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.moe_top_k = moe_top_k
        self.experts = nn.ModuleList([
            FeedForward(dim, inner_dim=ff_inner_dim, bias=ff_bias, dropout=dropout,
                        dtype=dtype, device=device, operations=operations)
            for _ in range(num_experts)
        ])
        self.gate = MoEGate(embed_dim=dim, num_experts=num_experts, num_experts_per_tok=moe_top_k)
        self.shared_experts = FeedForward(
            dim, inner_dim=ff_inner_dim, bias=ff_bias, dropout=dropout,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.moe_top_k, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                tmp = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = tmp.to(hidden_states.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.moe_top_k
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


# --------------------------------------------------------------------------- #
#  CrossAttention
# --------------------------------------------------------------------------- #

class CrossAttention(nn.Module):
    def __init__(self, qdim, kdim, num_heads, qkv_bias=True, qk_norm=False,
                 with_decoupled_ca=False, decoupled_ca_dim=16, decoupled_ca_weight=1.0,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0
        self.head_dim = self.qdim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = operations.Linear(qdim, qdim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(kdim, qdim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(kdim, qdim, bias=qkv_bias, dtype=dtype, device=device)

        self.q_norm = (
            operations.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            operations.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.out_proj = operations.Linear(qdim, qdim, bias=True, dtype=dtype, device=device)

        self.with_dca = with_decoupled_ca
        if self.with_dca:
            self.kv_proj_dca = operations.Linear(kdim, 2 * qdim, bias=qkv_bias, dtype=dtype, device=device)
            self.k_norm_dca = (
                operations.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
                if qk_norm else nn.Identity()
            )
            self.dca_dim = decoupled_ca_dim
            self.dca_weight = decoupled_ca_weight

    def forward(self, x, y):
        b, s1, c = x.shape

        if self.with_dca:
            token_len = y.shape[1]
            context_dca = y[:, -self.dca_dim:, :]
            kv_dca = self.kv_proj_dca(context_dca).view(b, self.dca_dim, 2, self.num_heads, self.head_dim)
            k_dca, v_dca = kv_dca.unbind(dim=2)
            k_dca = self.k_norm_dca(k_dca)
            y = y[:, :(token_len - self.dca_dim), :]

        _, s2, c = y.shape
        q = self.to_q(x)
        k = self.to_k(y)
        v = self.to_v(y)

        kv = torch.cat((k, v), dim=-1)
        split_size = kv.shape[-1] // self.num_heads // 2
        kv = kv.view(1, -1, self.num_heads, split_size * 2)
        k, v = torch.split(kv, split_size, dim=-1)

        q = q.view(b, s1, self.num_heads, self.head_dim)
        k = k.view(b, s2, self.num_heads, self.head_dim)
        v = v.view(b, s2, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rearrange(q, "b n h d -> b n (h d)")
        k = rearrange(k, "b n h d -> b n (h d)")
        v = rearrange(v, "b n h d -> b n (h d)")
        context = _attention(q, k, v, heads=self.num_heads)

        if self.with_dca:
            k_dca = rearrange(k_dca, "b n h d -> b n (h d)")
            v_dca = rearrange(v_dca, "b n h d -> b n (h d)")
            context_dca = _attention(q, k_dca, v_dca, heads=self.num_heads)
            context = context + self.dca_weight * context_dca

        out = self.out_proj(context)
        return out


# --------------------------------------------------------------------------- #
#  Self-Attention
# --------------------------------------------------------------------------- #

class LocalGlobalProcessor:
    def __init__(self, use_global=False):
        self.use_global = use_global

    def __call__(self, attn, hidden_states):
        if self.use_global:
            B_old, N_old, C_old = hidden_states.shape
            hidden_states = hidden_states.reshape(1, -1, C_old)
        B, N, C = hidden_states.shape

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)

        qkv = torch.cat((q, k, v), dim=-1)
        split_size = qkv.shape[-1] // attn.num_heads // 3
        qkv = qkv.view(1, -1, attn.num_heads, split_size * 3)
        q, k, v = torch.split(qkv, split_size, dim=-1)

        q = q.reshape(B, N, attn.num_heads, attn.head_dim)
        k = k.reshape(B, N, attn.num_heads, attn.head_dim)
        v = v.reshape(B, N, attn.num_heads, attn.head_dim)

        q = attn.q_norm(q)
        k = attn.k_norm(k)

        q = rearrange(q, "b n h d -> b n (h d)")
        k = rearrange(k, "b n h d -> b n (h d)")
        v = rearrange(v, "b n h d -> b n (h d)")
        hidden_states = _attention(q, k, v, heads=attn.num_heads)

        hidden_states = attn.out_proj(hidden_states)
        if self.use_global:
            hidden_states = hidden_states.reshape(B_old, N_old, -1)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=False,
                 use_global_processor=False,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.use_global_processor = use_global_processor
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0
        self.head_dim = self.dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, bias=qkv_bias, dtype=dtype, device=device)

        self.q_norm = (
            operations.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.k_norm = (
            operations.RMSNorm(self.head_dim, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            if qk_norm else nn.Identity()
        )
        self.out_proj = operations.Linear(dim, dim, dtype=dtype, device=device)

        self.processor = LocalGlobalProcessor(use_global=use_global_processor)

    def forward(self, x):
        return self.processor(self, x)


# --------------------------------------------------------------------------- #
#  AttentionPool (uses F.multi_head_attention_forward - no change needed)
# --------------------------------------------------------------------------- #

class AttentionPool(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.q_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.v_proj = operations.Linear(embed_dim, embed_dim, dtype=dtype, device=device)
        self.c_proj = operations.Linear(embed_dim, output_dim or embed_dim, dtype=dtype, device=device)
        self.num_heads = num_heads

    def forward(self, x, attention_mask=None):
        x = x.permute(1, 0, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).permute(1, 0, 2)
            global_emb = (x * attention_mask).sum(dim=0) / attention_mask.sum(dim=0)
            x = torch.cat([global_emb[None,], x], dim=0)
        else:
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None, bias_v=None,
            add_zero_attn=False, dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


# --------------------------------------------------------------------------- #
#  BboxEmbedder (conditional, rarely used - kept for state dict compat)
# --------------------------------------------------------------------------- #

class BboxEmbedder(nn.Module):
    def __init__(self, out_size, num_freqs=8,
                 dtype=None, device=None, operations=None):
        super().__init__()
        from .vae import FourierEmbedder
        self.fourier = FourierEmbedder(num_freqs=num_freqs, input_dim=6, include_input=True)
        self.mlp = nn.Sequential(
            operations.Linear(self.fourier.out_dim, out_size, dtype=dtype, device=device),
            nn.GELU(),
            operations.Linear(out_size, out_size, dtype=dtype, device=device),
        )

    def forward(self, bbox):
        return self.mlp(self.fourier(bbox))


# --------------------------------------------------------------------------- #
#  PartFormerDitBlock
# --------------------------------------------------------------------------- #

class PartFormerDitBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        use_self_attention=True,
        use_cross_attention=False,
        use_cross_attention_2=False,
        encoder_hidden_dim=1024,
        encoder_hidden2_dim=1024,
        qkv_bias=True,
        qk_norm=False,
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        skip_connection=False,
        timested_modulate=False,
        c_emb_size=0,
        use_moe=False,
        num_experts=8,
        moe_top_k=2,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        use_ele_affine = True

        # ========================= Self-Attention =========================
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6, dtype=dtype, device=device)
            self.attn1 = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                dtype=dtype, device=device, operations=operations,
            )

        # ========================= Timestep Modulation =========================
        self.timested_modulate = timested_modulate
        if self.timested_modulate:
            self.default_modulation = nn.Sequential(
                nn.SiLU(),
                operations.Linear(c_emb_size, hidden_size, bias=True, dtype=dtype, device=device),
            )

        # ========================= Cross-Attention =========================
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6, dtype=dtype, device=device)
            self.attn2 = CrossAttention(
                hidden_size, encoder_hidden_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                with_decoupled_ca=False,
                dtype=dtype, device=device, operations=operations,
            )

        self.use_cross_attention_2 = use_cross_attention_2
        if self.use_cross_attention_2:
            self.norm2_2 = operations.LayerNorm(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6, dtype=dtype, device=device)
            self.attn2_2 = CrossAttention(
                hidden_size, encoder_hidden2_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_norm=qk_norm,
                with_decoupled_ca=with_decoupled_ca,
                decoupled_ca_dim=decoupled_ca_dim,
                decoupled_ca_weight=decoupled_ca_weight,
                dtype=dtype, device=device, operations=operations,
            )

        # ========================= FFN =========================
        self.norm3 = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.use_moe = use_moe
        if self.use_moe:
            self.moe = MoEBlock(
                hidden_size, num_experts=num_experts, moe_top_k=moe_top_k,
                dropout=0.0, activation_fn="gelu", final_dropout=False,
                ff_inner_dim=int(hidden_size * 4.0), ff_bias=True,
                dtype=dtype, device=device, operations=operations,
            )
        else:
            self.mlp = MLP(width=hidden_size, dtype=dtype, device=device, operations=operations)

        # ========================= Skip Connection =========================
        if skip_connection:
            self.skip_norm = operations.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
            self.skip_linear = operations.Linear(2 * hidden_size, hidden_size, dtype=dtype, device=device)
        else:
            self.skip_linear = None

    def forward(self, hidden_states, encoder_hidden_states=None,
                encoder_hidden_states_2=None, temb=None, skip_value=None,
                transformer_options={}):
        # skip connection
        if self.skip_linear is not None:
            cat = torch.cat([skip_value, hidden_states], dim=-1)
            hidden_states = self.skip_linear(cat)
            hidden_states = self.skip_norm(hidden_states)

        # timestep modulation
        if self.timested_modulate:
            shift_msa = self.default_modulation(temb).unsqueeze(dim=1)
            hidden_states = hidden_states + shift_msa

        # self-attention
        if self.use_self_attention:
            attn_output = self.attn1(self.norm1(hidden_states))
            hidden_states = hidden_states + attn_output

        # cross-attention 1
        if self.use_cross_attention:
            original_cross_out = self.attn2(self.norm2(hidden_states), encoder_hidden_states)

        # cross-attention 2
        if self.use_cross_attention_2:
            cross_out_2 = self.attn2_2(self.norm2_2(hidden_states), encoder_hidden_states_2)

        hidden_states = (
            hidden_states
            + (original_cross_out if self.use_cross_attention else 0)
            + (cross_out_2 if self.use_cross_attention_2 else 0)
        )

        # FFN
        mlp_inputs = self.norm3(hidden_states)
        if self.use_moe:
            hidden_states = hidden_states + self.moe(mlp_inputs)
        else:
            hidden_states = hidden_states + self.mlp(mlp_inputs)

        return hidden_states


# --------------------------------------------------------------------------- #
#  FinalLayer
# --------------------------------------------------------------------------- #

class FinalLayer(nn.Module):
    def __init__(self, final_hidden_size, out_channels,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.final_hidden_size = final_hidden_size
        self.norm_final = operations.LayerNorm(final_hidden_size, elementwise_affine=True, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(final_hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.norm_final(x)
        x = x[:, 1:]
        x = self.linear(x)
        return x


# --------------------------------------------------------------------------- #
#  PartFormerDITPlain — top-level diffusion model
# --------------------------------------------------------------------------- #

class PartFormerDITPlain(nn.Module):
    def __init__(
        self,
        input_size=1024,
        in_channels=4,
        hidden_size=1024,
        use_self_attention=True,
        use_cross_attention=True,
        use_cross_attention_2=True,
        encoder_hidden_dim=1024,
        encoder_hidden2_dim=1024,
        depth=24,
        num_heads=16,
        qk_norm=False,
        qkv_bias=True,
        norm_type="layer",
        qk_norm_type="rms",
        with_decoupled_ca=False,
        decoupled_ca_dim=16,
        decoupled_ca_weight=1.0,
        use_pos_emb=False,
        guidance_cond_proj_dim=None,
        num_moe_layers=6,
        num_experts=8,
        moe_top_k=2,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # embedding
        self.x_embedder = operations.Linear(in_channels, hidden_size, bias=True, dtype=dtype, device=device)
        self.t_embedder = TimestepEmbedder(
            hidden_size, hidden_size * 4,
            cond_proj_dim=guidance_cond_proj_dim,
            dtype=dtype, device=device, operations=operations,
        )

        # positional embedding (fixed sin-cos)
        self.use_pos_emb = use_pos_emb
        if self.use_pos_emb:
            self.register_buffer("pos_embed", torch.zeros(1, input_size, hidden_size))
            pos = np.arange(self.input_size, dtype=np.float32)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # bbox conditioning (rarely used)
        self.use_bbox_cond = kwargs.get("use_bbox_cond", False)
        if self.use_bbox_cond:
            self.bbox_conditioner = BboxEmbedder(
                out_size=hidden_size,
                num_freqs=kwargs.get("num_freqs", 8),
                dtype=dtype, device=device, operations=operations,
            )

        # part id embedding
        self.use_part_embed = kwargs.get("use_part_embed", False)
        if self.use_part_embed:
            self.valid_num = kwargs.get("valid_num", 50)
            self.part_embed = nn.Parameter(torch.randn(self.valid_num, hidden_size))
            self.part_embed.data.zero_()

        # transformer blocks
        self.blocks = nn.ModuleList([
            PartFormerDitBlock(
                hidden_size, num_heads,
                use_self_attention=use_self_attention,
                use_cross_attention=use_cross_attention,
                use_cross_attention_2=use_cross_attention_2,
                encoder_hidden_dim=encoder_hidden_dim,
                encoder_hidden2_dim=encoder_hidden2_dim,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                with_decoupled_ca=with_decoupled_ca,
                decoupled_ca_dim=decoupled_ca_dim,
                decoupled_ca_weight=decoupled_ca_weight,
                skip_connection=layer > depth // 2,
                use_moe=True if depth - layer <= num_moe_layers else False,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                dtype=dtype, device=device, operations=operations,
            )
            for layer in range(depth)
        ])

        # set local-global processor on even blocks
        for layer, block in enumerate(self.blocks):
            if hasattr(block, "attn1") and (layer + 1) % 2 == 0:
                block.attn1.processor = LocalGlobalProcessor(use_global=True)

        self.depth = depth
        self.final_layer = FinalLayer(hidden_size, self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward(self, x, t, contexts, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                transformer_options,
            ),
        ).execute(x, t, contexts, transformer_options, **kwargs)

    def _forward(self, x, t, contexts, transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        # prepare input
        aabb = kwargs.get("aabb", None)
        object_context = contexts.get("obj_cond", None)
        geo_context = contexts.get("geo_cond", None)
        num_tokens = kwargs.get("num_tokens", None)

        # timestep embedding and input projection
        t = self.t_embedder(t, condition=kwargs.get("guidance_cond"))
        x = self.x_embedder(x)

        if self.use_pos_emb:
            pos_embed = self.pos_embed.to(x.dtype)
            x = x + pos_embed

        c = t

        # bounding box conditioning
        if self.use_bbox_cond:
            center_extent = torch.cat(
                [torch.mean(aabb, dim=-2), aabb[..., 1, :] - aabb[..., 0, :]], dim=-1
            )
            bbox_embeds = self.bbox_conditioner(center_extent)
            bbox_embeds = torch.repeat_interleave(bbox_embeds, repeats=num_tokens[0], dim=1)
            x = x + bbox_embeds

        # part id embedding
        if self.use_part_embed:
            num_parts = aabb.shape[1]
            random_idx = torch.randperm(self.valid_num)[:num_parts]
            part_embeds = self.part_embed[random_idx].unsqueeze(1)
            x = x + part_embeds

        x = torch.cat([c, x], dim=1)

        # block loop with block replacement support
        transformer_options["total_blocks"] = len(self.blocks)
        skip_value_list = []
        for layer, block in enumerate(self.blocks):
            transformer_options["block_index"] = layer
            skip_value = None if layer <= self.depth // 2 else skip_value_list.pop()

            if ("block", layer) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["hidden_states"] = block(
                        hidden_states=args["hidden_states"],
                        encoder_hidden_states=args.get("encoder_hidden_states"),
                        encoder_hidden_states_2=args.get("encoder_hidden_states_2"),
                        temb=args.get("temb"),
                        skip_value=args.get("skip_value"),
                        transformer_options=args.get("transformer_options", {}),
                    )
                    return out

                out = blocks_replace[("block", layer)](
                    {
                        "hidden_states": x,
                        "encoder_hidden_states": object_context,
                        "encoder_hidden_states_2": geo_context,
                        "temb": c,
                        "skip_value": skip_value,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                x = out["hidden_states"]
            else:
                x = block(
                    hidden_states=x,
                    encoder_hidden_states=object_context,
                    encoder_hidden_states_2=geo_context,
                    temb=c,
                    skip_value=skip_value,
                    transformer_options=transformer_options,
                )

            if layer < self.depth // 2:
                skip_value_list.append(x)

        x = self.final_layer(x)
        return x

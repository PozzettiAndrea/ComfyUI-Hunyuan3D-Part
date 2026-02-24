"""
Processing Nodes for Hunyuan3D-Part.

P3-SAM segmentation and X-Part generation nodes.
Loads models from config dicts passed by loader nodes.
"""

import torch
import numpy as np
import trimesh
import hashlib
import folder_paths
import os
import tempfile
import time
import concurrent.futures
import comfy.model_management
import comfy.utils

# Import utilities from core
from .mesh_utils import load_mesh, save_mesh, colorize_segmentation, get_temp_mesh_path
from .schedulers import FlowMatchEulerDiscreteScheduler

def _dbg(*args, **kwargs):
    """Print only when COMFYUI_DEBUG_NODES=1."""
    if os.environ.get("COMFYUI_DEBUG_NODES") == "1":
        print("[P3-SAM DEBUG]", *args, **kwargs)


def _vram_dbg(label=""):
    """Always print VRAM usage stats."""
    try:
        device = comfy.model_management.get_torch_device()
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"[VRAM] {label}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB")
    except Exception as e:
        print(f"[VRAM] {label}: could not read VRAM stats: {e}")


# Worker-process model caches (persist across node executions, same as TRELLIS2 pattern)
_p3sam_model_cache = {}
_xpart_model_cache = {}


def _enable_lowvram_cast(model):
    """Swap leaf modules to comfy.ops.disable_weight_init versions for lowvram support.

    ComfyUI's ModelPatcher can only partially-load modules that have the
    comfy_cast_weights attribute. Native ComfyUI models use comfy.ops layers,
    but third-party models use plain torch.nn.*. This retroactively swaps
    __class__ on every leaf module so ModelPatcher's lowvram streaming works.

    Also installs forward pre-hooks on non-leaf modules that own direct
    parameters or buffers (e.g. part_embed, pos_embed, non-persistent buffers)
    so they get moved to the input device on-the-fly during lowvram inference.
    """
    try:
        from comfy.ops import disable_weight_init
    except ImportError:
        return

    _CLASS_MAP = {
        torch.nn.Linear: disable_weight_init.Linear,
        torch.nn.Conv1d: disable_weight_init.Conv1d,
        torch.nn.Conv2d: disable_weight_init.Conv2d,
        torch.nn.Conv3d: disable_weight_init.Conv3d,
        torch.nn.GroupNorm: disable_weight_init.GroupNorm,
        torch.nn.LayerNorm: disable_weight_init.LayerNorm,
        torch.nn.ConvTranspose2d: disable_weight_init.ConvTranspose2d,
        torch.nn.ConvTranspose1d: disable_weight_init.ConvTranspose1d,
        torch.nn.Embedding: disable_weight_init.Embedding,
    }

    cast_count = 0
    for _name, module in model.named_modules():
        comfy_cls = _CLASS_MAP.get(type(module))
        if comfy_cls is not None:
            module.__class__ = comfy_cls
            cast_count += 1

    hook_count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'comfy_cast_weights'):
            continue
        direct_params = list(module.named_parameters(recurse=False))
        direct_bufs = list(module.named_buffers(recurse=False))
        if not direct_params and not direct_bufs:
            continue

        def _move_orphans_hook(mod, args, kwargs=None):
            device = None
            for a in args:
                if isinstance(a, torch.Tensor):
                    device = a.device
                    break
            if device is None and kwargs:
                for v in kwargs.values():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        break
            if device is None:
                for p in mod.parameters():
                    if p.device.type == 'cuda':
                        device = p.device
                        break
            if device is None:
                return
            for _, p in mod.named_parameters(recurse=False):
                if p.data.device != device:
                    p.data = p.data.to(device)
            for _, b in mod.named_buffers(recurse=False):
                if b.device != device:
                    b.data = b.data.to(device)

        module.register_forward_pre_hook(_move_orphans_hook, with_kwargs=True)
        hook_count += 1

    if cast_count or hook_count:
        print(f"[lowvram] Enabled: {cast_count} cast modules, {hook_count} orphan-param hooks")


def _get_p3sam_model(config):
    """Load or return cached P3-SAM model (ComfyUI-native: auto-dtype, ModelPatcher, load_models_gpu)."""
    import comfy.model_patcher

    precision = config.get('precision', 'auto')
    attn_backend = config.get('attn_backend', 'auto')
    enable_flash = attn_backend in ('auto', 'flash_attn')

    cache_key = f"{precision}_{attn_backend}"

    if cache_key in _p3sam_model_cache:
        patcher = _p3sam_model_cache[cache_key]
        comfy.model_management.load_models_gpu([patcher])
        return patcher.model

    from .p3sam.model import MultiHeadSegment
    from safetensors.torch import load_file

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    print(f"[P3-SAM] Building model (precision={precision}, attn_backend={attn_backend})...")
    model = MultiHeadSegment(
        in_channel=512,
        head_num=3,
        ignore_label=-100,
        enable_flash=enable_flash
    )

    state_dict = load_file(config['ckpt_path'], device="cpu")
    # Strip 'dit.' prefix from checkpoint keys (legacy checkpoint format)
    state_dict = {k.removeprefix("dit."): v for k, v in state_dict.items()}

    # Resolve dtype: "auto" = auto-detect, otherwise explicit override
    weight_dtype = next(iter(state_dict.values())).dtype
    if precision == 'auto':
        model_dtype = comfy.model_management.unet_dtype(
            device=device,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=weight_dtype,
        )
    else:
        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"[P3-SAM] precision={precision}, dtype={model_dtype}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(model_dtype)
    _enable_lowvram_cast(model)

    # Wrap in ModelPatcher for ComfyUI VRAM management
    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=device, offload_device=offload_device,
    )
    comfy.model_management.load_models_gpu([patcher])

    print(f"[P3-SAM] Model loaded on {device}")

    _p3sam_model_cache[cache_key] = patcher

    return patcher.model


def _get_sonata_model(config):
    """Load or return cached Sonata encoder (sonata + mlp only, no segmentation heads)."""
    import comfy.model_patcher
    from .p3sam.model import SonataEncoder
    from safetensors.torch import load_file

    precision = config.get('precision', 'auto')
    attn_backend = config.get('attn_backend', 'auto')
    enable_flash = attn_backend in ('auto', 'flash_attn')
    cache_key = f"sonata_{precision}_{attn_backend}"

    if cache_key in _p3sam_model_cache:
        patcher = _p3sam_model_cache[cache_key]
        comfy.model_management.load_models_gpu([patcher])
        return patcher.model

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    print(f"[Sonata Encoder] Building model (precision={precision}, attn_backend={attn_backend})...")
    model = SonataEncoder(enable_flash=enable_flash)

    state_dict = load_file(config['ckpt_path'], device="cpu")
    state_dict = {k.removeprefix("dit."): v for k, v in state_dict.items()}

    # Load Sonata backbone weights from the P3-SAM checkpoint.
    # The checkpoint stores its own copy of Sonata weights (453 keys).
    # These must be used instead of the generic facebook/sonata defaults to
    # ensure features are identical to what the full P3-SAM model produces.
    sonata_state = {k.removeprefix("sonata."): v for k, v in state_dict.items() if k.startswith("sonata.")}
    if sonata_state:
        missing, unexpected = model.sonata.load_state_dict(sonata_state, strict=False)
        if missing:
            print(f"[Sonata Encoder] Warning: {len(missing)} missing Sonata keys")
        if unexpected:
            print(f"[Sonata Encoder] Warning: {len(unexpected)} unexpected Sonata keys")
        print(f"[Sonata Encoder] Loaded {len(sonata_state)} Sonata backbone weights from P3-SAM checkpoint")
    else:
        print("[Sonata Encoder] Warning: no sonata.* keys found in checkpoint, using facebook/sonata defaults")

    # Load MLP projection weights
    mlp_state = {k.removeprefix("mlp."): v for k, v in state_dict.items() if k.startswith("mlp.")}
    model.mlp.load_state_dict(mlp_state, strict=True)
    model.eval()
    _enable_lowvram_cast(model)

    weight_dtype = next(iter(mlp_state.values())).dtype
    if precision == 'auto':
        model_dtype = comfy.model_management.unet_dtype(
            device=device,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=weight_dtype,
        )
    else:
        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    model.to(model_dtype)

    patcher = comfy.model_patcher.ModelPatcher(
        model, load_device=device, offload_device=offload_device,
    )
    comfy.model_management.load_models_gpu([patcher])
    print(f"[Sonata Encoder] Loaded on {device}, dtype={model_dtype}")

    _p3sam_model_cache[cache_key] = patcher
    return patcher.model


def _fix_meta_buffers(model, device):
    """Reinitialize any buffers left on meta device after assign=True loading."""
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            parent._buffers[parts[-1]] = torch.zeros_like(buf, device=device)


def _xpart_arch_config():
    """Architecture config for X-Part models (matches pretrained weights)."""
    num_latents = 1024
    z_scale_factor = 1.0039506158752403
    num_tokens_cond = 2048

    _shared_vae_params = dict(
        embed_dim=64, num_freqs=8, include_pi=False, heads=16,
        width=1024, num_encoder_layers=8, num_decoder_layers=16,
        qkv_bias=False, qk_norm=True, scale_factor=z_scale_factor,
        geo_decoder_mlp_expand_ratio=4, geo_decoder_downsample_ratio=1,
        geo_decoder_ln_post=True, point_feats=4,
    )

    return {
        "shapevae": {"params": {
            **_shared_vae_params,
            "num_latents": num_latents,
            "pc_size": 40960,
            "pc_sharpedge_size": 0,
        }},
        "conditioner": {"params": {
            "use_geo": True, "use_obj": True, "use_seg_feat": True,
            "geo_cfg": {
                "output_dim": 1024,
                "params": {
                    "use_local": True,
                    "local_feat_type": "latents_shape",
                    "num_tokens_cond": num_tokens_cond,
                    "local_geo_cfg": {"params": {
                        **_shared_vae_params,
                        "num_latents": num_tokens_cond,
                        "pc_size": 40960,
                        "pc_sharpedge_size": 0,
                    }},
                },
            },
            "obj_encoder_cfg": {
                "output_dim": 1024,
                "params": {
                    **_shared_vae_params,
                    "num_latents": 4096,
                    "pc_size": 40960,
                    "pc_sharpedge_size": 0,
                },
            },
            "seg_feat_cfg": {"params": {}},
        }},
        "model": {"params": {
            "use_self_attention": True, "use_cross_attention": True,
            "use_cross_attention_2": True, "use_bbox_cond": False,
            "num_freqs": 8, "use_part_embed": True, "valid_num": 50,
            "input_size": num_latents, "in_channels": 64,
            "hidden_size": 2048,
            "encoder_hidden_dim": 1024,
            "encoder_hidden2_dim": 1024,
            "depth": 21, "num_heads": 16,
            "qk_norm": True, "qkv_bias": False, "qk_norm_type": "rms",
            "with_decoupled_ca": False, "decoupled_ca_dim": num_tokens_cond,
            "decoupled_ca_weight": 1.0, "use_attention_pooling": False,
            "use_pos_emb": False,
            "num_moe_layers": 6, "num_experts": 8, "moe_top_k": 2,
        }},
        "scheduler": {"params": {"num_train_timesteps": 1000}},
    }


def _get_xpart_models(config, pc_size=40960):
    """Load or return cached X-Part models (ComfyUI-native: auto-dtype, ModelPatcher, load_models_gpu)."""
    import comfy.ops
    import comfy.model_patcher

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    precision = config.get('precision', 'auto')
    attn_backend = config.get('attn_backend', 'auto')
    enable_flash = attn_backend in ('auto', 'flash_attn')

    # Detect weight dtype from safetensors
    from safetensors import safe_open
    with safe_open(config['model_file'], framework="pt", device="cpu") as f:
        weight_dtype = f.get_tensor(list(f.keys())[0]).dtype

    # Resolve model dtype: "auto" = auto-detect, otherwise explicit override
    if precision == 'auto':
        model_dtype = comfy.model_management.unet_dtype(
            device=device,
            supported_dtypes=[torch.bfloat16, torch.float32],
            weight_dtype=weight_dtype,
        )
    else:
        model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        model_dtype, device, [torch.bfloat16, torch.float32],
    )
    ops = comfy.ops.pick_operations(model_dtype, manual_cast_dtype, load_device=device)

    print(f"[X-Part Models] precision={precision}, dtype={model_dtype}, attn_backend={attn_backend}, ops={ops.__name__}")

    cache_key = f"{precision}_{attn_backend}_{pc_size}"

    if cache_key in _xpart_model_cache:
        cached = _xpart_model_cache[cache_key]
        print(f"[X-Part Models] Using cached models")
        return cached

    from .hunyuan3d_part.model import PartFormerDITPlain
    from .hunyuan3d_part.vae import VolumeDecoderShapeVAE
    from .hunyuan3d_part.conditioner import Conditioner
    from safetensors.torch import load_file

    # Get architecture config and apply runtime overrides
    xpart_config = _xpart_arch_config()

    print(f"[X-Part Models] Overriding pc_size in config: {pc_size}")
    xpart_config["shapevae"]["params"]["pc_size"] = pc_size
    xpart_config["shapevae"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_size"] = pc_size
    xpart_config["conditioner"]["params"]["geo_cfg"]["params"]["local_geo_cfg"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_size"] = pc_size
    xpart_config["conditioner"]["params"]["obj_encoder_cfg"]["params"]["pc_sharpedge_size"] = 0
    xpart_config["conditioner"]["params"]["seg_feat_cfg"]["params"]["enable_flash"] = enable_flash

    print(f"[X-Part Models] Loading DiT, VAE, and Conditioner ({model_dtype}, meta-device)...")
    t0 = time.time()

    def load_dit():
        model_params = dict(xpart_config["model"]["params"])
        with torch.device("meta"):
            model = PartFormerDITPlain(**model_params, dtype=model_dtype, device="meta", operations=ops)
        sd = load_file(config['model_file'], device="cpu")
        model.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(model, device)
        model.to(dtype=model_dtype)
        model.eval()
        _enable_lowvram_cast(model)
        print(f"[X-Part Models] DiT loaded")
        return model

    def load_vae():
        vae_params = dict(xpart_config["shapevae"]["params"])
        with torch.device("meta"):
            vae = VolumeDecoderShapeVAE(**vae_params, dtype=model_dtype, device="meta", operations=ops)
        sd = load_file(config['vae_file'], device="cpu")
        vae.load_state_dict(sd, strict=False, assign=True)
        _fix_meta_buffers(vae, device)
        vae.to(dtype=model_dtype)
        vae.eval()
        _enable_lowvram_cast(vae)
        print(f"[X-Part Models] VAE loaded")
        return vae

    def load_cond():
        # Extract params directly (no instantiate_from_config)
        cond_cfg = dict(xpart_config["conditioner"]["params"])

        # Build geo_encoder_params from geo_cfg
        geo_encoder_params = None
        geo_output_dim = None
        if cond_cfg.get("use_geo") and "geo_cfg" in cond_cfg:
            raw_geo = dict(cond_cfg["geo_cfg"])
            geo_output_dim = raw_geo.get("output_dim")
            raw_geo_params = dict(raw_geo.get("params", {}))
            # Convert local_geo_cfg from nested {target, params} to flat params dict
            if "local_geo_cfg" in raw_geo_params:
                raw_local = raw_geo_params["local_geo_cfg"]
                raw_geo_params["local_geo_cfg"] = dict(raw_local.get("params", {}))
            geo_encoder_params = raw_geo_params

        # Build obj_encoder_params from obj_encoder_cfg
        obj_encoder_params = None
        obj_output_dim = None
        if cond_cfg.get("use_obj") and "obj_encoder_cfg" in cond_cfg:
            raw_obj = dict(cond_cfg["obj_encoder_cfg"])
            obj_output_dim = raw_obj.get("output_dim")
            obj_encoder_params = dict(raw_obj.get("params", {}))

        # Build seg_feat_encoder_params from seg_feat_cfg
        seg_feat_encoder_params = None
        seg_feat_output_dim = None
        if cond_cfg.get("use_seg_feat") and "seg_feat_cfg" in cond_cfg:
            raw_seg = dict(cond_cfg["seg_feat_cfg"])
            seg_feat_output_dim = raw_seg.get("output_dim")
            seg_feat_encoder_params = dict(raw_seg.get("params", {}))

        # No meta-device for Conditioner (Sonata uses spconv which needs real device)
        conditioner = Conditioner(
            use_image=cond_cfg.get("use_image", False),
            use_geo=cond_cfg.get("use_geo", True),
            use_obj=cond_cfg.get("use_obj", True),
            use_seg_feat=cond_cfg.get("use_seg_feat", False),
            geo_encoder_params=geo_encoder_params,
            geo_output_dim=geo_output_dim,
            obj_encoder_params=obj_encoder_params,
            obj_output_dim=obj_output_dim,
            seg_feat_encoder_params=seg_feat_encoder_params,
            seg_feat_output_dim=seg_feat_output_dim,
            dtype=model_dtype, device=device, operations=ops,
        )
        sd = load_file(config['cond_file'], device="cpu")
        conditioner.load_state_dict(sd, strict=False)

        # Keep seg_feat_encoder in float32 (Sonata backbone)
        if hasattr(conditioner, 'seg_feat_encoder') and conditioner.seg_feat_encoder is not None:
            conditioner.seg_feat_encoder = conditioner.seg_feat_encoder.to(dtype=torch.float32)

        conditioner.eval()
        _enable_lowvram_cast(conditioner)
        print(f"[X-Part Models] Conditioner loaded")
        return conditioner

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_dit = executor.submit(load_dit)
        future_vae = executor.submit(load_vae)
        future_cond = executor.submit(load_cond)
        dit = future_dit.result()
        vae = future_vae.result()
        cond = future_cond.result()

    total_time = time.time() - t0
    print(f"[X-Part Models] All models loaded in {total_time:.2f}s")

    # Wrap in ModelPatcher for ComfyUI VRAM management (caller stages GPU loading)
    dit_patcher = comfy.model_patcher.ModelPatcher(dit, load_device=device, offload_device=offload_device)
    vae_patcher = comfy.model_patcher.ModelPatcher(vae, load_device=device, offload_device=offload_device)
    cond_patcher = comfy.model_patcher.ModelPatcher(cond, load_device=device, offload_device=offload_device)

    result = {
        'dit': dit_patcher,
        'vae': vae_patcher,
        'conditioner': cond_patcher,
        'config': xpart_config,
        'dtype': model_dtype,
    }

    _xpart_model_cache[cache_key] = result

    return result


class ComputeMeshFeatures:
    """
    Compute mesh features for P3-SAM segmentation.

    Performs:
    - Mesh cleaning and adjacent faces computation (~1.4s)
    - Point cloud sampling (~0.03s)
    - Sonata feature extraction (~0.5s)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "sonata_config": ("SONATA_CONFIG",),
                "all_points": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use ALL mesh vertices instead of sampling. Enable for X-Part generation."
                }),
                "point_num": ("INT", {
                    "default": 100000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample (ignored if all_points=True)."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff,
                    "step": 1,
                    "tooltip": "Random seed for point sampling (ignored if all_points=True)."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh_with_features",)
    FUNCTION = "compute_features"
    CATEGORY = "Hunyuan3D/Processing"

    def compute_features(self, mesh, sonata_config, all_points, point_num, seed):
        """Compute mesh features using Sonata encoder."""
        try:
            # Load mesh if needed
            if isinstance(mesh, dict) and 'trimesh' in mesh:
                mesh_obj = mesh['trimesh']
            elif isinstance(mesh, str):
                mesh_obj = load_mesh(mesh)
            elif isinstance(mesh, trimesh.Trimesh):
                mesh_obj = mesh
            else:
                mesh_obj = load_mesh(mesh)

            print(f"[Compute Features] Computing mesh features...")

            # Load Sonata encoder
            sonata_model = _get_sonata_model(sonata_config)

            # Import required functions from core
            from .p3sam_processing import (
                build_adjacent_faces_numba,
                get_feat,
                clean_mesh,
                normalize_pc
            )

            # Clean mesh if needed
            print(f"[Compute Features] Cleaning mesh...")
            mesh_loaded = clean_mesh(mesh_obj)
            mesh_loaded = trimesh.Trimesh(vertices=mesh_loaded.vertices, faces=mesh_loaded.faces)

            # Build adjacent faces
            print(f"[Compute Features] Building adjacent faces...")
            face_adjacency = mesh_loaded.face_adjacency
            adjacent_faces = build_adjacent_faces_numba(face_adjacency)

            # Get points: either sample or use ALL vertices
            if all_points:
                # Use ALL mesh vertices
                print(f"[Compute Features] Using ALL {len(mesh_loaded.vertices)} mesh vertices...")
                _points = mesh_loaded.vertices
                normals = mesh_loaded.vertex_normals
                face_idx = None  # No face mapping when using vertices
            else:
                # Sample point cloud
                print(f"[Compute Features] Sampling {point_num} points...")
                _points, face_idx = trimesh.sample.sample_surface(mesh_loaded, point_num, seed=seed)
                normals = mesh_loaded.face_normals[face_idx]

            _points_normalized = normalize_pc(_points)

            # Extract features using P3-SAM's internal Sonata
            print(f"[Compute Features] Extracting Sonata features...")
            points = _points_normalized.astype(np.float32)
            normals = normals.astype(np.float32)

            # Get features using Sonata encoder
            t0 = time.time()
            feats = get_feat(sonata_model, points, normals)
            feat_time = time.time() - t0
            print(f"[Compute Features] Features computed ({feat_time:.2f}s)")

            # Store everything P3-SAM needs in mesh metadata
            feats_np = feats.detach().cpu().to(torch.float32).numpy()  # [N, 512]
            mesh_loaded.metadata['features'] = feats_np
            mesh_loaded.metadata['points'] = np.asarray(points)
            mesh_loaded.metadata['normals'] = np.asarray(normals)  # [N, 3]
            mesh_loaded.metadata['face_idx'] = np.asarray(face_idx) if face_idx is not None else None
            mesh_loaded.metadata['adjacent_faces'] = np.asarray(adjacent_faces)
            mesh_loaded.metadata['seed'] = seed

            # PCA4 on raw sample features (clean data, no zero-padding artifacts)
            print(f"[Compute Features] Computing PCA4 debug field...")
            X = feats_np - feats_np.mean(axis=0)  # [N, 512]
            C = (X.T @ X) / max(len(X), 1)        # [512, 512] covariance
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            top4 = eigenvectors[:, np.argsort(eigenvalues)[::-1][:4]]  # [512, 4]
            pca4_samples = (X @ top4).astype(np.float32)  # [N, 4]

            # Project PCA scores to vertices via face_idx scatter-average
            num_verts = len(mesh_loaded.vertices)
            if all_points:
                pca4_verts = pca4_samples  # N == V
            else:
                face_idx_arr = np.asarray(face_idx)
                # Vectorized scatter: for each sample, add its PCA scores to the 3 face vertices
                vert_pca = np.zeros((num_verts, 4), dtype=np.float32)
                vert_counts = np.zeros(num_verts, dtype=np.float32)
                sample_verts = mesh_loaded.faces[face_idx_arr]  # [N, 3]
                for c in range(3):
                    np.add.at(vert_pca, sample_verts[:, c], pca4_samples)
                    np.add.at(vert_counts, sample_verts[:, c], 1.0)
                covered = vert_counts > 0
                vert_pca[covered] /= vert_counts[covered, np.newaxis]
                pca4_verts = vert_pca

            if not hasattr(mesh_loaded, 'vertex_attributes') or mesh_loaded.vertex_attributes is None:
                mesh_loaded.vertex_attributes = {}
            mesh_loaded.vertex_attributes['features_pca4_debug'] = pca4_verts

            return (mesh_loaded,)

        except Exception as e:
            print(f"[Compute Features] Error: {e}")
            import traceback
            traceback.print_exc()
            raise



class P3SAMSegmentMesh:
    """
    Segment mesh into semantic parts using P3-SAM.

    Takes pre-computed mesh features and runs P3-SAM inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_with_features": ("TRIMESH",),
                "p3sam_config": ("P3SAM_CONFIG",),
                "prompt_num": ("INT", {
                    "default": 400,
                    "min": 50,
                    "max": 1000,
                    "step": 10,
                    "tooltip": "Number of prompt points sampled across the mesh surface. More prompts improve coverage: small or thin parts that receive no prompt point will be missed. Increase if parts are being skipped; decrease to speed up inference."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.7,
                    "max": 0.999,
                    "step": 0.01,
                    "tooltip": "Merge threshold."
                }),
                "post_process": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable post-processing."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of prompt points processed per forward pass. Higher values use more VRAM but speed up inference significantly. Start with 8-16 and increase if VRAM allows."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "BBOXES_3D")
    RETURN_NAMES = ("mesh", "bounding_boxes")
    FUNCTION = "segment"
    CATEGORY = "Hunyuan3D/Processing"

    def segment(self, mesh_with_features, p3sam_config, prompt_num, threshold, post_process, batch_size=1):
        """Segment mesh into parts using P3-SAM."""
        try:
            import fpsample
            from tqdm import tqdm
            from collections import defaultdict
            from concurrent.futures import ThreadPoolExecutor

            # Extract feature data from mesh metadata
            mesh_loaded = mesh_with_features
            face_idx = mesh_with_features.metadata['face_idx']
            adjacent_faces = mesh_with_features.metadata['adjacent_faces']
            feats = torch.from_numpy(mesh_with_features.metadata['features'])
            points = mesh_with_features.metadata['points']
            seed = mesh_with_features.metadata['seed']

            print(f"[P3-SAM Segment] Running segmentation with {prompt_num} prompts, batch_size={batch_size}...")
            _vram_dbg("before model load")

            # Load model from config (ModelPatcher handles GPU placement)
            p3sam = _get_p3sam_model(p3sam_config)
            device = comfy.model_management.get_torch_device()
            _vram_dbg("after model load")

            # Import functions from core
            from .p3sam_processing import (
                get_mask,
                cal_iou,
                fix_label,
                get_aabb_from_face_ids,
                do_post_process
            )

            # FPS sample prompt points
            fps_idx = fpsample.fps_sampling(points, prompt_num)
            _point_prompts = points[fps_idx]
            _dbg(f"FPS done, {len(fps_idx)} prompt points selected")

            # Pre-upload invariant tensors to GPU once (instead of per-batch)
            model_dtype = next(p3sam.parameters()).dtype
            feats_gpu = feats.to(device=device, dtype=model_dtype)       # [N, 512]
            points_gpu = torch.from_numpy(points).to(device=device, dtype=model_dtype)  # [N, 3]
            _vram_dbg("after tensor pre-upload")

            bs = batch_size
            step_num = (prompt_num + bs - 1) // bs
            mask_res = []
            iou_res = []
            print(f"[P3-SAM Segment] {prompt_num} prompts / batch_size {bs} = {step_num} steps")
            _vram_dbg("before inference loop")
            comfy_pbar = comfy.utils.ProgressBar(step_num)
            for i in tqdm(range(step_num), desc="P3-SAM Inference"):
                cur_prompt = _point_prompts[bs * i : bs * (i + 1)]
                if len(cur_prompt) == 0:
                    continue
                # get_mask returns GPU tensors: masks [N,K], pred_iou [K,3]
                mask_1, mask_2, mask_3, pred_iou = get_mask(
                    p3sam, feats_gpu, points_gpu, cur_prompt,
                    device=device, model_dtype=model_dtype
                )
                if i == 0:
                    _vram_dbg(f"after first batch (bs={len(cur_prompt)})")
                # Select best mask on GPU, transfer only 1x[N] bool per prompt
                pred_iou_cpu = pred_iou.detach().cpu().numpy()  # [K, 3] — tiny
                max_idx = np.argmax(pred_iou_cpu, axis=-1)      # [K]
                masks_stacked = torch.stack([mask_1, mask_2, mask_3], dim=-1)  # [N, K, 3] GPU
                for j in range(max_idx.shape[0]):
                    best_mask = (masks_stacked[:, j, max_idx[j]] > 0.5)  # [N] bool, GPU
                    mask_res.append(best_mask.cpu().numpy())
                    iou_res.append(pred_iou_cpu[j, max_idx[j]])
                _dbg(f"Batch {i + 1}/{step_num}, masks so far: {len(mask_res)}")
                comfy_pbar.update(1)
            _vram_dbg("after inference loop")
            del feats_gpu, points_gpu

            mask_res = np.stack(mask_res, axis=-1)

            # Sort by IOU
            mask_iou = [[mask_res[:, i], iou_res[i]] for i in range(len(iou_res))]
            mask_iou_sorted = sorted(mask_iou, key=lambda x: x[1], reverse=True)
            mask_sorted = [mask_iou_sorted[i][0] for i in range(len(iou_res))]
            iou_sorted = [mask_iou_sorted[i][1] for i in range(len(iou_res))]

            # NMS
            clusters = defaultdict(list)
            with ThreadPoolExecutor(max_workers=20) as executor:
                for i in tqdm(range(len(mask_sorted)), desc="NMS"):
                    _mask = mask_sorted[i]
                    futures = []
                    for j in clusters.keys():
                        futures.append(executor.submit(cal_iou, _mask, mask_sorted[j]))

                    for j, future in zip(clusters.keys(), futures):
                        if future.result() > 0.9:
                            clusters[j].append(i)
                            break
                    else:
                        clusters[i].append(i)

            print(f"[P3-SAM Segment] NMS complete: {len(clusters)} clusters")

            # Filter single mask clusters
            filtered_clusters = [i for i in clusters.keys() if len(clusters[i]) > 2]

            # Merge similar clusters
            merged_clusters = []
            for i in filtered_clusters:
                merged = False
                for j in range(len(merged_clusters)):
                    if cal_iou(mask_sorted[i], mask_sorted[merged_clusters[j]]) > threshold:
                        merged = True
                        break
                if not merged:
                    merged_clusters.append(i)

            _dbg(f"NMS clusters: {len(clusters)}, filtered: {len(filtered_clusters)}, merged: {len(merged_clusters)}")

            # Calculate point labels
            point_labels = np.zeros(len(points), dtype=np.int32) - 1
            for idx, cluster_id in enumerate(merged_clusters):
                mask = mask_sorted[cluster_id] > 0.5
                point_labels[mask] = idx

            # Project to mesh faces
            face_labels = np.zeros(len(mesh_loaded.faces), dtype=np.int32) - 1
            if face_idx is not None:
                for i, fid in enumerate(face_idx):
                    if point_labels[i] >= 0 and face_labels[fid] < 0:
                        face_labels[fid] = point_labels[i]

            _dbg(f"Face labels: {np.sum(face_labels >= 0)}/{len(face_labels)} assigned")

            # Fix unlabeled faces
            if post_process:
                face_ids = fix_label(face_labels, adjacent_faces, use_aabb=True, mesh=mesh_loaded, show_info=True)
            else:
                face_ids = face_labels

            # Get AABBs
            aabb = get_aabb_from_face_ids(mesh_loaded, face_ids)

            print(f"[P3-SAM Segment] Segmentation complete: found {len(aabb)} parts")

            # Add metadata
            processed_mesh = mesh_loaded

            # Store face part IDs as a face attribute so GeometryPack can visualize it
            if not hasattr(processed_mesh, 'face_attributes') or processed_mesh.face_attributes is None:
                processed_mesh.face_attributes = {}
            processed_mesh.face_attributes['part_id'] = face_ids.astype(np.int32)

            bboxes_output = {
                'bboxes': aabb,
                'num_parts': len(aabb)
            }

            return (processed_mesh, bboxes_output)

        except Exception as e:
            print(f"[P3-SAM Segment] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


class XPartGenerateParts:
    """
    Generate high-quality part meshes using X-Part.

    Takes mesh_with_features (from ComputeMeshFeatures with all_points=True),
    bounding boxes, and X-Part config as inputs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_with_features": ("TRIMESH",),
                "bounding_boxes": ("BBOXES_3D",),
                "xpart_config": ("XPART_CONFIG",),
                "octree_resolution": ("INT", {
                    "default": 256,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Mesh quality. 256=~8GB, 512=~12-16GB, 1024=~24GB+ VRAM"
                }),
                "num_inference_steps": ("INT", {
                    "default": 25,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Diffusion steps. 25=fast default, 50=high quality"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "-1.0 = disabled (fastest). 0-10 = enabled (slower, doubles compute)."
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducibility."
                }),
                "pc_size": ("INT", {
                    "default": 40960,
                    "min": 1024,
                    "max": 81920,
                    "step": 1024,
                    "tooltip": "Points per object/part. 40960=trained default, <20480=quality loss, <5120=very poor. Model reloads on change."
                }),
                "output_coordinate_system": (["Y-up (default)", "Z-up"], {
                    "default": "Y-up (default)",
                    "tooltip": "Output coordinate system. Use Z-up if your input mesh is Z-up (CAD convention like STL)."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("part_meshes", "parts_path", "bbox_path", "exploded_path")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D/Processing"

    def generate(self, mesh_with_features, bounding_boxes, xpart_config, octree_resolution, num_inference_steps,
                guidance_scale, seed, pc_size, output_coordinate_system):
        """Generate part meshes."""
        device = comfy.model_management.get_torch_device()
        try:
            # Read Sonata features and geometry from mesh metadata
            mesh_obj = mesh_with_features
            sonata_features = mesh_with_features.metadata.get('features')
            sonata_points = mesh_with_features.metadata.get('points')
            sonata_normals = mesh_with_features.metadata.get('normals')

            if sonata_features is None or sonata_points is None or sonata_normals is None:
                raise ValueError(
                    "mesh_with_features is missing Sonata features. "
                    "Connect a ComputeMeshFeatures node (with all_points=True recommended) to this input."
                )

            all_points_mode = mesh_with_features.metadata.get('face_idx') is None
            if not all_points_mode:
                print(f"[X-Part Generate] WARNING: features were computed with all_points=False. "
                      f"For best results, use ComputeMeshFeatures with all_points=True for X-Part generation.")

            print(f"[X-Part Generate] Using pre-computed Sonata features ({len(sonata_features)} points)")

            # Save mesh to temp file for pipeline
            mesh_path = get_temp_mesh_path(prefix="xpart_input_", suffix=".glb")
            save_mesh(mesh_obj, mesh_path)

            # Read bounding boxes from explicit BBOXES_3D input (from P3SAMSegmentMesh)
            aabb = bounding_boxes['bboxes']
            print(f"[X-Part Generate] Using {len(aabb)} bounding boxes")

            # Load models from config (returns ModelPatcher-wrapped models, NOT on GPU yet)
            models = _get_xpart_models(xpart_config, pc_size=pc_size)
            dit_patcher = models['dit']
            vae_patcher = models['vae']
            cond_patcher = models['conditioner']
            dit = dit_patcher.model
            vae = vae_patcher.model
            conditioner = cond_patcher.model
            xpart_cfg = models['config']
            dtype = models['dtype']

            print(f"[X-Part Generate] Using {dtype} precision")

            # Import from core
            from .xpart_pipeline import PartFormerPipeline, retrieve_timesteps, export_to_trimesh
            from .misc_utils import synchronize_timer
            from .geometry_utils import explode_mesh

            # Create scheduler directly
            scheduler_params = dict(xpart_cfg["scheduler"]["params"])
            scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_params)

            # Create minimal pipeline instance (references models, doesn't move them)
            pipeline = PartFormerPipeline(
                vae=vae,
                model=dit,
                scheduler=scheduler,
                conditioner=conditioner,
                bbox_predictor=None,
                device=device,
                dtype=dtype,
                verbose=True
            )

            print(f"[X-Part Generate] Running generation...")
            print(f"  - Octree resolution: {octree_resolution}")
            print(f"  - Inference steps: {num_inference_steps}")
            print(f"  - Guidance scale: {guidance_scale}")
            print(f"  - Seed: {seed}")

            # Convert aabb to tensor if provided
            if aabb is not None:
                aabb_tensor = torch.from_numpy(aabb).to(device=device, dtype=dtype)
            else:
                aabb_tensor = None

            # Prepare precomputed Sonata features for conditioner
            precomputed_sonata = {
                'features': torch.as_tensor(sonata_features).to(device=device, dtype=dtype),
                'points': torch.as_tensor(sonata_points).to(device=device, dtype=dtype),
                'normals': torch.as_tensor(sonata_normals).to(device=device, dtype=dtype),
            }

            # ── Staged generation: load each model to GPU only when needed ──

            do_classifier_free_guidance = guidance_scale >= 0 and not (
                hasattr(dit, "guidance_embed") and dit.guidance_embed is True
            )

            # 1. Check inputs and prepare geometry (CPU-side, no model needed)
            obj_surface, aabb_t, part_surface_inbbox, mesh_loaded, center, scale = pipeline.check_inputs(
                obj_surface=None, obj_surface_raw=None,
                mesh_path=mesh_path, mesh=None,
                aabb=aabb_tensor, part_surface_inbbox=None,
                seed=seed,
            )

            # Bbox visualization (before we move to GPU dtype)
            if pipeline.verbose:
                mesh_bbox = trimesh.Scene()
                if mesh_loaded is not None:
                    mesh_bbox.add_geometry(mesh_loaded)
                for bbox in aabb_t[0]:
                    box = trimesh.path.creation.box_outline()
                    box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
                    box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
                    mesh_bbox.add_geometry(box)

            obj_surface = obj_surface.to(device=device, dtype=dtype)
            aabb_t = aabb_t.to(device=device, dtype=dtype)
            part_surface_inbbox = part_surface_inbbox.to(device=device, dtype=dtype)
            batch_size, num_parts, N, dim = part_surface_inbbox.shape

            # Prepare latents and tokens (uses vae.latent_shape but no GPU compute)
            num_tokens = torch.tensor(
                [pipeline.allocate_tokens(x, vae.latent_shape[0]) for x in aabb_t],
                device=device,
            )
            generator = torch.Generator(device='cpu').manual_seed(seed)
            latents = pipeline.prepare_latents(
                num_parts, vae.latent_shape, dtype, device, generator
            )

            # ── Stage 1: Conditioning (load conditioner to GPU) ──
            print(f"[X-Part Generate] Stage 1: Conditioning")
            _vram_dbg("before conditioner load")
            comfy.model_management.load_models_gpu([cond_patcher])
            _vram_dbg("after conditioner load")

            cond = pipeline.encode_cond(
                part_surface_inbbox.reshape(batch_size * num_parts, N, dim),
                obj_surface.expand(batch_size * num_parts, -1, -1),
                do_classifier_free_guidance,
                precomputed_sonata_features=precomputed_sonata,
            )

            # Free conditioner inputs we no longer need
            del part_surface_inbbox, obj_surface, precomputed_sonata
            _vram_dbg("after conditioning")

            # ── Stage 2: Diffusion (load DiT to GPU, conditioner auto-evicted) ──
            print(f"[X-Part Generate] Stage 2: Diffusion ({num_inference_steps} steps)")
            comfy.model_management.load_models_gpu([dit_patcher])
            _vram_dbg("after DiT load")

            # Guidance conditioning
            guidance_cond = None
            if getattr(dit, "guidance_cond_proj_dim", None) is not None:
                guidance_scale_tensor = torch.tensor(guidance_scale - 1, device=device).repeat(batch_size)
                guidance_cond = pipeline.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=dit.guidance_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # Prepare timesteps
            sigmas = np.linspace(0, 1, num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                scheduler, num_inference_steps, device, sigmas=sigmas,
            )

            comfy.model_management.soft_empty_cache()
            aabb_orig = aabb_t

            # Denoising loop
            diffusion_pbar = comfy.utils.ProgressBar(len(timesteps))
            from tqdm import tqdm
            with synchronize_timer("Diffusion Sampling"):
                for i, t in enumerate(
                    tqdm(timesteps, desc="Diffusion Sampling:")
                ):
                    if do_classifier_free_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                        aabb_input = torch.repeat_interleave(aabb_orig, 2, dim=0)
                    else:
                        latent_model_input = latents
                        aabb_input = aabb_orig

                    timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                    timestep = timestep / scheduler.config.num_train_timesteps
                    noise_pred = dit(
                        latent_model_input, timestep, cond,
                        aabb=aabb_input, num_tokens=num_tokens,
                        guidance_cond=guidance_cond,
                    )

                    if do_classifier_free_guidance:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_cond - noise_pred_uncond
                        )

                    outputs = scheduler.step(noise_pred, t, latents)
                    latents = outputs.prev_sample

                    del noise_pred
                    if do_classifier_free_guidance:
                        del latent_model_input, aabb_input

                    diffusion_pbar.update(1)

            # Free diffusion-only tensors
            del cond, guidance_cond
            _vram_dbg("after diffusion")

            # ── Stage 3: VAE decode + marching cubes (load VAE to GPU) ──
            print(f"[X-Part Generate] Stage 3: VAE decode ({len(latents)} parts)")
            comfy.model_management.load_models_gpu([vae_patcher])
            _vram_dbg("after VAE load")

            export_pbar = comfy.utils.ProgressBar(len(latents))
            parts_list = []
            for i, part_latent in enumerate(latents):
                _vram_dbg(f"before export part {i}")
                try:
                    part_mesh = pipeline._export(
                        latents=part_latent.unsqueeze(0),
                        output_type="trimesh",
                        box_v=1.01,
                        mc_level=-1 / 512,
                        num_chunks=0,
                        octree_resolution=octree_resolution,
                        mc_algo="mc",
                        enable_pbar=True,
                    )[0]
                    random_color = np.random.randint(0, 255, size=3)
                    part_mesh.visual.face_colors = random_color
                    parts_list.append(part_mesh)
                except Exception as e:
                    print(f"[X-Part Generate] Failed to export part {i}: {e}")
                # Marching cubes allocates large temporary grids; release them between parts
                comfy.model_management.soft_empty_cache()
                _vram_dbg(f"after export part {i} (cache cleared)")
                export_pbar.update(1)

            # Denormalize
            print(f"Denormalize mesh: {center}, {scale}")
            for part_mesh in parts_list:
                part_mesh.vertices = part_mesh.vertices * scale + center

            # Build viz
            viz_tuple = None
            if pipeline.verbose:
                import copy as _copy
                temp_scene = trimesh.Scene()
                for p in parts_list:
                    temp_scene.add_geometry(p)
                explode_object = explode_mesh(_copy.deepcopy(temp_scene), explosion_scale=0.2)
                out_bbox = trimesh.Scene()
                out_bbox.add_geometry(temp_scene)
                for bbox in aabb_t[0]:
                    box = trimesh.path.creation.box_outline()
                    box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
                    box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
                    box.vertices = box.vertices * scale + center
                    out_bbox.add_geometry(box)
                viz_tuple = (out_bbox, mesh_bbox, explode_object)

            _vram_dbg("after VAE decode")

            print(f"[X-Part Generate] Generation complete! {len(parts_list)} parts")

            # Apply coordinate system transformation if requested
            if output_coordinate_system == "Z-up":
                print("[X-Part Generate] Converting output to Z-up coordinate system...")
                rotation_matrix = np.array([
                    [1,  0,  0, 0],
                    [0,  0,  1, 0],
                    [0, -1,  0, 0],
                    [0,  0,  0, 1]
                ])
                for part in parts_list:
                    part.apply_transform(rotation_matrix)
                print("[X-Part Generate] Converted to Z-up")

            # Save outputs
            output_dir = folder_paths.get_output_directory()

            # Build temp Scene for saving parts GLB
            parts_scene = trimesh.Scene()
            for p in parts_list:
                parts_scene.add_geometry(p)
            parts_path = os.path.join(output_dir, f"xpart_parts_{seed}.glb")
            save_mesh(parts_scene, parts_path)
            print(f"[X-Part Generate] Saved parts to: {parts_path}")

            # Save bbox viz and exploded view if available
            bbox_path = ""
            exploded_path = ""
            if viz_tuple is not None:
                out_bbox, mesh_gt_bbox, explode_object = viz_tuple
                if output_coordinate_system == "Z-up":
                    for name in list(out_bbox.geometry.keys()):
                        geom = out_bbox.geometry[name]
                        if hasattr(geom, 'apply_transform'):
                            geom.apply_transform(rotation_matrix)
                    for name in list(explode_object.geometry.keys()):
                        geom = explode_object.geometry[name]
                        if hasattr(geom, 'apply_transform'):
                            geom.apply_transform(rotation_matrix)

                bbox_path = os.path.join(output_dir, f"xpart_bbox_{seed}.glb")
                save_mesh(out_bbox, bbox_path)
                print(f"[X-Part Generate] Saved bbox viz to: {bbox_path}")

                exploded_path = os.path.join(output_dir, f"xpart_exploded_{seed}.glb")
                save_mesh(explode_object, exploded_path)
                print(f"[X-Part Generate] Saved exploded view to: {exploded_path}")

            # Clean up temp file
            if mesh_path and mesh_path.startswith(tempfile.gettempdir()):
                try:
                    os.remove(mesh_path)
                except (OSError, IOError) as cleanup_err:
                    print(f"[X-Part Generate] Warning: Failed to clean temp file: {cleanup_err}")

            # VRAM is managed by ComfyUI via ModelPatcher/load_models_gpu

            return (parts_list, parts_path, bbox_path, exploded_path)

        except Exception as e:
            print(f"[X-Part Generate] Error: {e}")
            import traceback
            traceback.print_exc()
            raise


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ComputeMeshFeatures": ComputeMeshFeatures,
    "P3SAMSegmentMesh": P3SAMSegmentMesh,
    "XPartGenerateParts": XPartGenerateParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComputeMeshFeatures": "Compute Mesh Features",
    "P3SAMSegmentMesh": "P3-SAM Segment Mesh",
    "XPartGenerateParts": "X-Part Generate Parts",
}

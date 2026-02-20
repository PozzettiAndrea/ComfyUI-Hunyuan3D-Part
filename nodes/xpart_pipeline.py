import torch
import comfy.model_management
from .misc_utils import logger, synchronize_timer
import inspect
from typing import List, Optional
import trimesh
import numpy as np
from tqdm import tqdm
import copy
from typing import List, Optional, Union
import os
from .geometry_utils import (
    SampleMesh,
    load_surface_points,
    sample_bbox_points_from_trimesh,
    explode_mesh,
    fix_mesh,
)
from .misc_utils import (
    init_from_ckpt,
    instantiate_from_config,
    get_config_from_file,
    smart_load_model,
)

from diffusers.utils.torch_utils import randn_tensor
from pathlib import Path


@synchronize_timer("Export to trimesh")
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                mesh_output = fix_mesh(mesh_output)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        mesh_output = fix_mesh(mesh_output)
        return mesh_output


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[Union[List[float], np.ndarray]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to"
            " set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps`"
                " does not support custom timestep schedules. Please check whether you"
                " are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps`"
                " does not support custom sigmas schedules. Please check whether you"
                " are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class TokenAllocMixin:

    def allocate_tokens(self, bboxes, num_latents=512):
        return np.array([num_latents] * bboxes.shape[0])


class PartFormerPipeline(TokenAllocMixin):

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        bbox_predictor=None,
        device="cuda",
        dtype=torch.float32,
        verbose=False,
        **kwargs,
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.bbox_predictor = bbox_predictor
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.verbose = verbose
        self.kwargs = kwargs

    @classmethod
    @synchronize_timer("Hunyuan3D PartGen Pipeline Model Loading")
    def from_single_file(
        cls,
        ckpt_path=None,
        config=None,
        device="cuda",
        dtype=torch.float32,
        use_safetensors=None,
        ignore_keys=(),
        **kwargs,
    ):
        # prepare config
        if config is None:
            config = get_config_from_file(
                str(
                    Path(__file__).parent.parent
                    / "config"
                    / "partformer_full_pipeline_512_with_sonata.yaml"
                )
            )
        # TODO:
        if ckpt_path is None:
            ckpt_path = str(
                Path(__file__).parent
                / "ckpts"
                / "partformer_full_pipeline_512_with_sonata.ckpt"
            )
        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace(".ckpt", ".safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch

            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split(".")[0]
                new_key = key[len(model_name) + 1 :]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            # ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=True)
            ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        # load model
        model = instantiate_from_config(config["model"])
        # model.load_state_dict(ckpt["model"])
        init_from_ckpt(model, ckpt, prefix="model", ignore_keys=ignore_keys)
        vae = instantiate_from_config(config["shapevae"])
        # vae.load_state_dict(ckpt["shapevae"], strict=False)
        init_from_ckpt(vae, ckpt, prefix="shapevae", ignore_keys=ignore_keys)
        if config.get("conditioner", None) is not None:
            conditioner = instantiate_from_config(config["conditioner"])
            init_from_ckpt(
                conditioner, ckpt, prefix="conditioner", ignore_keys=ignore_keys
            )
        else:
            conditioner = vae
        scheduler = instantiate_from_config(config["scheduler"])
        bbox_predictor = instantiate_from_config(config.get("bbox_predictor", None))
        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            bbox_predictor=bbox_predictor,  # TODO: add bbox predictor
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        config=None,
        dtype=torch.float32,
        ignore_keys=(),
        device="cuda",
        **kwargs,
    ):
        if config is None:
            config = get_config_from_file(
                str(
                    Path(__file__).parent.parent
                    / "config"
                    / "partformer_full_pipeline_512_with_sonata.yaml"
                )
            )
        ckpt_path = smart_load_model(
            model_path="tencent/Hunyuan3D-Part",
        )

        # Check if using old monolithic checkpoint or new safetensors format
        old_ckpt_file = os.path.join(ckpt_path, "xpart.pt")
        if os.path.exists(old_ckpt_file):
            print(f"[X-Part] Loading from old format (monolithic): {old_ckpt_file}")
            # Old format: load monolithic checkpoint
            import time
            t0 = time.time()
            ckpt = torch.load(old_ckpt_file, map_location="cuda", weights_only=False)
            print(f"[X-Part] [OK] Loaded checkpoint ({time.time()-t0:.2f}s)")

            # load model
            print(f"[X-Part] Loading model...")
            t0 = time.time()
            model = instantiate_from_config(config["model"])
            init_from_ckpt(model, ckpt, prefix="model", ignore_keys=ignore_keys)
            print(f"[X-Part] [OK] Model loaded ({time.time()-t0:.2f}s)")

            print(f"[X-Part] Loading VAE...")
            t0 = time.time()
            vae = instantiate_from_config(config["shapevae"])
            init_from_ckpt(vae, ckpt, prefix="shapevae", ignore_keys=ignore_keys)
            print(f"[X-Part] [OK] VAE loaded ({time.time()-t0:.2f}s)")

            if config.get("conditioner", None) is not None:
                print(f"[X-Part] Loading conditioner...")
                t0 = time.time()
                conditioner = instantiate_from_config(config["conditioner"])
                init_from_ckpt(
                    conditioner, ckpt, prefix="conditioner", ignore_keys=ignore_keys
                )
                print(f"[X-Part] [OK] Conditioner loaded ({time.time()-t0:.2f}s)")
            else:
                conditioner = vae
            p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam.ckpt")
        else:
            print(f"[X-Part] Loading from new format (safetensors): {ckpt_path}")
            # New format: load from individual safetensors files
            from safetensors.torch import load_file
            import time

            # Load model
            print(f"[X-Part] Loading model...")
            t0 = time.time()
            model = instantiate_from_config(config["model"])
            model_path = os.path.join(ckpt_path, "model", "model.safetensors")
            print(f"[X-Part]   Reading {model_path}")
            model_state = load_file(model_path, device="cuda")
            model.load_state_dict(model_state, strict=False)
            print(f"[X-Part] [OK] Model loaded ({time.time()-t0:.2f}s)")

            # Load shapevae
            print(f"[X-Part] Loading VAE...")
            t0 = time.time()
            vae = instantiate_from_config(config["shapevae"])
            vae_path = os.path.join(ckpt_path, "shapevae", "shapevae.safetensors")
            print(f"[X-Part]   Reading {vae_path}")
            vae_state = load_file(vae_path, device="cuda")
            vae.load_state_dict(vae_state, strict=False)
            print(f"[X-Part] [OK] VAE loaded ({time.time()-t0:.2f}s)")

            # Load conditioner
            if config.get("conditioner", None) is not None:
                print(f"[X-Part] Loading conditioner...")
                t0 = time.time()
                conditioner = instantiate_from_config(config["conditioner"])
                print(f"[X-Part] DEBUG: After instantiate_from_config, obj_encoder.encoder.pc_size = {conditioner.obj_encoder.encoder.pc_size}")
                conditioner_path = os.path.join(ckpt_path, "conditioner", "conditioner.safetensors")
                print(f"[X-Part]   Reading {conditioner_path}")
                conditioner_state = load_file(conditioner_path, device="cuda")
                conditioner.load_state_dict(conditioner_state, strict=False)
                print(f"[X-Part] DEBUG: After load_state_dict, obj_encoder.encoder.pc_size = {conditioner.obj_encoder.encoder.pc_size}")
                print(f"[X-Part] [OK] Conditioner loaded ({time.time()-t0:.2f}s)")
            else:
                conditioner = vae

            # Path to p3sam checkpoint (safetensors format)
            p3sam_ckpt_path = os.path.join(ckpt_path, "p3sam", "p3sam.safetensors")

        scheduler = instantiate_from_config(config["scheduler"])
        config["bbox_predictor"]["params"]["ckpt_path"] = p3sam_ckpt_path
        bbox_predictor = instantiate_from_config(config.get("bbox_predictor", None))

        # Set all models to eval mode for inference
        print(f"[X-Part] Setting models to eval mode...")
        model.eval()
        vae.eval()
        conditioner.eval()
        print(f"[X-Part] [OK] Models in eval mode")

        # Ensure models are on correct device and dtype
        print(f"[X-Part] Moving models to {device} with float32 precision...")
        model = model.to(device=device, dtype=dtype)
        vae = vae.to(device=device, dtype=dtype)
        conditioner = conditioner.to(device=device, dtype=dtype)
        print(f"[X-Part] [OK] Models ready on {device} in float32")

        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            bbox_predictor=bbox_predictor,  # TODO: add bbox predictor
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)
        return cls(**model_kwargs)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
            if self.bbox_predictor is not None:
                self.bbox_predictor.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)
            if self.bbox_predictor is not None:
                self.bbox_predictor.to(device)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to eta in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def predict_bbox(
        self, mesh: trimesh.Trimesh, scale_box=1.0, drop_normal=True, seed=42
    ):
        """
        Predict the bounding box of the object surface.
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3]
        Returns:
            `torch.Tensor`: [B, K, 2, 3] where K is the number of bounding boxes
        """
        if self.bbox_predictor is None:
            raise ValueError("bbox_predictor is not set.")
        aabb, face_ids, mesh = self.bbox_predictor.predict_aabb(
            mesh, post_process=True, seed=seed
        )
        # aabb, face_ids, mesh = self.bbox_predictor.predict_aabb(mesh, post_process=False)
        aabb = torch.from_numpy(aabb).to(self.device)
        return aabb

    def prepare_latents(
        self, batch_size, latent_shape, dtype, device, generator, latents=None
    ):
        # prepare latents for different parts
        shape = (batch_size, *latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but"
                f" requested an effective batch size of {batch_size}. Make sure the"
                " batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, "init_noise_sigma", 1.0)
        return latents

    @synchronize_timer("Encode cond")
    def encode_cond(
        self,
        part_surface_inbbox,
        object_surface,
        do_classifier_free_guidance,
        precomputed_sonata_features=None,
    ):
        bsz = object_surface.shape[0]
        cond = self.conditioner(part_surface_inbbox, object_surface, precomputed_sonata_features=precomputed_sonata_features)

        if do_classifier_free_guidance:
            # TODO: do_classifier_free_guidance, un_cond
            un_cond = {k: torch.zeros_like(v) for k, v in cond.items()}

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(self.dtype)
                out = {}
                for k in a.keys():
                    out[k] = cat_recursive(a[k], b[k])
                return out

            cond = cat_recursive(cond, un_cond)
        return cond

    def normalize_mesh(self, mesh):
        vertices = mesh.vertices
        min_xyz = np.min(vertices, axis=0)
        max_xyz = np.max(vertices, axis=0)
        center = (min_xyz + max_xyz) / 2.0
        # scale = np.max(np.linalg.norm(vertices - center, axis=1))
        scale = np.max(max_xyz - min_xyz) / 2 / 0.8
        vertices = (vertices - center) / scale
        mesh.vertices = vertices
        return mesh, center, scale

    def check_inputs(
        self,
        obj_surface=None,
        obj_surface_raw=None,
        mesh_path=None,
        mesh=None,
        aabb=None,
        part_surface_inbbox=None,
        seed=42,
    ):
        """
        Check the inputs of the pipeline.
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3+3+1]
            mesh_path (`str`): path to the mesh file
            mesh (`trimesh.Trimesh`): mesh object
            aabb (`torch.Tensor`): [B, K, 2, 3]
            part_surface_inbbox (`torch.Tensor`): [B, K,N, 3+3+1]
        """
        if obj_surface is None:
            if mesh_path is None and (mesh is None and obj_surface_raw is None):
                raise ValueError(
                    "obj_surface or mesh_path/mesh/obj_surface_raw must be provided."
                )
        elif aabb is None or part_surface_inbbox is None:
            raise ValueError(
                "aabb and part_surface_inbbox must be provided if obj_surface is"
                " provided."
            )
        else:
            assert aabb.shape[0] == part_surface_inbbox.shape[0], "Batch size mismatch."
        center = np.zeros(3)
        scale = 1.0
        # Load config values (used throughout this function)
        # pc_size is stored in the encoder, not directly on VAE
        config_pc_size = self.vae.encoder.pc_size
        config_pc_sharpedge_size = self.vae.encoder.pc_sharpedge_size
        config_total_points = config_pc_size + config_pc_sharpedge_size
        print(f"[X-Part] DEBUG: Config values from vae.encoder - pc_size={config_pc_size}, pc_sharpedge_size={config_pc_sharpedge_size}, total={config_total_points}")
        print(f"[X-Part] DEBUG: obj_encoder.encoder.pc_size = {self.conditioner.obj_encoder.encoder.pc_size}")
        print(f"[X-Part] DEBUG: geo_encoder local_encoder.encoder.pc_size = {self.conditioner.geo_encoder.local_encoder.encoder.pc_size}")

        # 1. Load object surface and sample
        if obj_surface is None:
            if obj_surface_raw is None:
                if mesh is not None:
                    obj_surface_raw = SampleMesh(
                        mesh.vertices, mesh.faces, -1, seed=seed
                    )
                elif mesh_path is not None:
                    mesh = trimesh.load(mesh_path, force="mesh")
                    mesh, center, scale = self.normalize_mesh(mesh)
                    print(f"Normalize mesh: {center}, {scale}")
                    obj_surface_raw = SampleMesh(
                        mesh.vertices, mesh.faces, -1, seed=seed
                    )
                else:
                    raise ValueError("obj_surface or mesh_path/mesh must be provided.")
            rng = np.random.default_rng(seed=seed)
            print(f"[X-Part] DEBUG: Loading object surface with {config_total_points} total points")
            obj_surface, _ = load_surface_points(
                rng,
                obj_surface_raw["random_surface"],
                obj_surface_raw["sharp_surface"],
                pc_size=config_pc_size,
                pc_sharpedge_size=config_pc_sharpedge_size,
                return_sharpedge_label=True,
                return_normal=True,
            )
            obj_surface = obj_surface.unsqueeze(0)
            print(f"[X-Part] DEBUG: Object surface loaded, shape={obj_surface.shape} (expected: [1, {config_total_points}, 7])")
        # 2. load aabb
        if aabb is None:
            aabb = self.predict_bbox(mesh, seed=seed)
            print(f"Get bbox from bbox_predictor: {aabb.shape}")
        else:
            if isinstance(aabb, np.ndarray):
                aabb = torch.from_numpy(aabb).to(self.device)
            # normalize aabb by mesh scale and center
            aabb = aabb.float()
            aabb = (aabb - torch.from_numpy(center).float().to(self.device)) / scale

        # 3. load part surface in bbox
        if part_surface_inbbox is None:
            # Use config values instead of hardcoded (already defined above)
            print(f"[X-Part] DEBUG: Sampling bbox points with num_points={config_total_points}")
            part_surface_inbbox, valid_parts_mask = sample_bbox_points_from_trimesh(
                mesh, aabb, num_points=config_total_points, seed=seed
            )
            aabb = aabb[valid_parts_mask]
            aabb = aabb.unsqueeze(0)
            part_surface_inbbox = part_surface_inbbox.unsqueeze(0)
        return (
            obj_surface,
            aabb,
            part_surface_inbbox,
            mesh,
            center,
            scale,
        )

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=self.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=self.device) * -emb)
        emb = w.to(device=self.device, dtype=dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def _export(
        self,
        latents,
        output_type="trimesh",
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo="mc",
        enable_pbar=True,
        **kwargs,
    ):
        if not output_type == "latent":
            latents = 1.0 / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latent2mesh_2(
                # outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                octree_depth=8,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_mode=mc_algo,
                # enable_pbar=enable_pbar,
                **kwargs,
            )
        else:
            outputs = latents

        if output_type == "trimesh":
            outputs = export_to_trimesh(outputs)

        return outputs

    @torch.no_grad()
    def __call__(
        self,
        obj_surface=None,
        obj_surface_raw=None,
        mesh_path=None,
        mesh=None,
        aabb=None,
        part_surface_inbbox=None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        eta: float = 0.0,
        # guidance_scale: float = 7.5,
        guidance_scale: float = -1.0,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        seed=42,
        # marching cubes
        box_v=1.01,
        octree_resolution=512,
        mc_level=-1 / 512,
        num_chunks=400000,
        mc_algo="mc",
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ):
        """
        Args:
            obj_surface (`torch.Tensor`): [B, N, 3+3+1]
            aabb (`torch.Tensor`): [B, K, 2, 3]
            part_surface_inbbox (`torch.Tensor`): [B, K,N, 3+3+1]
        Returns:
            `trimesh.Scene` : single object composed of multiple parts
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        precomputed_sonata_features = kwargs.pop("precomputed_sonata_features", None)
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True
        )
        # 1. Check inputs and predict bbox if not provided
        device = self.device
        dtype = self.dtype
        obj_surface, aabb, part_surface_inbbox, mesh, center, scale = self.check_inputs(
            obj_surface,
            obj_surface_raw,
            mesh_path,
            mesh,
            aabb,
            part_surface_inbbox,
            seed=seed,
        )
        if self.verbose:
            # return gt mesh with bbox
            mesh_bbox = trimesh.Scene()
            if mesh is not None:
                mesh_bbox.add_geometry(mesh)
            else:
                mesh = trimesh.points.PointCloud(
                    obj_surface[0, :, :3].float().cpu().numpy()
                )
                mesh_bbox.add_geometry(mesh)
            for bbox in aabb[0]:
                box = trimesh.path.creation.box_outline()
                box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
                box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
                mesh_bbox.add_geometry(box)
        #  Convert to device and dtype
        obj_surface = obj_surface.to(device=device, dtype=dtype)
        aabb = aabb.to(device=device, dtype=dtype)
        part_surface_inbbox = part_surface_inbbox.to(device=device, dtype=dtype)
        batch_size, num_parts, N, dim = part_surface_inbbox.shape
        # TODO: support batch size > 1
        assert batch_size == 1, "Batch size > 1 is not supported yet."
        # 2. Prepare latent variables
        # TODO:allocate tokens for each parts
        num_tokens = torch.tensor(
            [self.allocate_tokens(x, self.vae.latent_shape[0]) for x in aabb],
            device=device,
        )
        latent_shape = self.vae.latent_shape
        latents = self.prepare_latents(
            num_parts, latent_shape, dtype, device, generator
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 3. condition
        cond = self.encode_cond(
            part_surface_inbbox.reshape(batch_size * num_parts, N, dim),
            obj_surface.expand(batch_size * num_parts, -1, -1),
            do_classifier_free_guidance,
            precomputed_sonata_features=precomputed_sonata_features,
        )
        # 4. guidance_cond for controling sampling
        guidance_cond = None
        if getattr(self.model, "guidance_cond_proj_dim", None) is not None:
            logger.info("Using lcm guidance scale")
            guidance_scale_tensor = torch.tensor(guidance_scale - 1, device=device).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )

        comfy.model_management.soft_empty_cache()

        # Save original aabb for reuse in loop (to avoid exponential growth with CFG)
        aabb_orig = aabb

        # 6. Denoising loop
        with synchronize_timer("Diffusion Sampling"):
            for i, t in enumerate(
                tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")
            ):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    aabb_input = torch.repeat_interleave(aabb_orig, 2, dim=0)
                else:
                    latent_model_input = latents
                    aabb_input = aabb_orig

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(
                    latent_model_input,
                    timestep,
                    cond,
                    aabb=aabb_input,
                    num_tokens=num_tokens,
                    guidance_cond=guidance_cond,
                )

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                # Free intermediate tensors to reduce memory pressure
                del noise_pred
                if do_classifier_free_guidance:
                    del latent_model_input, aabb_input

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        # latents2mesh — build list of Trimesh (not Scene) for comfy_env serialization
        parts = []
        for i, part_latent in enumerate(latents):
            try:
                part_mesh = self._export(
                    latents=part_latent.unsqueeze(0),
                    output_type=output_type,
                    box_v=box_v,
                    mc_level=mc_level,
                    num_chunks=num_chunks,
                    octree_resolution=octree_resolution,
                    mc_algo=mc_algo,
                    enable_pbar=enable_pbar,
                )[0]
                random_color = np.random.randint(0, 255, size=3)
                part_mesh.visual.face_colors = random_color
                parts.append(part_mesh)
            except Exception as e:
                logger.error(f"Failed to export part {i} with error {e}")

        # Denormalize directly on list
        print(f"Denormalize mesh: {center}, {scale}")
        for part_mesh in parts:
            part_mesh.vertices = part_mesh.vertices * scale + center

        if self.verbose:
            # Build temp Scene only for explode_mesh + bbox viz (internal file saving)
            temp_scene = trimesh.Scene()
            for p in parts:
                temp_scene.add_geometry(p)
            explode_object = explode_mesh(copy.deepcopy(temp_scene), explosion_scale=0.2)
            out_bbox = trimesh.Scene()
            out_bbox.add_geometry(temp_scene)
            for bbox in aabb[0]:
                box = trimesh.path.creation.box_outline()
                box.vertices *= (bbox[1] - bbox[0]).float().cpu().numpy()
                box.vertices += (bbox[0] + bbox[1]).float().cpu().numpy() / 2
                box.vertices = box.vertices * scale + center
                out_bbox.add_geometry(box)
            return parts, (out_bbox, mesh_bbox, explode_object)
        else:
            return parts, None

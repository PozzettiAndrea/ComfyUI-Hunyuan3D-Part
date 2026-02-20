import os
import torch
import logging
import importlib
from typing import Union
from functools import wraps

from omegaconf import OmegaConf, DictConfig, ListConfig


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger("hy3dgen.partgen")


class synchronize_timer:
    """Synchronized timer to count the inference time of `nn.Module.forward`.

    Supports both context manager and decorator usage.

    Example as context manager:
    ```python
    with synchronize_timer('name') as t:
        run()
    ```

    Example as decorator:
    ```python
    @synchronize_timer('Export to trimesh')
    def export_to_trimesh(mesh_output):
        pass
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1":
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f"{self.name} takes {self.time} ms")

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if "base_config" in config_file.keys():
        if config_file["base_config"] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file["base_config"].endswith(".yaml"):
            base_config = get_config_from_file(config_file["base_config"])
        else:
            raise ValueError(
                f"{config_file} must be `.yaml` file or it contains `base_config` key."
            )

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)

    # Handle relative imports (starting with ".")
    if module.startswith("."):
        # Import relative to current package (get package name dynamically)
        # __package__ will be like 'custom_nodes.ComfyUI-Hunyuan3D-Part.nodes'
        package = __package__ if __package__ else __name__.rsplit(".", 1)[0]
        if reload:
            module_imp = importlib.import_module(module, package=package)
            importlib.reload(module_imp)
            return getattr(module_imp, cls)
        return getattr(importlib.import_module(module, package=package), cls)

    # Standard absolute import
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
            config["from_pretrained"],
            use_safetensors=config.get("use_safetensors", False),
            variant=config.get("variant", "fp16"),
        )

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    instance = cls(**kwargs)

    return instance


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model


def smart_load_model(
    model_path,
):
    """Ensure model files are available, downloading from HuggingFace if needed.

    Downloads only the 4 required .safetensors files flat into
    ComfyUI/models/hunyuan3d-part/{model,shapevae,conditioner,p3sam}.safetensors.
    Returns the local directory path.
    """
    try:
        import folder_paths
        default_base = os.path.join(folder_paths.models_dir, "hunyuan3d-part")
    except Exception:
        default_base = "~/.cache/xpart"

    model_dir = os.path.expanduser(
        os.environ.get("HY3DGEN_MODELS", default_base)
    )

    # repo path → flat local name
    required_files = {
        "model/model.safetensors": "model.safetensors",
        "shapevae/shapevae.safetensors": "shapevae.safetensors",
        "conditioner/conditioner.safetensors": "conditioner.safetensors",
        "p3sam/p3sam.safetensors": "p3sam.safetensors",
    }

    all_present = all(
        os.path.isfile(os.path.join(model_dir, local_name))
        for local_name in required_files.values()
    )
    if all_present:
        logger.info(f"Using local models at: {model_dir}")
        return model_dir

    logger.info(f"Downloading models to {model_dir}...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            f"Models not found at {model_dir}. "
            "Install huggingface_hub to auto-download, or download manually."
        )

    os.makedirs(model_dir, exist_ok=True)
    for repo_path, local_name in required_files.items():
        local_path = os.path.join(model_dir, local_name)
        if not os.path.isfile(local_path):
            logger.info(f"Downloading {local_name}...")
            hf_hub_download(
                repo_id=model_path,
                filename=repo_path,
                local_dir=model_dir,
            )
            # Flatten: hf_hub_download creates repo_path structure, move to root
            subfolder_path = os.path.join(model_dir, repo_path)
            if os.path.isfile(subfolder_path) and subfolder_path != local_path:
                os.rename(subfolder_path, local_path)
                # Remove empty parent dir
                try:
                    os.rmdir(os.path.dirname(subfolder_path))
                except OSError:
                    pass

    # Clean up HF metadata dir
    hf_cache = os.path.join(model_dir, ".cache")
    if os.path.isdir(hf_cache):
        import shutil
        shutil.rmtree(hf_cache)

    return model_dir


def init_from_ckpt(model, ckpt, prefix="model", ignore_keys=()):
    if "state_dict" not in ckpt:
        # deepspeed ckpt
        state_dict = {}
        ckpt = ckpt["module"] if "module" in ckpt else ckpt
        for k in ckpt.keys():
            new_k = k.replace("_forward_module.", "")
            state_dict[new_k] = ckpt[k]
    else:
        state_dict = ckpt["state_dict"]
    keys = list(state_dict.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik in k:
                print("Deleting key {} from state_dict.".format(k))
                del state_dict[k]
    state_dict = {
        k.replace(prefix + ".", ""): v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")

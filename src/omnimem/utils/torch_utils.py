import json
import os
import copy
from typing import Dict, Optional, Any, Type
from dataclasses import dataclass
from collections.abc import Mapping
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from safetensors.torch import load_file, save_file

from omnimem.models.transformers.wan_model import WanModel
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.misc import is_main_process, wait_for_everyone, get_world_size

# FSDP EMA
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FSDPModule

logger = get_logger()


def reshard_model(model):
    """Reshard all FSDPModule submodules in model."""
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


@torch.no_grad()
def update_ema_fsdp(targ_model, src_model, ema_decay=0.99, reshard_func=reshard_model, foreach=True):
    """EMA update from src_model to targ_model; reshards before and after.

    Args:
        targ_model: Target model to update.
        src_model: Source model weights.
        ema_decay: EMA decay factor.
        reshard_func: Function to reshard models.
        foreach: Use _foreach API for speed.
    """
    reshard_func(targ_model)
    reshard_func(src_model)
    if foreach:
        target_params = [param for param in targ_model.parameters()]
        source_params = [param.detach() for param in src_model.parameters()]
        torch._foreach_lerp_(target_params, source_params, 1.0 - ema_decay)
    else:
        target_params = targ_model.parameters()
        source_params = src_model.parameters()

        for targ, src in zip(target_params, source_params):
            targ.copy_(torch.lerp(targ.detach(), src.detach(), 1 - ema_decay))


def create_ema_model(model, module_cls: object):
    ema_config = dict(model.config)
    ema_config["_class_name"] = module_cls.__name__
    ema = module_cls.from_config(ema_config)
    state_dict = get_fsdp_state_dict(model, master_only=False)
    info = ema.load_state_dict(state_dict, strict=False)
    return ema, info


def resume_model(path, module: torch.nn.Module):
    from safetensors.torch import load_file
    import gc
    if module is None:
        return
    if not os.path.exists(path):
        return

    if os.path.isfile(path) and path.endswith('.safetensors'):
        state_dict = load_file(path, device="cpu")
    elif os.path.isdir(path):
        safetensors_path = os.path.join(path, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(path, "diffusion_pytorch_model.bin")
        index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")

        if os.path.exists(safetensors_path):
            state_dict = load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        elif os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shard_files = sorted(set(weight_map.values()))

            model_state = module.state_dict()
            loaded_keys = set()

            for shard_file in shard_files:
                shard_dict = load_file(os.path.join(path, shard_file), device="cpu")
                for key, tensor in shard_dict.items():
                    if key in model_state:
                        model_state[key].copy_(tensor)
                        loaded_keys.add(key)
                del shard_dict
                gc.collect()
            missing = [k for k in model_state if k not in loaded_keys]
            unexpected = [k for k in weight_map if k not in model_state]
            logger.info(f"Load ckpt from {path}")
            logger.info(f"missing key: {len(missing)}, {missing}")
            logger.info(f"unexpected_keys: {len(unexpected)}, {unexpected}")
            return
    else:
        state_dict = module.__class__.from_pretrained(path).state_dict()

    incompatible_keys = module.load_state_dict(state_dict, strict=False)
    logger.info(f"Load ckpt from {path}")
    logger.info(f"missing key: {len(incompatible_keys.missing_keys)}")
    logger.info(f"missing key: {incompatible_keys.missing_keys}")
    logger.info(f"unexpected_keys: {len(incompatible_keys.unexpected_keys)}")
    logger.info(f"unexpected_keys: {incompatible_keys.unexpected_keys}")


def from_key(ckpt: Dict, key: str) -> Optional[int]:
    if key in ckpt.keys():
        return ckpt[key]
    else:
        logger.info(f"Key '{key}' not found in checkpoint '{ckpt}', ignore")


def resume_step(path, key='global_step'):
    if os.path.exists(path):
        ckpt_step = from_key(torch.load(path, map_location="cpu", weights_only=True), key)
        if ckpt_step is not None:
            logger.info(f"Resume from {key} {ckpt_step}")
        else:
            ckpt_step = 0
    else:
        ckpt_step = 0
    return ckpt_step


def save_fsdp_checkpoint(model_state, optim_state, global_step, output_dir):
    torch.save({
        'unet': None,
        'optimizer': optim_state,
        'global_step': global_step,
    }, os.path.join(output_dir, "checkpoint.pth"))


def save_dcp_checkpoint(model, optimizer, global_step, output_dir):
    opt_state = get_optimizer_state_dict(model, optimizer)
    state_dict = {
        "optimizer": opt_state,
        "global_step": global_step
    }
    dcp.save(state_dict, checkpoint_id=output_dir)
    dist.barrier()


def resume_dcp_checkpoint(model, optimizer, path):
    if not os.path.isdir(path) or not os.listdir(path):
        return 0
    opt_state = get_optimizer_state_dict(model, optimizer)
    state = {
        "optimizer": opt_state,
        "global_step": 0,
        }
    dcp.load(state, checkpoint_id=path)
    set_optimizer_state_dict(model, optimizer, optim_state_dict=state["optimizer"])
    return state['global_step']
    

def get_fsdp_state_dict(model, dtype=torch.float32, master_only=True):
    sharded_sd = model.state_dict()
    state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        if isinstance(sharded_param, DTensor):
            full_param = sharded_param.full_tensor()
        elif isinstance(sharded_param, torch.Tensor):
            full_param = sharded_param
        else:
            raise RuntimeError(f"param {param_name} {type(sharded_param)} not supported")
        param_name = param_name.replace('._orig_mod.', '.').replace("._checkpoint_wrapped_module.", ".")
        if master_only:
            if torch.distributed.get_rank() == 0:
                state_dict[param_name] = full_param.to(dtype=dtype, device="cpu")
            else:
                del full_param
        else:
            state_dict[param_name] = full_param.to(dtype=dtype, device="cpu")
    return state_dict


@dataclass
class FullyShardDistributedParallelArgs(Mapping):
    kwargs: Optional[Dict[str, Any]] = None
    root_kwargs: Optional[Dict[str, Any]] = None

    def __getitem__(self, k):
        return self.kwargs[k]

    def __iter__(self):
        return iter(self.kwargs)

    def __len__(self):
        return len(self.kwargs)


def prepare_fsdp_kwargs(sharding_strategy, dtype) -> FullyShardDistributedParallelArgs:
    if sharding_strategy == 'opt_and_grad':
        reshard_after_forward = False
        mesh = None
    elif sharding_strategy == 'hybrid_shard':
        gpus_per_node = torch.cuda.device_count()
        world_size = dist.get_world_size()
        num_nodes = world_size // gpus_per_node
        if num_nodes > 1:
            mesh = dist.init_device_mesh(
                "cuda",
                (num_nodes, gpus_per_node),
                mesh_dim_names=("replicate", "shard"),
            )
        else:
            mesh = None
        reshard_after_forward = True
    elif sharding_strategy == 'fully_shard':
        reshard_after_forward = True
        mesh = None
    else:
        raise ValueError(
            f"Unknown sharding strategy {sharding_strategy}. expected 'opt_and_grad', 'hybrid_shard' or 'fully_shard'")
    logger.info(f"Sharding strategy {sharding_strategy}")

    fsdp_kwargs = dict(
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
        )
    )
    root_fsdp_kwargs = copy.deepcopy(fsdp_kwargs)
    root_fsdp_kwargs['reshard_after_forward'] = False
    return FullyShardDistributedParallelArgs(kwargs=fsdp_kwargs, root_kwargs=root_fsdp_kwargs)


def load_and_broadcast_diffuser(model_cls, model_name_or_path, device, **kwargs):
    if model_name_or_path.endswith('.json'):
        config = json.load(open(model_name_or_path, 'r'))
        config['_class_name'] = model_cls.__name__
        return model_cls.from_config(config)
    if is_main_process():
        model = model_cls.from_pretrained(model_name_or_path, **kwargs)
    else:
        config = model_cls.load_config(model_name_or_path)
        config['_class_name'] = model_cls.__name__
        model = model_cls.from_config(config, **kwargs)

    wait_for_everyone()

    for name, param in model.state_dict().items():
        tensor = param.contiguous().to(device)
        dist.broadcast(tensor, src=0)
        param.copy_(tensor).cpu()

    return model


def build_sharded_model(
        module_cls: Type[WanModel],
        model_dir,
        fsdp_kwargs: FullyShardDistributedParallelArgs,
        module='blocks'
):
    from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
    """
    FSDP2-friendly: meta-init -> wrap -> load (materialize) -> return sharded model.
    """
    logger.info(f"Start load {model_dir}")
    with torch.device("meta"):
        cfg = module_cls.load_config(model_dir)
        model = module_cls.from_config(cfg)

    for block in getattr(model, module):
        fully_shard(block, **fsdp_kwargs)  # leaf wrap
    fully_shard(model, **fsdp_kwargs.root_kwargs)  # root wrap

    if is_main_process():
        cpu_model = module_cls.from_pretrained(model_dir, device_map=None)  # fully materialized on CPU
        full_sd = {k: v.to("cpu") for k, v in cpu_model.state_dict().items()}
        del cpu_model
    else:
        full_sd = {}

    set_model_state_dict(
        model=model,
        model_state_dict=full_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )

    leftovers = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if leftovers:
        raise RuntimeError(f"Still on meta after load: {leftovers[:5]} ...")

    logger.info(f"Finish load {model_dir}")
    return model


def save_sharded_safetensors(state_dict, save_dir, filename_prefix="diffusion_pytorch_model", max_shard_size_gb=5):
    """Save a state dict as sharded safetensors files.

    Args:
        state_dict: Model state dict.
        save_dir: Output directory.
        filename_prefix: Prefix for shard file names.
        max_shard_size_gb: Max size per shard in GB.
    """
    os.makedirs(save_dir, exist_ok=True)
    max_shard_size = int(max_shard_size_gb * (1024 ** 3))

    tensor_sizes = {k: v.numel() * v.element_size() for k, v in state_dict.items()}
    total_size = sum(tensor_sizes.values())

    shards = []
    current_shard = {}
    current_size = 0

    for key in sorted(state_dict.keys()):
        tensor_size = tensor_sizes[key]

        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = state_dict[key]
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    if len(shards) == 1:
        path = os.path.join(save_dir, f"{filename_prefix}.safetensors")
        save_file(shards[0], path)
        return path

    weight_map = {}
    total_shards = len(shards)

    for i, shard_dict in enumerate(shards, 1):
        shard_name = f"{filename_prefix}-{i:05d}-of-{total_shards:05d}.safetensors"
        shard_path = os.path.join(save_dir, shard_name)
        save_file(shard_dict, shard_path)

        for key in shard_dict:
            weight_map[key] = shard_name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(save_dir, f"{filename_prefix}.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    return index_path

import functools
import random
from typing import Tuple

import numpy as np

from diffusers.utils import is_torch_version

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reshard_model(model):
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


@torch.no_grad()
def update_ema_fsdp(targ_model, src_model, ema_decay=0.99, reshard_func=reshard_model):
    """EMA update of targ_model toward src_model; reshards before updating."""
    reshard_func(targ_model)
    reshard_func(src_model)
    target_params = targ_model.parameters()
    source_params = src_model.parameters()
    for targ, src in zip(target_params, source_params):
        targ.copy_(torch.lerp(targ.detach(), src.detach(), 1 - ema_decay))


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def wait_for_everyone(func=None):
    if func is None:
        if is_distributed():
            torch.distributed.barrier()
        return None
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if is_distributed():
            torch.distributed.barrier()
        return result
    return wrapper


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_rank():
    if is_distributed():
        return dist.get_rank()
    else:
        return 0


@torch.compiler.disable()
@torch.no_grad()
def reduce_dict(dictionary, op=dist.ReduceOp.AVG):
    if dist.is_initialized() and isinstance(dictionary, dict):
        for k, v in dictionary.items():
            if torch.is_tensor(v):
                dist.all_reduce(v, op=op)


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def is_compiled_module(module):
    """Return True if module was compiled with torch.compile()."""
    if is_torch_version("<", "2.0.0") or not hasattr(torch, "_dynamo"):
        return False
    return isinstance(module, torch._dynamo.eval_frame.OptimizedModule)


def unwrap_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    if is_compiled_module(model):
        model = model._orig_mod

    return model


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


""" Helper function """
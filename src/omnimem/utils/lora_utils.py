import torch
import torch.distributed as dist
import peft
from peft import get_peft_model_state_dict


def configure_lora_for_model(transformer, block_class_names, lora_config, is_main_process=True):
    """Apply peft LoRA to all Linear layers inside the specified block classes.

    Args:
        transformer: the base model (before FSDP wrapping)
        block_class_names: list of class name strings to target, e.g. ['CausalWanNSAttentionBlock']
        lora_config: dict with keys rank, alpha, dropout, verbose, type
        is_main_process: controls logging
    Returns:
        PeftModel wrapping transformer
    """
    target_linear_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in block_class_names:
            for full_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_name)
    target_linear_modules = list(target_linear_modules)

    if is_main_process:
        print(f"LoRA target modules ({block_class_names}): {len(target_linear_modules)} Linear layers")
        if lora_config.get('verbose', False):
            for m in sorted(target_linear_modules):
                print(f"  - {m}")

    adapter_type = lora_config.get('type', 'lora')
    if adapter_type != 'lora':
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    peft_config = peft.LoraConfig(
        r=lora_config.get('rank', 16),
        lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
        lora_dropout=lora_config.get('dropout', 0.0),
        target_modules=target_linear_modules,
    )
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print(f'peft_config: {peft_config}')
        lora_model.print_trainable_parameters()

    return lora_model


def gather_lora_state_dict(lora_model):
    """Gather LoRA-only weights from model, handles FSDP and single-GPU."""
    from torch.distributed._tensor import DTensor

    raw_sd = lora_model.state_dict()
    full_sd = {}
    for k, v in raw_sd.items():
        clean_key = k.replace('._orig_mod.', '.').replace('._checkpoint_wrapped_module.', '.')
        full_sd[clean_key] = v.full_tensor().cpu() if isinstance(v, DTensor) else v.cpu()

    return get_peft_model_state_dict(lora_model, state_dict=full_sd)


def load_lora_weights(lora_model, lora_state_dict, is_main_process=True):
    """Load LoRA adapter weights via peft (call before FSDP wrapping)."""
    if is_main_process:
        print(f"Loading LoRA weights: {len(lora_state_dict)} keys")
    peft.set_peft_model_state_dict(lora_model, lora_state_dict)
    if is_main_process:
        print("LoRA weights loaded successfully")

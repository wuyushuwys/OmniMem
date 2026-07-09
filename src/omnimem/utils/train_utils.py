from typing import Union

import torch

from omnimem.models.autoencoders import AutoencoderKLWan
from omnimem.data.ode_dataset import OdeDataloader
from omnimem.data.text_dataset import TextDataset
from omnimem.utils.misc import get_world_size
from omnimem.utils.logging_tool import get_logger

logger = get_logger()


def prepare_dataloader(
        dataset_type,
        train_data,
        num_workers=8,
        pin_memory=True,
        num_train_examples=50499,
        conditioning_dropout_prob=0,
):
    if dataset_type == "ode_dataset":
        kwargs = dict()
        if train_data.get('item_config'):
            kwargs['item_config'] = train_data.get('item_config')
        train_dataloader = OdeDataloader(
            train_shards_path_or_url=train_data.train_shards_path_or_url,
            num_train_examples=num_train_examples,
            per_gpu_batch_size=train_data.batch_size,
            global_batch_size=train_data.batch_size * get_world_size(),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=8,
            **kwargs,
        ).train_dataloader
    elif dataset_type == "text_dataset":
        dataset = TextDataset(prompt_path=train_data.prompt_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        ) if get_world_size() > 1 else None
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_data.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} is not supported.")

    return train_dataloader


def pred_x0(
        model_output,
        timestep,
        sample,
        noise_scheduler=None,
):
    original_dtype = model_output.dtype
    model_output, timestep, sample = map(lambda x: x.double().to(model_output.device), [model_output, timestep, sample])
    if noise_scheduler is not None:
        dt = - timestep / noise_scheduler.config.num_train_timesteps
    else:
        dt = - timestep / 1000

    x_0 = sample + model_output * dt.reshape((model_output.shape[0], -1) + (1,) * (model_output.ndim - 2))
    return x_0.to(original_dtype)


@torch.no_grad()
def vae_encode(vae: Union[AutoencoderKLWan], pixel_values):
    assert pixel_values.ndim == 5, pixel_values.shape
    assert pixel_values.shape[1] == 3, pixel_values.shape
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        posterior = vae.encode(pixel_values).latent_dist
    latents = posterior.sample()

    if isinstance(vae, AutoencoderKLWan):
        latents_mean = torch.tensor(vae.config.latents_mean, device=vae.device, dtype=vae.dtype).reshape(1, -1, 1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=vae.device, dtype=vae.dtype).reshape(1, -1, 1, 1, 1)
        latents = (latents - latents_mean) / latents_std
    else:
        latents = latents / vae.scaling_factor
    return latents

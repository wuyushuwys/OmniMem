import gc
import os
import copy
import math
from typing import Union, Type
from itertools import islice, cycle

import wandb
import torch
import diffusers

import transformers
from transformers import T5EncoderModel, T5Tokenizer

from omnimem.models.transformers.wan_model import WanModel
from omnimem.models.transformers.causal_wan_model import CausalWanModel
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
from omnimem.pipelines import CausalWanT2VPipeline, WanTI2VPipeline
from omnimem.utils.io import save_videos_grid_pil
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.aws_handler import S3
from omnimem.utils.misc import get_world_size, get_rank, wait_for_everyone, is_main_process

diffusers.utils.logging.disable_progress_bar()
transformers.utils.logging.disable_progress_bar()


@torch.no_grad()
def evaluation_wan(
        transformer: Union[CausalWanModel, WanModel, CausalWanNSAModel],
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae,
        noise_scheduler,
        output_dir,
        global_step,
        global_seed,
        validation_data,
        commit=False,
        n_rows=4,
        tag='',
        original_cfg=None,
        original_step=None,
        pipeline_cls: Union[Type[WanTI2VPipeline], Type[CausalWanT2VPipeline]] = WanTI2VPipeline,
        frame_per_block=None,
        timestep_shift: int = 8,
        window_size=None,
        sink_size=None,
        enable_block_level_cache=False,
        enable_chunk_per_head_cache=False,
        enable_kv_evict=True,
        lru_max_size=10,
        device=None,
        s3_bucket=None,
        s3_dir=None,
        upload=True,
        _kv_cache=None,
):
    output_folder = f"{output_dir}/{tag.replace('/', '-')}samples"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    logger = get_logger()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device)
    generator.manual_seed(global_seed)

    validation_data = copy.deepcopy(validation_data)
    if original_cfg is not None:
        validation_data.update({'guidance_scale': original_cfg})

    if original_step is not None:
        validation_data.update({'num_inference_steps': original_step})

    transformer.eval()
    vae.eval()
    submodel_dict = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": noise_scheduler,
        "vae": vae,
    }
    if frame_per_block is not None:
        submodel_dict["frame_per_block"] = frame_per_block
    logger.info(f"Using {pipeline_cls.__name__}")
    pipeline = pipeline_cls(**submodel_dict)
    pipeline.transformer.eval()
    pipeline.set_progress_bar_config(disable=not is_main_process())

    fps = validation_data.get('fps', 24)
    rank = get_rank()
    world_size = get_world_size()

    num_frames = validation_data.get('num_frames')
    extension = 'mp4'
    prompts = [(i, p) for i, p in enumerate(validation_data.prompts)]
    num_prompts = len(prompts)
    extend_factor = math.ceil(len(prompts) / (world_size * n_rows))
    total_prompts = extend_factor * world_size * n_rows
    extended_prompts = list(islice(cycle(prompts), total_prompts))
    logger.info(f"Extend {len(prompts) = } to {len(extended_prompts) = }")
    height = validation_data.get('height', 480)
    width = validation_data.get('width', 640)
    additional_kwargs = {}

    if window_size:
        additional_kwargs['window_size'] = window_size
    if sink_size:
        additional_kwargs['sink_size'] = sink_size
    if enable_block_level_cache:
        additional_kwargs['enable_block_level_cache'] = True
    if enable_chunk_per_head_cache or isinstance(transformer, CausalWanNSAModel):
        additional_kwargs['enable_chunk_per_head_cache'] = True
    if not enable_kv_evict:
        additional_kwargs['enable_kv_evict'] = False
    if lru_max_size and lru_max_size > 0:
        additional_kwargs['lru_max_size'] = lru_max_size
    if _kv_cache is not None:
        additional_kwargs['_kv_cache'] = _kv_cache

    collect_gates = isinstance(transformer, CausalWanNSAModel) and hasattr(transformer, 'enable_gate_collection')
    all_gate_stats = {}

    num_inference_steps = validation_data.get('num_inference_steps')
    guide_scale = validation_data.get('guidance_scale', 6)

    for i in range(0, len(extended_prompts), world_size * n_rows):
        samples = []
        index = i + rank * n_rows
        end = index + n_rows if index + n_rows <= len(extended_prompts) else len(extended_prompts)

        logger.info(f"generate sample {index} -> {end - 1}")
        for idx, prompt in extended_prompts[index:end]:
            if collect_gates:
                transformer.enable_gate_collection()

            with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'):
                images = pipeline.generate(
                    prompt,
                    n_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                    size=(width, height),
                    frame_num=num_frames,
                    shift=timestep_shift,
                    sample_solver=validation_data.get("sample_solver", "unipc"),
                    sampling_steps=num_inference_steps,
                    guide_scale=guide_scale,
                    generator=generator,
                    offload_model=False,
                    **additional_kwargs
                ).images
                samples.append(images[0])

            if collect_gates:
                transformer.disable_gate_collection()
                gate_stats = transformer.get_gate_stats()
                if gate_stats:
                    prompt_key = prompt[:50].replace(' ', '_').replace('/', '_')
                    all_gate_stats[prompt_key] = gate_stats
                transformer.clear_gate_log()

        logger.info(f"finish sample {index} -> {end - 1}")
        if len(samples) > 0 and end <= num_prompts:
            output_video_path = f"{output_folder}/sample-{idx // n_rows:02d}.{extension}"
            save_videos_grid_pil(samples, output_video_path, fps=fps, n_rows=4, generic=False, release=True)

    wait_for_everyone()

    if collect_gates and world_size > 1:
        import torch.distributed as dist
        import pickle
        local_data = pickle.dumps(all_gate_stats)
        local_tensor = torch.ByteTensor(list(local_data)).cuda()
        local_size = torch.tensor([local_tensor.numel()], dtype=torch.long, device='cuda')
        all_sizes = [torch.zeros(1, dtype=torch.long, device='cuda') for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        padded = torch.zeros(max_size, dtype=torch.uint8, device='cuda')
        padded[:local_tensor.numel()] = local_tensor
        all_padded = [torch.zeros(max_size, dtype=torch.uint8, device='cuda') for _ in range(world_size)]
        dist.all_gather(all_padded, padded)
        if is_main_process():
            for r in range(world_size):
                if r == rank:
                    continue
                sz = all_sizes[r].item()
                remote_stats = pickle.loads(bytes(all_padded[r][:sz].cpu().tolist()))
                all_gate_stats.update(remote_stats)

    if is_main_process():
        if upload:
            with S3(bucket=s3_bucket, subdir=s3_dir) as s3:
                s3.upload_folder(output_folder, global_step)

        if wandb.run is not None:
            logger.info('Upload sample to wandb')
            wandb_video_logs = {}
            for i in range(0, len(validation_data.prompts), n_rows):
                output_video_path = f"{output_folder}/sample-{i // n_rows:02d}.{extension}"
                if extension == 'mp4':
                    wandb_video_logs[f"{tag}video_{i // n_rows:02d}"] = wandb.Video(output_video_path, format="mp4")
                elif extension == 'jpg':
                    wandb_video_logs[f"{tag}video_{i // n_rows:02d}"] = wandb.Image(output_video_path)
                else:
                    raise ValueError(f"Unknown video format: {extension}")
            wandb.log(
                wandb_video_logs,
                commit=False,
                step=global_step
            )

    wait_for_everyone()
    logger.info(f"Saved samples ...")
    """ >>> release memory >>> """
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    """ <<< release memory <<< """

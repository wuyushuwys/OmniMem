import argparse
import os
import uuid
from webdataset.writer import ShardWriter
from tqdm import tqdm
import datetime

import diffusers
from diffusers.utils import logging
import transformers

import torch
import torch.distributed as dist

from omnimem.models.autoencoders import get_autoencoder
from omnimem.utils.torch_utils import load_and_broadcast_diffuser
from omnimem.utils.aws_handler import AsyncS3UploadAndDeleteHook
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.misc import get_model_numel, format_numel_str
from omnimem.models.transformers.wan_model import WanModel
from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from omnimem.models.text_encoders.t5 import T5EncoderModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if __name__ == "__main__":

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    diffusers.utils.logging.disable_progress_bar()
    transformers.utils.logging.disable_progress_bar()

    OUTPUT_PATH = '/tmp'

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_folder", type=str, required=True)
    parser.add_argument("--prompt_file", nargs='+', type=str, required=True)
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--s3_bucket", type=str, required=True, help="s3 bucket name")
    parser.add_argument("--s3_dir", type=str, default="ode", help="s3 directory name")
    parser.add_argument("--text_encoder_path", type=str,
                        default="wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the text encoder model")
    parser.add_argument("--tokenizer_path", type=str, default="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl",
                        help="Path to the tokenizer")
    parser.add_argument('--height', type=int, default=480, help='Height of the video')
    parser.add_argument('--width', type=int, default=832, help='Width of the video')
    parser.add_argument('--num_frames', type=int, default=81, help='Number of frames of the video')
    parser.add_argument('--num_inference_steps', type=int, default=48, help='Number of inference steps')
    parser.add_argument('--shift', default=5, type=float, help='timestep shift')
    parser.add_argument('--target_step', nargs='+', type=int, default=None, help='Target timestep')
    parser.add_argument("--guidance_scale", type=int, default=5, help="Classifier Free Guidance Scale")
    parser.add_argument("--shard_index", type=int, default=None,
                        help="Shard index for manually sharding the prompt list")
    parser.add_argument("--num_shard", type=int, default=None,
                        help="Number of shard for manually sharding the prompt list")
    args = parser.parse_args()

    logger = get_logger(__name__, file_path='ode_generation')
    prompt_list = []
    for prompt_file in args.prompt_file:
        prompt_file = os.path.join(args.prompt_folder, prompt_file)
        logger.info(f"Reading {prompt_file}")
        with open(prompt_file, 'r') as f:
            video_prompts = [line.rstrip() for line in f.readlines()]

        prompt_list.extend(video_prompts)

    prompt_list = sorted(list(set(prompt_list)))

    logger.info(f"Load {len(prompt_list)} non-repeated prompts...")
    if args.num_shard is not None and args.shard_index is not None:
        assert args.num_shard > 0, "num_shard must be greater than 0"
        assert args.shard_index >= 0, "shard_index must be greater than or equal to 0"
        logger.info(f"Manually shard prompt list num_shard: {args.num_shard}, shard_index: {args.shard_index}.")
        prompt_list = prompt_list[args.shard_index::args.num_shard]
        logger.info(f"Assigned {len(prompt_list)} prompts to current shard.")

    if int(os.getenv("WORLD_SIZE", 1)) > 0:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(days=1))
    else:
        local_rank = 0

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    vae = get_autoencoder(args.vae_path)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
        checkpoint_path=args.text_encoder_path,
        tokenizer_path=args.tokenizer_path,
        shard_fn=None
    )
    tokenizer = text_encoder.tokenizer
    text_encoder.model.requires_grad_(False)
    text_encoder.model.eval()

    transformer = load_and_broadcast_diffuser(WanModel, args.pretrained, device=local_rank)
    model_numel, model_numel_trainable = get_model_numel(transformer)
    logger.info(
        f"[Diffusion] Trainable model params: {format_numel_str(model_numel_trainable)}, "
        f"Total model params: {format_numel_str(model_numel)}"
    )

    vae.requires_grad_(False)
    vae.eval()
    text_encoder.model.to(device=local_rank)
    transformer.requires_grad_(False)
    transformer.eval()

    model_name = args.model_name

    text_encoder.model.compile(mode="max-autotune-no-cudagraphs")
    transformer.compile(mode="max-autotune-no-cudagraphs")
    transformer.to(device=local_rank)

    height = args.height
    width = args.width
    num_frames = args.num_frames
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    shift = args.shift
    device = f"cuda:{local_rank}"

    target_step = args.target_step
    if target_step is None:
        target_step = list(range(args.num_inference_steps))
    # add clean image as well
    target_step.append(-1)
    logger.info(
        f"{model_name}_{args.name}_{num_frames}x{height}x{width}_{num_inference_steps}step_{guidance_scale}cfg_{shift}shift {target_step=}")

    s3_target_prefix = os.path.join(
        args.s3_dir,
        f"{model_name}_{args.name}_{num_frames}x{height}x{width}_{num_inference_steps}step_{guidance_scale}cfg_{shift}shift_{len(target_step) - 1}t"
    )
    s3_hook = AsyncS3UploadAndDeleteHook(bucket_name=args.s3_bucket, s3_prefix=s3_target_prefix)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'
    context_null, attn_null = text_encoder(sample_neg_prompt, device)

    size = (width, height)
    vae_stride = (vae.temporal_downscale_factor, vae.spatial_downscale_factor, vae.spatial_downscale_factor)
    target_shape = (vae.z_dim, (num_frames - 1) // vae_stride[0] + 1, size[1] // vae_stride[1],
                    size[0] // vae_stride[2])
    logger.info(f"Generate {target_shape} shape")
    rank_prompt_list = prompt_list[rank::world_size]
    logger.info(f"Generating {len(rank_prompt_list)} ODE in {rank =}...")

    if args.num_shard is not None and args.shard_index is not None:
        shard_name = f'shard_{args.num_shard:04d}_{args.shard_index:04d}_{rank:04d}_%06d.tar'
    else:
        shard_name = f'shard_{rank:04d}_%06d.tar'

    with ShardWriter(
            pattern=os.path.join(OUTPUT_PATH, shard_name),
            maxcount=1000,
            maxsize=2 * 1 << 30,
            post=s3_hook,
    ) as sink:
        for idx, prompt in tqdm(enumerate(rank_prompt_list, start=0), disable=rank != 0, total=len(rank_prompt_list)):
            with torch.no_grad():
                context, attn_cond = text_encoder(prompt, device)

            arg_c = {'context': context, 'seq_len': None}
            arg_null = {'context': context_null, 'seq_len': None}
            noise = torch.randn(
                1,
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device)
            sample_scheduler = FlowUniPCMultistepScheduler(shift=args.shift)
            sample_scheduler.set_timesteps(args.num_inference_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
            latents = noise
            ode = []
            with torch.no_grad():
                for _, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=True):
                    latent_model_input = latents
                    ode.append(latent_model_input)
                    timestep = [t]
                    timestep = torch.stack(timestep)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        noise_pred_cond = transformer(
                            latent_model_input,
                            t=timestep,
                            **arg_c,
                        )
                        if guidance_scale > 1:
                            noise_pred_uncond = transformer(
                                latent_model_input,
                                t=timestep,
                                **arg_null,
                            )

                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False
                    )[0]
                    latents = temp_x0.squeeze(0)
            ode.append(latents)
            timesteps = torch.cat([timesteps, torch.zeros_like(timesteps[:1])], dim=0)
            ode = torch.stack(ode, dim=1).squeeze(0)
            ode = ode[target_step]
            timesteps = timesteps[target_step]
            t5_embed = context[0]
            results = {
                "__key__": str(uuid.uuid4()),
                "ode.pyd": ode.clone().contiguous().to(torch.float16).cpu().numpy(),
                "timesteps.pyd": timesteps.clone().contiguous().to(torch.float16).cpu().numpy(),
                "t5_embed.pyd": t5_embed.clone().contiguous().to(torch.float16).cpu().numpy(),
            }

            sink.write(results)

    s3_hook.wait_and_close()
    logger.warning(f"Finished rank {rank:04d}.")
    dist.barrier()
    dist.destroy_process_group()
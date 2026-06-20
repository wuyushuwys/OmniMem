import os
import warnings
import datetime
import argparse
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import torch
import torch.distributed as dist


def setup_distributed():
    """Init torch.distributed under torchrun; pin the device to LOCAL_RANK first."""
    world_size = int(os.getenv("WORLD_SIZE", 1))
    if world_size > 1:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(minutes=60),
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# Distributed setup runs first so NUMA binding and CUDA init see the right GPU.
rank, world_size, local_rank = setup_distributed()

from omnimem.utils.numa_bind import bind_to_gpu_numa

result = bind_to_gpu_numa()
if not result and result.reason not in ("single-NUMA machine (no perf impact)",):
    if rank == 0:
        print(f"Warning: NUMA binding failed: {result.reason}")

import transformers
import diffusers
from omnimem.models.text_encoders.t5 import T5EncoderModel
from omnimem.models.transformers.causal_wan_model import CausalWanModel
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
from omnimem.models.autoencoders import get_autoencoder
from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.utils.io import save_videos_grid_pil
from omnimem.utils.aws_handler import S3
from omnimem.utils.logging_tool import get_logger
from omnimem.utils.misc import format_numel_str, get_model_numel
torch._dynamo.config.allow_unspec_int_on_nn_module = True

diffusers.utils.logging.disable_progress_bar()
transformers.utils.logging.disable_progress_bar()


parser = argparse.ArgumentParser(description='Text-to-Video Inference (multi-rank)')
# Model / data paths
parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
parser.add_argument('--vae_path', type=str, default='wan_models/vae-2.1/vae')
parser.add_argument('--t5_path', type=str,
                    default="wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
parser.add_argument('--tokenizer_path', type=str,
                    default="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl")
# Output / S3
parser.add_argument('--save_path', type=str, required=True, help='S3 prefix for outputs')
parser.add_argument('--prompt_path', type=str, default='configs/prompts/demo.txt')
parser.add_argument('--name', type=str,
                    default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    help='Name of the run folder under save_path')
# Video shape
parser.add_argument('--height', type=int, default=288, help='Height of the video')
parser.add_argument('--width', type=int, default=512, help='Width of the video')
parser.add_argument('--nframes', type=int, default=240, help='Number of frames of the video')
parser.add_argument('--fps', type=int, default=16, help='export video fps')
# Sampling
parser.add_argument('--cfg', type=float, default=7, help='classifier free guidance scale')
parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--shift', default=5, type=float, help='timestep shift')
parser.add_argument('--num_samples_per_prompt', type=int, default=1,
                    help='How many videos to sample per prompt')
# Causal / NSA runtime
parser.add_argument("--frame_per_block", default=3, type=int, help="frame per block")
parser.add_argument("--sink_size", default=None, type=int, help="attention sink block size")
parser.add_argument("--window_size", default=None, type=int, help="attention window block size")
parser.add_argument("--nsa", action='store_true', help="use NSA model")
parser.add_argument("--enable_kv_evict", action='store_true',
                    help="enable per-step KV cache eviction to CPU "
                         "(only window+sink+LRU stay on GPU)")
parser.add_argument("--lru_max_size", type=int, default=10,
                    help="LRU capacity for selection-attention chunk reuse")
parser.add_argument("--kv_evict_min_gpu_chunks", type=int, default=None,
                    help="floor on chunks kept on GPU during eviction "
                         "(None = window+sink only)")
args = parser.parse_args()

# Broadcast the run name so all ranks agree (the default is a per-rank timestamp).
if world_size > 1:
    name_list = [args.name]
    dist.broadcast_object_list(name_list, src=0)
    args.name = name_list[0]

logger = get_logger(__name__, master_only=False)
logger.info(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")
logger.info(args)

# S3 handler; subdir includes the run name so runs don't overwrite each other.
s3 = S3(
    bucket=os.getenv('S3_BUCKET', None),
    subdir=f"{args.save_path}/{args.name}",
    ignored_path='/tmp',
)

# Load prompts.
with open(args.prompt_path, 'r') as f:
    prompt_list = [line.strip() for line in f.readlines()]

# Build the pipeline.
noise_scheduler = FlowUniPCMultistepScheduler(shift=args.shift)
vae = get_autoencoder(args.vae_path)
text_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
    checkpoint_path=args.t5_path,
    tokenizer_path=args.tokenizer_path,
    shard_fn=None,
)
tokenizer = text_encoder.tokenizer
text_encoder.model.requires_grad_(False)
text_encoder.model.eval()


def get_model_name(pretrained_path: str) -> str:
    parts = pretrained_path.rstrip('/').split('/')
    skip = {'generator', 'ema', 'model', 'checkpoints', 'checkpoint'}
    for p in reversed(parts):
        if p and p not in skip:
            return p
    return parts[-1]  # fallback


model_name = get_model_name(args.pretrained)

model_cls = CausalWanNSAModel if args.nsa else CausalWanModel
transformer = model_cls.from_pretrained(args.pretrained)
model_numel, model_numel_trainable = get_model_numel(transformer)
if rank == 0:
    logger.info(
        f"[Diffusion] Trainable model params: {format_numel_str(model_numel_trainable)}, "
        f"Total model params: {format_numel_str(model_numel)}"
    )

vae.requires_grad_(False)
vae.eval()
transformer.requires_grad_(False)
transformer.eval()
transformer.compile(mode='max-autotune-no-cudagraphs')

pipeline = CausalWanT2VPipeline(
    transformer=transformer,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    scheduler=noise_scheduler,
    frame_per_block=args.frame_per_block,
)
pipeline.to(dtype=torch.bfloat16, device='cuda')

# Dict form caps only the full-res KV window (SWA); the cmp cache stays unlimited.
window_size_dict = {'kv': args.window_size, 'kv_cmp': None}
sink_size_dict = {'kv': args.sink_size, 'kv_cmp': None}

pipe_kwargs = dict(
    n_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    size=(args.width, args.height),
    frame_num=args.nframes,
    shift=args.shift,
    sample_solver='unipc',
    sampling_steps=args.num_inference_steps,
    guide_scale=args.cfg,
    offload_model=False,
    sink_size=sink_size_dict,
    window_size=window_size_dict,
    enable_block_level_cache=False,
    enable_chunk_per_head_cache=args.nsa,
    enable_kv_evict=args.enable_kv_evict,
    lru_max_size=args.lru_max_size,
    kv_evict_min_gpu_chunks=args.kv_evict_min_gpu_chunks,
)
if rank == 0:
    logger.info(f"pipe_kwargs: {pipe_kwargs}")

# Strided slice: rank r handles prompts r, r+N, r+2N, ...
my_prompts = prompt_list[rank::world_size]
generator = torch.Generator(device='cuda')

with torch.no_grad():
    for idx, prompt in enumerate(my_prompts, start=1):
        for index in range(args.num_samples_per_prompt):
            generator.manual_seed(args.seed + index)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sample = pipeline.generate(
                    input_prompt=prompt,
                    generator=generator,
                    **pipe_kwargs,
                ).images

            # Sanitize the prompt into a safe /tmp filename.
            safe_name = prompt[:200].replace('/', '_')
            cur_save_path = f'/tmp/{safe_name}-{index}.mp4'
            save_videos_grid_pil(sample, cur_save_path, fps=args.fps, n_rows=1)
            s3.upload(fpath=cur_save_path)
            os.remove(cur_save_path)

        logger.info(f'[{idx}/{len(my_prompts)}] - Generate Samples')

if world_size > 1:
    dist.barrier()

logger.info("Finished")

if world_size > 1:
    dist.destroy_process_group()
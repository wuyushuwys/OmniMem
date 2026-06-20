from omnimem.utils.numa_bind import bind_to_gpu_numa

result = bind_to_gpu_numa()
if not result and result.reason not in ("single-NUMA machine (no perf impact)",):
    print(f"Warning: NUMA binding failed: {result.reason}")
import warnings

warnings.filterwarnings("ignore")

import datetime
import os
import argparse
import logging

logging.basicConfig(level=logging.ERROR)

import torch
import transformers
import diffusers
from omnimem.models.text_encoders.t5 import T5EncoderModel
from omnimem.models.transformers.causal_wan_model import CausalWanModel
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel
from omnimem.models.autoencoders import get_autoencoder
from omnimem.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.utils.io import save_videos_grid_pil
from omnimem.utils.misc import format_numel_str, get_model_numel
torch._dynamo.config.allow_unspec_int_on_nn_module = True

diffusers.utils.logging.disable_progress_bar()
transformers.utils.logging.disable_progress_bar()


@torch.no_grad()
def generate(pipe: CausalWanT2VPipeline, prompts_path, pipe_kwargs, output_folder, tag='', n_rows=4, seed=0, fps=8):
    with open(prompts_path, 'r') as f:
        video_prompts = [line.rstrip() for line in f.readlines()]

    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)

    for i in range(0, len(video_prompts), n_rows):
        samples = []
        for p in video_prompts[i:i + n_rows]:
            sample = pipe.generate(
                input_prompt=p,
                generator=generator,
                **pipe_kwargs).images
            samples.extend(sample)
        save_videos_grid_pil(samples, f"{output_folder}/{tag}_{i // n_rows:02d}.mp4", n_rows=n_rows, fps=fps)


parser = argparse.ArgumentParser(description='Text-to-Video Inference')
parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
parser.add_argument('--vae_path', type=str, default="outputs/checkpoints/mobile_ltx_vae_v2", )
parser.add_argument('--height', type=int, default=288, help='Height of the video')
parser.add_argument('--width', type=int, default=512, help='Width of the video')
parser.add_argument('--nframes', type=int, default=240, help='Number of frames of the video')
parser.add_argument('--nrows', type=int, default=4, help='Number of rows to save video/image')
parser.add_argument('--cfg', type=float, default=7, help='classifier free guidance scale')
parser.add_argument('--num_inference_steps', type=int, default=25, help='Number of inference steps')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    help='Name of the folder')
parser.add_argument('--fps', type=int, default=16, help='export video fps')
parser.add_argument('--shift', default=5, type=float, help='timestep shift')
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
parser.add_argument("--t5_path", type=str,
                    default="wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth")
parser.add_argument("--tokenizer_path", type=str,
                    default="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl")

args = parser.parse_args()

vae_path = args.vae_path

noise_scheduler = FlowUniPCMultistepScheduler(shift=args.shift)
vae = get_autoencoder(vae_path)
text_encoder = T5EncoderModel(
    text_len=512,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
    checkpoint_path=args.t5_path,
    tokenizer_path=args.tokenizer_path,
    shard_fn=None
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
print(
    f"[Diffusion] Trainable model params: {format_numel_str(model_numel_trainable)}, Total model params: {format_numel_str(model_numel)}")

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

random_seed = args.seed

# Dict form caps only the full-res KV window (SWA); the cmp cache stays unlimited.
sink_size_dict = {'kv': args.sink_size, 'kv_cmp': None}
window_size_dict = {'kv': args.window_size, 'kv_cmp': None}

pipe_kwargs = dict(
    n_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    size=(args.width, args.height),
    frame_num=args.nframes,
    shift=args.shift,
    sample_solver='unipc',
    sampling_steps=args.num_inference_steps,
    guide_scale=args.cfg,
    offload_model=False,
    sink_size=sink_size_dict if args.nsa else args.sink_size,
    window_size=window_size_dict if args.nsa else args.window_size,
    enable_block_level_cache=False,
    enable_chunk_per_head_cache=args.nsa,
    enable_kv_evict=args.enable_kv_evict,
    lru_max_size=args.lru_max_size,
    kv_evict_min_gpu_chunks=args.kv_evict_min_gpu_chunks,
)
print(pipe_kwargs)

folder = (f"demo/{args.name}_"
          f"cfg{pipe_kwargs['guide_scale']}_"
          f"step{pipe_kwargs['sampling_steps']}_"
          f"{pipe_kwargs['frame_num']}x{args.height}x{args.width}_{model_name}")

os.makedirs(folder, exist_ok=True)
n_rows = args.nrows

generate(pipe=pipeline,
         prompts_path='configs/prompts/demo.txt',
         pipe_kwargs=pipe_kwargs,
         output_folder=folder,
         tag='demo',
         n_rows=n_rows,
         seed=random_seed,
         fps=args.fps)

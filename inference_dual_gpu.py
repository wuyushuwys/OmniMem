"""Single-video low-latency streaming inference on two GPUs (T2V, CFG == 1).

The DiT runs on --dit_device; the VAE decodes on --decode_device while the next chunk is denoised.

Example:

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python inference_dual_gpu.py \\
      --pretrained /path/to/checkpoint --vae_path /path/to/vae-2.1/vae \\
      --t5_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \\
      --tokenizer_path /path/to/google/umt5-xxl \\
      --prompt 'A cat sitting at a grand piano.' --nframes 249 --seed 42 --output demo/example.mp4
"""
import sys

from omnimem.utils.streaming_cli import build_parser, run


def main(argv=None) -> int:
    parser = build_parser(__doc__, tier=True)
    parser.add_argument("--dit_device", default="cuda:0",
                        help="GPU of the DiT and its KV history (default cuda:0)")
    parser.add_argument("--decode_device", default="cuda:1",
                        help="GPU of the VAE decode, which feeds the video writer (default cuda:1)")
    args = parser.parse_args(argv)
    return run(args, args.dit_device, args.decode_device)


if __name__ == "__main__":
    sys.exit(main())

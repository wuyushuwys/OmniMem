"""Single-video low-latency streaming inference on one GPU (T2V, CFG == 1).

The DiT and the VAE share --device; each chunk is decoded right after it is denoised.

Example:

    PYTHONPATH=src python inference_single_gpu.py \\
      --pretrained /path/to/checkpoint --vae_path /path/to/vae-2.1/vae \\
      --t5_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \\
      --tokenizer_path /path/to/google/umt5-xxl \\
      --prompt 'A cat sitting at a grand piano.' --nframes 249 --seed 42 --output demo/example.mp4
"""
import sys

from omnimem.utils.streaming_cli import build_parser, run


def main(argv=None) -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--device", default="cuda:0",
                        help="GPU of the DiT, its KV history and the VAE (default cuda:0)")
    args = parser.parse_args(argv)
    return run(args, args.device)


if __name__ == "__main__":
    sys.exit(main())

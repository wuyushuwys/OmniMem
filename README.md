<p align="center">
  <img src="docs/assets/omnimem-256.png" alt="OmniMem" width="140" />
</p>

# OmniMem: Scalable and Adaptive Memory Retrieval for Long Video Generation

> **OmniMem: Scalable and Adaptive Memory Retrieval for Long Video Generation**\
> [Lin Zhao*](https://lin-zhao-resolve.github.io/), [Yushu Wu*](http://wuyushuwys.github.io/), [Yifan Gong](https://yifanfanfanfan.github.io/), [Yanzhi Wang](https://www.yanzhiwang.com/), [Pu Zhao](https://puzhao.info/)\
> Northeastern University, Adobe Research

[[ArXiv](https://arxiv.org/abs/2605.30519)] [[Page](https://wuyushuwys.github.io/OmniMem/)]

OmniMem builds long-video generation on [Wan2.1](https://github.com/Wan-Video/Wan2.1) with adaptive memory retrieval.

## Installation

```bash
git clone git@github.com:wuyushuwys/OmniMem.git
cd OmniMem
git lfs pull   # fetch the prompt list

conda create -n omnimem python=3.10 -y
conda activate omnimem
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=16 pip install flash-attn --no-build-isolation
```

## Download weights

```bash
# base Wan2.1-T2V-1.3B + VAE
hf download Wan-AI/Wan2.1-T2V-1.3B           --exclude "Wan2.1_VAE.pth" --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*"          --local-dir wan_models/vae-2.1
```

## Credentials & storage (env vars)

Training and multi-GPU inference read credentials/storage from environment variables. Keep them in a gitignored `run.local.sh` and `source` it (see [`run.sh`](run.sh)):

```bash
export WANDB_KEY=<your wandb api key>     # Weights & Biases logging
export WANDB_PROJ=<your wandb project>
# export WANDB_HOST=<your wandb host>     # optional, defaults to api.wandb.ai

# optional: back up samples/checkpoints and inference_mp.py outputs to S3
# export S3_BUCKET=<your bucket>
# export S3_DIR=<your prefix>
```

## Training

Every stage launches through `train.py`:

```bash
num_gpus=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc_per_node=$num_gpus train.py --config <config> --task <task> --name <exp_name>
```

Two recipes are supported, both ending in a self-forcing stage — pick one:

**Flow 1 · ODE -> Self-Forcing**

| Step | `--task` | Config |
|------|----------|--------|
| ODE distillation | `nsa-ode` | `configs/ode/training_wan_nsa_ode.yaml` |
| Self-forcing | `nsa-self-forcing` | `configs/self_forcing/training_wan_nsa_self_forcing_4step_flow.yaml` |

**Flow 2 · Teacher-Forcing -> Consistency Distillation -> Self-Forcing**

| Step | `--task` | Config |
|------|----------|--------|
| Teacher-forcing (causal) | `nsa-causal` | `configs/causal/train_wan_tf_chunk_nsa_15x2_top16_qg15.yaml` |
| Consistency distillation | `nsa-cd` | `configs/causal/train_cd_tf_chunk_nsa_15x2_top16_qg15.yaml` |
| Self-forcing | `nsa-self-forcing` | `configs/self_forcing/training_wan_nsa_self_forcing_4step_flow.yaml` |

Optionally finetune for long video with `nsa-self-forcing-streaming` (`configs/self_forcing/training_wan_nsa_self_forcing_4step_flow_streaming.yaml`).

The ODE / teacher-forcing / consistency-distillation stages train on pre-generated ODE trajectories. Download the released dataset:

```bash
hf download omnimem/Wan2.1-T2V-1.3B_vidprom_81x480x832_40step_5cfg_5.0shift_4t \
    --repo-type dataset \
    --local-dir data/Wan2.1-T2V-1.3B_vidprom_81x480x832_40step_5cfg_5.0shift_4t
```

The configs read it from that `data/...` path. Or generate your own with `bash scripts/generate_ode_wan.sh`.

See [`run.sh`](run.sh) for a ready-to-edit end-to-end example.

## Inference

```bash
bash scripts/genv_wan_causal.sh
```

`inference_wan_causal.py` runs single-GPU and writes a demo grid under `demo/`. `inference_mp.py` (launched with `torchrun`) batches a prompt file across ranks for long-video generation and uploads results to S3. `--lru_max_size` sets the KV-cache memory budget.

## Citation

If you find our paper useful or relevant to your project and research, please kindly cite our paper:

```
@misc{zhao2026omnimemscalableadaptivememory,
      title={OmniMem: Scalable and Adaptive Memory Retrieval for Long Video Generation},
      author={Lin Zhao and Yushu Wu and Yifan Gong and Yanzhi Wang and Pu Zhao},
      year={2026},
      eprint={2605.30519},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.30519},
}
```
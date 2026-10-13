# Teacher-forcing training and resume

The `causal` and `nsa-causal` trainers save complete checkpoints through
`CausalCheckpointMixin`. This reuses the ODE atomic checkpoint store and training
state serialization, with additional checks for EMA and the causal training recipe.

## Checkpoint layout

```text
<checkpoint_dir>/step_00000100/
  model/config.json
  model/diffusion_pytorch_model*.safetensors
  ema/config.json                              # when ema_model: true
  ema/diffusion_pytorch_model*.safetensors
  optimizer/.metadata
  optimizer/*.distcp
  training_state.pt
  rng/rank_00000.pt                            # one file per rank
  COMPLETED.json
```

Weights may include a safetensors shard index. `COMPLETED.json` records file sizes
and SHA256 hashes. All payloads are written to a staging directory before atomic
publication. S3 receives the completion marker last. A partial save is not a
resumable checkpoint, and completed steps are never overwritten.

Use a shared `checkpoint_dir` for multiple nodes and an explicit, unique
`checkpoint_s3_prefix` for each run. On restart, the trainer finds the newest valid
local/S3 checkpoint, restores model weights before FSDP wrapping, restores the
optimizer and scheduler, then restores EMA before wrapping it. Python, NumPy,
CPU and CUDA RNG states are restored after the first lazy data-loader fetch.
Evaluation preserves the training RNG state.

Full resume requires the same world size, model configuration, teacher-forcing
recipe, EMA policy, optimizer settings and compatible LR schedule. A completed
run exits without another training step. Legacy checkpoint folders without the
complete state are rejected for resume; their model or EMA weights can still
initialize a separate run.

WebDataset worker positions and the sample cursor are not saved. The restarted
job samples a new data stream; fixed-input continuation tests do not imply an
identical sequence of real training samples after preemption.

## Compilation and gate initialization

The teacher-forcing loss is selected with `teacher_forcing: true`, and with
`nsa_kwargs.tf: true` for `nsa-causal` (see
`configs/causal/train_wan_tf_chunk_nsa_15x2_top16_qg15.yaml`).
`torch_compile: false` disables compilation of the transformer and of the T5
encoder. FlexAttention, which evaluates the block-causal training mask, is always
compiled: its `torch.compile` mode is `max-autotune-no-cudagraphs` unless the
environment variable `OMNIMEM_FLEX_COMPILE_MODE` names another mode (for example
`default`). The variable is read once, when the attention module is imported.

`nsa-causal` loads `model_path` (the Wan2.1-T2V-1.3B base weights in the shipped
config) and then initializes the NSA gates, synchronized across ranks.
`preserve_pretrained_gate: true` keeps the gates stored in `model_path` instead;
use it only when deliberately initializing a different run from a previously
trained NSA checkpoint. It must not be used to disguise missing pretrained gate
weights.

# Teacher-forcing training and resume

The `causal` and `nsa-causal` trainers save complete checkpoints through
`CausalCheckpointMixin`. This reuses the ODE atomic checkpoint store and training
state serialization, with additional checks for EMA and the causal training recipe;
see [ODE resume](ODE_RESUME.md#what-one-complete-checkpoint-contains) for the store,
its multi-node requirements and what resume guarantees.

## Checkpoint layout

A step directory holds the files of an ODE checkpoint plus `ema/` (config and
safetensors) when `ema_model: true`. On restart, the trainer restores model weights
before FSDP wrapping, restores the optimizer and scheduler, then restores EMA before
wrapping it.

Full resume requires the same world size, model configuration, teacher-forcing
recipe, EMA policy, optimizer settings and compatible LR schedule. Legacy checkpoint
folders without the complete state are rejected for resume; their model or EMA
weights can still initialize a separate run.

## Compilation and gate initialization

The teacher-forcing loss is selected with `teacher_forcing: true`, and with
`nsa_kwargs.tf: true` for `nsa-causal` (see
`configs/causal/train_wan_tf_chunk_nsa_15x2_top16_qg15.yaml`).
`torch_compile: false` disables compilation of the transformer and of the T5
encoder. FlexAttention, which evaluates the block-causal training mask, is always
compiled, in the mode `OMNIMEM_FLEX_COMPILE_MODE` sets (see the
[environment variables](STREAMING_INFERENCE.md#environment-variables)).

`nsa-causal` loads `model_path` (the Wan2.1-T2V-1.3B base weights in the shipped
config) and then initializes the NSA gates, synchronized across ranks.
`preserve_pretrained_gate: true` keeps the gates stored in `model_path` instead;
use it only when deliberately initializing a different run from a previously
trained NSA checkpoint. It must not be used to disguise missing pretrained gate
weights.

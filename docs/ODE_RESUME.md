# ODE training: checkpoint resume

This guide covers `WanODETrainer` and the NSA ODE trainer that inherits from it. A full resume needs the model, optimizer, scheduler, training step and every rank's random-number state from the same checkpoint; an export that holds model weights only is for initializing a new fine-tuning experiment.

## Initialization and resume

| Scenario | What to do |
| --- | --- |
| Start a new experiment from existing NSA ODE weights (for example with a different `num_selected_blocks`) | A new experiment output directory; point `model_path` at the existing model and set `preserve_pretrained_gate: true`; the optimizer and the step count start fresh |
| Continue the same experiment after preemption | Keep the experiment's identity and its persistent checkpoints; the full training state is restored |
| Only a legacy `checkpoint.pth`, whose `optimizer` is `None` | Not a full-resume source; the trainer raises an explicit error. To fine-tune further, use the model export as `model_path` with a new experiment output directory |
| The restored step has reached or passed `max_train_steps` | Training ends immediately, with no validation and no extra optimizer step |

Do not turn a resume error into a silent restart by deleting or forging checkpoint metadata. Confirm compatibility before changing the model architecture, the trainable parameters, or the optimizer or scheduler definition; a fine-tune that changes the meaning of the experiment should normally use a new experiment directory.

ODE training discovers checkpoints automatically; no separate resume switch is needed. The optional path settings are:

```yaml
# Every rank must see the same directory; defaults to output_dir/name/checkpoints.
checkpoint_dir: /shared/experiments/my-ode-run/checkpoints

# S3 object key prefix, without s3://bucket; the bucket is the existing s3_bucket setting.
checkpoint_s3_prefix: experiments/my-ode-run/checkpoints
```

When `checkpoint_s3_prefix` is omitted, the prefix is composed of `s3_dir`, the relative `output_dir`, the experiment `name` and `checkpoints`. If `output_dir` is absolute and S3 is enabled, `checkpoint_s3_prefix` must be set explicitly, so that a machine-local path is never used as an S3 key.

To continue an existing experiment, point these paths at its complete checkpoint root; there is no separate resume-source option. To start a new experiment from an existing model, set `model_path` or `model_checkpoint_path` and use a new experiment name and output directory.

When an NSA model loads trained weights through `model_path`, their gates must be kept explicitly:

```yaml
model_path: /path/to/exported-nsa-model
preserve_pretrained_gate: true
nsa_kwargs:
  num_selected_blocks: 12
  # The remaining pooling, query-group and window parameters follow the chosen experiment config.
```

`preserve_pretrained_gate` defaults to `false`, which keeps the gate initialization used when a plain model is converted to NSA. Fine-tuning an existing NSA checkpoint without the option makes `build_model()` re-initialize the gates; a full checkpoint resume then strictly loads all saved model weights afterwards.

## What one complete checkpoint contains

Each checkpoint is saved by training step into its own directory, such as `step_00000100`; the last resumable state is never overwritten in place. A save goes through a hidden `.incomplete-...` staging directory. The `COMPLETED.json` marker is published only after all required files have been written and validated; it records the format version, `global_step`, the world size, and the relative path, size and SHA-256 of every file.

```text
checkpoints/step_00000100/
  model/                 # Diffusers config and safetensors model export
  optimizer/             # PyTorch Distributed Checkpoint optimizer shards
  training_state.pt      # scheduler, step, format and compatibility information
  rng/rank_00000.pt      # one RNG state per rank
  ...
  COMPLETED.json         # integrity manifest, published last
```

`model/` can be used on its own for inference and to initialize a new experiment; a full resume needs the whole step directory.

Resume selects only completed, intact checkpoints. If the newest save was interrupted, an earlier complete checkpoint is tried; half-written new weights are never combined with an older step or optimizer. On S3 the completion marker is likewise uploaded last, and integrity is checked again after download. A storage error is never treated as "no checkpoint" followed by a start from scratch.

The persistent local directory and the configured S3 experiment directory are searched for resume points. A model export on S3 is not a complete resume point, and a legacy directory holding only `model/` and `checkpoint.pth` is not accepted just because it records a training step.

Discovery merges the local and S3 candidates in descending step order. For the same step a valid local copy is preferred, and S3 is used when the local copy is missing or corrupt. A newer complete S3 checkpoint takes precedence over an older local one. A candidate must first pass the file-list, size and SHA-256 checks.

If the S3 listing fails and a validated local checkpoint exists, that local copy is used after a warning; in that case it cannot be confirmed whether a newer step exists remotely. Without a valid local copy the run fails. Permission or service errors on a selected remote candidate are not disguised as a corrupt file; a candidate that is missing or fails validation falls back to an earlier complete save.

If only an unfinished first save exists and no complete checkpoint was ever published, training starts again from the configured initial model and logs that it does. If published checkpoints exist but all are corrupt, the run fails; a legacy `checkpoint.pth` in the checkpoint root is likewise rejected explicitly as incomplete state. A corrupt local directory is kept as `.corrupt-...` before the same step is repaired from a valid S3 copy. Old checkpoints are not cleaned up automatically; the one exception is `checkpoint_keep_local: N` with S3 upload enabled, which removes local copies beyond the newest `N` once their S3 copy has been verified.

## Multi-node requirements

- Every rank must access the same checkpoint directory; the per-rank files of the distributed checkpoint must be visible to all of them.
- Keep the checkpoint directory on persistent shared storage. A node-local SSD is lost when the node is released; it suits code, the software environment and caches of initial models that can be downloaded again.
- Per-rank RNG restore requires the same world size at resume as at save time. Even though a distributed checkpoint can be re-sharded, that does not make the random-number trajectory exactly restorable under a different world size.
- All ranks must take part in the save/restore collectives in the same order. A download or validation failure on the main rank is propagated to the other ranks, so that none of them keeps waiting.

Keep the same experiment output path to continue an experiment, and use a different directory for an independent one. Never run two training jobs that write to the same checkpoint root at the same time.

## What resume actually guarantees

The model and AdamW restore covers the saved parameters, moments and optimizer step count; the scheduler loads its saved state directly rather than being advanced by repeated `scheduler.step()` calls to guess the position.

Each rank's Python, NumPy, Torch CPU and CUDA random-number state is restored before the next training computation. Initialization, model loading, data-loader construction, the first batch fetch and validation do not consume the restored training RNG. WebDataset iterators are lazy: calling `iter(loader)` alone does not complete worker initialization, so the state is restored after the first batch has been fetched. A save point corresponds to a complete optimizer-step boundary; an intermediate state of gradient accumulation is not treated as a complete step.

**The WebDataset read position and the worker prefetch queues are not part of the resume guarantee.** After a resume the shard/sample stream is rebuilt, and samples that the next batches would have read may be repeated or skipped. A complete optimizer/scheduler/RNG restore therefore does not mean that the real training data stream continues sample by sample, and it does not guarantee results bit-identical to an uninterrupted run.

To verify resume correctness, use fixed inputs and compare the model, AdamW and scheduler state of "two consecutive steps" with that of "one step, save, rebuild the model and optimizer, restore, one more step"; check the random-number state and the step-count stop condition separately. A real multi-node launch must still confirm that all ranks load the same step.

## Preemption, requeue and checkpoint resume

These mechanisms are independent of each other:

| Setting | Effect |
| --- | --- |
| Preemptible resources (scheduler option) | Requests a preemptible resource allocation |
| Automatic requeue (scheduler option) | Lets the scheduler requeue and rerun the job after preemption |
| `torchrun --max-restarts` | Number of times the current torchrun agent retries after a worker failure |
| ODE checkpoint resume | A new process restores the model and training state from a persistent checkpoint |

A requeue only reruns the launch script; it does not save the optimizer or choose a checkpoint on behalf of the training code. `torchrun --max-restarts=0` does not prevent a separately enabled preemption requeue, and it provides no automatic retry for ordinary training failures either.

## Code entry points

- Training-state integration: `src/omnimem/trainer/wan_ode_trainer.py`
- NSA ODE subclass: `src/omnimem/trainer/wan_ode_nsa_trainer.py`

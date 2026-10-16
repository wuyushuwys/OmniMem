# NSA self-forcing: complete resume

Set `complete_sf_resume: true` for `nsa-self-forcing`, `nsa-self-forcing-streaming`, or `nsa-self-forcing-switch` with `dataset_type: text_dataset` and full parameter training; the two streaming trainers refuse to start without it. LoRA adapters are not supported by this mode. Set `torch_compile: false` when compilation is unwanted; all three trainers honor that flag.

An ODE checkpoint passed through `model_path` initializes generator weights only. A new SF checkpoint root starts at SF step 0 with fresh critic optimizer, generator optimizer, and schedulers. `checkpoint_dir` identifies the persistent, shared full-state SF checkpoint root; `checkpoint_s3_prefix` identifies its S3 mirror. Do not point either at an ODE run. `resume_ckpt_path` is rejected in this mode to avoid confusing weight initialization with full resume.

## Saved state and publication

Every completed optimizer/accumulation boundary may produce an immutable directory:

```text
step_00000200/
  model/                    # generator config and safetensors
  fake_net/                 # critic config and safetensors
  ema/                      # present only once delayed EMA is active
  optimizer/                # generator distributed optimizer state
  training_state.pt         # generator scheduler, recipe, step, data cursor
  rng/rank_00000.pt          # one file per rank
  critic_state/
    optimizer/              # critic distributed optimizer state
    training_state.pt       # critic scheduler, same recipe/step/data cursor
    rng/rank_00000.pt        # one file per rank
  COMPLETED.json            # SHA256 and size for every payload file
```

Publication, S3 upload and discovery use the ODE checkpoint store; see [ODE resume](ODE_RESUME.md#what-one-complete-checkpoint-contains). Before publishing, rank 0 also validates both model payloads, both optimizer states, rank RNG identities, step/world size, recipe, and EMA policy. Legacy checkpoints and stage-incompatible checkpoints are rejected.

Set `checkpoint_keep_local: 0` with S3 upload enabled to delete each completed local checkpoint after its remote manifest and every payload object's size and available SHA256 have been verified. Upload or verification failures preserve the local copy. Interrupted `.incomplete-*` directories are also preserved for recovery. A checkpoint downloaded during resume is removed only after all ranks have loaded model, optimizer, EMA, and streaming state. Unset retention keeps completed checkpoints; a positive value retains that many recent local copies. Without S3 upload nothing is pruned.

For a single-node run, use an explicit `checkpoint_dir` on local SSD plus a durable `checkpoint_s3_prefix`; all local ranks can share that SSD directory, and outputs and logs can live there too. Check user quota as well as filesystem free space before staging a full checkpoint.

## Continuation semantics

Both Adam states and both scheduler states are restored directly. With generator/critic ratio 5, the generator schedule counts generator updates and has `ceil(max_train_steps / 5)` total updates; the critic schedule counts every global update. In this mode a generator update does not clear critic gradients accumulated from preceding microbatches.

For `ema_start_step: 200`, the completed step-200 checkpoint already contains an initialized EMA. Earlier checkpoints require no EMA. At step 200→201 the first applicable generator update advances EMA normally. Resume restores the saved EMA rather than reconstructing it from the current generator.

Full resume requires the same world size, model configurations, and training recipe. The recipe includes both optimizers, schedule, update ratio, accumulation, rollout/cache settings, EMA policy, prompt contents SHA256, seed, negative prompt, and conditioning/teacher asset paths. The initialization generator path and output/checkpoint paths may change. Immutable source assets must remain available.

The deterministic text loader saves sampler epoch and consumed batch count. Resume rebuilds that epoch and skips consumed batches before restoring Python, NumPy, CPU torch, and current-device CUDA RNG. Distributed sampling calls `set_epoch`; loader/worker and single-rank shuffle generators use an isolated per-epoch seed. One flattened iterator carries partial gradient accumulation across epoch boundaries. A checkpoint at the last batch of an epoch advances correctly to the next epoch. Empty loaders and exhausted training epochs fail clearly. Evaluations preserve the training RNG streams.

A resume restores the saved state exactly: weights, optimizers, schedulers, EMA, RNG streams, the data cursor and, for the streaming trainers, the rollout and its KV cache. Whether the resumed run then reproduces the uninterrupted one bit for bit also depends on the computation itself being deterministic, so use the same world size, model/backend environment, and source assets. One known limit: the NSA selection backward sums gradient contributions in thread-arrival order. On that path (`lru_max_size=0`, or the dense selection path) the gradients of the `k`, `v` and `norm_k` parameters therefore vary between runs by up to about one bf16 unit at the tensor's scale; the forward never differs. FlashAttention's backward in its default non-deterministic mode and the autotuned Triton backward kernels are further sources of such differences.

## Streaming and prompt switching

`configs/self_forcing/training_wan_nsa_self_forcing_4step_flow_streaming.yaml` enables complete resume and saves every **20 completed optimizer steps**, even in the middle of a rollout. The final step also saves when it is not a multiple of the interval. An optimizer step includes all configured gradient accumulation microbatches; generator updates still follow `dfake_gen_update_ratio`.

Non-first streaming windows use a VAE boundary block: decode the previous window, re-encode its last `4 * frame_per_block - 3` RGB frames into `frame_per_block` boundary latents, and write their KV at the current sequence position before generating new latents. With a 21-latent window, blocks of 3 and `streaming_min_new_frame: 18`, each subsequent window contains 3 boundary latents and 18 new latents. Boundary latents are excluded from the generator loss; the sequence position advances by all 21. For NSA a boundary block is an ordinary block of the sequence: its global query position and SWA exclusion are computed as for any other block.

```bash
torchrun --nproc_per_node=8 train.py \
  --config configs/self_forcing/training_wan_nsa_self_forcing_4step_flow_streaming.yaml \
  --task nsa-self-forcing-streaming --name omnimem_nsa_sf_streaming
```

Restart with the same command and persistent checkpoint root. Use `--task nsa-self-forcing-switch --name omnimem_nsa_sf_switch` for a separate run with prompt switching. The switch trainer reads top-level `switch_prompt_file`, falling back to `train_data.switch_prompt_path`; `switch_choices` selects the latent-frame threshold, and switching happens at the first subsequent chunk boundary. `switch_mode: random_choice` is the supported mode. Each rollout switches at most once. Ordinary streaming ignores these switching options.

Streaming checkpoints add `streaming/rank_00000.pt` (one file per rank) under the same atomic completion marker. These files contain the current sequence position and selected maximum length, previous latent frames, positive/negative text embeddings, active and pending switch conditions, the evaluation cursor, and the actual KV cache. Cache tensors retain their CPU/GPU placement, LRU order, and eviction state; process-local pointer tables are rebuilt. The KV tensors are saved rather than recomputed because earlier chunks were generated using earlier model parameters. These checkpoints can therefore be substantially larger than ordinary SF checkpoints.

Resuming continues the saved rollout before drawing the next prompt. Prompt order and random streams continue from the saved cursor. Evaluation normally waits for a sequence boundary (or the final step); it preserves training RNG and restores the original cache, including when called on an active rollout. A resumed run does not repeat startup evaluation, and an already-completed run exits without another update or checkpoint.

Streaming and switch checkpoints reject cross-task resumes and changes to streaming lengths, boundary policy, switch schedule or prompt-file contents, as well as the common training recipe.

Checkpoints of the removed `complete_sf_resume: false` streaming loop are not complete: they lack the runtime/RNG state a complete resume needs and cannot be turned into one. To continue from such a checkpoint, use its generator weights as `model_path` in a new checkpoint root.

The recipe records `streaming_boundary_mode: vae_roundtrip_block`. A complete checkpoint whose recipe does not record this boundary mode is rejected, even when it was saved between rollouts. To train from its weights, pass its `model/` or `ema/` directory as `model_path` and use a fresh `checkpoint_dir` and S3 checkpoint prefix; this starts a new training run with fresh optimizer and rollout state.

## Distributed checkpoint metadata backend

Training-state DCP save/load and their error-message synchronization use a dedicated full-world Gloo group, reused across both optimizers and subsequent checkpoints. FSDP/DTensor model computation and tensor state collection retain their original groups. The metadata group has a 5-minute timeout and is rebuilt after explicit destruction or default-process-group reinitialization; the checkpoint format does not depend on it.

DCP exchanges its pickled planning metadata with `gather_object`/`scatter_object`. On an NCCL group these become CUDA point-to-point transfers, which can stall on multi-node saves; the CPU group avoids that.

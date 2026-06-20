set -e
[ -f run.local.sh ] && source run.local.sh   # WANDB_KEY/PROJ, S3_BUCKET/DIR (gitignored)

num_gpus=$(nvidia-smi --list-gpus | wc -l)
train="torchrun --nproc_per_node=$num_gpus train.py"

# ODE / teacher-forcing / CD stages train on ODE trajectories — generate first:
#   bash scripts/generate_ode_wan.sh

# ===== Flow 1: ODE -> Self-Forcing =====
# $train --config configs/ode/training_wan_nsa_ode.yaml --task nsa-ode --name omnimem_nsa_ode

# ===== Flow 2: Teacher-Forcing -> Consistency Distillation -> Self-Forcing =====
# $train --config configs/causal/train_wan_tf_chunk_nsa_15x2_top16_qg15.yaml --task nsa-causal --name omnimem_nsa_tf
# $train --config configs/causal/train_cd_tf_chunk_nsa_15x2_top16_qg15.yaml --task nsa-cd --name omnimem_nsa_cd

# ===== Self-Forcing (final stage of both flows) =====
$train --config configs/self_forcing/training_wan_nsa_self_forcing_4step_flow.yaml --task nsa-self-forcing --name omnimem_nsa_self_forcing

# ===== Long-video finetuning (optional) =====
# $train --config configs/self_forcing/training_wan_nsa_self_forcing_4step_flow_streaming.yaml --task nsa-self-forcing-streaming --name omnimem_nsa_sf_streaming
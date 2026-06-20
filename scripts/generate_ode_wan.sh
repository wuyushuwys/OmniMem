set -e
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Generate ODE trajectories from the base Wan2.1-T2V-1.3B teacher.
# These are consumed by the ODE / teacher-forcing training stages.
# For parallel generation, run multiple jobs with different --shard_index
# (0..num_shard-1) and the same --num_shard.

model_path=wan_models/Wan2.1-T2V-1.3B
vae_path=wan_models/vae-2.1/vae

torchrun --nproc_per_node=$num_gpus generate_ode_wan.py \
    --prompt_folder configs/prompts \
    --prompt_file vidprom_filtered_extended.txt \
    --name vidprom \
    --model_name Wan2.1-T2V-1.3B \
    --pretrained $model_path \
    --vae_path $vae_path \
    --height 480 --width 832 --num_frames 81 \
    --num_inference_steps 40 --guidance_scale 5 --shift 5 \
    --target_step 0 10 20 30 \
    --shard_index 0 --num_shard 1
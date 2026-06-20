set -e
# export CUDA_VISIBLE_DEVICES=0

vae_path='wan_models/vae-2.1/vae'
model_path="outputs/omnimem_nsa_cd/checkpoints/ema"

# --- Single-GPU inference (writes a demo grid under demo/) ---
python inference_wan_causal.py \
    --pretrained $model_path \
    --vae_path $vae_path \
    --height 480 --width 832 --nframes 81 \
    --fps 16 --cfg 1 --num_inference_steps 4 --shift 5 \
    --frame_per_block 3 --window_size 3 --nrows 1 --nsa \
    --enable_kv_evict --lru_max_size 7 --kv_evict_min_gpu_chunks 4 \
    --name genv

# --- Multi-GPU batched inference over a prompt file (e.g. long video) ---
# Outputs are uploaded to S3; set S3_BUCKET in your env first.
# export S3_BUCKET=<your-bucket>
# torchrun --nproc_per_node=8 inference_mp.py \
#     --pretrained $model_path \
#     --vae_path $vae_path \
#     --save_path logs/omnimem/nsa \
#     --prompt_path configs/prompts/MovieGenVideoBench.txt \
#     --height 480 --width 832 --nframes 960 \
#     --fps 16 --cfg 1 --num_inference_steps 4 --shift 5 \
#     --frame_per_block 3 --window_size 3 --nsa \
#     --enable_kv_evict --lru_max_size 7 --kv_evict_min_gpu_chunks 4
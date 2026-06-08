#!/bin/bash

#SBATCH --job-name=glorcq_quant
#SBATCH --output=/home/qyyang/repo/GLoRCQ/logs/%j.out
#SBATCH --error=/home/qyyang/repo/GLoRCQ/logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=LocalQ
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

export HF_HOME=/home/qyyang/resource_dir/hf_cache/
export TRANSFORMERS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_DATASETS_CACHE=/home/qyyang/resource_dir/hf_cache/
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_DISABLE_CACHING=0

cd /home/qyyang/repo/GLoRCQ

model_path="${MODEL_PATH:-Qwen/Qwen1.5-MoE-A2.7B}"
save_dir="/home/qyyang/resource_dir/GLoRCQ_out"
qbit="${QBIT:-2}"
rank="${RANK:-64}"
rank_attn="${RANK_ATTN:-512}"
n_iter="${N_ITER:-5}"
G_moe="${G_MOE:-128}"
G_attn="${G_ATTN:-4}"
u_bits="${U_BITS:-4}"
sv_bits="${SV_BITS:-4}"
u_bits_attn="${U_BITS_ATTN:-8}"
sv_bits_attn="${SV_BITS_ATTN:-8}"
recon_weight="${RECON_WEIGHT:-0.0}"

logname="glorcq_${qbit}bit_rank${rank}_rattn${rank_attn}_u${u_bits}_sv${sv_bits}_uattn${u_bits_attn}_svattn${sv_bits_attn}_iter${n_iter}"
output_path="$save_dir/${model_path}-glorcq-${qbit}bit-rank${rank}-rattn${rank_attn}-u${u_bits}-sv${sv_bits}-uattn${u_bits_attn}-svattn${sv_bits_attn}"

uv run python run_quantize.py \
    --model_path "$model_path" \
    --output_path "$output_path" \
    --qbit $qbit \
    --w_clip \
    --rank $rank \
    --rank_attn $rank_attn \
    --G_moe $G_moe \
    --G_attn $G_attn \
    --nsamples 128 \
    --n_iter $n_iter \
    --hessian_svd \
    --use_turboquant \
    --search_act_alpha \
    --u_bits $u_bits \
    --sv_bits $sv_bits \
    --u_bits_attn $u_bits_attn \
    --sv_bits_attn $sv_bits_attn \
    --recon_weight $recon_weight \
    2>&1 | tee ./logs/${logname}.log

uv run python evaluate/eval_ppl.py \
    --model_path "$output_path" \
    2>&1 | tee -a ./logs/${logname}.log


#!/bin/bash
# Mixtral-8x7B fake-quant eval using 2 GPUs (model is ~88GB FP16, > single A100)
# Usage: bash run_mixtral_eval.sh <gpu_a> <gpu_b>
# Example: bash run_mixtral_eval.sh 4 5
#
# Requirements:
#   - Two free A100 80GB GPUs (model needs ~90GB total)
#   - model_loader.py supports --device auto → device_map="auto"
cd /home/qyyang/repo/GLoRCQ

GPU_A=${1:-4}
GPU_B=${2:-5}
DEVICE="auto"
MODEL=/home/qyyang/resource_dir/GLoRCQ_out/mixtral_8x7b_sota

echo "[$(date)] Mixtral eval using GPUs $GPU_A + $GPU_B"
echo "[$(date)] Model: $MODEL"
echo "[$(date)] Note: device_map=auto distributes across both GPUs"

# PPL eval
echo "[$(date)] === Mixtral PPL eval ==="
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} .venv/bin/python evaluate/eval_ppl.py \
    --model_path "$MODEL" \
    --device $DEVICE \
    > logs/mixtral_ppl.log 2>&1
echo "[$(date)] PPL done: $(grep 'WikiText-2 PPL' logs/mixtral_ppl.log)"

# FP16 baseline PPL
echo "[$(date)] === Mixtral FP16 PPL eval ==="
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} .venv/bin/python evaluate/eval_ppl.py \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --device $DEVICE \
    > logs/mixtral_fp16_ppl.log 2>&1
echo "[$(date)] FP16 PPL done: $(grep 'WikiText-2 PPL' logs/mixtral_fp16_ppl.log)"

# Zero-shot eval
echo "[$(date)] === Mixtral zero-shot eval ==="
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path "$MODEL" \
    --device $DEVICE \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/mixtral_zeroshot.json \
    > logs/mixtral_zeroshot.log 2>&1
echo "[$(date)] Zero-shot done"
grep -E "Average|arc|wino|hella|piqa" logs/mixtral_zeroshot.log | tail -8

# FP16 baseline zero-shot
echo "[$(date)] === Mixtral FP16 zero-shot eval ==="
CUDA_VISIBLE_DEVICES=${GPU_A},${GPU_B} .venv/bin/python evaluate/eval_zeroshot.py \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --device $DEVICE \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --batch_size 1 \
    --output_json logs/mixtral_fp16_zeroshot.json \
    > logs/mixtral_fp16_zeroshot.log 2>&1
echo "[$(date)] FP16 zero-shot done"
grep -E "Average|arc|wino|hella|piqa" logs/mixtral_fp16_zeroshot.log | tail -8

echo "[$(date)] === All Mixtral evals done ==="
echo "PPL: $(grep 'WikiText-2 PPL' logs/mixtral_ppl.log)"
echo "FP16 PPL: $(grep 'WikiText-2 PPL' logs/mixtral_fp16_ppl.log)"

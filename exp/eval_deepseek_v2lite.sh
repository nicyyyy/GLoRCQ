#!/bin/bash
# Eval DeepSeek-V2-Lite GLoRCQ fake-quant: WikiText-2 PPL + 5-task zero-shot.
# Also runs fp16 baseline PPL for the quantization gap.
cd "$(dirname "$0")/.."

FAKE=/mnt/Data/yqy/resource_dir/glorcq_paper_exp/deepseek_v2lite_fake
FP16=/mnt/Data/yqy/resource_dir/deepseek-v2-lite
OUT=logs/deepseek_v2lite
mkdir -p $OUT
export HF_HOME=/mnt/Data/yqy/resource_dir/hf_cache

GPU=${CUDA_VISIBLE_DEVICES:-4}

echo "[$(date)] === PPL (fake-quant) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path $FAKE --device cuda:0 \
    --max_length 2048 --stride 512 --output_json $OUT/ppl_fake.json 2>&1 | tee $OUT/eval_ppl_fake.log

echo "[$(date)] === PPL (fp16 baseline) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_ppl.py --model_path $FP16 --device cuda:0 \
    --max_length 2048 --stride 512 --output_json $OUT/ppl_fp16.json 2>&1 | tee $OUT/eval_ppl_fp16.log

echo "[$(date)] === Zero-shot 5-task (fake-quant) ==="
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python evaluate/eval_zeroshot.py --model_path $FAKE --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --num_fewshot 0 --batch_size 1 --output_json $OUT/zeroshot_fake.json 2>&1 | tee $OUT/eval_zs_fake.log

echo "[$(date)] DEEPSEEK_EVAL_DONE"

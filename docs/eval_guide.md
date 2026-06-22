# GLoRCQ Evaluation Guide

How to reproduce GLoRCQ evaluation results on a remote machine.  
All models are uploaded as standard HuggingFace checkpoints (FP16 fake-quant format) and can be evaluated with the scripts in this repository.

---

## 1. Hardware Requirements

| Model | Eval type | Minimum GPU memory | Recommended |
|-------|-----------|--------------------|-------------|
| Qwen1.5-MoE-A2.7B | PPL + zero-shot | 40 GB | 80 GB A100 |
| Mixtral-8x7B-v0.1 | PPL + zero-shot | 80 GB | 80 GB A100 |
| Qwen3-30B-A3B | PPL + zero-shot | 80 GB × 2 | H100 80 GB |

> **Note**: The fake-quant checkpoint is a standard FP16 model (same size as the original).  
> Peak GPU memory during eval = model size + activation buffer (~5-10 GB extra).

---

## 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your_org>/GLoRCQ.git
cd GLoRCQ

# Install dependencies (requires Python 3.10+, CUDA 12+)
pip install uv
uv sync

# Verify lm-eval is available
python -c "import lm_eval; print(lm_eval.__version__)"
```

---

## 3. Available Models on HuggingFace

> **TODO**: Fill in actual HuggingFace repo paths after upload.

| Model | HF Repo | Description |
|-------|---------|-------------|
| Qwen1.5-MoE-A2.7B (GLoRCQ 2-bit) | `<hf_user>/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit` | Fake-quant FP16, ready to eval |
| Mixtral-8x7B-v0.1 (GLoRCQ 2-bit) | `<hf_user>/Mixtral-8x7B-v0.1-GLoRCQ-2bit` | Fake-quant FP16, ready to eval |
| Qwen3-30B-A3B (GLoRCQ 2-bit) | `<hf_user>/Qwen3-30B-A3B-GLoRCQ-2bit` | Fake-quant FP16, needs 2×80 GB |

---

## 4. Quick Start (One-Command Eval)

The `eval_all.sh` script runs PPL + zero-shot (5 tasks, 0-shot) + MMLU (5-shot) in sequence.

```bash
# Qwen1.5-MoE-A2.7B — single GPU (≥40 GB)
bash evaluate/eval_all.sh <hf_user>/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit logs/qwen_moe_eval 0

# Mixtral-8x7B — single GPU (80 GB)
bash evaluate/eval_all.sh <hf_user>/Mixtral-8x7B-v0.1-GLoRCQ-2bit logs/mixtral_eval 0

# Qwen3-30B-A3B — requires multi-GPU or very large memory
CUDA_VISIBLE_DEVICES=0,1 bash evaluate/eval_all.sh \
    <hf_user>/Qwen3-30B-A3B-GLoRCQ-2bit logs/qwen3_eval 0
```

Results are saved to `logs/<model_name>/`:
- `ppl.json` — WikiText-2 PPL
- `zeroshot_5task.json` — ARC-C, ARC-E, WinoGrande, HellaSwag, PIQA
- `zeroshot_mmlu.json` — MMLU (5-shot)
- `summary.txt` — Human-readable table

---

## 5. Running Evaluations Individually

### 5.1 WikiText-2 Perplexity

```bash
python evaluate/eval_ppl.py \
    --model_path <hf_user>/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit \
    --device cuda:0 \
    --max_length 2048 \
    --stride 512 \
    --output_json logs/ppl.json
```

### 5.2 Zero-Shot Tasks (ARC, WinoGrande, HellaSwag, PIQA)

```bash
python evaluate/eval_zeroshot.py \
    --model_path <hf_user>/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit \
    --device cuda:0 \
    --tasks arc_challenge,arc_easy,winogrande,hellaswag,piqa \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_json logs/zeroshot_5task.json
```

### 5.3 MMLU (5-shot)

```bash
python evaluate/eval_zeroshot.py \
    --model_path <hf_user>/Qwen1.5-MoE-A2.7B-GLoRCQ-2bit \
    --device cuda:0 \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_json logs/zeroshot_mmlu.json
```

---

## 6. Expected Results (Sanity Check)

Use these numbers to verify your setup is correct.

### Qwen1.5-MoE-A2.7B

| Metric | GLoRCQ 2-bit | FP16 baseline |
|--------|-------------|---------------|
| WikiText-2 PPL | **8.48** | 6.79 |
| ARC-Challenge | 40.19% | 44.54% |
| ARC-Easy | 65.07% | 69.19% |
| WinoGrande | 67.32% | 69.06% |
| HellaSwag (acc_norm) | 69.78% | 77.28% |
| PIQA | 77.04% | 80.52% |
| MMLU (5-shot) | *TBD* | 61.2%* |
| Avg (5 tasks, no MMLU) | **63.88%** | 68.12% |

*FP16 MMLU from TileQ Table 1.

### Mixtral-8x7B-v0.1

| Metric | GLoRCQ 2-bit | FP16 baseline |
|--------|-------------|---------------|
| WikiText-2 PPL | *TBD (running)* | 3.87 |
| 6-task Avg | *TBD* | 74.75% |

---

## 7. Notes on Evaluation Protocol

- **HellaSwag**: We use `acc_norm`. TileQ may use `acc` (approximately 5-10% lower).  
  When comparing with TileQ, note this discrepancy.
- **MMLU**: Use exactly 5-shot (`--num_fewshot 5`). 0-shot MMLU results are not comparable.
- **Batch size**: Use `--batch_size 1` for reproducibility. Larger batch sizes give different  
  results due to padding effects in lm-eval.
- **WikiText-2**: Use `--max_length 2048 --stride 512` (standard sliding-window protocol).

---

## 8. Troubleshooting

**OOM during PPL eval**:
```bash
# Reduce max_length to fit in memory
python evaluate/eval_ppl.py --model_path ... --max_length 1024 --stride 256
```

**Slow MMLU eval**:  
MMLU has 57 subjects × ~100 questions each. Expect 2-4 hours at batch_size=1 on a single GPU.  
You can test a subset with `--tasks mmlu_abstract_algebra` (one subject) first.

**Model loading fails for Mixtral/Qwen on multi-GPU**:
```bash
# Use device_map to spread across GPUs
# (modify the --device argument or use CUDA_VISIBLE_DEVICES)
CUDA_VISIBLE_DEVICES=0,1 python evaluate/eval_ppl.py \
    --model_path ... --device auto
```
Note: `--device auto` triggers `device_map="auto"` in the model loader (requires modification  
to `utils/model_loader.py` for very large models — see §9 below).

---

## 9. Loading Very Large Models (Qwen3-30B-A3B)

The default model loader uses `.to(device)` which requires the full model to fit in one GPU.
For Qwen3-30B-A3B (~60 GB FP16), use multi-GPU or CPU offload:

```python
# Temporary patch: use device_map="auto" for multi-GPU distribution
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",           # distribute across available GPUs
    trust_remote_code=True,
)
```

Alternatively, run the eval scripts with `CUDA_VISIBLE_DEVICES=0,1` and modify  
`utils/model_loader.py` to pass `device_map="auto"` when `device == "auto"`.

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"

echo "================================================================="
echo "  Exp 5: Full FT Causality — PR Collapse → Benchmark Degradation"
echo "  Qwen3-0.6B | Full Fine-Tuning DPO | 800 steps"
echo "  Benchmarks: arc_easy, hellaswag via lm-eval-harness"
echo "================================================================="
echo "Start: $(date)"
echo "GPU: 4 (train) + 5 (ref)"
echo ""

export CUDA_VISIBLE_DEVICES=4,5
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1

$PYTHON "$SCRIPT_DIR/exp5_fullft_causality.py" 2>&1 | tee "$SCRIPT_DIR/results/exp5_fullft_causality/exp5.log"

echo ""
echo "================================================================="
echo "  Exp 5 DONE: $(date)"
echo "================================================================="

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
LOG_DIR="$SCRIPT_DIR/results/exp5b_fullft_1.7B"
mkdir -p "$LOG_DIR"

echo "================================================================="
echo "  Exp 5b: Full FT Causality — Qwen3-1.7B (8-bit Adam)"
echo "  GPU 6 = train model, GPU 7 = ref model"
echo "  800 steps | Benchmarks: arc_easy, hellaswag"
echo "================================================================="
echo "Start: $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=6,7
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1

$PYTHON "$SCRIPT_DIR/exp5b_fullft_1.7B.py" 2>&1 | tee "$LOG_DIR/exp5b.log"

echo ""
echo "================================================================="
echo "  Exp 5b DONE: $(date)"
echo "================================================================="

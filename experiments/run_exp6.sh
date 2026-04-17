#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
LOG_DIR="$SCRIPT_DIR/results/exp6_sensitivity"
mkdir -p "$LOG_DIR"

echo "================================================================="
echo "  Exp 6: Sensitivity Battery — Qwen3-1.7B"
echo "  Tier 1: Diversity (Distinct-N, TTR, Rep Rate, PPL)"
echo "  Tier 2: ARC-Challenge via lm-eval"
echo "  Tier 3: LoRA Plasticity (step 0, 200, 800)"
echo "  Checkpoints: 0 (baseline), 100, 200, 400, 600, 800"
echo "  GPU: 4"
echo "================================================================="
echo "Start: $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=4
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0

$PYTHON "$SCRIPT_DIR/exp6_sensitivity_battery.py" 2>&1 | tee "$LOG_DIR/exp6.log"

echo ""
echo "================================================================="
echo "  Exp 6 DONE: $(date)"
echo "================================================================="

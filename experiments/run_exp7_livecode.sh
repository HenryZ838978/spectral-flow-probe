#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
LOG_DIR="$SCRIPT_DIR/results/exp7_ood_probe"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT=https://hf-mirror.com

echo "================================================================="
echo "  Exp 7B: LiveCodeBench Easy — Contamination-Free OOD Probe"
echo "  GPU 6: Qwen3-1.7B × 43 Easy problems (post-2025-01)"
echo "  GPU 7: Qwen3-0.6B × 43 Easy problems (post-2025-01)"
echo "================================================================="
echo "Start: $(date)"
echo ""

# GPU 6: 1.7B LiveCodeBench
echo "Launching 1.7B LiveCodeBench on GPU 6..."
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$SCRIPT_DIR/exp7_livecode.py" \
    --model-size 1.7B --gpu-id 0 \
    > "$LOG_DIR/livecode_1.7B.log" 2>&1 &
PID_17B=$!
echo "  PID: $PID_17B → log: $LOG_DIR/livecode_1.7B.log"

# GPU 7: 0.6B LiveCodeBench
echo "Launching 0.6B LiveCodeBench on GPU 7..."
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$SCRIPT_DIR/exp7_livecode.py" \
    --model-size 0.6B --gpu-id 0 \
    > "$LOG_DIR/livecode_0.6B.log" 2>&1 &
PID_06B=$!
echo "  PID: $PID_06B → log: $LOG_DIR/livecode_0.6B.log"

echo ""
echo "Both LiveCodeBench jobs launched. PIDs: $PID_17B (1.7B), $PID_06B (0.6B)"
echo "Monitor: tail -f $LOG_DIR/livecode_1.7B.log"
echo "Monitor: tail -f $LOG_DIR/livecode_0.6B.log"

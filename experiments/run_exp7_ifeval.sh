#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
LOG_DIR="$SCRIPT_DIR/results/exp7_ood_probe"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT=https://hf-mirror.com

echo "================================================================="
echo "  Exp 7A: IFEval OOD Probe — Parallel Launch"
echo "  GPU 4: Qwen3-1.7B × IFEval (step 0, 200, 400, 800)"
echo "  GPU 5: Qwen3-0.6B × IFEval (step 0, 200, 400, 800)"
echo "================================================================="
echo "Start: $(date)"
echo ""

# GPU 4: 1.7B IFEval
echo "Launching 1.7B IFEval on GPU 4..."
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$SCRIPT_DIR/exp7_ifeval.py" \
    --model-size 1.7B --gpu-id 0 \
    > "$LOG_DIR/ifeval_1.7B.log" 2>&1 &
PID_17B=$!
echo "  PID: $PID_17B → log: $LOG_DIR/ifeval_1.7B.log"

# GPU 5: 0.6B IFEval
echo "Launching 0.6B IFEval on GPU 5..."
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$SCRIPT_DIR/exp7_ifeval.py" \
    --model-size 0.6B --gpu-id 0 \
    > "$LOG_DIR/ifeval_0.6B.log" 2>&1 &
PID_06B=$!
echo "  PID: $PID_06B → log: $LOG_DIR/ifeval_0.6B.log"

echo ""
echo "Both IFEval jobs launched. PIDs: $PID_17B (1.7B), $PID_06B (0.6B)"
echo "Monitor: tail -f $LOG_DIR/ifeval_1.7B.log"
echo "Monitor: tail -f $LOG_DIR/ifeval_0.6B.log"

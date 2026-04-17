#!/bin/bash
# Exp 9: 7-Band Phased Array Radar — Overnight Scan
# 4 GPUs in parallel, each scanning one model independently
#
# GPU 4: Qwen2.5-7B (base)       — the key reference
# GPU 5: Qwen2.5-7B-Instruct     — the key comparison (full RLHF)
# GPU 6: Qwen3-1.7B base + DPO   — our weak DPO (sequential on same GPU)
# GPU 7: Qwen3-0.6B base + DPO   — our weak DPO small model (sequential)

set -e
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/exp9_radar_scan.py"
LOG_DIR="$SCRIPT_DIR/results/exp9_radar"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="https://hf-mirror.com"

echo "===== Exp 9: 7-Band Radar Scan — $(date) ====="
echo "Launching 4 parallel GPU scans..."

# GPU 4: Qwen2.5-7B base
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$SCRIPT" \
    --model /cache/zhangjing/models/Qwen2.5-7B \
    --label "Qwen2.5-7B-base" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_qwen25_7b_base.log" 2>&1 &
PID1=$!
echo "  GPU 4: Qwen2.5-7B base       PID=$PID1"

# GPU 5: Qwen2.5-7B-Instruct
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$SCRIPT" \
    --model /cache/zhangjing/models/Qwen2.5-7B-Instruct \
    --label "Qwen2.5-7B-Instruct" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_qwen25_7b_instruct.log" 2>&1 &
PID2=$!
echo "  GPU 5: Qwen2.5-7B-Instruct   PID=$PID2"

# GPU 6: Qwen3-1.7B base then DPO step 800 (sequential, both small)
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$SCRIPT" \
    --model /cache/zhangjing/models/Qwen3-1.7B \
    --label "Qwen3-1.7B-base" \
    --also-model "$SCRIPT_DIR/results/exp5b_fullft_1.7B/checkpoints/step_800" \
    --also-label "Qwen3-1.7B-DPO800" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_qwen3_17b.log" 2>&1 &
PID3=$!
echo "  GPU 6: Qwen3-1.7B base+DPO   PID=$PID3"

# GPU 7: Qwen3-0.6B base then DPO step 800 (sequential, both tiny)
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$SCRIPT" \
    --model /cache/zhangjing/models/Qwen3-0.6B \
    --label "Qwen3-0.6B-base" \
    --also-model "$SCRIPT_DIR/results/exp5_fullft_causality/checkpoints/step_800" \
    --also-label "Qwen3-0.6B-DPO800" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_qwen3_06b.log" 2>&1 &
PID4=$!
echo "  GPU 7: Qwen3-0.6B base+DPO   PID=$PID4"

echo ""
echo "All 4 scans launched. PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Logs:"
echo "  tail -f $LOG_DIR/scan_qwen25_7b_base.log"
echo "  tail -f $LOG_DIR/scan_qwen25_7b_instruct.log"
echo "  tail -f $LOG_DIR/scan_qwen3_17b.log"
echo "  tail -f $LOG_DIR/scan_qwen3_06b.log"
echo ""
echo "Monitor: watch -n 30 'for f in $LOG_DIR/scan_*.log; do echo \"=== \$(basename \$f) ===\"; tail -3 \$f; echo; done'"
echo ""
echo "Expected runtime: ~30min for 7B models, ~10min for small models"

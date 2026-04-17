#!/bin/bash
# Rerun the 4 failed universality jobs in parallel
# Mistral-7B DPO/KTO (chat_template fixed) + Qwen KTO (batch_size fixed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
OUTPUT_DIR="$SCRIPT_DIR/results/exp2_universality"

echo "=== Rerunning 4 failed universality jobs ==="
echo "Start: $(date)"

# Mistral-7B DPO on GPU 4
CUDA_VISIBLE_DEVICES=4 $PYTHON "$SCRIPT_DIR/_exp2_worker.py" \
    --model_path /cache/zhangjing/models/Mistral-7B-v0.1 \
    --model_name Mistral-7B --method DPO --gpu_id 4 \
    --max_steps 200 --probe_every 25 \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/Mistral-7B_DPO.log" 2>&1 &
PID1=$!

# Mistral-7B KTO on GPU 5
CUDA_VISIBLE_DEVICES=5 $PYTHON "$SCRIPT_DIR/_exp2_worker.py" \
    --model_path /cache/zhangjing/models/Mistral-7B-v0.1 \
    --model_name Mistral-7B --method KTO --gpu_id 5 \
    --max_steps 200 --probe_every 25 \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/Mistral-7B_KTO.log" 2>&1 &
PID2=$!

# Qwen3-0.6B KTO on GPU 6
CUDA_VISIBLE_DEVICES=6 $PYTHON "$SCRIPT_DIR/_exp2_worker.py" \
    --model_path /cache/zhangjing/models/Qwen3-0.6B \
    --model_name Qwen3-0.6B --method KTO --gpu_id 6 \
    --max_steps 200 --probe_every 25 \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/Qwen3-0.6B_KTO.log" 2>&1 &
PID3=$!

# Qwen3-4B KTO on GPU 7
CUDA_VISIBLE_DEVICES=7 $PYTHON "$SCRIPT_DIR/_exp2_worker.py" \
    --model_path /cache/zhangjing/models/Qwen3-4B \
    --model_name Qwen3-4B --method KTO --gpu_id 7 \
    --max_steps 200 --probe_every 25 \
    --output_dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/Qwen3-4B_KTO.log" 2>&1 &
PID4=$!

echo "Launched 4 jobs: PIDs $PID1 $PID2 $PID3 $PID4"
echo "Waiting..."

wait $PID1; echo "[$(date)] Mistral-7B DPO: exit=$?"
wait $PID2; echo "[$(date)] Mistral-7B KTO: exit=$?"
wait $PID3; echo "[$(date)] Qwen3-0.6B KTO: exit=$?"
wait $PID4; echo "[$(date)] Qwen3-4B KTO: exit=$?"

echo "=== All done: $(date) ==="

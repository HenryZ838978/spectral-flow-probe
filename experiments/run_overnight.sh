#!/bin/bash
# ============================================================
# SFP Overnight Experiment Suite
# ============================================================
# Runs Exp 4 (Regularizer), Exp 2 (Universality), Exp 3 (Causality)
# in sequence with full logging.
#
# Usage:
#   nohup bash experiments/run_overnight.sh > experiments/results/overnight.log 2>&1 &
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"

mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "SFP OVERNIGHT EXPERIMENT SUITE"
echo "Started: $(date)"
echo "Python: $PYTHON"
echo "========================================================"

# ── Exp 4: Regularizer Sweep (highest value, parallel across 5 GPUs) ──
echo ""
echo "========================================================"
echo "[$(date)] STARTING Exp 4: Regularizer Sweep"
echo "========================================================"
$PYTHON "$SCRIPT_DIR/exp4_regularizer.py" 2>&1
EXP4_RC=$?
echo "[$(date)] Exp 4 finished with exit code $EXP4_RC"

# ── Exp 2: Universality Matrix (3 models x 3 methods, parallel waves) ──
echo ""
echo "========================================================"
echo "[$(date)] STARTING Exp 2: Universality Matrix"
echo "========================================================"
$PYTHON "$SCRIPT_DIR/exp2_universality.py" 2>&1
EXP2_RC=$?
echo "[$(date)] Exp 2 finished with exit code $EXP2_RC"

# ── Exp 3: Causality Bridge (serial, single GPU) ──
echo ""
echo "========================================================"
echo "[$(date)] STARTING Exp 3: Causality Bridge"
echo "========================================================"
CUDA_VISIBLE_DEVICES=7 $PYTHON "$SCRIPT_DIR/exp3_causality.py" 2>&1
EXP3_RC=$?
echo "[$(date)] Exp 3 finished with exit code $EXP3_RC"

# ── Summary ──
echo ""
echo "========================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Finished: $(date)"
echo "Exit codes: Exp4=$EXP4_RC  Exp2=$EXP2_RC  Exp3=$EXP3_RC"
echo ""
echo "Results:"
echo "  Exp 4 (Regularizer):  $RESULTS_DIR/exp4_regularizer/"
echo "  Exp 2 (Universality): $RESULTS_DIR/exp2_universality/"
echo "  Exp 3 (Causality):    $RESULTS_DIR/exp3_causality/"
echo "========================================================"

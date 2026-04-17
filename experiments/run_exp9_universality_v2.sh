#!/bin/bash
# Exp 9 Universality v2: 3-Family Radar + SVD
# ==============================================
# Family 1: Qwen2.5-7B base/instruct — DONE
# Family 2: Mistral-7B base/instruct — downloading
# Family 3: Yi-1.5-6B base/chat — downloading
#
# Script waits for downloads, then runs:
#   1. 7-Band Radar on all 4 models (4 GPUs parallel)
#   2. SVD weight comparison for each family
#   3. Band 0 comparison for each family

set -e
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RADAR="$SCRIPT_DIR/exp9_radar_scan.py"
LOG_DIR="$SCRIPT_DIR/results/exp9_radar"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="https://hf-mirror.com"

echo "===== Exp 9 Universality v2 — $(date) ====="

# ── Wait for all downloads ──
echo "Waiting for model downloads..."

wait_for_model() {
    local path=$1
    local name=$2
    local max_wait=3600  # 1 hour max
    local waited=0
    while [ ! -f "$path/config.json" ] || [ -d "$path/.cache" ]; do
        sleep 30
        waited=$((waited + 30))
        if [ $waited -ge $max_wait ]; then
            echo "  TIMEOUT waiting for $name"
            return 1
        fi
        # Check if actual model files exist (not just config)
        local n_safetensors=$(find "$path" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l)
        local n_bins=$(find "$path" -maxdepth 1 -name "*.bin" 2>/dev/null | wc -l)
        local has_incomplete=$(find "$path" -name "*.incomplete" 2>/dev/null | wc -l)
        echo "  $name: safetensors=$n_safetensors bins=$n_bins incomplete=$has_incomplete (${waited}s)"
        if [ $n_safetensors -gt 0 ] || [ $n_bins -gt 0 ]; then
            if [ $has_incomplete -eq 0 ]; then
                echo "  $name: READY!"
                return 0
            fi
        fi
    done
    echo "  $name: READY!"
    return 0
}

wait_for_model "/cache/zhangjing/models/Mistral-7B-Instruct-v0.1" "Mistral-7B-Instruct"
MISTRAL_OK=$?
wait_for_model "/cache/zhangjing/models/Yi-1.5-6B" "Yi-1.5-6B"
YI_BASE_OK=$?
wait_for_model "/cache/zhangjing/models/Yi-1.5-6B-Chat" "Yi-1.5-6B-Chat"
YI_CHAT_OK=$?

sleep 30  # extra buffer for filesystem sync

echo ""
echo "===== Phase 1: 7-Band Radar Scans — $(date) ====="

# GPU 4: Mistral base
if [ -f "/cache/zhangjing/models/Mistral-7B-v0.1/config.json" ]; then
    CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$RADAR" \
        --model /cache/zhangjing/models/Mistral-7B-v0.1 \
        --label "Mistral-7B-base" \
        --gpu-id 0 \
        > "$LOG_DIR/scan_mistral_base.log" 2>&1 &
    PID1=$!
    echo "  GPU 4: Mistral-7B base         PID=$PID1"
fi

# GPU 5: Mistral instruct
if [ $MISTRAL_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$RADAR" \
        --model /cache/zhangjing/models/Mistral-7B-Instruct-v0.1 \
        --label "Mistral-7B-Instruct" \
        --gpu-id 0 \
        > "$LOG_DIR/scan_mistral_instruct.log" 2>&1 &
    PID2=$!
    echo "  GPU 5: Mistral-7B-Instruct     PID=$PID2"
fi

# GPU 6: Yi base
if [ $YI_BASE_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$RADAR" \
        --model /cache/zhangjing/models/Yi-1.5-6B \
        --label "Yi-1.5-6B-base" \
        --gpu-id 0 \
        > "$LOG_DIR/scan_yi_base.log" 2>&1 &
    PID3=$!
    echo "  GPU 6: Yi-1.5-6B base          PID=$PID3"
fi

# GPU 7: Yi chat
if [ $YI_CHAT_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$RADAR" \
        --model /cache/zhangjing/models/Yi-1.5-6B-Chat \
        --label "Yi-1.5-6B-Chat" \
        --gpu-id 0 \
        > "$LOG_DIR/scan_yi_chat.log" 2>&1 &
    PID4=$!
    echo "  GPU 7: Yi-1.5-6B-Chat          PID=$PID4"
fi

echo "  Waiting for radar scans..."
wait $PID1 $PID2 $PID3 $PID4 2>/dev/null
echo "  All radar scans done! $(date)"

echo ""
echo "===== Phase 2: SVD Weight Comparison — $(date) ====="

# Generic SVD comparison function as a Python script
cat > /tmp/svd_compare.py << 'PYEOF'
import json, time, sys, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

def load_sd(p):
    path = Path(p)
    sd = {}
    # Try safetensors first
    st_files = sorted(path.glob("*.safetensors"))
    if st_files:
        for f in st_files:
            sd.update(load_file(str(f), device="cpu"))
        return sd
    # Fallback to pytorch bins
    bin_files = sorted(path.glob("pytorch_model*.bin"))
    if bin_files:
        for f in bin_files:
            sd.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return sd
    raise FileNotFoundError(f"No model files in {p}")

def classify(name):
    if "lm_head" in name: return "lm_head", None
    if "embed" in name: return "embed", None
    if "norm" in name and "layers" not in name: return "final_norm", None
    layer_idx = None
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i+1 < len(parts):
            try: layer_idx = int(parts[i+1])
            except: pass
    for tag in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
        if tag in name: return tag, layer_idx
    if "norm" in name.lower(): return "layer_norm", layer_idx
    return "other", layer_idx

def run(base_path, inst_path, family, gpu_id, out_dir):
    print(f"  SVD: {family}")
    t0 = time.time()
    sd_b = load_sd(base_path)
    sd_i = load_sd(inst_path)
    common = sorted(set(sd_b) & set(sd_i))
    print(f"    {len(common)} common params")

    by_comp = defaultdict(list)
    by_layer = defaultdict(list)
    svd_data = []
    device = f"cuda:{gpu_id}"
    INTERESTING = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"}

    for i, key in enumerate(common):
        wb, wi = sd_b[key], sd_i[key]
        if wb.shape != wi.shape: continue
        comp, lidx = classify(key)
        base_norm = torch.norm(wb.float()).item()
        diff_norm = torch.norm((wi.float()-wb.float())).item()
        rel = diff_norm/base_norm if base_norm > 0 else 0
        by_comp[comp].append(rel)
        if lidx is not None: by_layer[lidx].append(rel)

        if wb.dim()>=2 and comp in INTERESTING and min(wb.shape)>=64:
            sv_b = torch.linalg.svdvals(wb.float().to(device))[:100].cpu().numpy()
            sv_i = torch.linalg.svdvals(wi.float().to(device))[:100].cpu().numpy()
            torch.cuda.empty_cache()
            k = min(len(sv_b),len(sv_i))
            sv_bn = sv_b[:k]/(sv_b[:k].sum()+1e-12)
            sv_in = sv_i[:k]/(sv_i[:k].sum()+1e-12)
            spec_diff = float(np.linalg.norm(sv_bn-sv_in))
            pr_b = float((sv_b[:k].sum()**2)/((sv_b[:k]**2).sum()+1e-12))
            pr_i = float((sv_i[:k].sum()**2)/((sv_i[:k]**2).sum()+1e-12))
            svd_data.append({"comp":comp,"layer":lidx,"spec_diff":spec_diff,
                           "pr_base":pr_b,"pr_inst":pr_i,
                           "pr_pct":(pr_i-pr_b)/pr_b*100 if pr_b>0 else 0})
        if (i+1)%50==0: print(f"    {i+1}/{len(common)}... ({time.time()-t0:.0f}s)")

    total_b = sum(torch.norm(sd_b[k].float()).item()**2 for k in common)**0.5
    total_d = sum(torch.norm((sd_i[k].float()-sd_b[k].float())).item()**2 for k in common if sd_b[k].shape==sd_i[k].shape)**0.5
    global_rel = total_d/total_b

    comp_sum = {c:{"mean":float(np.mean(v)),"n":len(v)} for c,v in by_comp.items()}
    svd_by_comp = defaultdict(list)
    for r in svd_data: svd_by_comp[r["comp"]].append(r)
    svd_sum = {c:{"mean_spec_diff":float(np.mean([i["spec_diff"] for i in items])),
                   "mean_pr_pct":float(np.mean([i["pr_pct"] for i in items]))}
                for c,items in svd_by_comp.items()}

    result = {"family":family, "global_pct":global_rel*100, "comp":comp_sum, "svd":svd_sum}
    out = Path(out_dir)/f"weight_svd_{family.replace(' ','_')}.json"
    with open(out,"w") as f: json.dump(result, f, indent=2)
    print(f"    Global change: {global_rel*100:.4f}%")
    for c,s in sorted(svd_sum.items(), key=lambda x:-x[1]["mean_spec_diff"]):
        print(f"    {c:15s} spec_diff={s['mean_spec_diff']:.6f} PR_shift={s['mean_pr_pct']:+.2f}%")
    print(f"    Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
PYEOF

# Mistral SVD on GPU 4
if [ $MISTRAL_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=4 $PYTHON /tmp/svd_compare.py \
        /cache/zhangjing/models/Mistral-7B-v0.1 \
        /cache/zhangjing/models/Mistral-7B-Instruct-v0.1 \
        "Mistral-7B" 0 "$LOG_DIR" \
        > "$LOG_DIR/svd_mistral.log" 2>&1
    echo "  Mistral SVD done"
fi

# Yi SVD on GPU 6
if [ $YI_BASE_OK -eq 0 ] && [ $YI_CHAT_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=6 $PYTHON /tmp/svd_compare.py \
        /cache/zhangjing/models/Yi-1.5-6B \
        /cache/zhangjing/models/Yi-1.5-6B-Chat \
        "Yi-1.5-6B" 0 "$LOG_DIR" \
        > "$LOG_DIR/svd_yi.log" 2>&1
    echo "  Yi SVD done"
fi

echo ""
echo "===== Phase 3: Band 0 — $(date) ====="

# Band 0 comparison script
cat > /tmp/band0_compare.py << 'PYEOF'
import gc, json, os, sys, numpy as np, torch
sys.path.insert(0, os.environ.get("SCRIPT_DIR",""))
sys.path.insert(0, os.path.dirname(os.environ.get("SCRIPT_DIR","")))
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers

BAND0 = [
    "The following is an excerpt from a scientific paper on quantum field theory:",
    "In the year 1847, the city of London was experiencing rapid industrial growth.",
    "Chapter 3: The fundamental problem with recursive descent parsing is that",
    "Once upon a time, in a kingdom far beyond the mountains, there lived",
    "Abstract: We present a novel approach to distributed systems that achieves",
    "The mitochondrial electron transport chain consists of four major complexes",
    "WASHINGTON (Reuters) — Federal Reserve officials signaled they would keep",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid =",
    "The history of mathematics traces back to ancient Mesopotamia, where clay tablets",
    "Ingredients: 2 cups flour, 1 cup butter. Instructions: 1. Preheat oven to",
]
INSTR = [
    "What is the capital of France?",
    "Explain quantum entanglement in simple terms.",
    "Write a Python function to reverse a linked list.",
    "List 5 benefits of regular exercise.",
    "What would happen if the speed of light were halved?",
    "Compare TCP and UDP protocols.",
    "Write a haiku about the ocean.",
    "How does photosynthesis work?",
    "Describe supervised vs unsupervised learning.",
    "Translate hello world to Japanese.",
]

def measure_pr(model, tokenizer, prompts, device):
    _, layers, _, _ = find_decoder_layers(model)
    caps = []
    def hook(m,i,o):
        h = o[0] if isinstance(o,tuple) else o
        caps.append(h[:,-1,:].detach().float().cpu().numpy())
    handle = layers[-1].register_forward_hook(hook)
    try:
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=512, padding=False).to(device)
            with torch.no_grad(): model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
    finally: handle.remove()
    if len(caps)<3: return 0.0
    ls = run_pca_layer(np.vstack(caps))
    return float(ls.pr) if ls else 0.0

def scan(path, label, gpu_id):
    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map={"":gpu_id})
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    r = {"label":label, "band0":measure_pr(model,tok,BAND0,device), "instr":measure_pr(model,tok,INSTR,device)}
    r["ratio"] = r["band0"]/r["instr"] if r["instr"]>0 else 0
    del model; gc.collect(); torch.cuda.empty_cache()
    return r

base_path, inst_path, family, gpu_id, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rb = scan(base_path, f"{family}-base", gpu_id)
ri = scan(inst_path, f"{family}-instruct", gpu_id)
print(f"  {family} base:     raw={rb['band0']:.2f} instr={rb['instr']:.2f} ratio={rb['ratio']:.3f}")
print(f"  {family} instruct: raw={ri['band0']:.2f} instr={ri['instr']:.2f} ratio={ri['ratio']:.3f}")
from pathlib import Path
with open(Path(out_dir)/f"band0_{family.replace(' ','_')}.json","w") as f:
    json.dump({"base":rb,"instruct":ri},f,indent=2)
PYEOF

export SCRIPT_DIR="$SCRIPT_DIR"

if [ $MISTRAL_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=4 $PYTHON /tmp/band0_compare.py \
        /cache/zhangjing/models/Mistral-7B-v0.1 \
        /cache/zhangjing/models/Mistral-7B-Instruct-v0.1 \
        "Mistral-7B" 0 "$LOG_DIR" \
        > "$LOG_DIR/band0_mistral.log" 2>&1
    echo "  Mistral Band 0 done"
fi

if [ $YI_BASE_OK -eq 0 ] && [ $YI_CHAT_OK -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=6 $PYTHON /tmp/band0_compare.py \
        /cache/zhangjing/models/Yi-1.5-6B \
        /cache/zhangjing/models/Yi-1.5-6B-Chat \
        "Yi-1.5-6B" 0 "$LOG_DIR" \
        > "$LOG_DIR/band0_yi.log" 2>&1
    echo "  Yi Band 0 done"
fi

echo ""
echo "===== ALL COMPLETE — $(date) ====="
echo ""
echo "Results:"
ls -la "$LOG_DIR"/radar_*Mistral* "$LOG_DIR"/radar_*Yi* "$LOG_DIR"/weight_svd_*.json "$LOG_DIR"/band0_*.json 2>/dev/null
echo ""
echo "=== UNIVERSALITY CHECK ==="
echo "Qwen2.5-7B: already done (see earlier results)"
for f in "$LOG_DIR"/weight_svd_Mistral*.json "$LOG_DIR"/weight_svd_Yi*.json; do
    if [ -f "$f" ]; then
        echo "$(basename $f): $(python3 -c "import json; d=json.load(open('$f')); print(f'global_change={d[\"global_pct\"]:.4f}%')" 2>/dev/null)"
    fi
done

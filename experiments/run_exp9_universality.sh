#!/bin/bash
# Exp 9 Universality: 3-Family Radar + SVD — Overnight Script
# ==============================================================
# Downloads complete → radar scan → SVD weight comparison
# 3 families: Qwen2.5 (done), Llama-3.1, Mistral-7B
#
# GPU 4: Llama-3.1-8B base       (radar)
# GPU 5: Llama-3.1-8B-Instruct   (radar)
# GPU 6: Mistral-7B base         (radar)
# GPU 7: Mistral-7B-Instruct     (radar)
# Then: SVD comparison on GPU 6 (sequential)

set -e
PYTHON="/cache/zhangjing/miniconda3/envs/sde_scan/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RADAR="$SCRIPT_DIR/exp9_radar_scan.py"
LOG_DIR="$SCRIPT_DIR/results/exp9_radar"
mkdir -p "$LOG_DIR"

export HF_ENDPOINT="https://hf-mirror.com"

echo "===== Exp 9 Universality — $(date) ====="

# ── Wait for downloads ──
echo "Waiting for Llama-3.1-8B-Instruct..."
while [ ! -f "/cache/zhangjing/models/Llama-3.1-8B-Instruct/config.json" ]; do
    sleep 30
    echo "  Still waiting... $(date +%H:%M:%S)"
done
echo "  Llama-3.1-8B-Instruct ready!"

echo "Waiting for Mistral-7B-Instruct-v0.1..."
while [ ! -f "/cache/zhangjing/models/Mistral-7B-Instruct-v0.1/config.json" ]; do
    sleep 30
    echo "  Still waiting... $(date +%H:%M:%S)"
done

# Extra wait to ensure all shards are fully written
echo "  Both models detected. Waiting 60s for file system sync..."
sleep 60

echo ""
echo "===== Phase 1: 7-Band Radar Scans — $(date) ====="

# GPU 4: Llama base
CUDA_VISIBLE_DEVICES=4 nohup $PYTHON "$RADAR" \
    --model /cache/zhangjing/models/Llama-3.1-8B \
    --label "Llama-3.1-8B-base" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_llama_base.log" 2>&1 &
PID1=$!
echo "  GPU 4: Llama-3.1-8B base         PID=$PID1"

# GPU 5: Llama instruct
CUDA_VISIBLE_DEVICES=5 nohup $PYTHON "$RADAR" \
    --model /cache/zhangjing/models/Llama-3.1-8B-Instruct \
    --label "Llama-3.1-8B-Instruct" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_llama_instruct.log" 2>&1 &
PID2=$!
echo "  GPU 5: Llama-3.1-8B-Instruct     PID=$PID2"

# GPU 6: Mistral base
CUDA_VISIBLE_DEVICES=6 nohup $PYTHON "$RADAR" \
    --model /cache/zhangjing/models/Mistral-7B-v0.1 \
    --label "Mistral-7B-base" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_mistral_base.log" 2>&1 &
PID3=$!
echo "  GPU 6: Mistral-7B base           PID=$PID3"

# GPU 7: Mistral instruct
CUDA_VISIBLE_DEVICES=7 nohup $PYTHON "$RADAR" \
    --model /cache/zhangjing/models/Mistral-7B-Instruct-v0.1 \
    --label "Mistral-7B-Instruct" \
    --gpu-id 0 \
    > "$LOG_DIR/scan_mistral_instruct.log" 2>&1 &
PID4=$!
echo "  GPU 7: Mistral-7B-Instruct       PID=$PID4"

echo ""
echo "  Radar scans launched. Waiting for all to finish..."
wait $PID1 $PID2 $PID3 $PID4
echo "  All radar scans complete! $(date)"

echo ""
echo "===== Phase 2: SVD Weight Comparison — $(date) ====="

# Llama SVD on GPU 4
CUDA_VISIBLE_DEVICES=4 $PYTHON -c "
import json, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

RESULTS_DIR = Path('$LOG_DIR')

def load_sd(p):
    sd = {}
    for f in sorted(Path(p).glob('*.safetensors')):
        sd.update(load_file(str(f), device='cpu'))
    return sd

def classify(name):
    if 'lm_head' in name: return 'lm_head', None
    if 'embed' in name: return 'embed', None
    if 'norm' in name and 'layers' not in name: return 'final_norm', None
    layer_idx = None
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            try: layer_idx = int(parts[i+1])
            except: pass
    for tag in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
        if tag in name: return tag, layer_idx
    if 'norm' in name.lower(): return 'layer_norm', layer_idx
    return 'other', layer_idx

def run_svd_comparison(base_path, instruct_path, family_name, gpu_id=0):
    print(f'  SVD: {family_name}')
    t0 = time.time()
    sd_b = load_sd(base_path)
    sd_i = load_sd(instruct_path)
    common = sorted(set(sd_b) & set(sd_i))
    print(f'    {len(common)} common params')

    by_comp = defaultdict(list)
    by_layer = defaultdict(list)
    svd_data = []
    device = f'cuda:{gpu_id}'

    INTERESTING = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'}

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
            svd_data.append({
                'key':key, 'comp':comp, 'layer':lidx,
                'rel_frob':rel, 'spectral_diff':spec_diff,
                'pr_base':pr_b, 'pr_instruct':pr_i,
                'pr_change_pct':(pr_i-pr_b)/pr_b*100 if pr_b>0 else 0,
            })
        if (i+1)%50==0: print(f'    {i+1}/{len(common)}... ({time.time()-t0:.0f}s)')

    total_b = sum(torch.norm(sd_b[k].float()).item()**2 for k in common)**0.5
    total_d = sum(torch.norm((sd_i[k].float()-sd_b[k].float())).item()**2 for k in common if sd_b[k].shape==sd_i[k].shape)**0.5
    global_rel = total_d/total_b

    # Summaries
    comp_sum = {c:{'mean':float(np.mean(v)),'max':float(np.max(v)),'n':len(v)} for c,v in by_comp.items()}
    layer_sum = {str(l):{'mean':float(np.mean(v))} for l,v in sorted(by_layer.items())}
    svd_by_comp = defaultdict(list)
    for r in svd_data: svd_by_comp[r['comp']].append(r)
    svd_sum = {c:{'mean_spec_diff':float(np.mean([i['spectral_diff'] for i in items])),
                   'mean_pr_change':float(np.mean([i['pr_change_pct'] for i in items]))}
                for c,items in svd_by_comp.items()}

    result = {
        'family': family_name,
        'global_rel_change': global_rel,
        'global_rel_change_pct': global_rel*100,
        'component_summary': comp_sum,
        'layer_summary': layer_sum,
        'svd_summary': svd_sum,
    }
    out = RESULTS_DIR / f'weight_svd_{family_name.replace(\" \",\"_\").replace(\"-\",\"_\")}.json'
    with open(out,'w') as f: json.dump(result, f, indent=2)
    print(f'    Global change: {global_rel*100:.4f}%')
    for c,s in sorted(svd_sum.items(), key=lambda x:-x[1]['mean_spec_diff']):
        print(f'    {c:15s} spec_diff={s[\"mean_spec_diff\"]:.6f} PR_shift={s[\"mean_pr_change\"]:+.2f}%')
    print(f'    Done in {time.time()-t0:.0f}s → {out}')
    return result

# Run both
run_svd_comparison(
    '/cache/zhangjing/models/Llama-3.1-8B',
    '/cache/zhangjing/models/Llama-3.1-8B-Instruct',
    'Llama-3.1-8B', gpu_id=0
)
" > "$LOG_DIR/svd_llama.log" 2>&1
echo "  Llama SVD done"

# Mistral SVD on GPU 6
CUDA_VISIBLE_DEVICES=6 $PYTHON -c "
import json, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

RESULTS_DIR = Path('$LOG_DIR')

def load_sd(p):
    sd = {}
    for f in sorted(Path(p).glob('*.safetensors')):
        sd.update(load_file(str(f), device='cpu'))
    return sd

def classify(name):
    if 'lm_head' in name: return 'lm_head', None
    if 'embed' in name: return 'embed', None
    if 'norm' in name and 'layers' not in name: return 'final_norm', None
    layer_idx = None
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            try: layer_idx = int(parts[i+1])
            except: pass
    for tag in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
        if tag in name: return tag, layer_idx
    if 'norm' in name.lower(): return 'layer_norm', layer_idx
    return 'other', layer_idx

def run_svd_comparison(base_path, instruct_path, family_name, gpu_id=0):
    print(f'  SVD: {family_name}')
    t0 = time.time()
    sd_b = load_sd(base_path)
    sd_i = load_sd(instruct_path)
    common = sorted(set(sd_b) & set(sd_i))
    print(f'    {len(common)} common params')

    by_comp = defaultdict(list)
    by_layer = defaultdict(list)
    svd_data = []
    device = f'cuda:{gpu_id}'

    INTERESTING = {'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'}

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
            svd_data.append({
                'key':key, 'comp':comp, 'layer':lidx,
                'rel_frob':rel, 'spectral_diff':spec_diff,
                'pr_base':pr_b, 'pr_instruct':pr_i,
                'pr_change_pct':(pr_i-pr_b)/pr_b*100 if pr_b>0 else 0,
            })
        if (i+1)%50==0: print(f'    {i+1}/{len(common)}... ({time.time()-t0:.0f}s)')

    total_b = sum(torch.norm(sd_b[k].float()).item()**2 for k in common)**0.5
    total_d = sum(torch.norm((sd_i[k].float()-sd_b[k].float())).item()**2 for k in common if sd_b[k].shape==sd_i[k].shape)**0.5
    global_rel = total_d/total_b

    comp_sum = {c:{'mean':float(np.mean(v)),'max':float(np.max(v)),'n':len(v)} for c,v in by_comp.items()}
    layer_sum = {str(l):{'mean':float(np.mean(v))} for l,v in sorted(by_layer.items())}
    svd_by_comp = defaultdict(list)
    for r in svd_data: svd_by_comp[r['comp']].append(r)
    svd_sum = {c:{'mean_spec_diff':float(np.mean([i['spectral_diff'] for i in items])),
                   'mean_pr_change':float(np.mean([i['pr_change_pct'] for i in items]))}
                for c,items in svd_by_comp.items()}

    result = {
        'family': family_name,
        'global_rel_change': global_rel,
        'global_rel_change_pct': global_rel*100,
        'component_summary': comp_sum,
        'layer_summary': layer_sum,
        'svd_summary': svd_sum,
    }
    out = RESULTS_DIR / f'weight_svd_{family_name.replace(\" \",\"_\").replace(\"-\",\"_\")}.json'
    with open(out,'w') as f: json.dump(result, f, indent=2)
    print(f'    Global change: {global_rel*100:.4f}%')
    for c,s in sorted(svd_sum.items(), key=lambda x:-x[1]['mean_spec_diff']):
        print(f'    {c:15s} spec_diff={s[\"mean_spec_diff\"]:.6f} PR_shift={s[\"mean_pr_change\"]:+.2f}%')
    print(f'    Done in {time.time()-t0:.0f}s → {out}')
    return result

run_svd_comparison(
    '/cache/zhangjing/models/Mistral-7B-v0.1',
    '/cache/zhangjing/models/Mistral-7B-Instruct-v0.1',
    'Mistral-7B', gpu_id=0
)
" > "$LOG_DIR/svd_mistral.log" 2>&1
echo "  Mistral SVD done"

echo ""
echo "===== Phase 3: Band 0 Comparison — $(date) ====="

# Band 0 for Llama on GPU 4
CUDA_VISIBLE_DEVICES=4 $PYTHON -c "
import gc, json, os, sys, time, numpy as np, torch
sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers
from pathlib import Path

RESULTS_DIR = Path('$LOG_DIR')

BAND0 = [
    'The following is an excerpt from a scientific paper on quantum field theory published in Physical Review Letters:',
    'In the year 1847, the city of London was experiencing rapid industrial growth. The factories along the Thames',
    'Chapter 3: The Algorithm. The fundamental problem with recursive descent parsing is that',
    'Once upon a time, in a kingdom far beyond the mountains where the rivers flow upward, there lived',
    'Abstract: We present a novel approach to large-scale distributed systems that achieves consensus in',
    'The mitochondrial electron transport chain consists of four major protein complexes embedded in the inner',
    'WASHINGTON (Reuters) — Federal Reserve officials on Wednesday signaled they would keep interest rates',
    'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left =',
    'The history of mathematics can be traced back to ancient Mesopotamia, where clay tablets from around 1800 BCE',
    'Ingredients: 2 cups flour, 1 cup butter, 3/4 cup sugar. Instructions: 1.',
]
INSTR = [
    'What is the capital of France?',
    'Explain quantum entanglement in simple terms.',
    'Write a Python function to reverse a linked list.',
    'List 5 benefits of regular exercise.',
    'Translate hello world to Japanese.',
    'What would happen if the speed of light were halved?',
    'Compare and contrast TCP and UDP protocols.',
    'Write a haiku about the ocean.',
    'How does photosynthesis work?',
    'Describe the difference between supervised and unsupervised learning.',
]

def measure_pr(model, tokenizer, prompts, device):
    _, layers, _, _ = find_decoder_layers(model)
    caps = []
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        caps.append(h[:,-1,:].detach().float().cpu().numpy())
    handle = layers[-1].register_forward_hook(hook)
    try:
        for p in prompts:
            enc = tokenizer(p, return_tensors='pt', truncation=True, max_length=512, padding=False).to(device)
            with torch.no_grad(): model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
    finally: handle.remove()
    if len(caps)<3: return 0.0
    ls = run_pca_layer(np.vstack(caps))
    return float(ls.pr) if ls else 0.0

def scan(path, label, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':gpu_id})
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    r = {'label': label, 'band0_raw_completion': measure_pr(model, tok, BAND0, device), 'band_instruction': measure_pr(model, tok, INSTR, device)}
    r['ratio'] = r['band0_raw_completion']/r['band_instruction'] if r['band_instruction']>0 else 0
    del model; gc.collect(); torch.cuda.empty_cache()
    return r

rb = scan('/cache/zhangjing/models/Llama-3.1-8B', 'Llama-3.1-8B-base', 0)
ri = scan('/cache/zhangjing/models/Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct', 0)
print(f'Llama base:     raw={rb[\"band0_raw_completion\"]:.2f}  instr={rb[\"band_instruction\"]:.2f}  ratio={rb[\"ratio\"]:.3f}')
print(f'Llama instruct: raw={ri[\"band0_raw_completion\"]:.2f}  instr={ri[\"band_instruction\"]:.2f}  ratio={ri[\"ratio\"]:.3f}')
with open(RESULTS_DIR/'band0_llama.json','w') as f: json.dump({'base':rb,'instruct':ri},f,indent=2)
" > "$LOG_DIR/band0_llama.log" 2>&1
echo "  Llama Band 0 done"

# Band 0 for Mistral on GPU 6
CUDA_VISIBLE_DEVICES=6 $PYTHON -c "
import gc, json, os, sys, time, numpy as np, torch
sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$(dirname $SCRIPT_DIR)')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModelForCausalLM, AutoTokenizer
from spectral_flow_probe.core import run_pca_layer
from spectral_flow_probe._compat import find_decoder_layers
from pathlib import Path

RESULTS_DIR = Path('$LOG_DIR')

BAND0 = [
    'The following is an excerpt from a scientific paper on quantum field theory published in Physical Review Letters:',
    'In the year 1847, the city of London was experiencing rapid industrial growth. The factories along the Thames',
    'Chapter 3: The Algorithm. The fundamental problem with recursive descent parsing is that',
    'Once upon a time, in a kingdom far beyond the mountains where the rivers flow upward, there lived',
    'Abstract: We present a novel approach to large-scale distributed systems that achieves consensus in',
    'The mitochondrial electron transport chain consists of four major protein complexes embedded in the inner',
    'WASHINGTON (Reuters) — Federal Reserve officials on Wednesday signaled they would keep interest rates',
    'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left =',
    'The history of mathematics can be traced back to ancient Mesopotamia, where clay tablets from around 1800 BCE',
    'Ingredients: 2 cups flour, 1 cup butter, 3/4 cup sugar. Instructions: 1.',
]
INSTR = [
    'What is the capital of France?',
    'Explain quantum entanglement in simple terms.',
    'Write a Python function to reverse a linked list.',
    'List 5 benefits of regular exercise.',
    'Translate hello world to Japanese.',
    'What would happen if the speed of light were halved?',
    'Compare and contrast TCP and UDP protocols.',
    'Write a haiku about the ocean.',
    'How does photosynthesis work?',
    'Describe the difference between supervised and unsupervised learning.',
]

def measure_pr(model, tokenizer, prompts, device):
    _, layers, _, _ = find_decoder_layers(model)
    caps = []
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        caps.append(h[:,-1,:].detach().float().cpu().numpy())
    handle = layers[-1].register_forward_hook(hook)
    try:
        for p in prompts:
            enc = tokenizer(p, return_tensors='pt', truncation=True, max_length=512, padding=False).to(device)
            with torch.no_grad(): model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'])
    finally: handle.remove()
    if len(caps)<3: return 0.0
    ls = run_pca_layer(np.vstack(caps))
    return float(ls.pr) if ls else 0.0

def scan(path, label, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':gpu_id})
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    r = {'label': label, 'band0_raw_completion': measure_pr(model, tok, BAND0, device), 'band_instruction': measure_pr(model, tok, INSTR, device)}
    r['ratio'] = r['band0_raw_completion']/r['band_instruction'] if r['band_instruction']>0 else 0
    del model; gc.collect(); torch.cuda.empty_cache()
    return r

rb = scan('/cache/zhangjing/models/Mistral-7B-v0.1', 'Mistral-7B-base', 0)
ri = scan('/cache/zhangjing/models/Mistral-7B-Instruct-v0.1', 'Mistral-7B-Instruct', 0)
print(f'Mistral base:     raw={rb[\"band0_raw_completion\"]:.2f}  instr={rb[\"band_instruction\"]:.2f}  ratio={rb[\"ratio\"]:.3f}')
print(f'Mistral instruct: raw={ri[\"band0_raw_completion\"]:.2f}  instr={ri[\"band_instruction\"]:.2f}  ratio={ri[\"ratio\"]:.3f}')
with open(RESULTS_DIR/'band0_mistral.json','w') as f: json.dump({'base':rb,'instruct':ri},f,indent=2)
" > "$LOG_DIR/band0_mistral.log" 2>&1
echo "  Mistral Band 0 done"

echo ""
echo "===== ALL COMPLETE — $(date) ====="
echo "Results in: $LOG_DIR"
echo ""
echo "Files generated:"
ls -la "$LOG_DIR"/radar_Llama* "$LOG_DIR"/radar_Mistral* "$LOG_DIR"/weight_svd_*.json "$LOG_DIR"/band0_*.json 2>/dev/null

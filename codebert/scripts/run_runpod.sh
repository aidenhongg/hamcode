#!/usr/bin/env bash
# Runpod 4090 bootstrap: end-to-end pipeline from raw fetch to final binary head.
#
# Layout assumption: this script lives at codebert/scripts/run_runpod.sh and
# is invoked from the codebert/ directory. Override DATA_DIR / RUN_ROOT to
# place artifacts elsewhere.
#
# Phases:
#   0. install deps
#   1. fetch sources (leetcode, kamyu, codecomplex, mbxp)
#   2. parse + normalize + filter + balance + split
#   3. emit pointwise + pairwise parquets + audit
#   4. Phase A: full FT LongCoder
#   5. Phase B: per-language LoRA (11 runs)
#   6. Phase C: AST features + LoRA pointwise feature extraction
#   7. Phase D: binary stacking head training
#
# Usage:
#   bash scripts/run_runpod.sh              # full run
#   STAGE=lora bash scripts/run_runpod.sh   # skip earlier stages
#
# Environment:
#   DATA_DIR=data                   (default)
#   RUN_ROOT=runs                   (default)
#   STAGE={all,fetch,parse,fullft,lora,extract,head}  (default: all)
#   LANGUAGES="python java cpp ..." (default: all 12)

set -euo pipefail

DATA_DIR=${DATA_DIR:-data}
RUN_ROOT=${RUN_ROOT:-runs}
STAGE=${STAGE:-all}

LANGUAGES=${LANGUAGES:-"python java cpp c csharp go javascript typescript php ruby rust swift"}

TS=$(date +%Y%m%d-%H%M%S)
FULLFT_DIR=${FULLFT_DIR:-${RUN_ROOT}/multi-fullft-${TS}}
LORA_DIR=${LORA_DIR:-${RUN_ROOT}/lora-${TS}}
HEAD_EXTRACTION=${HEAD_EXTRACTION:-${RUN_ROOT}/heads/extraction}
HEAD_RUN=${HEAD_RUN:-${RUN_ROOT}/heads/binary-${TS}}

# A bit of resilience: print + exit on any unset variable
echo "[runpod] STAGE=${STAGE} TS=${TS}"
echo "[runpod] FULLFT_DIR=${FULLFT_DIR}"
echo "[runpod] LORA_DIR=${LORA_DIR}"
echo "[runpod] LANGUAGES=${LANGUAGES}"

stage_at_or_after() {
    local target=$1
    local stages=(all fetch parse fullft lora extract head)
    local stage_idx=-1
    local target_idx=-1
    for i in "${!stages[@]}"; do
        if [[ "${stages[i]}" == "${STAGE}" ]]; then stage_idx=$i; fi
        if [[ "${stages[i]}" == "${target}" ]]; then target_idx=$i; fi
    done
    if [[ ${stage_idx} -le ${target_idx} ]]; then return 0; else return 1; fi
}

# ---------------------------------------------------------------- Phase 0: deps
# Runs unconditionally regardless of STAGE — fresh Runpod pods don't have these,
# and even on resumed pods this is idempotent (pip and apt skip what's installed).
echo
echo "=== [0a] system packages ==="
SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo"; fi
if command -v apt-get >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get update -y -qq
    DEBIAN_FRONTEND=noninteractive ${SUDO} apt-get install -y -qq --no-install-recommends \
        git git-lfs curl ca-certificates build-essential pkg-config \
        python3 python3-pip python3-venv
    git lfs install --skip-repo || true
fi

echo
echo "=== [0b] python deps ==="
# Pin pip / wheel / setuptools first (older base images ship pip < 23 which
# can't resolve some recent tree-sitter wheels).
python -m pip install --quiet --upgrade pip wheel setuptools

# CUDA-matched torch from the cu128 index. cu128 wheels run fine on Ada (4090,
# sm_89), Hopper, and Blackwell; if you're on Turing/Volta swap to cu121.
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cu128 torch

# Strip torchvision/torchaudio if the base image bundled them — they almost
# certainly won't match the cu128 torch we just installed and will explode on
# transformers' lazy-vision import.
python -m pip uninstall --quiet -y torchvision torchaudio || true

# Repo deps (peft, tree-sitter, transformers, datasets, etc.)
python -m pip install --quiet -r requirements.txt

echo
echo "=== [0c] pre-warm HF cache (LongCoder backbone + tokenizer) ==="
# Pull the model/tokenizer once so a mid-training network blip can't fail us.
# Idempotent: HF caches under ~/.cache/huggingface/.
python - <<'PY'
import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LongformerModel
name = "microsoft/longcoder-base"
print(f"  tokenizer: {name}")
AutoTokenizer.from_pretrained(name)
print(f"  weights:   {name}")
LongformerModel.from_pretrained(name)
print("  hf cache warm.")
PY

echo
echo "=== [0d] guardrails ==="
# Catch the usual fresh-pod failures BEFORE we burn an hour on Phase A.
python - <<'PY'
import sys, importlib
# 1) CUDA visible to PyTorch
import torch
if not torch.cuda.is_available():
    print(f"FATAL: torch.cuda.is_available() == False (torch={torch.__version__})")
    sys.exit(1)
dev = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  cuda: {dev}  ({mem:.1f} GB)")
# 2) bf16 supported (mandatory for our config)
if not torch.cuda.is_bf16_supported():
    print(f"FATAL: bf16 not supported on {dev}; configs/point.yaml expects it")
    sys.exit(1)
print("  bf16: OK")
# 3) peft + safetensors importable
import peft, safetensors
print(f"  peft={peft.__version__} safetensors={safetensors.__version__}")
# 4) tree-sitter per-language packages all load
sys.path.insert(0, ".")
from common.parsers import _self_test
_self_test()
# 5) AST feature schema didn't drift
from stacking.features.ast_features import N_FEATURES
assert N_FEATURES == 21, N_FEATURES
print("  ast schema: 21 dims OK")
print("  guardrails: PASS")
PY

# ---------------------------------------------------------------- Phase 1: fetch
if stage_at_or_after fetch; then
    echo
    echo "=== [1] fetch raw sources ==="
    python pipeline/01_fetch_sources.py --raw_dir "${DATA_DIR}/raw"
fi

# ---------------------------------------------------------------- Phase 2: parse
if stage_at_or_after parse; then
    echo
    echo "=== [2a] parse leetcode (multi-fence) ==="
    python pipeline/02_parse_leetcode.py \
        --raw_dir "${DATA_DIR}/raw/leetcode" \
        --out "${DATA_DIR}/interim/parsed/leetcode.jsonl" \
        --fail_log "${DATA_DIR}/audit/leetcode_parse_failures.jsonl"

    echo "=== [2b] parse codecomplex (python + java) ==="
    python pipeline/03_parse_codecomplex.py \
        --raw_paths "${DATA_DIR}/raw/codecomplex/python.jsonl" \
                    "${DATA_DIR}/raw/codecomplex/java.jsonl" \
        --out "${DATA_DIR}/interim/parsed/codecomplex.jsonl"

    echo "=== [2c] parse kamyu104 ==="
    python pipeline/12_parse_kamyu.py \
        --raw_dir "${DATA_DIR}/raw/kamyu" \
        --out "${DATA_DIR}/interim/parsed/kamyu.jsonl" \
        --fail_log "${DATA_DIR}/audit/kamyu_parse_failures.jsonl"

    echo "=== [2d] parse MBXP (label transfer from labeled python rows) ==="
    # Anchor uses the just-emitted leetcode.jsonl python rows + codecomplex python.
    python pipeline/13_parse_mbxp.py \
        --raw_dir "${DATA_DIR}/raw/mbxp" \
        --anchor "${DATA_DIR}/interim/parsed/leetcode.jsonl" \
        --out "${DATA_DIR}/interim/parsed/mbxp.jsonl"

    echo "=== [2e] supplemental synthetic templates ==="
    python pipeline/04_parse_supplemental.py \
        --out "${DATA_DIR}/interim/parsed/supplemental.jsonl"

    echo "=== [2f] normalize labels ==="
    python pipeline/05_normalize_labels.py \
        --in_dir "${DATA_DIR}/interim/parsed" \
        --out "${DATA_DIR}/interim/normalized/combined.jsonl"

    echo "=== [2g] dedupe + filter ==="
    python pipeline/06_dedupe_filter.py \
        --in_path "${DATA_DIR}/interim/normalized/combined.jsonl" \
        --out "${DATA_DIR}/interim/filtered.jsonl"

    echo "=== [2h] balance + augment per (lang,label) ==="
    python pipeline/07_balance_augment.py \
        --in_path "${DATA_DIR}/interim/filtered.jsonl" \
        --out "${DATA_DIR}/interim/balanced.jsonl"

    echo "=== [2i] split (stratified by lang,label; problem_id pinned) ==="
    python pipeline/08_split.py \
        --in_path "${DATA_DIR}/interim/balanced.jsonl" \
        --out "${DATA_DIR}/interim/split.jsonl"

    echo "=== [2j] emit pointwise parquet ==="
    python pipeline/09_make_pointwise.py \
        --in_path "${DATA_DIR}/interim/split.jsonl" \
        --out "${DATA_DIR}/processed/pointwise.parquet"

    echo "=== [2k] emit pairwise parquet (within-language only) ==="
    python pipeline/10_make_pairwise.py \
        --in_path "${DATA_DIR}/processed/pointwise.parquet" \
        --out "${DATA_DIR}/processed/pairwise.parquet"

    echo "=== [2l] audit report ==="
    python pipeline/11_audit_report.py \
        --pointwise "${DATA_DIR}/processed/pointwise.parquet" \
        --pairwise "${DATA_DIR}/processed/pairwise.parquet" \
        --out "${DATA_DIR}/audit/stats.json"
fi

# ---------------------------------------------------------------- Phase 3: full FT
if stage_at_or_after fullft; then
    echo
    echo "=== [3] Phase A: full FT LongCoder on combined dataset ==="
    mkdir -p "${FULLFT_DIR}"
    python -u train.py \
        --data_dir "${DATA_DIR}/processed" \
        --output_dir "${FULLFT_DIR}" \
        --num_workers 8 --prefetch_factor 4
    echo "[runpod] Phase A best/ at: ${FULLFT_DIR}/best"
fi

# ---------------------------------------------------------------- Phase 4: per-language LoRA
if stage_at_or_after lora; then
    echo
    echo "=== [4] Phase B: per-language LoRA (11 + python) ==="
    if [[ ! -d "${FULLFT_DIR}/best" ]]; then
        echo "[runpod] FATAL: ${FULLFT_DIR}/best not found. Set FULLFT_DIR or run STAGE=fullft."
        exit 2
    fi
    mkdir -p "${LORA_DIR}"
    for lang in ${LANGUAGES}; do
        echo "[runpod]   --- LoRA[${lang}] ---"
        python -u lora_train.py \
            --base_run "${FULLFT_DIR}/best" \
            --language "${lang}" \
            --data_dir "${DATA_DIR}/processed" \
            --output_root "${LORA_DIR}" \
            --num_workers 4
    done
    echo "[runpod] Phase B bundle: ${LORA_DIR}/{python,java,cpp,...}"

    echo
    echo "=== [4b] consolidated bundle report (Phase A vs Phase B per language) ==="
    python scripts/report_lora_bundle.py \
        --fullft_run "${FULLFT_DIR}" \
        --lora_root  "${LORA_DIR}" \
        --out_dir    "${LORA_DIR}"
fi

# ---------------------------------------------------------------- Phase 5: features
if stage_at_or_after extract; then
    echo
    echo "=== [5a] AST features (multi-language) ==="
    python -m stacking.features.ast_features \
        --in_splits "${DATA_DIR}/processed" \
        --out_dir "${HEAD_EXTRACTION}"

    echo "=== [5b] LoRA pointwise feature extraction ==="
    python -m stacking.features.extract_lora_features \
        --base_run "${FULLFT_DIR}/best" \
        --lora_root "${LORA_DIR}" \
        --in_splits "${DATA_DIR}/processed" \
        --out_dir "${HEAD_EXTRACTION}" \
        --batch 8
fi

# ---------------------------------------------------------------- Phase 6: head
if stage_at_or_after head; then
    echo
    echo "=== [6] Phase D: train binary head ==="
    python -m stacking.train_head \
        --in_dir "${HEAD_EXTRACTION}" \
        --pair_dir "${DATA_DIR}/processed" \
        --out_dir "${HEAD_RUN}" \
        --variant v1
fi

echo
echo "[runpod] === ALL DONE ==="
echo "[runpod] Phase A: ${FULLFT_DIR}/best/"
echo "[runpod] Phase B: ${LORA_DIR}/"
echo "[runpod] Phase C: ${HEAD_EXTRACTION}/"
echo "[runpod] Phase D: ${HEAD_RUN}/"

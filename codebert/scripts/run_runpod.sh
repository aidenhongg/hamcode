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

# Phase-C extraction mode for stacker features:
#   leaky  (default) — extract_lora_features.py: one pass per language using the
#                      full Phase-B LoRA. Train logits are over-confident
#                      because the LoRA was trained on those rows. Cheap (~10
#                      min total). Stacker train metrics will be inflated.
#   oof              — oof_lora.py: K-fold per language. Trains K LoRAs per
#                      language to produce out-of-fold train logits with no
#                      train-set leakage. K=3 (default) ~= 18 h on a 4090;
#                      K=5 ~= 30 h. Recommended only when binary-head
#                      generalization is the deliverable.
EXTRACT_MODE=${EXTRACT_MODE:-leaky}
OOF_FOLDS=${OOF_FOLDS:-3}

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

# Fail fast with a clear next-step suggestion when a stage is asked to run
# but its prereqs aren't on disk. Better than letting train.py blow up
# 200 lines deep in pyarrow.
require_file() {
    local path=$1
    local what=$2
    local fix=$3
    if [[ ! -e "${path}" ]]; then
        echo "[runpod] FATAL: missing prereq for STAGE=${STAGE}"
        echo "[runpod]   expected: ${path}  (${what})"
        echo "[runpod]   fix: ${fix}"
        exit 2
    fi
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
# Pin pip / wheel only. Don't upgrade setuptools — torch 2.10+ caps it at <82,
# and the base image's setuptools (typically 65–70) already satisfies that.
# Upgrading to the latest setuptools (82+) silently breaks torch's dep resolver
# and leaves you with a half-installed environment.
python -m pip install --quiet --upgrade pip wheel

# CUDA-matched torch, pinned to torch==2.6.0+cu124. This is the unique sweet
# spot satisfying every constraint in this stack:
#
#   1. Driver compat: cu124 wheel runs on driver >= 12.4 (Runpod 4090 default).
#      cu128 wheels are off the table — that index serves +cu130 builds at HEAD
#      which need driver 12.8+.
#
#   2. transformers >= 4.51 (Apr 2025) added check_torch_load_is_safe() in
#      response to CVE-2025-32434. It refuses to load pytorch_model.bin under
#      torch < 2.6 even with weights_only=True. microsoft/longcoder-base is
#      shipped as .bin (no safetensors mirror on HF), so we MUST be on >= 2.6.
#
#   3. setuptools compat: torch 2.10+ caps setuptools<82 which fights modern
#      pip envs. 2.6.0 predates the cap.
#
#   4. NCCL ABI: torch 2.7 upgraded its bundled NCCL to 2.21+, which calls
#      ncclCommWindowDeregister. If a CUDA base image ships an older system
#      libnccl that gets shadow-loaded ahead of torch's bundled one, that
#      symbol is missing at import time. torch 2.6.0 bundles NCCL 2.20.x and
#      doesn't reference the new symbols.
#
# --force-reinstall guards against any previously-broken torch from prior
# bootstrap attempts (e.g. 2.11+cu130 or 2.4.1).
python -m pip install --quiet --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0"

# Strip torchvision/torchaudio if the base image bundled them — they almost
# certainly won't match the cu124 torch we just installed and will explode on
# transformers' lazy-vision import.
python -m pip uninstall --quiet -y torchvision torchaudio || true

# Belt-and-braces: ensure torch's bundled libnccl/libcudart are preferred
# over any system libs the CUDA base image leaves on the loader path. This
# only matters if a future torch bump drags NCCL forward; harmless on 2.6.0.
TORCH_LIB="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__) + "/lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "  LD_LIBRARY_PATH prepended with: ${TORCH_LIB}"

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
    require_file "${DATA_DIR}/processed/train.parquet" "Phase 1+2 output" \
        "drop the STAGE flag (\`bash $0\`) to run fetch+parse first, or run STAGE=parse"
    require_file "${DATA_DIR}/processed/val.parquet"   "Phase 1+2 output" \
        "drop the STAGE flag (\`bash $0\`) to run fetch+parse first, or run STAGE=parse"
    require_file "${DATA_DIR}/processed/test.parquet"  "Phase 1+2 output" \
        "drop the STAGE flag (\`bash $0\`) to run fetch+parse first, or run STAGE=parse"
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
    require_file "${FULLFT_DIR}/best/codebert_meta.json" "Phase A best/ checkpoint" \
        "set FULLFT_DIR=runs/multi-fullft-<ts> to point at an existing run, or STAGE=fullft to train Phase A"
    require_file "${DATA_DIR}/processed/train.parquet" "Phase 1+2 output" \
        "drop the STAGE flag to run fetch+parse, or STAGE=parse"
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
    require_file "${DATA_DIR}/processed/train.parquet" "Phase 1+2 output" \
        "drop the STAGE flag to run fetch+parse, or STAGE=parse"
    python -m stacking.features.ast_features \
        --in_splits "${DATA_DIR}/processed" \
        --out_dir "${HEAD_EXTRACTION}"

    echo "=== [5b] LoRA pointwise feature extraction (mode=${EXTRACT_MODE}) ==="
    require_file "${FULLFT_DIR}/best/codebert_meta.json" "Phase A best/ checkpoint" \
        "set FULLFT_DIR=runs/multi-fullft-<ts> or STAGE=fullft"
    require_file "${LORA_DIR}/python" "Phase B LoRA bundle" \
        "set LORA_DIR=runs/lora-<ts> or STAGE=lora"
    case "${EXTRACT_MODE}" in
        leaky)
            python -m stacking.features.extract_lora_features \
                --base_run "${FULLFT_DIR}/best" \
                --lora_root "${LORA_DIR}" \
                --in_splits "${DATA_DIR}/processed" \
                --out_dir "${HEAD_EXTRACTION}" \
                --batch 8
            ;;
        oof)
            echo "[runpod] OOF mode: K=${OOF_FOLDS} folds per language."
            echo "[runpod] Expect ~$((OOF_FOLDS * 6)) h total on a 4090."
            python -m stacking.features.oof_lora \
                --base_run "${FULLFT_DIR}/best" \
                --full_lora_root "${LORA_DIR}" \
                --in_splits "${DATA_DIR}/processed" \
                --out_dir "${HEAD_EXTRACTION}" \
                --n_folds "${OOF_FOLDS}" \
                --batch 8 \
                --resume
            ;;
        *)
            echo "[runpod] FATAL: EXTRACT_MODE='${EXTRACT_MODE}' (expected 'leaky' or 'oof')"
            exit 2
            ;;
    esac
fi

# ---------------------------------------------------------------- Phase 6: head
if stage_at_or_after head; then
    echo
    echo "=== [6] Phase D: train binary head ==="
    require_file "${HEAD_EXTRACTION}/point_logits_train.parquet" "Phase C extraction output" \
        "STAGE=extract first"
    require_file "${DATA_DIR}/processed/pair_train.parquet" "pairwise parquet" \
        "drop the STAGE flag to run fetch+parse, or STAGE=parse"
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

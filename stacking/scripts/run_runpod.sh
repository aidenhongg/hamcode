#!/usr/bin/env bash
# End-to-end stacking experiment on a fresh RunPod RTX 5090 pod.
#
# Assumes: CUDA 12.8+ torch already installed (see requirements.txt note),
# this repo checked out to the pod, and you are inside codebert/.
#
# What it does:
#   1. Build the dataset (run_pipeline.sh)
#   2. Extract AST features (CPU, ~15s)
#   3. OOF pointwise: K-fold train pointwise BERT + emit unbiased train logits,
#      then train full-train pointwise model and emit val/test logits + CLS.
#   4. Semantic similarity from CLS (CPU, ~1s)
#   5. Run the head sweep (top-4 heads x 3 seeds = 12 experiments).
#
# Pairwise BERT fine-tuning was retired — heads now consume only pointwise
# logits + AST diffs + CLS similarity.
#
# Total GPU time on 5090 ballpark (dataset ~5K point):
#   - OOF point:  5 folds x ~10min train + extraction = ~60min
#   - Per-language head sweep: ~30-60 min CPU
#     (12 languages x 4 heads x 3 seeds; CPU-bound)
#   ≈ 1.5-2 hours wallclock for the full pipeline.
#
# Usage:
#   bash stacking/scripts/run_runpod.sh [--skip-data] [--n_folds 5]
#
# Flags:
#   --skip-data     : dataset already built under data/processed/
#   --skip-encoder-sweep : skip the encoder sweep (OOF + semantic + head sweep)
#   --n_folds N     : number of folds (default 5)
#   --extraction_dir DIR : override output root (default runs/heads/extraction)

set -euo pipefail

# ---------------------------------------------------------------------------
# Kill transformers' background safetensors-conversion thread.
# Root cause: transformers spawns Thread-auto_conversion that HEAD-requests
# model.safetensors on every open PR ref (refs/pr/N). For models that ship
# only pytorch_model.bin (microsoft/longcoder-base does), PR-ref LFS redirects
# can hang indefinitely and block training on network I/O.
#
# - DISABLE_SAFETENSORS_CONVERSION=1 short-circuits the thread entirely
#   (see transformers/modeling_utils.py:_get_resolved_checkpoint_files,
#   can_auto_convert gate).
# - HF_HUB_DOWNLOAD_TIMEOUT=30 bounds any residual HEAD/GET — default is 10s
#   but LFS redirects have no upper bound without this.
# These must be exported BEFORE any python process that imports transformers.
# ---------------------------------------------------------------------------
export DISABLE_SAFETENSORS_CONVERSION=1
export HF_HUB_DOWNLOAD_TIMEOUT=30

SKIP_DATA=0
SKIP_ENCODER_SWEEP=0
N_FOLDS=5
ENCODER_CONFIG="stacking/configs/encoder_sweep.yaml"
EXTRACTION_DIR="runs/heads/extraction"
OUT_ROOT="runs/heads"

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-data) SKIP_DATA=1 ;;
        --skip-encoder-sweep) SKIP_ENCODER_SWEEP=1 ;;
        --encoder-config) ENCODER_CONFIG="$2"; shift ;;
        --n_folds) N_FOLDS="$2"; shift ;;
        --extraction_dir) EXTRACTION_DIR="$2"; shift ;;
        --out_root) OUT_ROOT="$2"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$EXTRACTION_DIR" "$OUT_ROOT"

echo "=== [0/4] Environment sanity ==="
# torchvision / torchaudio preinstalled on the base image nearly always mismatch
# the cu128 torch we install on top, causing transformers' lazy-vision import
# to crash with "operator torchvision::nms does not exist". We never use them —
# remove them so transformers' optional-dep probe gracefully skips vision.
if python -c "import torchvision" 2>/dev/null; then
    echo "  purging torchvision/torchaudio (not used; avoids ABI mismatch with cu128 torch)"
    pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true
fi

echo "=== [0.5/4] GPU + transformers sanity (still ONLINE for pre-download) ==="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, 'bf16:', torch.cuda.is_bf16_supported())"

echo "=== [1/4] Pre-download microsoft/longcoder-base from main (no PR refs) ==="
# snapshot_download walks repo_id@revision and fetches only the allow-listed
# files. It never falls through to PR refs — that fallback is strictly a
# transformers.from_pretrained behaviour, not huggingface_hub's.
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    'microsoft/longcoder-base',
    revision='main',
    allow_patterns=[
        'config.json',
        'pytorch_model.bin',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
        'merges.txt',
        'special_tokens_map.json',
        'added_tokens.json',
    ],
)
print('cached at', p)
"

# Lock the rest of the pipeline into offline mode.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== [1.5/4] Offline-mode smoke: load encoder from cache only ==="
python -c "
from transformers import AutoTokenizer, LongformerModel
_t = AutoTokenizer.from_pretrained('microsoft/longcoder-base')
_m = LongformerModel.from_pretrained('microsoft/longcoder-base')
print('encoder + tokenizer load OK from cache (offline)')
"

if [ "$SKIP_DATA" -eq 0 ]; then
    echo "=== [2/4] Building dataset ==="
    bash run_pipeline.sh
else
    echo "=== [2/4] Skipping dataset build (--skip-data) ==="
fi

echo "=== [3/4] AST feature extraction ==="
python -m stacking.features.ast_features \
    --in_splits data/processed \
    --out_dir "$EXTRACTION_DIR"

if [ "$SKIP_ENCODER_SWEEP" -eq 0 ]; then
    echo "=== [4/4] Encoder recipe sweep ($ENCODER_CONFIG) ==="
    # Iterates each recipe: (optional cache prewarm) -> OOF -> semantic -> head sweep.
    # Resume-aware: recipes whose <out_root>/<recipe>/SUMMARY.md exists are skipped.
    python -m stacking.encoder_sweep \
        --config "$ENCODER_CONFIG" \
        --data_dir data/processed \
        --out_root "$OUT_ROOT" \
        --extraction_root "$EXTRACTION_DIR"
else
    echo "=== [4/4] Skipping encoder sweep (--skip-encoder-sweep) ==="
fi

echo ""
echo "=== DONE ==="
echo "Encoder sweep:        $OUT_ROOT/ENCODER_SUMMARY.md"
echo "Encoder winner:       $OUT_ROOT/ENCODER_WINNER.json"
echo "Per-recipe heads:     $OUT_ROOT/<recipe>/PER_LANGUAGE_SUMMARY.md"
echo "Per-recipe flat:      $OUT_ROOT/<recipe>/SUMMARY.md"
echo "OOF artifacts:        $EXTRACTION_DIR/<recipe>/oof/"

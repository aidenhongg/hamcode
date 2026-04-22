#!/usr/bin/env bash
# End-to-end stacking experiment on a fresh RunPod RTX 5090 pod.
#
# Assumes: CUDA 12.8+ torch already installed (see requirements.txt note),
# this repo checked out to the pod, and you are inside codebert/.
#
# What it does (leakage-fixed, unlike the legacy path):
#   1. Build the dataset (run_pipeline.sh)
#   2. Extract AST features (CPU, ~15s)
#   3. OOF pointwise: K-fold train pointwise BERT + emit unbiased train logits,
#      then train full-train pointwise model and emit val/test logits + CLS.
#      (Replaces legacy bert_logits --point which leaks train features.)
#   4. Semantic similarity from CLS (CPU, ~1s)
#   5. OOF pairwise: same idea for the pair BERT.
#      (Replaces legacy bert_logits --pair which leaks train features.)
#   6. Run the 72-experiment head sweep.
#
# Total GPU time on 5090 ballpark (dataset ~5K point / ~30K pair):
#   - OOF point:  5 folds x ~10min train + extraction = ~60min
#   - OOF pair:   5 folds x ~25min train + extraction = ~140min
#   - Sweep:      ~10min CPU
#   ≈ 3.5 hours wallclock.
#
# Usage:
#   bash stacking/scripts/run_runpod.sh [--skip-data] [--skip-oof-pair] [--n_folds 5]
#
# Flags:
#   --skip-data     : dataset already built under data/processed/
#   --skip-oof-point: OOF pointwise already done (resume into pair + sweep)
#   --skip-oof-pair : skip OOF for pair (accepts pair-side leakage as a known limitation)
#   --n_folds N     : number of folds (default 5)
#   --extraction_dir DIR : override output root (default runs/heads/extraction)

set -euo pipefail

SKIP_DATA=0
SKIP_POINT=0
SKIP_PAIR=0
N_FOLDS=5
EXTRACTION_DIR="runs/heads/extraction"
OUT_ROOT="runs/heads"

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-data) SKIP_DATA=1 ;;
        --skip-oof-point) SKIP_POINT=1 ;;
        --skip-oof-pair) SKIP_PAIR=1 ;;
        --n_folds) N_FOLDS="$2"; shift ;;
        --extraction_dir) EXTRACTION_DIR="$2"; shift ;;
        --out_root) OUT_ROOT="$2"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$EXTRACTION_DIR" "$OUT_ROOT"

echo "=== [0/6] Environment sanity ==="
# torchvision / torchaudio preinstalled on the base image nearly always mismatch
# the cu128 torch we install on top, causing transformers' lazy-vision import
# to crash with "operator torchvision::nms does not exist". We never use them —
# remove them so transformers' optional-dep probe gracefully skips vision.
if python -c "import torchvision" 2>/dev/null; then
    echo "  purging torchvision/torchaudio (not used; avoids ABI mismatch with cu128 torch)"
    pip uninstall -y torchvision torchaudio >/dev/null 2>&1 || true
fi

echo "=== [1/6] GPU check ==="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, 'bf16:', torch.cuda.is_bf16_supported())"
# Fail fast if transformers can't import RobertaModel — this catches the
# torch/torchvision ABI mismatch even when the import happened deep inside
# subprocess-spawned train.py, where the traceback is harder to read.
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/graphcodebert-base'); print('transformers + encoder load OK')" 2>&1 | tail -5

if [ "$SKIP_DATA" -eq 0 ]; then
    echo "=== [2/6] Building dataset ==="
    bash run_pipeline.sh
else
    echo "=== [2/6] Skipping dataset build (--skip-data) ==="
fi

echo "=== [3/6] AST feature extraction ==="
python -m stacking.features.ast_features \
    --in_splits data/processed \
    --out_dir "$EXTRACTION_DIR"

if [ "$SKIP_POINT" -eq 0 ]; then
    echo "=== [4/6] OOF pointwise BERT (leakage fix) ==="
    # Tuned for 5090 / 32GB VRAM. Warm-starts off fresh GraphCodeBERT-base per
    # fold. Epochs is a generous upper bound — patience=3 is the real stopper.
    python -m stacking.features.oof_point \
        --data_dir data/processed \
        --out_dir "$EXTRACTION_DIR" \
        --n_folds "$N_FOLDS" \
        --epochs 50 \
        --batch_size 16 \
        --grad_accum 2 \
        --lr 2e-5 \
        --bf16 \
        --num_workers 4 \
        --max_seq_len 512 \
        --eval_every_steps 200 \
        --patience 3 \
        --extract_batch 64 \
        --resume
else
    echo "=== [4/6] Skipping OOF pointwise (--skip-oof-point) ==="
fi

echo "=== [5/6] CLS-based pair similarity ==="
python -m stacking.features.semantic \
    --in_splits data/processed \
    --extraction_dir "$EXTRACTION_DIR"

if [ "$SKIP_PAIR" -eq 0 ]; then
    echo "=== [6/6a] OOF pairwise BERT (leakage fix) ==="
    # Warm-starts from the OOF full pointwise encoder (matches legacy recipe).
    # Epochs is a generous upper bound — patience=3 is the real stopper.
    python -m stacking.features.oof_pair \
        --data_dir data/processed \
        --out_dir "$EXTRACTION_DIR" \
        --n_folds "$N_FOLDS" \
        --epochs 30 \
        --batch_size 12 \
        --grad_accum 2 \
        --lr 1e-5 \
        --label_smoothing 0.05 \
        --class_weights none \
        --bf16 \
        --num_workers 4 \
        --max_seq_len 512 \
        --eval_every_steps 200 \
        --patience 3 \
        --extract_batch 32 \
        --warm_start_from "$EXTRACTION_DIR/oof/full/best" \
        --resume
else
    echo "=== [6/6a] Skipping OOF pairwise (--skip-oof-pair) ==="
    echo "   WARNING: pair logit features on pair_train will be over-confident."
fi

echo "=== [6/6b] Head sweep (8 heads x 3 variants x 3 seeds = 72 experiments) ==="
python -m stacking.sweep \
    --config stacking/configs/sweep.yaml \
    --in_splits data/processed \
    --extraction_dir "$EXTRACTION_DIR" \
    --out_dir "$OUT_ROOT"

echo ""
echo "=== DONE ==="
echo "Summary: $OUT_ROOT/SUMMARY.md"
echo "Per-experiment dirs: $OUT_ROOT/<head>-<variant>-s<seed>/"
echo "OOF artifacts: $EXTRACTION_DIR/oof/ (pointwise), $EXTRACTION_DIR/oof_pair/ (pairwise)"

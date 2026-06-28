#!/usr/bin/env bash
#
# Run the SMD anomaly evaluation (forward.py) for either the ZERO-SHOT Chronos-2
# model or a FINE-TUNED checkpoint, on the held-out test pkl in this folder.
#
# This script lives in SMD_run; forward.py imports `chronos` and `VUS_ROC_VUS_PR`
# which live one level up (rajib_work_space), so we add that to PYTHONPATH.
#
# Usage:
#   ./run_forward_smd.sh                                   # zero-shot
#   CHECKPOINT=chronos2-single-stage_SMD/finetuned-ckpt ./run_forward_smd.sh   # fine-tuned
#   USE_NORMAL_PREFIX=0 ./run_forward_smd.sh               # ablation: context only
#   SCORE_METHOD=mse AGG_METHOD=max ./run_forward_smd.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/SMD_run
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space (chronos + VUS pkgs)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — edit here or override via environment
# ─────────────────────────────────────────────────────────────────────────────
TEST_PKL="${TEST_PKL:-${SCRIPT_DIR}/test_model_inputs.pkl}"
META_PKL="${META_PKL:-${SCRIPT_DIR}/test_series_meta.pkl}"
OUT_CSV="${OUT_CSV:-${SCRIPT_DIR}/eval_results.csv}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
# CHECKPOINT empty -> zero-shot (MODEL_ID). Set to a path -> evaluate fine-tuned model.

CHECKPOINT="${CHECKPOINT:-}" # Zero-shot
# CHECKPOINT="${CHECKPOINT:-/home/rajib/Sir_git_TSAD/TSFM-anomaly/Chronos_Finetuning/rajib_work_space/SMD_run/chronos2-single-stage_SMD/finetuned-ckpt}" # The fine-tuned model

DEVICE="${DEVICE:-cuda}"

# Sequence layout — must match the data-prep values used to build the pkl
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
# Must match INPUT_PATCH_SIZE in run_finetune_smd.sh. Used to restore the SEP patch
# index (= NORMAL_SIGNAL_LENGTH / INPUT_PATCH_SIZE) on the fine-tuned checkpoint,
# since a LoRA adapter does not persist sep_patch_index.
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"
# Whether to feed the per-series normal signal as [normal|context].
#   1 = feed [normal|context]   (the instruction setting the model was fine-tuned on)
#   0 = context only            (standard forecasting input)
# Default is MODE-AWARE: the base zero-shot model cannot interpret the normal
# prefix (no [SEP], no training) and the prefix's discontinuity can actively
# mislead it, so zero-shot defaults to context-only; the fine-tuned model defaults
# to [normal|context]. Override explicitly to run the 2x2 ablation, e.g.
#   USE_NORMAL_PREFIX=1 ./run_forward_smd.sh           # zero-shot + prefix (ablation)
#   CHECKPOINT=... USE_NORMAL_PREFIX=0 ./run_forward_smd.sh   # fine-tuned, no prefix
if [ -n "${CHECKPOINT:-}" ]; then _DEFAULT_PREFIX=1; else _DEFAULT_PREFIX=0; fi
USE_NORMAL_PREFIX="${USE_NORMAL_PREFIX:-${_DEFAULT_PREFIX}}"

# Inference batching
BATCH_WINDOWS="${BATCH_WINDOWS:-32}"
PREDICT_BATCH_SIZE="${PREDICT_BATCH_SIZE:-160}"

# Scoring
SCORE_METHOD="${SCORE_METHOD:-mse}"     # mse | interval | normalized_deviation | smape
AGG_METHOD="${AGG_METHOD:-l2}"        # l2 | max | mean | topk_mean
TOPK="${TOPK:-4}"
SMOOTH_WINDOW="${SMOOTH_WINDOW:-5}"

# VUS
SLIDING_WINDOW_VUS="${SLIDING_WINDOW_VUS:-100}"
VUS_VERSION="${VUS_VERSION:-opt}"
VUS_THRE="${VUS_THRE:-250}"

# ─────────────────────────────────────────────────────────────────────────────
#  Print config
# ─────────────────────────────────────────────────────────────────────────────
if [ -n "${CHECKPOINT}" ]; then MODE="FINE-TUNED (${CHECKPOINT})"; else MODE="ZERO-SHOT (${MODEL_ID})"; fi
echo "======================================================"
echo "  SMD Chronos-2 Anomaly Evaluation"
echo "======================================================"
echo "  MODE              = ${MODE}"
echo "  DEVICE            = ${DEVICE}"
echo "  TEST_PKL          = ${TEST_PKL}"
echo "  META_PKL          = ${META_PKL}"
echo "  NORMAL/CTX/PRED   = ${NORMAL_SIGNAL_LENGTH} / ${CONTEXT_LENGTH} / ${PREDICTION_LENGTH}"
echo "  USE_NORMAL_PREFIX = ${USE_NORMAL_PREFIX}"
echo "  SCORE / AGG       = ${SCORE_METHOD} / ${AGG_METHOD} (topk=${TOPK}, smooth=${SMOOTH_WINDOW})"
echo "  VUS               = win=${SLIDING_WINDOW_VUS}, version=${VUS_VERSION}, thre=${VUS_THRE}"
echo "  OUT_CSV           = ${OUT_CSV}"
echo "======================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ensure the local rajib_work_space/chronos + VUS_ROC_VUS_PR are importable
# ─────────────────────────────────────────────────────────────────────────────
export PYTHONPATH="${WORK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Clear stale bytecode so edits are always picked up
find "$WORK_ROOT/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
#  Build args and run
# ─────────────────────────────────────────────────────────────────────────────
ARGS=(
    --test_pkl            "${TEST_PKL}"
    --meta_pkl            "${META_PKL}"
    --model_id            "${MODEL_ID}"
    --device              "${DEVICE}"
    --normal_signal_length "${NORMAL_SIGNAL_LENGTH}"
    --context_length      "${CONTEXT_LENGTH}"
    --prediction_length   "${PREDICTION_LENGTH}"
    --input_patch_size    "${INPUT_PATCH_SIZE}"
    --batch_windows       "${BATCH_WINDOWS}"
    --predict_batch_size  "${PREDICT_BATCH_SIZE}"
    --score_method        "${SCORE_METHOD}"
    --agg_method          "${AGG_METHOD}"
    --topk                "${TOPK}"
    --smooth_window       "${SMOOTH_WINDOW}"
    --sliding_window_VUS  "${SLIDING_WINDOW_VUS}"
    --vus_version         "${VUS_VERSION}"
    --vus_thre            "${VUS_THRE}"
    --out_csv             "${OUT_CSV}"
)

# Fine-tuned checkpoint (omit entirely for zero-shot)
[ -n "${CHECKPOINT}" ] && ARGS+=(--checkpoint "${CHECKPOINT}")

# Ablation flag
[ "${USE_NORMAL_PREFIX}" = "0" ] && ARGS+=(--no_normal_prefix)

python -u "${SCRIPT_DIR}/forward.py" "${ARGS[@]}"

#!/usr/bin/env bash
# Single-stage anomaly-aware fine-tuning for Chronos-2.
#
# Uses ONE combined dataset (normal + anomaly future pairs mixed). The loss is a
# margin (hinge) objective per sample based on future_type:
#   L_total = L_good + lambda * max(0, tau - L_bad)
#   future_type == 0 (normal)  → L_good : minimised (predict normal well)
#   future_type == 1 (anomaly) → L_bad  : pushed UP toward margin tau, then saturates
#
# Assumes the prepared data dir contains the COMBINED model inputs:
#   <PREPARED_DIR>/train_model_inputs.pkl
#   <PREPARED_DIR>/val_model_inputs.pkl
# Run run_prepare_labeled.sh (inst_data_prepare_labeled.py) first if they don't exist.
#
# Usage examples:
#   bash run_finetune.sh                                     # LoRA, all defaults
#   FINETUNE_MODE=full bash run_finetune.sh                  # full fine-tuning
#   BATCH_SIZE=2 GRAD_ACCUM=16 bash run_finetune.sh          # tighter memory budget
#   NUM_STEPS=8000 bash run_finetune.sh                      # more training steps
#   ENABLE_SEP_TOKEN=1 NORMAL_SIGNAL_LENGTH=256 CONTEXT_LENGTH=768 \
#       bash run_finetune.sh                                 # turn on [SEP] token
#                                                              (sequence becomes
#                                                               [normal][SEP][ctx][REG][future])

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — edit here or export before running
# ─────────────────────────────────────────────────────────────────────────────

# This script lives in rajib_work_space/SMD_run. The training entrypoint
# (finetune_anomaly_simple.py) and the `chronos` package live one level up, in
# rajib_work_space. Anchor everything to absolute paths so it runs from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rajib_work_space/SMD_run
WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                  # .../rajib_work_space (code root)

# Paths
# PREPARED_DIR must contain train_model_inputs.pkl (and val_model_inputs.pkl unless
# NO_VALIDATION=1). The SMD pkls live next to this script, in SMD_run.
PREPARED_DIR="${PREPARED_DIR:-${SCRIPT_DIR}}"                # train (& val) data = SMD_run
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/chronos2-single-stage_SMD}"

# Optional third dataset (full path to a .pkl). When set, it is evaluated and
# logged every eval step exactly like validation (no weight updates), as eval_test_*.
# TEST_DATA="${TEST_DATA:-${REPO_ROOT}/rajib_work_space/prepared_data_labeled/test_model_inputs.pkl}" # test data, ignored if set empty
TEST_DATA="${TEST_DATA:-}"
# Model
MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

# Fine-tuning mode
FINETUNE_MODE="${FINETUNE_MODE:-lora}"          # "lora" or "full"

# Training hyperparameters
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
# NOTE: CONTEXT_LENGTH here is N + C (normal signal + actual context).
#       It must equal data-prep NORMAL_SIGNAL_LENGTH + data-prep CONTEXT_LENGTH
#       e.g.  CONTEXT_LENGTH (768) = NORMAL_SIGNAL_LENGTH (256) + CONTEXT_LENGTH (512)
CONTEXT_LENGTH="${CONTEXT_LENGTH:-768}"
NUM_STEPS="${NUM_STEPS:-4000}"
LR="${LR:-1e-5}"                                    # blank → script default (1e-5 lora / 1e-6 full)
BATCH_SIZE="${BATCH_SIZE:-160}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"                   # effective batch = BATCH_SIZE * GRAD_ACCUM
LOGGING_STEPS="${LOGGING_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-100}"      # validate + log eval_loss every N steps (must divide 100)
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"                               # 1 = fp16 mixed precision, 0 = disable
NO_VALIDATION="${NO_VALIDATION:-1}"             # 1 = disable validation (SMD_run has no val pkl)
DEBUG="${DEBUG:-0}"                             # set to 1 to truncate train/val to 50 samples (smoke test)

# A future window is labeled anomalous (future_type=1) iff it contains at least
# ANOMALY_THRESHOLD anomalous timesteps (out of PREDICTION_LENGTH); else normal.
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"

# Margin (hinge) loss: L_good + MARGIN_LAMBDA * max(0, MARGIN_TAU - L_bad).
# MARGIN_TAU must sit ABOVE the normal-point loss (~3-4 here) to matter — use ~10-15,
# NOT 2. The hinge self-saturates once L_bad >= tau, so no ceiling clamp is needed.
MARGIN_TAU="${MARGIN_TAU:-6.0}"
MARGIN_LAMBDA="${MARGIN_LAMBDA:-1.0}"

# LoRA hyperparameters (ignored when FINETUNE_MODE=full)
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.01}"

# [SEP] token between normal signal and context
# When enabled, every target must be laid out as
#   [normal (NORMAL_SIGNAL_LENGTH) | context | future]
# and CONTEXT_LENGTH must equal NORMAL_SIGNAL_LENGTH + actual_context_length.
ENABLE_SEP_TOKEN="${ENABLE_SEP_TOKEN:-1}"            # set to 1 to enable
NORMAL_SIGNAL_LENGTH="${NORMAL_SIGNAL_LENGTH:-256}"  # length of normal signal prefix
INPUT_PATCH_SIZE="${INPUT_PATCH_SIZE:-16}"           # model's input_patch_size

# ─────────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────────

if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    if [ "${NORMAL_SIGNAL_LENGTH}" -le 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH must be > 0 when ENABLE_SEP_TOKEN=1"
        exit 1
    fi
    if [ $(( NORMAL_SIGNAL_LENGTH % INPUT_PATCH_SIZE )) -ne 0 ]; then
        echo "ERROR: NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH) must be a multiple of INPUT_PATCH_SIZE ($INPUT_PATCH_SIZE)"
        exit 1
    fi
    if [ "${CONTEXT_LENGTH}" -le "${NORMAL_SIGNAL_LENGTH}" ]; then
        echo "ERROR: CONTEXT_LENGTH ($CONTEXT_LENGTH) must be greater than NORMAL_SIGNAL_LENGTH ($NORMAL_SIGNAL_LENGTH)"
        echo "       It must equal NORMAL_SIGNAL_LENGTH + actual_context_length."
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
#  Print config
# ─────────────────────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Chronos-2 Single-Stage Anomaly Fine-Tuning"
echo "======================================================"
echo "  PREPARED_DIR      = $PREPARED_DIR"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  MODEL_ID          = $MODEL_ID"
echo "  DEVICE            = $DEVICE"
echo "  FINETUNE_MODE     = $FINETUNE_MODE"
echo "  PREDICTION_LENGTH = $PREDICTION_LENGTH"
echo "  CONTEXT_LENGTH    = $CONTEXT_LENGTH"
echo "  NUM_STEPS         = $NUM_STEPS"
echo "  LR                = ${LR:-<default>}"
echo "  BATCH_SIZE        = $BATCH_SIZE"
echo "  GRAD_ACCUM        = $GRAD_ACCUM  (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  LOGGING_STEPS     = $LOGGING_STEPS"
echo "  EVAL_STEPS        = $EVAL_STEPS"
echo "  WARMUP_RATIO      = $WARMUP_RATIO"
echo "  LR_SCHEDULER      = $LR_SCHEDULER"
echo "  FP16              = $FP16"
echo "  NO_VALIDATION     = $NO_VALIDATION"
echo "  DEBUG             = $DEBUG  (1 = truncate train/val to 50 samples)"
echo "  ANOMALY_THRESHOLD = $ANOMALY_THRESHOLD  (>= this many anomalous steps -> anomaly window)"
echo "  MARGIN_TAU        = $MARGIN_TAU"
echo "  MARGIN_LAMBDA     = $MARGIN_LAMBDA"
if [ "$FINETUNE_MODE" = "lora" ]; then
echo "------------------------------------------------------"
echo "  LORA_R            = $LORA_R"
echo "  LORA_ALPHA        = $LORA_ALPHA"
echo "  LORA_DROPOUT      = $LORA_DROPOUT"
fi
echo "------------------------------------------------------"
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
echo "  [SEP] token       = ENABLED"
echo "  NORMAL_SIGNAL_LEN = $NORMAL_SIGNAL_LENGTH  (sep_patch_index = $(( NORMAL_SIGNAL_LENGTH / INPUT_PATCH_SIZE )))"
echo "  INPUT_PATCH_SIZE  = $INPUT_PATCH_SIZE"
echo "  Sequence layout   = [normal][SEP][context][REG][future]"
else
echo "  [SEP] token       = disabled"
echo "  Sequence layout   = [context][REG][future]"
fi
echo "======================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
#  Ensure the local rajib_work_space/chronos is used, not any installed package
# ─────────────────────────────────────────────────────────────────────────────

export PYTHONPATH="${WORK_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ─────────────────────────────────────────────────────────────────────────────
#  Build argument list and run
# ─────────────────────────────────────────────────────────────────────────────

FINETUNE_ARGS=(
    --model_id                    "$MODEL_ID"
    --device                      "$DEVICE"
    --data_dir                    "$PREPARED_DIR"
    --output_dir                  "$OUTPUT_DIR"
    --finetune_mode               "$FINETUNE_MODE"
    --prediction_length           "$PREDICTION_LENGTH"
    --context_length              "$CONTEXT_LENGTH"
    --num_steps                   "$NUM_STEPS"
    --batch_size                  "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --logging_steps               "$LOGGING_STEPS"
    --eval_steps                  "$EVAL_STEPS"
    --warmup_ratio                "$WARMUP_RATIO"
    --lr_scheduler_type           "$LR_SCHEDULER"
    --anomaly_threshold           "$ANOMALY_THRESHOLD"
    --margin_tau                  "$MARGIN_TAU"
    --margin_lambda               "$MARGIN_LAMBDA"
)

# Learning rate — omit entirely to use the script's built-in default
[ -n "${LR}" ] && FINETUNE_ARGS+=(--lr "$LR")

# Optional third (test) dataset — evaluated like validation, logged as eval_test_*
[ -n "${TEST_DATA}" ] && FINETUNE_ARGS+=(--test_data "$TEST_DATA")

# Flags
[ "${NO_VALIDATION}" = "1" ] && FINETUNE_ARGS+=(--no_validation)
[ "${DEBUG}" = "1" ] && FINETUNE_ARGS+=(--debug)
[ "${FP16}" = "1" ] && FINETUNE_ARGS+=(--fp16) || FINETUNE_ARGS+=(--no_fp16)

# LoRA hyperparameters
if [ "$FINETUNE_MODE" = "lora" ]; then
    FINETUNE_ARGS+=(
        --lora_r       "$LORA_R"
        --lora_alpha   "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
    )
fi

# [SEP] token
if [ "${ENABLE_SEP_TOKEN}" = "1" ]; then
    FINETUNE_ARGS+=(
        --enable_sep_token
        --normal_signal_length "$NORMAL_SIGNAL_LENGTH"
        --input_patch_size     "$INPUT_PATCH_SIZE"
    )
fi

# Clear stale bytecode so edits made on Windows are always picked up in WSL
find "$WORK_ROOT/chronos" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

python -u "$WORK_ROOT/finetune_anomaly_simple.py" "${FINETUNE_ARGS[@]}"

# ─────────────────────────────────────────────────────────────────────────────
#  Done
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo "  Single-stage fine-tuning complete!"
echo "  Checkpoint : $OUTPUT_DIR/finetuned-ckpt  ← use for inference"
echo ""
echo "  Load the final model with:"
echo "    from chronos import BaseChronosPipeline"
echo "    pipeline = BaseChronosPipeline.from_pretrained("
echo "        '$OUTPUT_DIR/finetuned-ckpt', device_map='cuda')"
echo "======================================================"
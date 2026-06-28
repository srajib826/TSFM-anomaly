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

# Root of the chronos-forecasting repo (parent of this script's directory)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Paths
# Output dir from inst_data_prepare_labeled.py — must contain the combined
# train_model_inputs.pkl / val_model_inputs.pkl at its root (each sample carrying
# the per-timestep future_labels array).
PREPARED_DIR="${PREPARED_DIR:-${REPO_ROOT}/rajib_work_space/prepared_data_labeled}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/rajib_work_space/chronos2-single-stage_NS1000_V4}"

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
NUM_STEPS="${NUM_STEPS:-3000}"
LR="${LR:-}"                                    # blank → script default (1e-5 lora / 1e-6 full)
BATCH_SIZE="${BATCH_SIZE:-160}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"                   # effective batch = BATCH_SIZE * GRAD_ACCUM
LOGGING_STEPS="${LOGGING_STEPS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
FP16="${FP16:-1}"                               # 1 = fp16 mixed precision, 0 = disable
NO_VALIDATION="${NO_VALIDATION:-0}"             # set to 1 to disable validation

# A future window is labeled anomalous (future_type=1) iff it contains at least
# ANOMALY_THRESHOLD anomalous timesteps (out of PREDICTION_LENGTH); else normal.
ANOMALY_THRESHOLD="${ANOMALY_THRESHOLD:-10}"

# Margin (hinge) loss: L_good + MARGIN_LAMBDA * max(0, MARGIN_TAU - L_bad).
# MARGIN_TAU must sit ABOVE the normal-point loss (~3-4 here) to matter — use ~10-15,
# NOT 2. The hinge self-saturates once L_bad >= tau, so no ceiling clamp is needed.
MARGIN_TAU="${MARGIN_TAU:-2.0}"
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
echo "  WARMUP_RATIO      = $WARMUP_RATIO"
echo "  LR_SCHEDULER      = $LR_SCHEDULER"
echo "  FP16              = $FP16"
echo "  NO_VALIDATION     = $NO_VALIDATION"
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
    --warmup_ratio                "$WARMUP_RATIO"
    --lr_scheduler_type           "$LR_SCHEDULER"
    --anomaly_threshold           "$ANOMALY_THRESHOLD"
    --margin_tau                  "$MARGIN_TAU"
    --margin_lambda               "$MARGIN_LAMBDA"
)

# Learning rate — omit entirely to use the script's built-in default
[ -n "${LR}" ] && FINETUNE_ARGS+=(--lr "$LR")

# Flags
[ "${NO_VALIDATION}" = "1" ] && FINETUNE_ARGS+=(--no_validation)
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

python finetune_anomaly_simple.py "${FINETUNE_ARGS[@]}"

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

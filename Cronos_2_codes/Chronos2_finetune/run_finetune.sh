#!/usr/bin/env bash
# Fine-tune Chronos-2 on pre-processed mTSBench data.
# Assumes prepared_data/ already exists (run prepare_data.py separately if not).
#
# Usage:
#   bash run_finetune.sh                      # LoRA, all defaults
#   FINETUNE_MODE=full bash run_finetune.sh   # full fine-tuning
#   BATCH_SIZE=2 GRAD_ACCUM=16 bash run_finetune.sh  # tighter memory budget

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit here or export before running
# ---------------------------------------------------------------------------
PREPARED_DIR="${PREPARED_DIR:-./prepared_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./chronos2-finetuned}"

MODEL_ID="${MODEL_ID:-amazon/chronos-2}"
DEVICE="${DEVICE:-cuda}"

FINETUNE_MODE="${FINETUNE_MODE:-lora}"          # "lora" or "full"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-24}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
NUM_STEPS="${NUM_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"

# Memory-efficient defaults (effective batch = BATCH_SIZE * GRAD_ACCUM = 32)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

# LoRA-specific (ignored when FINETUNE_MODE=full)
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

# Learning rate default: 1e-5 for LoRA, 1e-6 for full
if [ "$FINETUNE_MODE" = "lora" ]; then
    LEARNING_RATE="${LEARNING_RATE:-1e-5}"
else
    LEARNING_RATE="${LEARNING_RATE:-1e-6}"
fi


echo "======================================================"
echo "  Chronos-2 Fine-Tuning"
echo "======================================================"
echo "  PREPARED_DIR      = $PREPARED_DIR"
echo "  OUTPUT_DIR        = $OUTPUT_DIR"
echo "  MODEL_ID          = $MODEL_ID"
echo "  DEVICE            = $DEVICE"
echo "  FINETUNE_MODE     = $FINETUNE_MODE"
echo "  PREDICTION_LENGTH = $PREDICTION_LENGTH"
echo "  CONTEXT_LENGTH    = $CONTEXT_LENGTH"
echo "  NUM_STEPS         = $NUM_STEPS"
echo "  LEARNING_RATE     = $LEARNING_RATE"
echo "  BATCH_SIZE        = $BATCH_SIZE"
echo "  GRAD_ACCUM        = $GRAD_ACCUM  (effective batch = $((BATCH_SIZE * GRAD_ACCUM)))"
if [ "$FINETUNE_MODE" = "lora" ]; then
echo "  LORA_R            = $LORA_R"
echo "  LORA_ALPHA        = $LORA_ALPHA"
echo "  LORA_DROPOUT      = $LORA_DROPOUT"
fi
echo "======================================================"
echo ""


FINETUNE_ARGS=(
    --model_id          "$MODEL_ID"
    --device            "$DEVICE"
    --data_dir          "$PREPARED_DIR"
    --output_dir        "$OUTPUT_DIR"
    --finetune_mode     "$FINETUNE_MODE"
    --prediction_length "$PREDICTION_LENGTH"
    --context_length    "$CONTEXT_LENGTH"
    --num_steps         "$NUM_STEPS"
    --learning_rate     "$LEARNING_RATE"
    --batch_size        "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --logging_steps     "$LOGGING_STEPS"
)

if [ "$FINETUNE_MODE" = "lora" ]; then
    FINETUNE_ARGS+=(
        --lora_r       "$LORA_R"
        --lora_alpha   "$LORA_ALPHA"
        --lora_dropout "$LORA_DROPOUT"
    )
fi

python finetune.py "${FINETUNE_ARGS[@]}"

echo ""
echo "======================================================"
echo "  Fine-tuning complete!"
echo "  Checkpoint saved to: $OUTPUT_DIR/finetuned-ckpt"
echo "======================================================"

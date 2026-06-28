#!/usr/bin/env bash
#
# SMD-only data preparation with a FILE-BASED 70/30 train/test split.
#
# Usage:
#   ./run_prepare_smd.sh                       # 70/30 train/test, NO validation
#   CREATE_VAL=1 ./run_prepare_smd.sh          # also carve a validation set
#   CREATE_VAL=1 VAL_FRACTION=0.15 ./run_prepare_smd.sh
#   DATA_DIR=/path OUTPUT_DIR=/path ./run_prepare_smd.sh
#
set -euo pipefail

# Run from the directory this script lives in (so outputs land in SMD_run).
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/home/rajib/mTSBench/Datasets/mTSBench/SMD}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
MIN_LENGTH="${MIN_LENGTH:-50}"
TEST_FRACTION="${TEST_FRACTION:-0.3}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-64}"
SEED="${SEED:-42}"

# Validation toggle:
#   CREATE_VAL=0 (default) -> no validation data (full 70% used for training)
#   CREATE_VAL=1           -> carve VAL_FRACTION of the 70% train files into a val set
CREATE_VAL="${CREATE_VAL:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"

if [ "${CREATE_VAL}" = "1" ]; then
    EFFECTIVE_VAL_FRACTION="${VAL_FRACTION}"
else
    EFFECTIVE_VAL_FRACTION="0"
fi

echo "Data dir          : ${DATA_DIR}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Test fraction     : ${TEST_FRACTION}  (file-based)"
echo "Create validation : ${CREATE_VAL}  (val_fraction=${EFFECTIVE_VAL_FRACTION})"
echo "Context length    : ${CONTEXT_LENGTH}"
echo "Prediction length : ${PREDICTION_LENGTH}"
echo "Stride            : ${STRIDE}"
echo "Seed              : ${SEED}"
echo

python prepare_smd_split.py \
    --data_dir          "${DATA_DIR}" \
    --output_dir        "${OUTPUT_DIR}" \
    --min_length        "${MIN_LENGTH}" \
    --test_fraction     "${TEST_FRACTION}" \
    --val_fraction      "${EFFECTIVE_VAL_FRACTION}" \
    --context_length    "${CONTEXT_LENGTH}" \
    --prediction_length "${PREDICTION_LENGTH}" \
    --stride            "${STRIDE}" \
    --seed              "${SEED}"

echo
echo "Done. Outputs written to ${OUTPUT_DIR}"

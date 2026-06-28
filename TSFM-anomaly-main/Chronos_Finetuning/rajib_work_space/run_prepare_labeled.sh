#!/usr/bin/env bash
#
# Run the per-timestamp-labeled sliding-window data preparation.
#
# Usage:
#   ./run_prepare_labeled.sh                # use defaults below
#   DATA_DIR=/path OUTPUT_DIR=/path ./run_prepare_labeled.sh
#
set -euo pipefail

# Run from the directory this script lives in (so relative ./prepared_data_labeled paths work).
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-/home/rajib/mTSBench/Datasets/mTSBench}"
OUTPUT_DIR="${OUTPUT_DIR:-./prepared_data_labeled}"
MIN_LENGTH="${MIN_LENGTH:-50}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
PREDICTION_LENGTH="${PREDICTION_LENGTH:-64}"
STRIDE="${STRIDE:-576}"

echo "Data dir          : ${DATA_DIR}"
echo "Output dir        : ${OUTPUT_DIR}"
echo "Context length    : ${CONTEXT_LENGTH}"
echo "Prediction length : ${PREDICTION_LENGTH}"
echo "Stride            : ${STRIDE}"
echo

python inst_data_prepare_labeled.py \
    --data_dir          "${DATA_DIR}" \
    --output_dir        "${OUTPUT_DIR}" \
    --min_length        "${MIN_LENGTH}" \
    --val_fraction      "${VAL_FRACTION}" \
    --context_length    "${CONTEXT_LENGTH}" \
    --prediction_length "${PREDICTION_LENGTH}" \
    --stride            "${STRIDE}"

echo
echo "Done. Outputs written to ${OUTPUT_DIR}"

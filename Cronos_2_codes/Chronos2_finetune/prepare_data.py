"""
Data preparation script for fine-tuning Chronos-2 on mTSBench data.

Loads all *train.csv files from the mTSBench dataset directory, extracts feature
columns (excluding timestamp and is_anomaly), and saves train/val splits as
pickle files ready for chronos2 pipeline.fit().

Additionally creates instruction tuning pairs using a sliding window approach:
  - context : {'target': array(num_features, context_length)}   <- INSTRUCTION
  - future  : {'target': array(num_features, pred_length)}      <- OUTPUT

Both context and future maintain the same {'target': array} format as the
original data, so they are directly compatible with the Chronos2 pipeline.

Usage:
    python prepare_data.py [--data_dir ...] [--output_dir ...] [--min_length ...]
                           [--val_fraction ...] [--context_length ...]
                           [--prediction_length ...] [--stride ...]
"""

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd


log_path = os.path.join("./prepared_data/log", "prepare_data.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ]
)
logger = logging.getLogger(__name__)


def load_csv_as_multivariate(csv_path: str) -> np.ndarray | None:
    """
    Load one *train.csv and return float32 array of shape (n_variates, time_steps).

    Excludes 'timestamp' and 'is_anomaly' columns — only feature columns are kept.
    The resulting shape is:
        rows    = number of features  (n_variates)
        columns = number of time steps
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None
    else:
        try:
            return df[feature_cols].values.T.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error occurred while processing {csv_path}: {e}")
            return None


def create_instruction_pairs(
    series_dict: dict,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> list[dict]:
    """
    Convert one {'target': array(num_features, T)} entry into a list of
    instruction tuning pairs using a sliding window along the TIME axis (axis=1).

    The FEATURE axis (axis=0) is never touched — shape is always preserved.

    Each pair looks like:
        {
            'context': {'target': array(num_features, context_length)},  <- INSTRUCTION
            'future' : {'target': array(num_features, prediction_length)} <- OUTPUT
        }

    Parameters
    ----------
    series_dict      : one entry from your data list, {'target': array(F, T)}
    context_length   : number of time steps in the context window (instruction)
    prediction_length: number of future time steps to predict (output)
    stride           : how many steps to advance the window each iteration
                       - stride=1          → maximum overlap, most pairs, most redundancy
                       - stride=pred_length → no overlap between targets, most diverse
                       - stride=pred_length//2 → balanced (recommended default)

    Returns
    -------
    list of dicts, each with 'context' and 'future' keys
    Both context['target'] and future['target'] are shape (num_features, length)
    """
    data = series_dict["target"]          # shape: (num_features, time_steps)
    num_features, total_time_steps = data.shape

    window_size = context_length + prediction_length

    # Need at least one full window to create any pairs
    if total_time_steps < window_size:
        logger.debug(
            f"Series too short for windowing: "
            f"length={total_time_steps}, required={window_size}. Skipping."
        )
        return []

    pairs = []

    # Slide window along the TIME axis (axis=1) only
    # Start positions for each window
    start_positions = range(0, total_time_steps - window_size + 1, stride)

    for start in start_positions:
        context_end = start + context_length
        future_end  = context_end + prediction_length

        # Slice along axis=1 (time axis) — rows (features) stay intact
        context_array = data[:, start:context_end]   # (num_features, context_length)
        future_array  = data[:, context_end:future_end]  # (num_features, pred_length)

        # Wrap back into the same {'target': array} format
        pair = {
            "context": {"target": context_array},
            "future" : {"target": future_array},
        }
        pairs.append(pair)

    return pairs


def prepare_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    context_length: int,
    prediction_length: int,
    stride: int,
    seed: int = 42,
):
    """
    Load all *train.csv files, build raw series list, split into train/val,
    and create instruction tuning pairs for each split.

    Returns
    -------
    train_inputs       : list of {'target': array(F, T)} — raw series for train
    val_inputs         : list of {'target': array(F, T)} — raw series for val
    train_pairs        : list of {'context': ..., 'future': ...} for train
    val_pairs          : list of {'context': ..., 'future': ...} for val
    """
    csv_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*train.csv"), recursive=True)
    )
    logger.info(f"Found {len(csv_files)} *train.csv files under {data_dir}")

    # ------------------------------------------------------------------ #
    #  Step 1 — Load all CSVs into raw series list                        #
    # ------------------------------------------------------------------ #
    all_inputs = []
    skipped = 0

    for path in csv_files:
        try:
            data = load_csv_as_multivariate(path)

            # Skip if empty or too short for even one window
            min_required = max(min_length, context_length + prediction_length)
            if data is None or data.shape[1] < min_required:
                logger.debug(
                    f"Skipping {os.path.basename(path)}: "
                    f"length={data.shape[1] if data is not None else 'None'}, "
                    f"required={min_required}"
                )
                skipped += 1
                continue

            all_inputs.append({"target": data})

        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1

    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")

    if len(all_inputs) == 0:
        raise ValueError(
            "No usable series found. Check your data_dir and min_length settings."
        )

    # ------------------------------------------------------------------ #
    #  Step 2 — Train / Val split on raw series                           #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_inputs))
    n_val = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())

    train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_set]
    val_inputs   = [all_inputs[i] for i in val_set]

    logger.info(f"Train series: {len(train_inputs)} | Val series: {len(val_inputs)}")

    # ------------------------------------------------------------------ #
    #  Step 3 — Create instruction pairs from each split                  #
    #                                                                      #
    #  Each raw series {'target': array(F, T)} produces multiple pairs:   #
    #    {'context': {'target': array(F, context_length)},                #
    #     'future' : {'target': array(F, pred_length)}}                   #
    #                                                                      #
    #  Window slides along axis=1 (time axis) only.                       #
    #  Feature axis (axis=0) is never modified.                           #
    # ------------------------------------------------------------------ #
    logger.info(
        f"Creating instruction pairs: "
        f"context_length={context_length}, "
        f"prediction_length={prediction_length}, "
        f"stride={stride}"
    )

    train_pairs = []
    for series in train_inputs:
        pairs = create_instruction_pairs(
            series_dict=series,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )
        train_pairs.extend(pairs)

    val_pairs = []
    for series in val_inputs:
        pairs = create_instruction_pairs(
            series_dict=series,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
        )
        val_pairs.extend(pairs)

    logger.info(
        f"Instruction pairs created — "
        f"Train: {len(train_pairs)} pairs from {len(train_inputs)} series | "
        f"Val: {len(val_pairs)} pairs from {len(val_inputs)} series"
    )

    return train_inputs, val_inputs, train_pairs, val_pairs


def log_dataset_statistics(
    train_inputs: list,
    val_inputs: list,
    train_pairs: list,
    val_pairs: list,
) -> None:
    """Log shape statistics for both raw series and instruction pairs."""

    # Raw series statistics
    all_series = train_inputs + val_inputs
    lengths   = [s["target"].shape[1] for s in all_series]
    variates  = [s["target"].shape[0] for s in all_series]

    logger.info("=" * 60)
    logger.info("RAW SERIES STATISTICS")
    logger.info(f"  Total series : {len(all_series)}")
    logger.info(f"  Time steps   : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
    logger.info(f"  Num features : min={min(variates)}, max={max(variates)}, mean={np.mean(variates):.1f}")

    # Instruction pair statistics
    all_pairs = train_pairs + val_pairs
    if all_pairs:
        ctx_lengths  = [p["context"]["target"].shape[1] for p in all_pairs]
        fut_lengths  = [p["future"]["target"].shape[1]  for p in all_pairs]
        ctx_features = [p["context"]["target"].shape[0] for p in all_pairs]

        logger.info("=" * 60)
        logger.info("INSTRUCTION PAIR STATISTICS")
        logger.info(f"  Total pairs          : {len(all_pairs)}")
        logger.info(f"  Train pairs          : {len(train_pairs)}")
        logger.info(f"  Val pairs            : {len(val_pairs)}")
        logger.info(f"  Context shape        : ({ctx_features[0]}, {ctx_lengths[0]})  [features, context_length]")
        logger.info(f"  Future shape         : ({ctx_features[0]}, {fut_lengths[0]})   [features, pred_length]")
        logger.info(f"  Avg pairs per series : {len(all_pairs) / len(all_series):.1f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mTSBench data for Chronos-2 instruction tuning."
    )
    parser.add_argument(
        "--data_dir",
        default="/home/rajib/mTSBench/Datasets/mTSBench",
        help="Root directory of the mTSBench dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./prepared_data",
        help="Directory to write output pickle files",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=50,
        help="Minimum time-series length; shorter series are discarded",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Fraction of series held out for validation",
    )

    # --- New arguments for instruction pair creation ---
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help=(
            "Number of past time steps used as context (instruction). "
        ),
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=64,
        help=(
            "Number of future time steps to predict (output). "
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Window stride for sliding window. "
            "Defaults to prediction_length // 2 if not set. "
            "stride=1 gives maximum pairs but high redundancy. "
            "stride= prediction_length gives no target overlap."
        ),
    )

    args = parser.parse_args()

    # Default stride = prediction_length // 2 (balanced overlap)
    if args.stride is None:
        args.stride = max(1, args.prediction_length // 2)
        logger.info(f"Stride not set — using default: stride={args.stride}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run preparation
    train_inputs, val_inputs, train_pairs, val_pairs = prepare_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
    )

    # ------------------------------------------------------------------ #
    #  Save all outputs                                                    #
    #                                                                      #
    #  train_inputs.pkl / val_inputs.pkl                                  #
    #    → original format, list of {'target': array(F, T)}              #
    #    → used directly with pipeline.fit() if no windowing needed       #
    #                                                                      #
    #  train_pairs.pkl / val_pairs.pkl                                    #
    #    → instruction pairs for fine-tuning                              #
    #    → list of {'context': {'target': array(F, ctx)},                #
    #               'future' : {'target': array(F, pred)}}               #
    # ------------------------------------------------------------------ #
    files_to_save = {
        "train_inputs.pkl": train_inputs,
        "val_inputs.pkl"  : val_inputs,
        "train_pairs.pkl" : train_pairs,
        "val_pairs.pkl"   : val_pairs,
    }

    for filename, data in files_to_save.items():
        path = os.path.join(args.output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {len(data):>6} entries → {path}")

    # Log full statistics
    log_dataset_statistics(train_inputs, val_inputs, train_pairs, val_pairs)


if __name__ == "__main__":
    main()
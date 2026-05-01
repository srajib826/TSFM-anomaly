"""
Data preparation script for fine-tuning Chronos-2 on mTSBench data.

Loads all *test.csv files from the mTSBench dataset directory, extracts feature
columns (excluding timestamp and is_anomaly), and saves train/val splits as
pickle files ready for chronos2 pipeline.fit().

Usage:
    python prepare_data.py [--data_dir ...] [--output_dir ...] [--min_length ...]
                           [--val_fraction ...]
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
    """Load one *test.csv and return float32 array of shape (n_variates, time_steps)."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None
    return df[feature_cols].values.T.astype(np.float32)


def prepare_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    seed: int = 42,
):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} *test.csv files under {data_dir}")

    all_inputs = []
    skipped = 0
    for path in csv_files:
        try:
            data = load_csv_as_multivariate(path)
            if data is None or data.shape[1] < min_length:
                skipped += 1
                continue
            all_inputs.append({"target": data})
        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1

    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_inputs))
    n_val = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())

    train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_set]
    val_inputs = [all_inputs[i] for i in val_set]

    return train_inputs, val_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/rajib/mTSBench/Datasets/mTSBench",
        help="Root directory of the mTSBench dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="./prepared_data",
        help="Directory to write train_inputs.pkl and val_inputs.pkl",
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_inputs, val_inputs = prepare_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
    )

    train_path = os.path.join(args.output_dir, "train_inputs.pkl")
    val_path = os.path.join(args.output_dir, "val_inputs.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train_inputs, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_inputs, f)

    logger.info(f"Saved {len(train_inputs)} train series  → {train_path}")
    logger.info(f"Saved {len(val_inputs)} val series     → {val_path}")

    lengths = [inp["target"].shape[1] for inp in train_inputs + val_inputs]
    variates = [inp["target"].shape[0] for inp in train_inputs + val_inputs]
    logger.info(
        f"Length  min={min(lengths):5d}  max={max(lengths):6d}  mean={np.mean(lengths):.0f}"
    )
    logger.info(
        f"Variates min={min(variates):3d}  max={max(variates):4d}  mean={np.mean(variates):.1f}"
    )


if __name__ == "__main__":
    main()

"""
Data preparation script for fine-tuning Chronos-2 on mTSBench data.

Loads all *test.csv files from the mTSBench dataset directory, extracts feature
columns (excluding timestamp and is_anomaly), and saves train/val splits as
pickle files ready for chronos2 pipeline.fit().

NOTE: mTSBench *train.csv files contain only normal data (is_anomaly=0 throughout).
Anomaly labels (required for Type B and C pairs) only exist in *test.csv files,
so only those are loaded.

Creates THREE types of instruction tuning pairs using anomaly ground truth labels:

  Type A — Normal-to-Normal   : context=normal, future=normal        (60% target)
  Type B — Pre-Anomaly        : context=normal, future=anomaly onset  (30% target)
  Type C — Anomaly-Context    : context=anomaly, future=normal        (10% target)

All pairs maintain the {'target': array(num_features, length)} format.

Usage:
    python prepare_data.py [--data_dir ...] [--output_dir ...] [--min_length ...]
                           [--val_fraction ...] [--context_length ...]
                           [--prediction_length ...] [--stride ...]
                           [--ratio_a ...] [--ratio_b ...] [--ratio_c ...]
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


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(
    csv_path: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load one *train.csv file.

    Returns
    -------
    features       : float32 array of shape (n_variates, time_steps)
                     Only feature columns — timestamp and is_anomaly excluded.
    anomaly_labels : int32 array of shape (time_steps,)
                     1 = anomaly, 0 = normal.
                     All zeros if 'is_anomaly' column is absent.
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]

    if not feature_cols:
        return None, None

    try:
        features = df[feature_cols].values.T.astype(np.float32)

        if "is_anomaly" in df.columns:
            anomaly_labels = df["is_anomaly"].values.astype(np.int32)
        else:
            # No label column — treat entire series as normal
            anomaly_labels = np.zeros(df.shape[0], dtype=np.int32)

        return features, anomaly_labels

    except Exception as e:
        logger.warning(f"Error processing {csv_path}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Boundary Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(
    anomaly_labels: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Find contiguous anomaly regions from the binary label array.

    Returns
    -------
    List of (start, end) tuples where is_anomaly == 1.
    End index is EXCLUSIVE (Python-slice style).

    Example
    -------
    labels = [0,0,0,1,1,1,0,0,1,1,0]
    returns [(3, 6), (8, 10)]
    """
    boundaries = []
    in_anomaly = False
    start = 0

    for i, label in enumerate(anomaly_labels):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start = i
        elif label == 0 and in_anomaly:
            in_anomaly = False
            boundaries.append((start, i))

    # Handle series that ends while still in anomaly
    if in_anomaly:
        boundaries.append((start, len(anomaly_labels)))

    return boundaries


def get_normal_zones(
    anomaly_boundaries: list[tuple[int, int]],
    total_length: int,
) -> list[tuple[int, int]]:
    """
    Return the normal (non-anomaly) zones as (start, end) pairs.

    These are the gaps between anomalies, plus the leading and trailing
    normal segments at the edges of the series.

    Example
    -------
    anomaly_boundaries = [(3, 6), (8, 10)], total_length = 12
    returns [(0, 3), (6, 8), (10, 12)]
    """
    normal_zones = []
    prev_end = 0

    for anom_start, anom_end in anomaly_boundaries:
        if anom_start > prev_end:
            normal_zones.append((prev_end, anom_start))
        prev_end = anom_end

    # Trailing normal zone after last anomaly
    if prev_end < total_length:
        normal_zones.append((prev_end, total_length))

    return normal_zones


# ─────────────────────────────────────────────────────────────────────────────
#  Type A — Normal-to-Normal Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_a_pairs(
    data: np.ndarray,
    normal_zones: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    stride: int,
) -> list[dict]:
    """
    Type A — Normal-to-Normal instances.

    Sliding window is applied ONLY within normal zones.
    Both context and future are guaranteed to be anomaly-free.

    Teaches the model: "this is what normal continuation looks like."

    Parameters
    ----------
    data         : feature array, shape (num_features, time_steps)
    normal_zones : list of (start, end) for normal regions
    """
    pairs = []
    window_size = context_length + prediction_length

    for zone_start, zone_end in normal_zones:
        zone_length = zone_end - zone_start

        # Zone must fit at least one full window
        if zone_length < window_size:
            logger.debug(
                f"Normal zone [{zone_start}, {zone_end}] too short "
                f"(length={zone_length}, need={window_size}). Skipping."
            )
            continue

        for start in range(zone_start, zone_end - window_size + 1, stride):
            context_end = start + context_length
            future_end  = context_end + prediction_length

            # Guard: future must not spill outside the normal zone
            if future_end > zone_end:
                break

            pairs.append({
                "context": {"target": data[:, start:context_end]},
                "future" : {"target": data[:, context_end:future_end]},
                "type"   : "normal_to_normal",
            })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type B — Pre-Anomaly Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_b_pairs(
    data: np.ndarray,
    anomaly_labels: np.ndarray,
    anomaly_boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    pre_anomaly_offsets: list[int],
) -> list[dict]:
    """
    Type B — Pre-Anomaly instances.

    Context is entirely in the normal zone immediately before an anomaly.
    Future window begins just before (or at) the anomaly onset and extends
    into it by 'offset' steps.

    Multiple offset values produce multiple pairs per anomaly event,
    which is critical because anomalies are rare in the dataset.

    Teaches the model: "given normal history, the correct future is still
    normal" — the gap between prediction and actual anomalous values
    becomes the anomaly detection signal at inference time.

    Parameters
    ----------
    pre_anomaly_offsets : list of int
        How many steps INTO the anomaly the future window's end reaches.
        offset=1            -> future mostly normal, 1 anomaly step at end.
        offset=pred_length  -> future fully inside anomaly region.
    """
    pairs = []
    total_length = len(anomaly_labels)

    for anom_idx, (anom_start, anom_end) in enumerate(anomaly_boundaries):

        # Safe start of the preceding normal zone
        preceding_normal_start = anomaly_boundaries[anom_idx - 1][1] if anom_idx > 0 else 0

        for offset in pre_anomaly_offsets:

            # Future window: ends 'offset' steps into the anomaly
            future_end   = anom_start + offset
            future_start = future_end - prediction_length
            context_end  = future_start

            # ── Boundary checks ───────────────────────────────────────────── #
            # Clamp context_start to the available normal zone; this allows
            # shorter-than-context_length contexts when the anomaly occurs early
            # in the series or close after a previous anomaly.
            context_start = max(context_end - context_length,
                                preceding_normal_start, 0)

            # Skip if the available normal context is too short to be useful
            if context_end <= 0 or (context_end - context_start) < prediction_length:
                continue

            if future_end > total_length:
                continue                           # future goes beyond series end

            # Context must be entirely normal (hard constraint)
            if np.any(anomaly_labels[context_start:context_end] == 1):
                continue

            pairs.append({
                "context": {"target": data[:, context_start:context_end]},
                "future" : {"target": data[:, future_start:future_end]},
                "type"   : "pre_anomaly",
            })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type C — Anomaly-Context Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_c_pairs(
    data: np.ndarray,
    anomaly_labels: np.ndarray,
    anomaly_boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    normal_lead: int,
    normal_tail: int,
) -> list[dict]:
    """
    Type C — Anomaly-Context instances.

    Context window CONTAINS the anomaly event (with a normal lead-in and
    normal tail after it). Future window is the post-anomaly normal behavior.

    Teaches the model: "after an anomaly, what does recovery look like."
    Also prevents the model from confusing anomaly residuals with normal patterns.

    Parameters
    ----------
    normal_lead : int
        Normal timesteps BEFORE the anomaly to include in the context window.
    normal_tail : int
        Normal timesteps AFTER the anomaly to include in the context window,
        before the future window begins.
    """
    pairs = []
    total_length = len(anomaly_labels)

    for anom_idx, (anom_start, anom_end) in enumerate(anomaly_boundaries):

        # ── Build context window ─────────────────────────────────────────── #
        # Cap context_end so the future window always fits inside the series.
        # Without this cap, context_end = context_start + context_length can
        # push future_end beyond total_length, silently dropping all Type C
        # pairs for anomalies near the end of the series.
        max_context_end = total_length - prediction_length
        context_end = min(anom_end + normal_tail, max_context_end)

        if context_end <= anom_start:
            continue                              # anomaly too close to series end

        context_start = max(0, context_end - context_length)
        context_end   = context_start + context_length  # exact length (safe: total_length >= ctx+pred)

        # ── Build future window ──────────────────────────────────────────── #
        future_start = context_end
        future_end   = future_start + prediction_length

        # ── Boundary checks ──────────────────────────────────────────────── #
        if future_end > total_length:
            continue                              # not enough series after anomaly

        # Context must actually contain the anomaly
        if not np.any(anomaly_labels[context_start:context_end] == 1):
            continue

        # Future must not overlap with the NEXT anomaly
        if anom_idx + 1 < len(anomaly_boundaries):
            next_anom_start = anomaly_boundaries[anom_idx + 1][0]
            if future_end > next_anom_start:
                continue

        # Future must be entirely normal
        if np.any(anomaly_labels[future_start:future_end] == 1):
            continue

        pairs.append({
            "context": {"target": data[:, context_start:context_end]},
            "future" : {"target": data[:, future_start:future_end]},
            "type"   : "anomaly_context",
        })

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Balancing and Shuffling
# ─────────────────────────────────────────────────────────────────────────────

def balance_and_shuffle(
    type_a_pairs: list[dict],
    type_b_pairs: list[dict],
    type_c_pairs: list[dict],
    ratio_a: float,
    ratio_b: float,
    ratio_c: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Balance the three types to the target ratio, then shuffle.

    Strategy:
      - Type B and C are rare — always keep ALL of them.
      - Type A is abundant — downsample to match the target ratio
        relative to how many B and C pairs are available.

    Parameters
    ----------
    ratio_a / ratio_b / ratio_c : target proportions, must sum to 1.0
    """
    n_a = len(type_a_pairs)
    n_b = len(type_b_pairs)
    n_c = len(type_c_pairs)

    logger.info(
        f"  Before balancing — "
        f"A (normal): {n_a}  B (pre-anom): {n_b}  C (anom-ctx): {n_c}  "
        f"| target {ratio_a:.0%}/{ratio_b:.0%}/{ratio_c:.0%}"
    )

    # Edge case: no rare pairs — return all Type A unbalanced
    if n_b == 0 and n_c == 0:
        logger.warning(
            "No Type B or C pairs found for this series. "
            "Returning all Type A pairs. "
            "Check if is_anomaly column exists and has positive labels."
        )
        return [type_a_pairs[i] for i in rng.permutation(n_a)]

    # Determine target counts to satisfy the ratio.
    # Two cases:
    #   (a) A is abundant  → downsample A, keep all B and C.
    #   (b) A is limiting  → keep all A, downsample B and C proportionally.
    # This ensures the per-series ratio is always close to the target,
    # preventing B/C from dominating when A pairs are scarce.
    rare_total = n_b + n_c
    rare_ratio = ratio_b + ratio_c
    n_a_target = int(rare_total * ratio_a / rare_ratio) if rare_ratio > 0 else n_a

    if n_a >= n_a_target:
        # Case (a): enough A — downsample A, keep all B and C
        n_a_sampled = n_a_target
        n_b_sampled = n_b
        n_c_sampled = n_c
    else:
        # Case (b): A is the bottleneck — scale B and C down to match
        n_a_sampled = n_a
        n_b_sampled = min(n_b, int(n_a * ratio_b / ratio_a))
        n_c_sampled = min(n_c, int(n_a * ratio_c / ratio_a))

    a_indices = rng.choice(n_a, size=n_a_sampled, replace=False)
    b_indices = rng.choice(n_b, size=n_b_sampled, replace=False)
    c_indices = rng.choice(n_c, size=n_c_sampled, replace=False)

    final_pairs = (
        [type_a_pairs[i] for i in a_indices]
        + [type_b_pairs[i] for i in b_indices]
        + [type_c_pairs[i] for i in c_indices]
    )

    logger.info(
        f"  After  balancing — "
        f"A: {n_a_sampled}  B: {n_b_sampled}  C: {n_c_sampled}  Total: {len(final_pairs)}"
    )

    # Shuffle so all types are interleaved during training
    return [final_pairs[i] for i in rng.permutation(len(final_pairs))]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-Series Pair Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_for_series(
    data: np.ndarray,
    anomaly_labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
    pre_anomaly_offsets: list[int],
    normal_lead: int,
    normal_tail: int,
    ratio_a: float,
    ratio_b: float,
    ratio_c: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Build all three types of instruction pairs for a single time series,
    then balance and shuffle.

    Parameters
    ----------
    data           : feature array, shape (num_features, time_steps)
    anomaly_labels : binary label array, shape (time_steps,)

    Returns
    -------
    Balanced and shuffled list of instruction pair dicts.
    Each dict has keys: 'context', 'future', 'type'
      - 'context' and 'future' each contain {'target': array(F, length)}
      - 'type' is for analysis only — strip before passing to pipeline.fit()
    """
    anomaly_boundaries = extract_anomaly_boundaries(anomaly_labels)
    normal_zones       = get_normal_zones(anomaly_boundaries, len(anomaly_labels))

    logger.debug(
        f"  Anomaly events: {len(anomaly_boundaries)} | "
        f"Normal zones: {len(normal_zones)}"
    )

    type_a = create_type_a_pairs(
        data=data,
        normal_zones=normal_zones,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )

    type_b = create_type_b_pairs(
        data=data,
        anomaly_labels=anomaly_labels,
        anomaly_boundaries=anomaly_boundaries,
        context_length=context_length,
        prediction_length=prediction_length,
        pre_anomaly_offsets=pre_anomaly_offsets,
    )

    type_c = create_type_c_pairs(
        data=data,
        anomaly_labels=anomaly_labels,
        anomaly_boundaries=anomaly_boundaries,
        context_length=context_length,
        prediction_length=prediction_length,
        normal_lead=normal_lead,
        normal_tail=normal_tail,
    )

    return balance_and_shuffle(
        type_a_pairs=type_a,
        type_b_pairs=type_b,
        type_c_pairs=type_c,
        ratio_a=ratio_a,
        ratio_b=ratio_b,
        ratio_c=ratio_c,
        rng=rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main Preparation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    context_length: int,
    prediction_length: int,
    stride: int,
    ratio_a: float,
    ratio_b: float,
    ratio_c: float,
    seed: int = 42,
):
    """
    Full pipeline:
      1. Load all *train.csv files (features + anomaly labels separately)
      2. Split into train / val at the series level
      3. For each split, build three-type instruction pairs per series
      4. Return raw series lists and instruction pair lists

    Returns
    -------
    train_inputs : list of {'target': array(F, T)}  — original format
    val_inputs   : list of {'target': array(F, T)}  — original format
    train_pairs  : list of {'context':..., 'future':..., 'type':...}
    val_pairs    : list of {'context':..., 'future':..., 'type':...}
    """
    rng = np.random.default_rng(seed)

    # Use only *test.csv files: these carry the anomaly labels (is_anomaly=1)
    # required for Type B and Type C pair generation.
    csv_files = sorted(
        glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True)
    )
    logger.info(f"Found {len(csv_files)} *test.csv files under {data_dir}")

    # Pre-anomaly offsets: multiple windows per anomaly event
    # Covers the full spectrum from "barely touching" to "fully inside" anomaly
    pre_anomaly_offsets = sorted(set([
        1,
        max(1, prediction_length // 4),
        max(1, prediction_length // 2),
        prediction_length,
    ]))
    logger.info(f"Pre-anomaly offsets (Type B): {pre_anomaly_offsets}")

    # Normal lead/tail for Type C context construction
    normal_lead = max(10, context_length // 4)
    normal_tail = max(5,  prediction_length // 4)
    logger.info(f"Type C — normal_lead={normal_lead}, normal_tail={normal_tail}")

    # ── Step 1: Load all CSVs ────────────────────────────────────────────── #
    all_inputs         = []   # list of {'target': array(F, T)}
    all_anomaly_labels = []   # parallel list of anomaly label arrays
    skipped = 0

    min_required = max(min_length, context_length + prediction_length)

    for path in csv_files:
        try:
            features, anomaly_labels = load_csv_as_multivariate(path)

            if features is None or features.shape[1] < min_required:
                logger.debug(
                    f"Skipping {os.path.basename(path)}: "
                    f"length={features.shape[1] if features is not None else 'None'}, "
                    f"required={min_required}"
                )
                skipped += 1
                continue

            all_inputs.append({"target": features})
            all_anomaly_labels.append(anomaly_labels)

        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1

    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")

    if len(all_inputs) == 0:
        raise ValueError("No usable series found. Check data_dir and min_length.")

    # ── Step 2: Train / Val split ────────────────────────────────────────── #
    idx     = rng.permutation(len(all_inputs))
    n_val   = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())

    train_inputs = [all_inputs[i]         for i in range(len(all_inputs)) if i not in val_set]
    val_inputs   = [all_inputs[i]         for i in val_set]
    train_labels = [all_anomaly_labels[i] for i in range(len(all_inputs)) if i not in val_set]
    val_labels   = [all_anomaly_labels[i] for i in val_set]

    logger.info(f"Train series: {len(train_inputs)} | Val series: {len(val_inputs)}")

    # ── Step 3: Build instruction pairs ─────────────────────────────────── #
    logger.info(
        f"Building instruction pairs — "
        f"context={context_length}, pred={prediction_length}, stride={stride}, "
        f"ratio A/B/C = {ratio_a:.0%}/{ratio_b:.0%}/{ratio_c:.0%}"
    )

    train_pairs = []
    for i, (series, labels) in enumerate(zip(train_inputs, train_labels)):
        logger.debug(f"Train series {i+1}/{len(train_inputs)}")
        pairs = build_pairs_for_series(
            data=series["target"],
            anomaly_labels=labels,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            pre_anomaly_offsets=pre_anomaly_offsets,
            normal_lead=normal_lead,
            normal_tail=normal_tail,
            ratio_a=ratio_a,
            ratio_b=ratio_b,
            ratio_c=ratio_c,
            rng=rng,
        )
        train_pairs.extend(pairs)

    val_pairs = []
    for i, (series, labels) in enumerate(zip(val_inputs, val_labels)):
        logger.debug(f"Val series {i+1}/{len(val_inputs)}")
        pairs = build_pairs_for_series(
            data=series["target"],
            anomaly_labels=labels,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            pre_anomaly_offsets=pre_anomaly_offsets,
            normal_lead=normal_lead,
            normal_tail=normal_tail,
            ratio_a=ratio_a,
            ratio_b=ratio_b,
            ratio_c=ratio_c,
            rng=rng,
        )
        val_pairs.extend(pairs)

    logger.info(
        f"Instruction pairs — Train: {len(train_pairs)} | Val: {len(val_pairs)}"
    )

    return train_inputs, val_inputs, train_pairs, val_pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_dataset_statistics(
    train_inputs: list,
    val_inputs: list,
    train_pairs: list,
    val_pairs: list,
) -> None:
    """Log shape and type-distribution statistics for the full dataset."""

    all_series = train_inputs + val_inputs
    lengths  = [s["target"].shape[1] for s in all_series]
    variates = [s["target"].shape[0] for s in all_series]

    logger.info("=" * 60)
    logger.info("RAW SERIES STATISTICS")
    logger.info(f"  Total series : {len(all_series)}")
    logger.info(f"  Time steps   : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
    logger.info(f"  Num features : min={min(variates)}, max={max(variates)}, mean={np.mean(variates):.1f}")

    all_pairs = train_pairs + val_pairs
    if all_pairs:
        type_counts: dict[str, int] = {}
        for p in all_pairs:
            t = p.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        ctx_shape = all_pairs[0]["context"]["target"].shape
        fut_shape = all_pairs[0]["future"]["target"].shape

        logger.info("=" * 60)
        logger.info("INSTRUCTION PAIR STATISTICS")
        logger.info(f"  Total pairs          : {len(all_pairs)}")
        logger.info(f"  Train pairs          : {len(train_pairs)}")
        logger.info(f"  Val pairs            : {len(val_pairs)}")
        logger.info(f"  Context shape        : {ctx_shape}  [features x context_length]")
        logger.info(f"  Future  shape        : {fut_shape}  [features x pred_length]")
        logger.info(f"  Avg pairs per series : {len(all_pairs) / len(all_series):.1f}")
        logger.info("  Type distribution:")
        for type_name, count in sorted(type_counts.items()):
            pct = count / len(all_pairs) * 100
            logger.info(f"    {type_name:<22} : {count:>6}  ({pct:.1f}%)")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

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
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Number of past time steps used as context (instruction)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=64,
        help="Number of future time steps to predict (output)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Window stride for Type A sliding window. "
            "Defaults to prediction_length // 2 if not set."
        ),
    )
    parser.add_argument(
        "--ratio_a",
        type=float,
        default=0.60,
        help="Target fraction of Normal-to-Normal (Type A) pairs. Default: 0.60",
    )
    parser.add_argument(
        "--ratio_b",
        type=float,
        default=0.30,
        help="Target fraction of Pre-Anomaly (Type B) pairs. Default: 0.30",
    )
    parser.add_argument(
        "--ratio_c",
        type=float,
        default=0.10,
        help="Target fraction of Anomaly-Context (Type C) pairs. Default: 0.10",
    )

    args = parser.parse_args()

    # Validate ratios sum to 1.0
    total_ratio = args.ratio_a + args.ratio_b + args.ratio_c
    if not np.isclose(total_ratio, 1.0, atol=1e-3):
        raise ValueError(
            f"ratio_a + ratio_b + ratio_c must sum to 1.0 (got {total_ratio:.3f})"
        )

    # Default stride = prediction_length // 2
    if args.stride is None:
        args.stride = max(1, args.prediction_length // 2)
        logger.info(f"Stride not set — using default: stride={args.stride}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_inputs, val_inputs, train_pairs, val_pairs = prepare_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        ratio_a=args.ratio_a,
        ratio_b=args.ratio_b,
        ratio_c=args.ratio_c,
    )

    # ── Save outputs ─────────────────────────────────────────────────────── #
    #
    #  train_inputs.pkl / val_inputs.pkl
    #    Original format → list of {'target': array(F, T)}
    #    Use directly with pipeline.fit() for baseline training
    #
    #  train_pairs.pkl / val_pairs.pkl
    #    Three-type instruction pairs for fine-tuning
    #    Format: {'context': {'target': array(F, ctx)},
    #             'future':  {'target': array(F, pred)},
    #             'type':    str}
    #    NOTE: Strip the 'type' key before passing to pipeline.fit()
    #
    # ─────────────────────────────────────────────────────────────────────── #
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

    log_dataset_statistics(train_inputs, val_inputs, train_pairs, val_pairs)


if __name__ == "__main__":
    main()
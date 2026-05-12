"""
Two-stage data preparation for Chronos-2 anomaly fine-tuning on mTSBench.

Loads all *test.csv files from the mTSBench dataset directory, extracts feature
columns (excluding timestamp and is_anomaly), and saves train/val splits as
pickle files ready for chronos2 pipeline.fit().

NOTE: mTSBench *train.csv files contain only normal data (is_anomaly=0 throughout).
Anomaly labels (required for Type B/C/D pairs) only exist in *test.csv files.

Creates FOUR types of instruction tuning pairs using anomaly ground truth labels.
ALL pairs from each type are kept as-is (no balancing or downsampling) so that
the raw pair counts per type are visible for analysis.

  Stage 1 — normal futures (gradient descent, loss minimised):
    Type A — Normal-to-Normal   : context=normal,  future=normal
    Type C — Anomaly-Context    : context=anomaly, future=normal

  Stage 2 — anomalous futures (gradient ascent, loss maximised):
    Type B — Pre-Anomaly        : context=normal,  future=anomaly onset
    Type D — Anomaly-to-Anomaly : context=anomaly, future=anomaly

At inference time, high prediction error on a region => high anomaly score.

Usage:
    python inst_data_prepare.py [--data_dir ...] [--output_dir ...]
                                [--pre_anomaly_offsets 1 16 32 64 128]
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
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(csv_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load one *test.csv file.

    Returns
    -------
    features : float32 array (n_variates, time_steps) — timestamp/is_anomaly excluded.
    labels   : int32 array (time_steps,), 1=anomaly 0=normal (all-zero if column absent).
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "is_anomaly")]
    if not feature_cols:
        return None, None
    try:
        features = df[feature_cols].values.T.astype(np.float32)
        labels = df["is_anomaly"].values.astype(np.int32) if "is_anomaly" in df.columns \
            else np.zeros(df.shape[0], dtype=np.int32)
        return features, labels
    except Exception as e:
        logger.warning(f"Error processing {csv_path}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly Boundary Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(labels: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous anomaly regions from the binary label array.

    Returns list of (start, end) where end is EXCLUSIVE (Python-slice style).
    Example: [0,0,0,1,1,1,0,0,1,1,0] → [(3,6), (8,10)]
    """
    boundaries, in_anom, start = [], False, 0
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            in_anom, start = True, i
        elif v == 0 and in_anom:
            in_anom = False
            boundaries.append((start, i))
    if in_anom:
        boundaries.append((start, len(labels)))
    return boundaries


def get_normal_zones(boundaries: list[tuple[int, int]], total: int) -> list[tuple[int, int]]:
    """
    Return normal (non-anomaly) zones as (start, end) pairs.

    Example: boundaries=[(3,6),(8,10)], total=12 → [(0,3),(6,8),(10,12)]
    """
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(
    data: np.ndarray,
    normal_zones: list[tuple[int, int]],
    length: int,
) -> np.ndarray | None:
    """
    Return a (F, length) reference normal signal sampled from the series' normal zones.

    Strategy:
      1. If a single normal zone is long enough, take its last `length` timesteps.
      2. Otherwise concatenate normal zones (longest first) until we have enough.
      3. If still short, left-pad with NaN — the model masks NaN inputs out.

    Returns None if there are no normal zones at all.
    """
    if not normal_zones:
        return None

    sorted_zones = sorted(normal_zones, key=lambda z: z[1] - z[0], reverse=True)
    s, e = sorted_zones[0]
    if e - s >= length:
        return data[:, e - length:e].astype(np.float32, copy=False)

    chunks = []
    collected = 0
    for s, e in sorted_zones:
        chunks.append(data[:, s:e])
        collected += e - s
        if collected >= length:
            break

    combined = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if combined.shape[1] >= length:
        return combined[:, -length:]

    F = combined.shape[0]
    pad = np.full((F, length - combined.shape[1]), np.nan, dtype=np.float32)
    return np.concatenate([pad, combined], axis=1)


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
    Type A — Normal-to-Normal.

    Sliding window applied ONLY within normal zones; both context and future
    are guaranteed anomaly-free. Teaches the model what normal continuation looks like.
    """
    pairs, win = [], context_length + prediction_length
    for zs, ze in normal_zones:
        if ze - zs < win:
            continue
        for s in range(zs, ze - win + 1, stride):
            ce = s + context_length
            if ce + prediction_length > ze:
                break
            pairs.append({"context": {"target": data[:, s:ce]},
                          "future":  {"target": data[:, ce:ce + prediction_length]},
                          "type": "normal_to_normal"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type B — Pre-Anomaly Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_b_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    pre_anomaly_offsets: list[int],
) -> list[dict]:
    """
    Type B — Pre-Anomaly (normal context → anomaly onset).

    Context is entirely normal; future window begins just before the anomaly onset
    and extends into it by 'offset' steps. Multiple offsets per event compensate
    for anomaly rarity. The gap between model prediction and actual anomalous values
    becomes the anomaly detection signal at inference time.

    pre_anomaly_offsets : steps INTO the anomaly the future window's end reaches.
    """
    pairs, total = [], len(labels)
    seen: set[tuple[int, int, int, int]] = set()

    for idx, (anom_s, anom_e) in enumerate(boundaries):
        prev_end = boundaries[idx - 1][1] if idx > 0 else 0
        anom_len = anom_e - anom_s

        for offset in pre_anomaly_offsets:
            # Clamp offset to the actual anomaly length so the future window never
            # spills past this anomaly's end into a subsequent normal or anomaly region.
            effective_offset = min(offset, anom_len)
            if effective_offset <= 0:
                continue

            fe = anom_s + effective_offset
            fs = fe - prediction_length
            ce = fs
            # Clamp context start to the available normal zone
            cs = max(ce - context_length, prev_end, 0)
            if ce <= 0 or (ce - cs) < prediction_length or fe > total:
                continue
            if np.any(labels[cs:ce] == 1):   # context must be entirely normal
                continue

            # Clamping can map multiple offsets to the same window for short anomalies;
            # skip duplicates to avoid redundant training examples.
            key = (cs, ce, fs, fe)
            if key in seen:
                continue
            seen.add(key)

            pairs.append({"context": {"target": data[:, cs:ce]},
                          "future":  {"target": data[:, fs:fe]},
                          "type": "pre_anomaly"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type C — Anomaly-Context Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_c_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    boundaries: list[tuple[int, int]],
    context_length: int,
    prediction_length: int,
    normal_lead: int,
    normal_tail: int,
) -> list[dict]:
    """
    Type C — Anomaly-Context (anomaly context → normal future).

    Context window CONTAINS the anomaly event (with a normal lead-in and tail).
    Future window is post-anomaly normal behavior. Teaches the model what recovery
    looks like and prevents confusing anomaly residuals with normal patterns.

    normal_lead : normal timesteps BEFORE the anomaly to include in context.
    normal_tail : normal timesteps AFTER the anomaly before the future window starts.
    """
    pairs, total = [], len(labels)
    for idx, (anom_s, anom_e) in enumerate(boundaries):
        # Cap context_end so the future window always fits inside the series.
        max_ce = total - prediction_length
        ce = min(anom_e + normal_tail, max_ce)
        if ce <= anom_s:
            continue                              # anomaly too close to series end

        cs = max(0, ce - context_length)
        ce = cs + context_length                  # normalise to exact context_length

        # Enforce normal_lead: context must have at least normal_lead normal steps
        # before the anomaly starts, otherwise the lead-in is uninformative.
        if anom_s - cs < normal_lead:
            continue

        # Skip pairs where the normalised ce has drifted far past the intended
        # anom_e + normal_tail boundary (happens when the anomaly is early in the
        # series and cs clamps to 0, pushing ce to context_length). In that case
        # the future window starts hundreds of steps after the anomaly rather than
        # normal_tail steps, defeating the purpose of this pair type.
        if ce > anom_e + normal_tail + context_length // 2:
            continue

        fs, fe = ce, ce + prediction_length
        if not np.any(labels[cs:ce] == 1):        # context must contain the anomaly
            continue
        if idx + 1 < len(boundaries) and fe > boundaries[idx + 1][0]:
            continue                              # future must not reach next anomaly
        if np.any(labels[fs:fe] == 1):            # future must be entirely normal
            continue
        pairs.append({"context": {"target": data[:, cs:ce]},
                      "future":  {"target": data[:, fs:fe]},
                      "type": "anomaly_context"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Type D — Anomaly-to-Anomaly Pairs
# ─────────────────────────────────────────────────────────────────────────────

def create_type_d_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
) -> list[dict]:
    """
    Type D — Anomaly-to-Anomaly.

    Sliding window; kept only when BOTH context and future contain anomalous
    timesteps. Covers two natural scenarios:
      1. A long anomaly spanning both windows.
      2. Adjacent anomalies — context contains one event, future reaches the next.

    The future must contain at least prediction_length // 4 anomalous steps.
    Using np.any (1 step minimum) would admit futures that are overwhelmingly
    normal, diluting the Stage 2 gradient-ascent signal and causing the model
    to learn high error on normal timesteps — the opposite of the intent.

    Used exclusively in Stage 2 (gradient ascent).
    """
    pairs, win = [], context_length + prediction_length
    # Minimum anomaly content required in the future window so that gradient
    # ascent is dominated by anomalous timesteps, not normal ones.
    min_fut_anom = max(1, prediction_length // 4)

    for s in range(0, data.shape[1] - win + 1, stride):
        ce, fe = s + context_length, s + win
        if (np.any(labels[s:ce])
                and np.sum(labels[ce:fe]) >= min_fut_anom):
            pairs.append({"context": {"target": data[:, s:ce]},
                          "future":  {"target": data[:, ce:fe]},
                          "type": "anomaly_to_anomaly"})
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Model-Ready Input Conversion
# ─────────────────────────────────────────────────────────────────────────────

def pairs_to_model_inputs(
    pairs: list[dict],
    context_length: int,
    normal_signal_length: int = 0,
) -> list[dict]:
    """
    Convert instruction pairs to fixed-length model inputs for pipeline.fit().

    When `normal_signal_length > 0`, each pair must contain a `normal_signal` field
    of shape (F, normal_signal_length). The output `target` is laid out as:

        [normal_signal (N) | context (C) | future (P)]

    so the total target length is N + C + P. At training time, set
    `context_length = N + C` and `min_past = N + C` so that the dataset slices
    exactly at the future boundary, and pass `sep_patch_index = N // input_patch_size`
    to the model so it inserts [SEP] between the normal and context patches.

    Short Type B contexts are NaN-padded on the left so the model masks them out.
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        F, clen = ctx.shape
        if clen < context_length:
            pad = np.full((F, context_length - clen), np.nan, dtype=np.float32)
            ctx = np.concatenate([pad, ctx], axis=1)
        elif clen > context_length:
            ctx = ctx[:, -context_length:]

        if normal_signal_length > 0:
            normal = p.get("normal_signal", None)
            if normal is None:
                # No normal zone available for this series — fill with NaN; model masks it out.
                normal = np.full((F, normal_signal_length), np.nan, dtype=np.float32)
            elif normal.shape[1] != normal_signal_length:
                raise ValueError(
                    f"normal_signal length {normal.shape[1]} != expected {normal_signal_length}"
                )
            target = np.concatenate([normal, ctx, fut], axis=1)
        else:
            target = np.concatenate([ctx, fut], axis=1)

        out.append({"target": target})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Stage-Specific Per-Series Pair Construction
# ─────────────────────────────────────────────────────────────────────────────

def _attach_normal_signal(pairs: list[dict], normal_sig: np.ndarray | None) -> None:
    """In-place: add the same per-series normal_signal reference to every pair."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def build_stage1_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
    normal_lead: int,
    normal_tail: int,
    normal_signal_length: int = 0,
) -> list[dict]:
    """
    Stage 1 — ALL pairs with NORMAL futures (gradient descent).

    Type A: context=normal,  future=normal
    Type C: context=anomaly, future=normal

    Returns every pair produced by both types with no downsampling, so the
    raw counts are preserved for analysis.
    """
    bounds = extract_anomaly_boundaries(labels)
    zones  = get_normal_zones(bounds, len(labels))
    type_a = create_type_a_pairs(data, zones, context_length, prediction_length, stride)
    type_c = create_type_c_pairs(data, labels, bounds, context_length, prediction_length,
                                  normal_lead, normal_tail)
    pairs = type_a + type_c
    if normal_signal_length > 0:
        normal_sig = extract_normal_signal(data, zones, normal_signal_length)
        _attach_normal_signal(pairs, normal_sig)
    logger.debug(f"  Stage 1 — A: {len(type_a)}  C: {len(type_c)}")
    return pairs


def build_stage2_pairs(
    data: np.ndarray,
    labels: np.ndarray,
    context_length: int,
    prediction_length: int,
    stride: int,
    pre_anomaly_offsets: list[int],
    normal_signal_length: int = 0,
) -> list[dict]:
    """
    Stage 2 — ALL pairs with ANOMALOUS futures (gradient ascent).

    Type B: context=normal,  future=anomaly onset
    Type D: context=anomaly, future=anomaly

    Returns every pair produced by both types with no downsampling, so the
    raw counts are preserved for analysis.
    """
    bounds = extract_anomaly_boundaries(labels)
    type_b = create_type_b_pairs(data, labels, bounds, context_length, prediction_length,
                                  pre_anomaly_offsets)
    type_d = create_type_d_pairs(data, labels, context_length, prediction_length, stride)
    pairs = type_b + type_d
    if normal_signal_length > 0:
        zones = get_normal_zones(bounds, len(labels))
        normal_sig = extract_normal_signal(data, zones, normal_signal_length)
        _attach_normal_signal(pairs, normal_sig)
    logger.debug(f"  Stage 2 — B: {len(type_b)}  D: {len(type_d)}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
#  Two-Stage Preparation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_two_stage_inputs(
    data_dir: str,
    min_length: int,
    val_fraction: float,
    context_length: int,
    prediction_length: int,
    stride: int,
    pre_anomaly_offsets: list[int],
    seed: int = 42,
    normal_signal_length: int = 0,
):
    """
    Produce separate Stage 1 and Stage 2 datasets for two-stage anomaly fine-tuning.

    Stage 1 — NORMAL futures (gradient descent, loss minimised):
        Type A: normal context  → normal future
        Type C: anomaly context → normal future

    Stage 2 — ANOMALOUS futures (gradient ascent, loss maximised):
        Type B: normal context  → anomaly future (onset)
        Type D: anomaly context → anomaly future

    All pairs are kept as-is — no balancing or downsampling — so raw type counts
    are visible in the statistics log.

    Returns
    -------
    train_inputs, val_inputs,
    s1_train_pairs, s1_val_pairs, s2_train_pairs, s2_val_pairs,
    s1_train_model_inputs, s1_val_model_inputs,
    s2_train_model_inputs, s2_val_model_inputs
    """
    rng = np.random.default_rng(seed)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*test.csv"), recursive=True))
    logger.info(f"Found {len(csv_files)} *test.csv files under {data_dir}")

    # Normal lead/tail for Type C context construction
    normal_lead = max(10, context_length // 4)
    normal_tail = max(5,  prediction_length // 4)
    logger.info(f"Pre-anomaly offsets (Type B): {pre_anomaly_offsets}")
    logger.info(f"Type C — normal_lead={normal_lead}, normal_tail={normal_tail}")

    # ── Load all CSVs ────────────────────────────────────────────────────── #
    all_inputs, all_labels, skipped = [], [], 0
    min_req = max(min_length, context_length + prediction_length)
    for path in csv_files:
        try:
            feat, lbl = load_csv_as_multivariate(path)
            if feat is None or feat.shape[1] < min_req:
                logger.debug(
                    f"Skipping {os.path.basename(path)}: "
                    f"length={feat.shape[1] if feat is not None else 'None'}, required={min_req}"
                )
                skipped += 1
                continue
            all_inputs.append({"target": feat})
            all_labels.append(lbl)
        except Exception as exc:
            logger.warning(f"Skipping {path}: {exc}")
            skipped += 1
    logger.info(f"Usable series: {len(all_inputs)}  (skipped {skipped})")
    if not all_inputs:
        raise ValueError("No usable series found. Check data_dir and min_length.")

    # ── Train / Val split ────────────────────────────────────────────────── #
    idx     = rng.permutation(len(all_inputs))
    n_val   = max(1, int(len(all_inputs) * val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_inputs = [all_inputs[i] for i in range(len(all_inputs)) if i not in val_set]
    val_inputs   = [all_inputs[i] for i in val_set]
    train_labels = [all_labels[i] for i in range(len(all_inputs)) if i not in val_set]
    val_labels   = [all_labels[i] for i in val_set]
    logger.info(f"Train series: {len(train_inputs)} | Val series: {len(val_inputs)}")

    # ── Build stage-specific pairs (all pairs, no balancing) ─────────────── #
    common = dict(context_length=context_length, prediction_length=prediction_length,
                  stride=stride)

    s1_tr, s2_tr = [], []
    for i, (series, lbl) in enumerate(zip(train_inputs, train_labels)):
        logger.debug(f"Train series {i + 1}/{len(train_inputs)}")
        s1_tr.extend(build_stage1_pairs(series["target"], lbl, **common,
                                         normal_lead=normal_lead, normal_tail=normal_tail,
                                         normal_signal_length=normal_signal_length))
        s2_tr.extend(build_stage2_pairs(series["target"], lbl, **common,
                                         pre_anomaly_offsets=pre_anomaly_offsets,
                                         normal_signal_length=normal_signal_length))

    s1_val, s2_val = [], []
    for i, (series, lbl) in enumerate(zip(val_inputs, val_labels)):
        logger.debug(f"Val series {i + 1}/{len(val_inputs)}")
        s1_val.extend(build_stage1_pairs(series["target"], lbl, **common,
                                          normal_lead=normal_lead, normal_tail=normal_tail,
                                          normal_signal_length=normal_signal_length))
        s2_val.extend(build_stage2_pairs(series["target"], lbl, **common,
                                          pre_anomaly_offsets=pre_anomaly_offsets,
                                          normal_signal_length=normal_signal_length))

    logger.info(f"Stage 1 pairs — Train: {len(s1_tr)} | Val: {len(s1_val)}")
    logger.info(f"Stage 2 pairs — Train: {len(s2_tr)} | Val: {len(s2_val)}")

    # ── Convert to fixed-length model inputs ─────────────────────────────── #
    # Each entry: {'target': array(F, context_length + prediction_length)}.
    # Short Type B contexts are NaN-padded on the left; pass to pipeline.fit()
    # with min_past=context_length to lock the cut at the intended boundary.
    logger.info("Converting to fixed-length model inputs (NaN-padding short contexts)...")
    s1_tr_in  = pairs_to_model_inputs(s1_tr,  context_length, normal_signal_length=normal_signal_length)
    s1_val_in = pairs_to_model_inputs(s1_val, context_length, normal_signal_length=normal_signal_length)
    s2_tr_in  = pairs_to_model_inputs(s2_tr,  context_length, normal_signal_length=normal_signal_length)
    s2_val_in = pairs_to_model_inputs(s2_val, context_length, normal_signal_length=normal_signal_length)
    logger.info(f"Stage 1 model inputs — Train: {len(s1_tr_in)} | Val: {len(s1_val_in)}")
    logger.info(f"Stage 2 model inputs — Train: {len(s2_tr_in)} | Val: {len(s2_val_in)}")

    return (train_inputs, val_inputs,
            s1_tr, s1_val, s2_tr, s2_val,
            s1_tr_in, s1_val_in, s2_tr_in, s2_val_in)


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_statistics(
    train_inputs: list,
    val_inputs: list,
    s1_train_pairs: list,
    s1_val_pairs: list,
    s2_train_pairs: list,
    s2_val_pairs: list,
) -> None:
    """Log shape and type-distribution statistics for both stages."""
    series   = train_inputs + val_inputs
    lengths  = [s["target"].shape[1] for s in series]
    variates = [s["target"].shape[0] for s in series]

    logger.info("=" * 60)
    logger.info("RAW SERIES STATISTICS")
    logger.info(f"  Total series : {len(series)}")
    logger.info(f"  Time steps   : min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f}")
    logger.info(f"  Num features : min={min(variates)}, max={max(variates)}, mean={np.mean(variates):.1f}")

    for stage_name, tr_pairs, val_pairs in [
        ("STAGE 1", s1_train_pairs, s1_val_pairs),
        ("STAGE 2", s2_train_pairs, s2_val_pairs),
    ]:
        all_pairs = tr_pairs + val_pairs
        if not all_pairs:
            continue
        counts: dict[str, int] = {}
        for p in all_pairs:
            t = p.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
        logger.info("=" * 60)
        logger.info(f"{stage_name} INSTRUCTION PAIR STATISTICS")
        logger.info(f"  Train: {len(tr_pairs)}  Val: {len(val_pairs)}  Total: {len(all_pairs)}")
        logger.info(f"  Avg per series : {len(all_pairs) / len(series):.1f}")
        logger.info("  Type distribution (raw counts, no balancing):")
        for type_name, count in sorted(counts.items()):
            logger.info(f"    {type_name:<22} : {count:>6}  ({count / len(all_pairs) * 100:.1f}%)")
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Two-stage data prep for Chronos-2 anomaly fine-tuning."
    )
    p.add_argument("--data_dir",          default="/home/rajib/mTSBench/Datasets/mTSBench",
                   help="Root directory of the mTSBench dataset")
    p.add_argument("--output_dir",        default="./prepared_data",
                   help="Root output directory; stage1/ and stage2/ subdirs are created inside")
    p.add_argument("--min_length",        type=int,   default=50,
                   help="Minimum series length; shorter series are discarded")
    p.add_argument("--val_fraction",      type=float, default=0.1,
                   help="Fraction of series held out for validation")
    p.add_argument("--context_length",    type=int,   default=512,
                   help="Number of past time steps used as context")
    p.add_argument("--prediction_length", type=int,   default=64,
                   help="Number of future time steps to predict")
    p.add_argument("--stride",            type=int,   default=None,
                   help="Sliding-window stride (default: prediction_length // 2)")
    p.add_argument("--pre_anomaly_offsets", type=int, nargs="+",
                   default=[1, 16, 32, 64, 128],
                   help="Steps INTO the anomaly the Type B future window's end reaches. "
                        "Each value produces one pair per anomaly event. "
                        "Values exceeding the anomaly length are clamped. "
                        "Default: 1 16 32 64 128")
    p.add_argument("--normal_signal_length", type=int, default=0,
                   help="If > 0, prepend a per-series normal reference signal of this length "
                        "to each pair's target. The output target layout becomes "
                        "[normal (N) | context (C) | future (P)]. Pass the same value to the "
                        "model via sep_patch_index=N/input_patch_size and set "
                        "context_length=N+C during fine-tuning. Default: 0 (disabled).")
    args = p.parse_args()

    if args.stride is None:
        args.stride = max(1, args.prediction_length // 2)
        logger.info(f"Stride not set — using default: stride={args.stride}")

    # Sort and deduplicate offsets; keep only positive values
    args.pre_anomaly_offsets = sorted(set(o for o in args.pre_anomaly_offsets if o > 0))
    if not args.pre_anomaly_offsets:
        raise ValueError("--pre_anomaly_offsets must contain at least one positive integer.")
    logger.info(f"Pre-anomaly offsets: {args.pre_anomaly_offsets}")

    os.makedirs(args.output_dir, exist_ok=True)

    (train_inputs, val_inputs,
     s1_tr, s1_val, s2_tr, s2_val,
     s1_tr_in, s1_val_in, s2_tr_in, s2_val_in) = prepare_two_stage_inputs(
        data_dir=args.data_dir,
        min_length=args.min_length,
        val_fraction=args.val_fraction,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        pre_anomaly_offsets=args.pre_anomaly_offsets,
        normal_signal_length=args.normal_signal_length,
    )

    # Save stage1/ and stage2/ outputs
    # train_model_inputs.pkl / val_model_inputs.pkl  — USE THESE for pipeline.fit()
    #   Format: {'target': array(F, context_length + prediction_length)}
    # train_pairs.pkl / val_pairs.pkl  — for analysis/inspection
    #   Format: {'context': {'target': ...}, 'future': {'target': ...}, 'type': str}
    for stage, (pairs_tr, pairs_val, inputs_tr, inputs_val) in {
        "stage1": (s1_tr, s1_val, s1_tr_in, s1_val_in),
        "stage2": (s2_tr, s2_val, s2_tr_in, s2_val_in),
    }.items():
        stage_dir = os.path.join(args.output_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        for fname, data in [
            ("train_pairs.pkl",        pairs_tr),
            ("val_pairs.pkl",          pairs_val),
            ("train_model_inputs.pkl", inputs_tr),
            ("val_model_inputs.pkl",   inputs_val),
        ]:
            path = os.path.join(stage_dir, fname)
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"{stage} — {len(data):>6} entries → {path}")

    log_statistics(train_inputs, val_inputs, s1_tr, s1_val, s2_tr, s2_val)


if __name__ == "__main__":
    main()

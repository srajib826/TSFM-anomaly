"""
SMD-only sliding-window data preparation for Chronos-2 anomaly fine-tuning,
with a FILE-BASED 70/30 train/test split over the SMD *test.csv files.

This mirrors inst_data_prepare_labeled.py exactly (same NORMAL_SIGNAL_LENGTH,
same sliding-window pairing, same per-timestamp future labels, same
[normal_signal | context | future] model-input layout). The ONLY differences are:

  1. Source is restricted to the SMD folder's *test.csv files.
  2. The split is FILE-BASED: 70% of the files go to the training pool and 30%
     to the test set. No window from a given CSV ever appears in two splits.
  3. Optionally a small validation set is carved out of the TRAINING-POOL files
     (still file-based) so the existing run_finetune.sh — which expects a
     val_model_inputs.pkl — works out of the box. Set --val_fraction 0 to skip
     it and use the full 70% for training.

Outputs (written to --output_dir, default ./ i.e. SMD_run):
    train_model_inputs.pkl   ← from (70% files) minus the val hold-out  [SHUFFLED]
    val_model_inputs.pkl     ← from val_fraction of the 70% files       [SHUFFLED, omitted if 0]
    test_model_inputs.pkl    ← from the 30% files                       [ORDERED + metadata]
    test_series_meta.pkl     ← per-series ground truth for the test files

train/val entries (order irrelevant — shuffled):
    {"target": (F, NORMAL_SIGNAL_LENGTH + context_length + prediction_length),
     "future_labels": (prediction_length,) int 0/1}

TEST entries are NOT shuffled: windows stay grouped by series and in temporal
order, and each carries positional metadata so the per-window 64-step predictions
can be scattered back onto the original series timeline to build a per-timestamp
anomaly-score vector for series-based metrics (VUS-PR, range-AUC, etc.):
    {"target": ..., "future_labels": ...,
     "series_id": <csv basename>, "future_start": int, "future_end": int,
     "series_length": int}

test_series_meta.pkl is a dict  series_id -> {
     "length": int,                    # total timesteps of the series
     "labels": (length,) int 0/1,      # FULL per-timestamp ground truth
     "n_features": int,
     "context_length": int}            # first `context_length` steps are never a
                                       # forecast target (no score) -> mask or
                                       # neutral-fill them when scoring.

Reconstruction sketch for VUS-PR (per test series_id):
    meta  = test_series_meta[series_id]
    score = np.full(meta["length"], np.nan)          # NaN = uncovered
    for w in [windows with this series_id, in order]:
        pred = model(w["target"])                    # 64-step forecast
        score[w["future_start"]:w["future_end"]] = per_step_anomaly_score(pred, actual)
    # mask uncovered (incl. leading context_length) or neutral-fill, then:
    vus_pr(score, meta["labels"])

Usage:
    python prepare_smd_split.py
    python prepare_smd_split.py --data_dir /path/to/SMD --output_dir .
    python prepare_smd_split.py --test_fraction 0.3 --val_fraction 0.1 --seed 42
"""

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# Same instruction-prefix length as inst_data_prepare_labeled.py
NORMAL_SIGNAL_LENGTH = 256

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Loading  (identical to inst_data_prepare_labeled.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_as_multivariate(csv_path: str):
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
#  Anomaly Boundary / Normal Zone Helpers  (identical)
# ─────────────────────────────────────────────────────────────────────────────

def extract_anomaly_boundaries(labels: np.ndarray):
    """Contiguous anomaly regions as (start, end) with end EXCLUSIVE."""
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


def get_normal_zones(boundaries, total: int):
    """Normal (non-anomaly) zones as (start, end) pairs."""
    zones, prev = [], 0
    for s, e in boundaries:
        if s > prev:
            zones.append((prev, s))
        prev = e
    if prev < total:
        zones.append((prev, total))
    return zones


def extract_normal_signal(data: np.ndarray, normal_zones, length: int):
    """
    Return a (F, length) reference normal signal sampled from the series' normal zones.

      1. If a single normal zone is long enough, take its last `length` timesteps.
      2. Otherwise concatenate normal zones (longest first) until enough.
      3. If still short, left-pad with NaN.

    Returns None if there are no normal zones at all.
    """
    if not normal_zones:
        return None

    sorted_zones = sorted(normal_zones, key=lambda z: z[1] - z[0], reverse=True)
    s, e = sorted_zones[0]
    if e - s >= length:
        return data[:, e - length:e].astype(np.float32, copy=False)

    chunks, collected = [], 0
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
#  Pair Construction  (identical)
# ─────────────────────────────────────────────────────────────────────────────

def create_pairs(data, labels, context_length, prediction_length, stride):
    """
    Slide a window over the series. For each start t (from context_length onward):

      context        = data[:, t - context_length : t]      (always full, real steps)
      future         = data[:, t : t + prediction_length]   (full window only)
      future_labels  = labels[t : t + prediction_length]    (one label per future step)

    Windows with fewer than `prediction_length` future steps remaining are skipped.
    """
    pairs = []
    total = data.shape[1]
    for t in range(context_length, total, stride):
        fut_end = t + prediction_length
        if fut_end > total:
            break
        ctx = data[:, t - context_length:t].astype(np.float32, copy=False)
        fut = data[:, t:fut_end].astype(np.float32, copy=False)
        fut_labels = labels[t:fut_end].astype(np.int32, copy=False)
        pairs.append({
            "context": {"target": ctx},
            "future":  {"target": fut},
            "future_labels": fut_labels,
            "future_start": int(t),         # global index of first future step in the series
            "future_end":   int(fut_end),   # exclusive
        })
    return pairs


def _attach_normal_signal(pairs, normal_sig):
    """In-place: attach the same per-series normal_signal reference to every pair."""
    for p in pairs:
        p["normal_signal"] = normal_sig


def pairs_to_model_inputs(pairs, include_meta: bool = False):
    """
    Convert pairs to fixed-length model inputs:

        [normal_signal (256) | context (C) | future (P)]

    Each output dict carries `future_labels` (P,) int array (0=normal, 1=anomaly).

    When `include_meta=True` (used for the TEST split) each entry also carries the
    positional metadata needed to scatter the 64 per-step predictions back onto the
    original series timeline for series-based metrics (VUS-PR etc.):
        series_id, future_start, future_end, series_length
    """
    out = []
    for p in pairs:
        ctx, fut = p["context"]["target"], p["future"]["target"]
        normal = p.get("normal_signal")
        if normal is None:
            normal = np.full((ctx.shape[0], NORMAL_SIGNAL_LENGTH), np.nan, dtype=np.float32)
        target = np.concatenate([normal, ctx, fut], axis=1)
        entry = {"target": target, "future_labels": p["future_labels"]}
        if include_meta:
            entry["series_id"]     = p.get("series_id")
            entry["future_start"]  = p.get("future_start")
            entry["future_end"]    = p.get("future_end")
            entry["series_length"] = p.get("series_length")
        out.append(entry)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Per-file pair building
# ─────────────────────────────────────────────────────────────────────────────

def build_pairs_for_files(files, context_length, prediction_length, stride, min_req, tag):
    """
    Load each CSV (in the given order), build pairs with the per-series normal
    prefix attached. Each pair is tagged with its series_id and series_length, so
    the windows of a series stay grouped and temporally ordered.

    Also returns `series_meta`: series_id -> {length, labels (full per-timestamp
    ground truth), n_features, context_length}. This is the ground truth used to
    compute series-based metrics (VUS-PR), including the per-step labels for the
    leading `context_length` steps that are never a forecast target.
    """
    all_pairs, used, skipped, series_meta = [], [], 0, {}
    for path in tqdm(files, desc=f"Building {tag} pairs", unit="file"):
        feat, lbl = load_csv_as_multivariate(path)
        if feat is None or feat.shape[1] < min_req:
            length = feat.shape[1] if feat is not None else "None"
            logger.info(f"  skip {os.path.basename(path)} (length={length} < {min_req})")
            skipped += 1
            continue
        sid = os.path.basename(path)
        pairs = create_pairs(feat, lbl, context_length, prediction_length, stride)
        zones = get_normal_zones(extract_anomaly_boundaries(lbl), len(lbl))
        normal_sig = extract_normal_signal(feat, zones, NORMAL_SIGNAL_LENGTH)
        _attach_normal_signal(pairs, normal_sig)
        for p in pairs:
            p["series_id"] = sid
            p["series_length"] = int(feat.shape[1])
        all_pairs.extend(pairs)
        used.append(sid)
        series_meta[sid] = {
            "length": int(feat.shape[1]),
            "labels": lbl.astype(np.int32, copy=False),
            "n_features": int(feat.shape[0]),
            "context_length": int(context_length),
        }
    return all_pairs, used, skipped, series_meta


def dump_split(pairs, output_dir, fname, rng=None, include_meta=False):
    """
    Convert pairs → model inputs, pickle, and log distribution.

    rng is not None  → SHUFFLE (train/val): order does not matter.
    rng is None      → KEEP ORDER (test): windows stay grouped by series and in
                       temporal order so predictions can be reassembled into a
                       per-timestamp score series for VUS-PR.
    include_meta     → attach series_id / future_start / future_end / series_length.
    """
    model_inputs = pairs_to_model_inputs(pairs, include_meta=include_meta)
    if rng is not None:
        rng.shuffle(model_inputs)
    path = os.path.join(output_dir, fname)
    with open(path, "wb") as f:
        pickle.dump(model_inputs, f)
    anom = sum(int(d["future_labels"].sum()) for d in model_inputs)
    total = sum(d["future_labels"].size for d in model_inputs)
    pct = (anom / total * 100) if total else 0.0
    order = "shuffled" if rng is not None else "ordered"
    logger.info(
        f"{fname:<26} {len(model_inputs):>8} windows ({order})  "
        f"(future steps: normal={total - anom}, anomaly={anom} [{pct:.1f}%]) → {path}"
    )
    return len(model_inputs)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SMD-only data prep with a file-based 70/30 train/test split."
    )
    p.add_argument("--data_dir", default="/home/rajib/mTSBench/Datasets/mTSBench/SMD",
                   help="SMD folder containing the *test.csv files")
    p.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Where to write the .pkl files (default: this script's folder = SMD_run)")
    p.add_argument("--min_length", type=int, default=50,
                   help="Minimum series length; shorter series are discarded")
    p.add_argument("--test_fraction", type=float, default=0.3,
                   help="Fraction of files held out for testing (file-based)")
    p.add_argument("--val_fraction", type=float, default=0.0,
                   help="Fraction of the TRAINING-POOL files used for validation "
                        "(file-based). Default 0 = no validation (full 70%% for "
                        "training, no val_model_inputs.pkl). Set >0 to carve a val set.")
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--prediction_length", type=int, default=64)
    p.add_argument("--stride", type=int, default=576)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "prepare_smd_split.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    # ── Discover SMD test.csv files ──────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(args.data_dir, "**", "*test.csv"),
                                 recursive=True))
    if not csv_files:
        raise ValueError(f"No *test.csv files found under {args.data_dir}")
    logger.info(f"Found {len(csv_files)} SMD *test.csv files under {args.data_dir}")

    # ── FILE-BASED 70/30 split (deterministic via seed) ──────────────────────
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(csv_files))
    n_test = max(1, int(round(len(csv_files) * args.test_fraction)))
    test_idx = set(perm[:n_test].tolist())
    train_pool = [csv_files[i] for i in range(len(csv_files)) if i not in test_idx]
    test_files = [csv_files[i] for i in sorted(test_idx)]

    # Optional val carved from the TRAINING-POOL files (still file-based)
    if args.val_fraction > 0 and len(train_pool) > 1:
        perm_tp = rng.permutation(len(train_pool))
        n_val = max(1, int(round(len(train_pool) * args.val_fraction)))
        n_val = min(n_val, len(train_pool) - 1)  # keep >=1 training file
        val_set = set(perm_tp[:n_val].tolist())
        val_files = [train_pool[i] for i in sorted(val_set)]
        train_files = [train_pool[i] for i in range(len(train_pool)) if i not in val_set]
    else:
        val_files = []
        train_files = train_pool

    logger.info("=" * 70)
    logger.info("FILE-BASED SPLIT")
    logger.info(f"  Total files : {len(csv_files)}")
    logger.info(f"  Train files : {len(train_files)}  -> {[os.path.basename(f) for f in train_files]}")
    logger.info(f"  Val files   : {len(val_files)}  -> {[os.path.basename(f) for f in val_files]}")
    logger.info(f"  Test files  : {len(test_files)}  -> {[os.path.basename(f) for f in test_files]}")
    logger.info("=" * 70)

    min_req = max(args.min_length, args.context_length + args.prediction_length)

    train_pairs, train_used, _, _ = build_pairs_for_files(
        train_files, args.context_length, args.prediction_length, args.stride, min_req, "train")
    val_pairs, val_used, _, _ = build_pairs_for_files(
        val_files, args.context_length, args.prediction_length, args.stride, min_req, "val")
    test_pairs, test_used, _, test_series_meta = build_pairs_for_files(
        test_files, args.context_length, args.prediction_length, args.stride, min_req, "test")

    logger.info(f"Pairs — train: {len(train_pairs)} | val: {len(val_pairs)} | test: {len(test_pairs)}")

    # ── Write pickles ─────────────────────────────────────────────────────────
    # train/val are SHUFFLED (order irrelevant for training).
    # test is written UNSHUFFLED, grouped by series and temporally ordered, with
    # positional metadata, so predictions can be reassembled per series for VUS-PR.
    shuf = np.random.default_rng(args.seed)
    logger.info("Writing model-input pickles...")
    dump_split(train_pairs, args.output_dir, "train_model_inputs.pkl", rng=shuf)
    if val_pairs:
        dump_split(val_pairs, args.output_dir, "val_model_inputs.pkl", rng=shuf)
    else:
        logger.info("val_fraction=0 → no val_model_inputs.pkl written "
                    "(run finetune with NO_VALIDATION=1)")
    dump_split(test_pairs, args.output_dir, "test_model_inputs.pkl",
               rng=None, include_meta=True)

    # Per-series ground truth for series-based metrics (full per-timestamp labels,
    # series length, context_length). Keyed by series_id (CSV basename).
    meta_path = os.path.join(args.output_dir, "test_series_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(test_series_meta, f)
    logger.info(f"test_series_meta.pkl        {len(test_series_meta):>8} series "
                f"(per-timestamp ground-truth labels) → {meta_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

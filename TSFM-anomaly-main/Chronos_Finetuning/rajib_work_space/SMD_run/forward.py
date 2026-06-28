"""
Zero-shot Chronos-2 anomaly evaluation on the held-out SMD TEST split.

Consumes the ORDERED test pkl produced by SMD_run/prepare_smd_split.py
(test_model_inputs.pkl + test_series_meta.pkl), runs Chronos-2 zero-shot via
pipeline.predict(), turns the per-window forecast error into a per-timestamp
anomaly score, reassembles those scores onto each series' original timeline, and
computes VUS-PR / VUS-ROC / AUC-PR / AUC-ROC (+ F1 variants) per series.

Methodology — instruction fine-tuning with a series-level normal signal:
  - Each test entry's target is laid out as
        [ normal(NORMAL_SIGNAL_LENGTH) | context(CONTEXT_LENGTH) | future(PRED) ]
    where the `normal` prefix is the SAME reference normal signal for every window
    of a series (built once per series by extract_normal_signal during data prep).
  - We ASSUME the model has access to the tested series' normal signal, so EVERY
    window is fed [ normal | context ] and asked to predict `future`. This is the
    instruction the fine-tuned model was trained to follow (a [SEP] token is
    inserted automatically after the normal prefix when the loaded checkpoint's
    config has use_sep_token=True; the base zero-shot model simply sees the prefix
    as extra history). Feeding identical inputs to BOTH models is what makes the
    zero-shot vs fine-tuned comparison fair.
  - Windows tile each series contiguously (stride == prediction_length), so the
    per-window 64-step forecast errors concatenate into a gap-free anomaly-score
    vector over the covered region [context_length, last_future_end). The leading
    `context_length` steps are never forecast and are excluded from scoring.
  - VUS-PR/-ROC and AUC-PR/-ROC are computed per series from the forecast-derived
    anomaly score vs the per-timestamp ground-truth labels.

This script lives in SMD_run; the `chronos` and `VUS_ROC_VUS_PR` packages live one
level up (rajib_work_space), so run it via run_forward_smd.sh (which sets PYTHONPATH),
or set PYTHONPATH to the parent dir yourself.

The SAME script evaluates either model:
    # zero-shot base model
    bash run_forward_smd.sh
    # fine-tuned checkpoint (instruction model)
    CHECKPOINT=chronos2-single-stage_SMD/finetuned-ckpt bash run_forward_smd.sh

Other examples:
    python forward.py --score_method interval --agg_method topk_mean
    python forward.py --no_normal_prefix          # ablation: context only, no normal signal
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
#  Args
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot Chronos-2 anomaly eval on SMD test pkl")
    p.add_argument("--test_pkl", default=os.path.join(_HERE, "test_model_inputs.pkl"),
                   help="Ordered test windows pkl from prepare_smd_split.py")
    p.add_argument("--meta_pkl", default=os.path.join(_HERE, "test_series_meta.pkl"),
                   help="Per-series ground-truth meta pkl")
    p.add_argument("--model_id", default="amazon/chronos-2",
                   help="HF id or local path for the ZERO-SHOT model")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a fine-tuned checkpoint (e.g. .../finetuned-ckpt). "
                        "When set, this is loaded INSTEAD of --model_id. SEP handling "
                        "is driven by the checkpoint's own config.")
    p.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu")

    # Sequence layout (must match the data-prep values used to build the pkl)
    p.add_argument("--normal_signal_length", type=int, default=256,
                   help="Length of the per-series normal-signal prefix in the target")
    p.add_argument("--context_length", type=int, default=512,
                   help="Number of real context steps after the normal prefix")
    p.add_argument("--prediction_length", type=int, default=64,
                   help="Future horizon (must equal the pkl's future length)")
    p.add_argument("--input_patch_size", type=int, default=16,
                   help="Model input_patch_size; used to restore the SEP patch index "
                        "(= normal_signal_length / input_patch_size) on a fine-tuned "
                        "checkpoint, since a LoRA adapter does not persist sep_patch_index.")
    p.add_argument("--no_normal_prefix", dest="use_normal_prefix", action="store_false",
                   help="Ablation: feed only the context (drop the normal signal). "
                        "Default feeds [normal | context] to both models.")
    p.set_defaults(use_normal_prefix=True)

    # Inference
    p.add_argument("--batch_windows", type=int, default=32,
                   help="How many windows to send to predict() per call (progress granularity)")
    p.add_argument("--predict_batch_size", type=int, default=256,
                   help="predict() batch_size (counts SERIES = windows * n_variates)")

    # Scoring (mirrors the reference script's options)
    p.add_argument("--score_method", default="interval",
                   choices=["mse", "interval", "normalized_deviation", "smape"])
    p.add_argument("--agg_method", default="topk_mean",
                   choices=["l2", "max", "mean", "topk_mean"])
    p.add_argument("--topk", type=int, default=4, help="k for agg_method=topk_mean")
    p.add_argument("--smooth_window", type=int, default=5, help="1 = no smoothing")

    # VUS
    p.add_argument("--sliding_window_VUS", type=int, default=100)
    p.add_argument("--vus_version", default="opt", choices=["opt", "opt_mem"])
    p.add_argument("--vus_thre", type=int, default=250)

    p.add_argument("--out_csv", default=os.path.join(_HERE, "eval_results.csv"))
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Per-feature scoring / aggregation / normalization (per-feature -> per-step)
# ──────────────────────────────────────────────────────────────────────────────
def compute_feature_score(actual, q10, q50, q90, method):
    """All inputs shape (n_features, P). Returns per-feature per-step score (n_features, P)."""
    if method == "mse":
        return (actual - q50) ** 2
    if method == "smape":
        eps = 1e-8
        return np.abs(actual - q50) / (np.abs(actual) + np.abs(q50) + eps)
    if method == "interval":
        upper = np.maximum(0.0, actual - q90)
        lower = np.maximum(0.0, q10 - actual)
        return upper + lower
    # normalized_deviation
    band = (q90 - q10) + 1e-8
    return np.abs(actual - q50) / band


def robust_normalize_rows(mat):
    """Robust-normalize each feature row (1st-99th pct clip) over time. mat: (n_features, T)."""
    out = np.empty_like(mat, dtype=float)
    for f in range(mat.shape[0]):
        row = mat[f]
        p1, p99 = np.percentile(row, 1), np.percentile(row, 99)
        denom = p99 - p1
        if denom < 1e-8:
            out[f] = 0.0
        else:
            out[f] = (np.clip(row, p1, p99) - p1) / denom
    return out


def aggregate_features(mat, method, k):
    """Aggregate per-feature scores (n_features, T) -> per-step score (T,)."""
    if method == "l2":
        return np.sqrt((mat ** 2).sum(axis=0))
    if method == "max":
        return mat.max(axis=0)
    if method == "mean":
        return mat.mean(axis=0)
    # topk_mean
    k = min(k, mat.shape[0])
    topk = np.sort(mat, axis=0)[-k:, :]      # k largest per column
    return topk.mean(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    from chronos import BaseChronosPipeline
    from chronos.chronos2 import Chronos2Pipeline
    from transformers.utils.peft_utils import find_adapter_config_file
    from VUS_ROC_VUS_PR.metrics import get_metrics

    # ── Load data ────────────────────────────────────────────────────────────
    with open(args.test_pkl, "rb") as f:
        windows = pickle.load(f)
    with open(args.meta_pkl, "rb") as f:
        meta = pickle.load(f)
    print(f"Loaded {len(windows)} test windows across {len(meta)} series.")

    N, C, P = args.normal_signal_length, args.context_length, args.prediction_length
    fut_lo, fut_hi = N + C, N + C + P            # future (ground-truth) slice within target
    in_lo = 0 if args.use_normal_prefix else N   # feed [normal|context] or [context] only
    model_context = fut_lo - in_lo               # 768 with prefix, 512 without
    print(f"Input layout per window: target[:, {in_lo}:{fut_lo}] "
          f"({'[normal|context]' if args.use_normal_prefix else '[context only]'}, "
          f"len={model_context}); predict {P} steps; compare to target[:, {fut_lo}:{fut_hi}]")

    # ── Load model: fine-tuned checkpoint if given, else zero-shot base ───────
    src = args.checkpoint or args.model_id
    mode = "FINE-TUNED" if args.checkpoint else "ZERO-SHOT"
    print(f"Loading [{mode}] model from: {src}  (device={args.device}) ...")
    # A LoRA/PEFT fine-tuned checkpoint contains only adapter_config.json +
    # adapter_model.safetensors (no Chronos config.json). BaseChronosPipeline
    # reads the PEFT config via AutoConfig and raises "Not a Chronos config
    # file" before it can apply the adapter. Chronos2Pipeline.from_pretrained
    # detects the adapter, builds the base model and merges the adapter in, so
    # route adapter checkpoints there directly.
    if find_adapter_config_file(src) is not None:
        pipeline = Chronos2Pipeline.from_pretrained(src, device_map=args.device)
    else:
        pipeline = BaseChronosPipeline.from_pretrained(src, device_map=args.device)
    use_sep = bool(getattr(pipeline.model.chronos_config, "use_sep_token", False))
    sep_idx = getattr(pipeline.model.chronos_config, "sep_patch_index", None)
    # A LoRA adapter persists neither use_sep_token nor sep_patch_index in a
    # config.json: Chronos2Pipeline recovers use_sep_token from the expanded
    # `shared` embedding, but sep_patch_index falls back to the base-model default
    # (0/None). When evaluating a fine-tuned checkpoint we therefore RESTORE the
    # SEP boundary used during training, which is fixed by the data layout as
    # normal_signal_length / input_patch_size (e.g. 256 / 16 = 16). Without this
    # the [SEP] token is inserted at patch 0 -> the model sees [SEP][normal][ctx]
    # instead of the trained [normal][SEP][ctx] and the instruction is broken.
    if use_sep and args.use_normal_prefix:
        if N % args.input_patch_size != 0:
            raise ValueError(
                f"normal_signal_length ({N}) is not a multiple of "
                f"input_patch_size ({args.input_patch_size}); cannot place [SEP] "
                "on a patch boundary. Check that these match the training config."
            )
        expected_sep = N // args.input_patch_size
        if sep_idx != expected_sep:
            print(f"  Restoring sep_patch_index {sep_idx} -> {expected_sep} "
                  f"(= {N} / {args.input_patch_size}); LoRA adapters do not persist it.")
            pipeline.model.chronos_config.sep_patch_index = expected_sep
            sep_idx = expected_sep
        print(f"  SEP boundary = {sep_idx} patches = {sep_idx * args.input_patch_size} "
              f"steps -> matches normal prefix ({N}) ✓")
    elif use_sep:
        print(f"  use_sep_token=True (sep_patch_index={sep_idx}); "
              "normal prefix disabled, SEP placement not adjusted.")
    quantiles = list(pipeline.model.chronos_config.quantiles)
    qi = {q: quantiles.index(q) for q in (0.1, 0.5, 0.9)}   # indices into the quantile axis

    # ── Per-series accumulators: per-feature score matrix over the full timeline
    feat_scores = {sid: np.full((m["n_features"], m["length"]), np.nan, dtype=np.float32)
                   for sid, m in meta.items()}
    covered = {sid: [None, None] for sid in meta}   # [min future_start, max future_end]

    # ── Run zero-shot prediction in chunks, scatter scores ───────────────────
    for i in tqdm(range(0, len(windows), args.batch_windows), desc="Predicting", unit="batch"):
        chunk = windows[i:i + args.batch_windows]
        # Feed [normal | context] (default) so the model receives the series-level
        # normal signal as its instruction; pass context_length so the full input
        # is kept (no left-truncation -> SEP stays aligned for the fine-tuned model).
        inputs = [{"target": w["target"][:, in_lo:fut_lo]} for w in chunk]    # (n_var, model_context)
        preds = pipeline.predict(inputs, prediction_length=P,
                                 context_length=model_context,
                                 batch_size=args.predict_batch_size)
        for w, pred in zip(chunk, preds):
            pred = np.asarray(pred.float().cpu())            # (n_var, n_quantiles, P)
            q10, q50, q90 = pred[:, qi[0.1], :], pred[:, qi[0.5], :], pred[:, qi[0.9], :]
            actual = np.asarray(w["target"][:, fut_lo:fut_hi], dtype=np.float32)  # (n_var, P)
            fscore = compute_feature_score(actual, q10, q50, q90, args.score_method)  # (n_var, P)

            sid, fs, fe = w["series_id"], w["future_start"], w["future_end"]
            feat_scores[sid][:, fs:fe] = fscore
            cl, ch = covered[sid]
            covered[sid][0] = fs if cl is None else min(cl, fs)
            covered[sid][1] = fe if ch is None else max(ch, fe)

    # ── Per-series scoring + metrics ─────────────────────────────────────────
    results = defaultdict(list)
    for sid, m in meta.items():
        lo, hi = covered[sid]
        if lo is None:
            print(f"  {sid}: no windows, skipping")
            continue

        fmat = feat_scores[sid][:, lo:hi]                   # (n_features, covered_len)
        # per-feature robust normalization (skip for smape, which is already scale-free)
        if args.score_method != "smape":
            fmat = robust_normalize_rows(fmat)
        fmat = np.nan_to_num(fmat, nan=0.0)

        y_score = aggregate_features(fmat, args.agg_method, args.topk)   # (covered_len,)
        if args.smooth_window > 1:
            y_score = uniform_filter1d(y_score, size=args.smooth_window)

        y_true = m["labels"][lo:hi].astype(int)
        if y_true.sum() == 0:
            print(f"  {sid}: no anomalies in covered region, skipping")
            continue

        res = get_metrics(y_score, y_true,
                          slidingWindow=args.sliding_window_VUS,
                          version=args.vus_version, thre=args.vus_thre)
        print(f"  {sid:<26} VUS-PR={res['VUS-PR']:.4f}  VUS-ROC={res['VUS-ROC']:.4f}  "
              f"AUC-PR={res['AUC-PR']:.4f}  AUC-ROC={res['AUC-ROC']:.4f}")

        results["series_id"].append(sid)
        for k in ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
                  "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"):
            results[k].append(res[k])

    # ── Summary ──────────────────────────────────────────────────────────────
    if not results["series_id"]:
        print("No series scored (no anomalies in any covered region).")
        return

    print("\n================ MEAN OVER {} SERIES ================".format(len(results["series_id"])))
    for k in ("VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC",
              "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1", "Affiliation-F"):
        print(f"  {k:<16}: {np.mean(results[k]):.4f}")

    # Save per-series table
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False)
        print(f"\nPer-series results written to {args.out_csv}")
    except Exception as e:
        print(f"(could not write csv: {e})")


if __name__ == "__main__":
    main()

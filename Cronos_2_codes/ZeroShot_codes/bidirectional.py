import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

from chronos import BaseChronosPipeline, Chronos2Pipeline
from VUS_ROC_VUS_PR.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Chronos SMD Bidirectional Anomaly Detection (VUS metrics)")
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Train/Test split ratio (e.g., 0.2 means 20%% train, 80%% test)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Chronos will predict timestamps"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Number of past/future timestamps to use as context for predictions"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--score_method",
        type=str,
        default="interval",
        choices=["mse", "interval", "normalized_deviation", "smape"],
        help=(
            "Anomaly scoring method per feature:\n"
            "  mse                 - squared error vs median\n"
            "  interval            - violation beyond [0.1, 0.9] quantile band\n"
            "  normalized_deviation- |actual - median| / band_width\n"
            "  smape               - symmetric MAPE vs median"
        )
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="topk_mean",
        choices=["l2", "max", "mean", "topk_mean"],
        help=(
            "How to aggregate per-feature scores into a single time-series score:\n"
            "  l2        - L2 norm\n"
            "  max       - maximum across features\n"
            "  mean      - mean across features\n"
            "  topk_mean - mean of top-k features"
        )
    )
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="max",
        choices=["max", "mean", "min", "fwd_only", "bwd_only"],
        help=(
            "How to fuse forward and backward per-feature anomaly scores:\n"
            "  max      - elementwise max (an anomaly fires if either direction is surprised)\n"
            "  mean     - elementwise mean\n"
            "  min      - elementwise min (anomaly only if both directions agree)\n"
            "  fwd_only - ignore backward (ablation)\n"
            "  bwd_only - ignore forward (ablation)"
        )
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=10,
        help="Uniform smoothing window for final anomaly score (1 = no smoothing)"
    )
    parser.add_argument(
        "--sliding_window_VUS",
        type=int,
        default=100,
        help="Sliding-window size used by VUS metrics"
    )
    parser.add_argument(
        "--vus_version",
        type=str,
        default="opt",
        choices=["opt", "opt_mem"],
        help="VUS computation backend"
    )
    parser.add_argument(
        "--vus_thre",
        type=int,
        default=250,
        help="Number of thresholds used in VUS curve generation"
    )
    return parser.parse_args()


# -------------------- GPU SETUP --------------------
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# -------------------- LOAD CHRONOS --------------------
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda"
)


# -------------------- DATA PREPARATION --------------------
def prepare_df_test(df):
    df = df.copy()
    df["timestamp"] = pd.date_range(
        start="2000-02-01",
        periods=len(df),
        freq="1s"
    )
    df = df.sort_values("timestamp")

    ts = df["timestamp"]
    assert ts.is_monotonic_increasing
    assert ts.diff().dropna().nunique() == 1

    return df


def split_dataset(df, split_ratio):
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    return df_train, df_test


# -------------------- BACKWARD HELPERS --------------------
def reverse_context(df, feature_list):
    # Reverse feature values row-wise while keeping timestamps forward-increasing
    # so Chronos (forward-only) accepts the series. Row 0 of the result holds
    # the feature values that were originally at the LAST row of `df`.
    df = df.reset_index(drop=True)
    rev = df[feature_list].iloc[::-1].reset_index(drop=True)
    rev["timestamp"] = df["timestamp"].values
    rev["id"] = df["id"].iloc[0]
    return rev


def flip_predictions(pred_df):
    # Predictions for a reversed context come out in reverse real-time order.
    # Flip quantile values per feature so row i aligns with target_start + i.
    q_cols = ["0.1", "0.5", "0.9"]
    parts = []
    for _, grp in pred_df.groupby("target_name", sort=False):
        grp = grp.reset_index(drop=True)
        flipped = grp.copy()
        flipped[q_cols] = grp[q_cols].iloc[::-1].reset_index(drop=True).values
        parts.append(flipped)
    return pd.concat(parts, ignore_index=True)


# -------------------- PREDICTION --------------------
def generate_forward_prediction(df_train, df_test, feature_list, prediction_length, context_length):
    """
    Forward forecast over df_test. For each window [s, s+W), use the last
    `context_length` rows ending just before s as context.
    Always covers every row of df_test.
    """
    window_length    = prediction_length
    all_predictions  = []
    id_column        = "id"
    timestamp_column = "timestamp"
    num_windows      = len(df_test) // window_length
    remainder        = len(df_test) %  window_length

    for i in range(num_windows):
        start = i * window_length

        if i == 0:
            df_train_window = df_train.copy()
        else:
            df_past_test    = df_test.iloc[:start].copy()
            df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        pred_df = pipeline.predict_df(
            df_train_window,
            future_df=None,
            prediction_length=window_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=feature_list,
            context_length=context_length,
            validate_inputs=False
        )

        pred_df["window_id"] = i
        all_predictions.append(pred_df)

    if remainder > 0:
        start           = num_windows * window_length
        df_past_test    = df_test.iloc[:start].copy()
        df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        pred_df = pipeline.predict_df(
            df_train_window,
            future_df=None,
            prediction_length=remainder,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=feature_list,
            context_length=context_length,
            validate_inputs=False
        )

        pred_df["window_id"] = num_windows
        all_predictions.append(pred_df)

    return pd.concat(all_predictions, ignore_index=True)


def generate_backward_prediction(df_train, df_test, feature_list, prediction_length, context_length):
    """
    Backward forecast over df_test. For each window [s, s+W), take up to
    `context_length` rows AFTER the window, reverse them, and let Chronos predict
    W steps in fake-time. Flip predictions back to forward order.

    Windows without sufficient future context are SKIPPED here (no forward
    fallback) — fusion handles those rows by falling back to the forward
    prediction. Coverage is contiguous from row 0 because the future window
    only shrinks toward the end of df_test.

    Returns (pred_df, n_backward_rows). pred_df may be empty if no window
    has enough future.
    """
    window_length    = prediction_length
    all_predictions  = []
    id_column        = "id"
    timestamp_column = "timestamp"

    full_df   = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    train_len = len(df_train)
    test_len  = len(df_test)

    num_windows = test_len // window_length
    remainder   = test_len %  window_length

    n_backward_rows = 0

    for i in range(num_windows):
        start        = i * window_length
        window_start = train_len + start
        window_end   = window_start + window_length

        fut_start = window_end
        fut_end   = min(fut_start + context_length, len(full_df))

        if fut_end - fut_start < window_length:
            # No sufficient future for this (or any later) window — stop.
            break

        ctx = full_df.iloc[fut_start:fut_end]
        ctx = reverse_context(ctx, feature_list)

        pred_df = pipeline.predict_df(
            ctx,
            future_df=None,
            prediction_length=window_length,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=feature_list,
            context_length=context_length,
            validate_inputs=False
        )
        pred_df = flip_predictions(pred_df)
        pred_df["window_id"] = i
        all_predictions.append(pred_df)
        n_backward_rows += window_length

    # Try the remainder window if all full windows were covered.
    if remainder > 0 and n_backward_rows == num_windows * window_length:
        start        = num_windows * window_length
        window_start = train_len + start
        window_end   = window_start + remainder

        fut_start = window_end
        fut_end   = min(fut_start + context_length, len(full_df))

        if fut_end > fut_start:
            ctx = full_df.iloc[fut_start:fut_end]
            ctx = reverse_context(ctx, feature_list)

            pred_df = pipeline.predict_df(
                ctx,
                future_df=None,
                prediction_length=remainder,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column=id_column,
                timestamp_column=timestamp_column,
                target=feature_list,
                context_length=context_length,
                validate_inputs=False
            )
            pred_df = flip_predictions(pred_df)
            pred_df["window_id"] = num_windows
            all_predictions.append(pred_df)
            n_backward_rows += remainder

    if all_predictions:
        return pd.concat(all_predictions, ignore_index=True), n_backward_rows
    return None, 0


# -------------------- ANOMALY SCORING --------------------
def compute_feature_score(y_actual, group_df, method="mse"):
    y_median = group_df["0.5"].values
    y_lower  = group_df["0.1"].values
    y_upper  = group_df["0.9"].values

    if method == "mse":
        return (y_actual - y_median) ** 2

    elif method == "smape":
        eps = 1e-8
        return np.abs(y_actual - y_median) / (
            np.abs(y_actual) + np.abs(y_median) + eps
        )

    elif method == "interval":
        upper_violation = np.maximum(0.0, y_actual - y_upper)
        lower_violation = np.maximum(0.0, y_lower  - y_actual)
        return upper_violation + lower_violation

    else:  # normalized_deviation
        band_width = y_upper - y_lower + 1e-8
        deviation  = np.abs(y_actual - y_median)
        return deviation / band_width


def aggregate_scores(anomaly_df, method="l2"):
    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1)).values

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:  # topk_mean
        k = 4
        return anomaly_df.apply(
            lambda row: row.nlargest(k).mean(), axis=1
        ).values


def robust_normalize(series):
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


def fuse_anomaly_dfs(fwd_df, bwd_df, method):
    # Both inputs are (test_len x n_features). bwd_df has NaN rows beyond
    # backward coverage. Returned DataFrame is (test_len x n_features).
    if method == "fwd_only":
        return fwd_df.copy()
    if method == "bwd_only":
        # Bwd-only ablation: rows without backward fall back to forward so
        # downstream evaluation still has a score for every test row.
        out = bwd_df.copy()
        missing = out.isna().any(axis=1)
        out.loc[missing] = fwd_df.loc[missing]
        return out

    fwd_arr = fwd_df.values
    bwd_arr = bwd_df.values

    if method == "max":
        fused = np.fmax(fwd_arr, bwd_arr)   # NaN-safe; falls back to fwd where bwd is NaN
    elif method == "mean":
        fused = np.nanmean(np.stack([fwd_arr, bwd_arr], axis=0), axis=0)
    else:  # min
        fused = np.fmin(fwd_arr, bwd_arr)

    return pd.DataFrame(fused, columns=fwd_df.columns)


# -------------------- PATHS --------------------
data_path = "/home/rajib/mTSBench/Datasets/mTSBench/SMD/*test.csv"


# -------------------- MAIN LOOP --------------------
file_list         = glob.glob(data_path)
dic_for_each_file = defaultdict(list)
prediction_length = args.horizon
context_length    = args.context_length


for f in tqdm(file_list, desc="Processing SMD files", unit="file"):
    file_name = os.path.basename(f).replace(".csv", "")
    print(f"\nProcessing: {file_name}")

    df_original = pd.read_csv(f)
    df_original = prepare_df_test(df_original)

    feature_list = [
        c for c in df_original.columns
        if c not in ["timestamp", "is_anomaly"]
    ]

    df_train, df_test = split_dataset(df_original, args.split_ratio)
    df_train["id"] = "SMD"
    df_test["id"]  = "SMD"

    test_len = len(df_test)

    # --- forward + backward predictions ---
    fwd_pred_df = generate_forward_prediction(
        df_train, df_test, feature_list, prediction_length, context_length
    )
    bwd_pred_df, n_backward_rows = generate_backward_prediction(
        df_train, df_test, feature_list, prediction_length, context_length
    )

    print(f"  backward coverage: {n_backward_rows}/{test_len} rows "
          f"({(n_backward_rows / test_len * 100 if test_len else 0):.1f}%)")

    # --- per-feature scores: forward (full length) + backward (partial) ---
    fwd_scores = {}
    bwd_scores = {}

    fwd_groups = {name: g.reset_index(drop=True) for name, g in fwd_pred_df.groupby("target_name")}
    bwd_groups = (
        {name: g.reset_index(drop=True) for name, g in bwd_pred_df.groupby("target_name")}
        if bwd_pred_df is not None else {}
    )

    for feature_name in feature_list:
        y_actual = df_test[feature_name].values

        fwd_scores[feature_name] = compute_feature_score(
            y_actual, fwd_groups[feature_name], method=args.score_method
        )

        bwd_full = np.full(test_len, np.nan, dtype=float)
        if feature_name in bwd_groups and n_backward_rows > 0:
            bwd_full[:n_backward_rows] = compute_feature_score(
                y_actual[:n_backward_rows],
                bwd_groups[feature_name],
                method=args.score_method,
            )
        bwd_scores[feature_name] = bwd_full

    fwd_anomaly_df = pd.DataFrame(fwd_scores)
    bwd_anomaly_df = pd.DataFrame(bwd_scores)

    # --- normalize each direction independently so fusion is on the same scale ---
    if args.score_method != "smape":
        fwd_anomaly_df = fwd_anomaly_df.apply(
            lambda col: pd.Series(robust_normalize(col.values)), axis=0
        )

        # Normalize backward only over rows where it is defined; rows beyond
        # coverage stay NaN so fusion treats them as missing.
        def _normalize_bwd(col):
            arr = col.values
            mask = ~np.isnan(arr)
            out = np.full_like(arr, np.nan, dtype=float)
            if mask.any():
                out[mask] = robust_normalize(arr[mask])
            return pd.Series(out)

        bwd_anomaly_df = bwd_anomaly_df.apply(_normalize_bwd, axis=0)

    # --- fuse, fill, aggregate ---
    anomaly_df = fuse_anomaly_dfs(fwd_anomaly_df, bwd_anomaly_df, args.fusion_method)
    anomaly_df = anomaly_df.fillna(0)

    y_score = aggregate_scores(anomaly_df, method=args.agg_method)

    if args.smooth_window > 1:
        y_score = uniform_filter1d(y_score, size=args.smooth_window)

    y_true = df_test["is_anomaly"].values.astype(int)

    if y_true.sum() == 0:
        print(f"Skipping {file_name}: no anomalies in ground truth")
        continue

    evaluation_result = get_metrics(
        y_score, y_true,
        slidingWindow=args.sliding_window_VUS,
        version=args.vus_version,
        thre=args.vus_thre,
    )

    vus_roc = evaluation_result["VUS-ROC"]
    vus_pr  = evaluation_result["VUS-PR"]
    auroc   = evaluation_result["AUC-ROC"]
    auprc   = evaluation_result["AUC-PR"]
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
          f"VUS-ROC: {vus_roc:.4f} | VUS-PR: {vus_pr:.4f}")

    dic_for_each_file["file_name"].append(file_name)
    dic_for_each_file["AUROC"].append(auroc)
    dic_for_each_file["AUPRC"].append(auprc)
    dic_for_each_file["VUS-ROC"].append(vus_roc)
    dic_for_each_file["VUS-PR"].append(vus_pr)


# -------------------- SUMMARY --------------------
auroc_list   = dic_for_each_file["AUROC"]
auprc_list   = dic_for_each_file["AUPRC"]
vus_roc_list = dic_for_each_file["VUS-ROC"]
vus_pr_list  = dic_for_each_file["VUS-PR"]

mean_auroc   = float(np.mean(auroc_list))   if auroc_list   else float("nan")
mean_auprc   = float(np.mean(auprc_list))   if auprc_list   else float("nan")
mean_vus_roc = float(np.mean(vus_roc_list)) if vus_roc_list else float("nan")
mean_vus_pr  = float(np.mean(vus_pr_list))  if vus_pr_list  else float("nan")

print("\nFinished processing all SMD files")
print("\nPer-file results:")
for name, ar, ap, vr, vp in zip(
    dic_for_each_file["file_name"],
    auroc_list, auprc_list, vus_roc_list, vus_pr_list,
):
    print(f"  {name}: AUROC={ar:.4f} | AUPRC={ap:.4f} | "
          f"VUS-ROC={vr:.4f} | VUS-PR={vp:.4f}")

print("\nMean metrics:")
print(f"Mean AUROC  : {mean_auroc:.4f}")
print(f"Mean AUPRC  : {mean_auprc:.4f}")
print(f"Mean VUS-ROC: {mean_vus_roc:.4f}")
print(f"Mean VUS-PR : {mean_vus_pr:.4f}")

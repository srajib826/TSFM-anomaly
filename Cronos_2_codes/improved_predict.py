import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Chronos SMD Anomaly Detection")
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
        default=512,                        # IMPROVED: increased from 256 → more normal pattern context
        help="Number of past timestamps to use as context for predictions"
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
        choices=["mse", "interval", "normalized_deviation"],
        help=(
            "Anomaly scoring method per feature:\n"
            "  mse                 – squared error vs median (original)\n"
            "  interval            – violation beyond [0.1, 0.9] quantile band\n"
            "  normalized_deviation– |actual - median| / band_width  (default)"
        )
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="topk_mean",
        choices=["l2", "max", "mean", "topk_mean"],
        help=(
            "How to aggregate per-feature scores into a single time-series score:\n"
            "  l2        – L2 norm (original)\n"
            "  max       – maximum across features\n"
            "  mean      – mean across features\n"
            "  topk_mean – mean of top-25%% features  (default)"
        )
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Uniform smoothing window for final anomaly score (1 = no smoothing)"
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


# -------------------- PREDICTION --------------------
def generate_prediction(df_train, df_test, feature_list, prediction_length, context_length):
    window_length   = prediction_length
    all_predictions = []
    id_column       = "id"
    timestamp_column = "timestamp"
    num_windows     = len(df_test) // window_length
    remainder       = len(df_test) % window_length

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

    # -------- Handle remainder window --------
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

    final_predictions = pd.concat(all_predictions, ignore_index=True)
    return final_predictions


# -------------------- ANOMALY SCORING --------------------
def compute_feature_score(y_actual, group_df, method="mse"):
    """
    Compute per-feature anomaly score.

    Methods
    -------
    mse
        (y - median)^2  — original approach, ignores uncertainty.

    interval
        How far the actual value lies *outside* the [0.1, 0.9] prediction
        band.  Values inside the band score 0; violations are penalised
        proportionally to the excess.
            score = max(0, y - q90)  +  max(0, q10 - y)

    normalized_deviation  (recommended)
        Absolute deviation from the median, normalised by the band width.
        This rewards the model for being confident on easy timesteps and
        focuses attention on surprising deviations.
            score = |y - q50| / (q90 - q10 + ε)
    """
    y_median = group_df["0.5"].values
    y_lower  = group_df["0.1"].values
    y_upper  = group_df["0.9"].values

    if method == "mse":
        return (y_actual - y_median) ** 2

    elif method == "interval":
        upper_violation = np.maximum(0.0, y_actual - y_upper)
        lower_violation = np.maximum(0.0, y_lower  - y_actual)
        return upper_violation + lower_violation

    else:  # normalized_deviation (default)
        band_width = y_upper - y_lower + 1e-8
        deviation  = np.abs(y_actual - y_median)
        return deviation / band_width


def aggregate_scores(anomaly_df, method="l2"):
    """
    Aggregate per-feature scores into a single scalar per timestep.

    Methods
    -------
    l2        L2 norm across features (original).
    max       Maximum — any strongly anomalous feature triggers an alert.
    mean      Mean — spread signal across all features.
    topk_mean Mean of the top-25 % features — robust to noisy features
              while still capturing concentrated anomalies.
    """
    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1))

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:  # topk_mean (default)
        k = max(1, anomaly_df.shape[1] // 4)
        return anomaly_df.apply(
            lambda row: row.nlargest(k).mean(), axis=1
        ).values


def robust_normalize(series):
    """
    Clip to [1st, 99th] percentile before min-max normalization to reduce
    the influence of extreme outliers on the score distribution.
    """
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


# -------------------- PATHS --------------------
data_path  = "/home/rajib/mTSBench/Datasets/mTSBench/Exathlon/*test.csv"
output_dir = "/home/rajib/mTSBench/results/chronos/Exathlon/parallel_prediction_Exathlon"
os.makedirs(output_dir, exist_ok=True)

results_file = os.path.join(
    output_dir,
    f"chronos_results_{int(args.split_ratio * 100)}%_"
    f"{args.score_method}_{args.agg_method}.csv"
)

if not os.path.exists(results_file):
    pd.DataFrame(columns=["file_name", "AUROC", "AUPRC"]).to_csv(
        results_file, index=False
    )


# -------------------- MAIN LOOP --------------------
file_list         = glob.glob(data_path)
auroc_list        = []
auprc_list        = []
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

    # -------- Predictions --------
    prediction_df = generate_prediction(
        df_train, df_test, feature_list, prediction_length, context_length
    )

    # -------- Per-feature anomaly scores --------
    anomaly_scores = {}

    for feature_name, group_df in prediction_df.groupby("target_name"):
        group_df = group_df.reset_index(drop=True)
        y_actual = df_test[feature_name].values

        score = compute_feature_score(y_actual, group_df, method=args.score_method)
        anomaly_scores[feature_name] = score

    # -------- Build score matrix & robust-normalize each feature --------
    anomaly_df = pd.DataFrame(anomaly_scores)

    # IMPROVED: robust percentile normalization instead of plain min-max
    anomaly_df = anomaly_df.apply(
        lambda col: pd.Series(robust_normalize(col.values)), axis=0
    )
    anomaly_df = anomaly_df.fillna(0)

    # -------- Aggregate across features --------
    y_score = aggregate_scores(anomaly_df, method=args.agg_method)

    # IMPROVED: temporal smoothing to suppress point-wise noise
    if args.smooth_window > 1:
        y_score = uniform_filter1d(y_score, size=args.smooth_window)

    y_true = df_test["is_anomaly"].values

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    dic_for_each_file["file_name"].append(file_name)
    dic_for_each_file["AUROC"].append(auroc)
    dic_for_each_file["AUPRC"].append(auprc)

    auroc_list.append(auroc)
    auprc_list.append(auprc)

    row = pd.DataFrame([{
        "file_name": file_name,
        "AUROC": auroc,
        "AUPRC": auprc
    }])
    row.to_csv(
        results_file,
        mode="a",
        header=not os.path.exists(results_file),
        index=False
    )

# -------- Save per-file results --------
df_results = pd.DataFrame(dic_for_each_file)
df_results.to_csv("alternate_results_file.csv", index=False)

print("\nFinished processing all SMD files")
print(f"Results saved to: {results_file}")

# -------------------- SAVE MEAN SCORES --------------------
mean_auroc = np.mean(auroc_list)
mean_auprc = np.mean(auprc_list)

score_file = os.path.join(
    output_dir,
    f"Exathlon_chronos_score_{int(args.split_ratio * 100)}"
    f"_{args.score_method}_{args.agg_method}.txt"
)

with open(score_file, "w") as fh:
    fh.write(f"Split ratio       : {args.split_ratio}\n")
    fh.write(f"Horizon           : {args.horizon}\n")
    fh.write(f"Context length    : {args.context_length}\n")
    fh.write(f"Score method      : {args.score_method}\n")
    fh.write(f"Aggregation method: {args.agg_method}\n")
    fh.write(f"Smooth window     : {args.smooth_window}\n")
    fh.write(f"Mean AUROC        : {mean_auroc:.6f}\n")
    fh.write(f"Mean AUPRC        : {mean_auprc:.6f}\n")

print("\nMean scores saved")
print(f"Mean AUROC : {mean_auroc:.4f}")
print(f"Mean AUPRC : {mean_auprc:.4f}")
print(f"Saved at   : {score_file}")
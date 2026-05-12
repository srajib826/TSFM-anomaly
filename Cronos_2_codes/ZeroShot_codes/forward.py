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
    parser = argparse.ArgumentParser(description="Chronos SMD Anomaly Detection (VUS metrics)")
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
        "--smooth_window",
        type=int,
        default=5,
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


# -------------------- PREDICTION --------------------
def generate_prediction(df_train, df_test, feature_list, prediction_length, context_length):
    window_length    = prediction_length
    all_predictions  = []
    id_column        = "id"
    timestamp_column = "timestamp"
    num_windows      = len(df_test) // window_length
    remainder        = len(df_test) % window_length

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

    prediction_df = generate_prediction(
        df_train, df_test, feature_list, prediction_length, context_length
    )

    anomaly_scores = {}
    for feature_name, group_df in prediction_df.groupby("target_name"):
        group_df = group_df.reset_index(drop=True)
        y_actual = df_test[feature_name].values
        anomaly_scores[feature_name] = compute_feature_score(
            y_actual, group_df, method=args.score_method
        )

    anomaly_df = pd.DataFrame(anomaly_scores)

    if args.score_method != "smape":
        anomaly_df = anomaly_df.apply(
            lambda col: pd.Series(robust_normalize(col.values)), axis=0
        )
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
print(f"Mean AUROC: {mean_auroc:.4f}")
print(f"Mean AUPRC: {mean_auprc:.4f}")
print(f"Mean VUS-ROC: {mean_vus_roc:.4f}")
print(f"Mean VUS-PR : {mean_vus_pr:.4f}")

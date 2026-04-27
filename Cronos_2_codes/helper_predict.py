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



def generate_prediction(pipeline, df_train, df_test, feature_list,
                        prediction_length, context_length):

    window_length   = prediction_length
    all_predictions = []
    id_column       = "id"
    timestamp_column = "timestamp"

    num_windows = len(df_test) // window_length
    remainder   = len(df_test) % window_length

    for i in range(num_windows):
        start = i * window_length

        if i == 0:
            df_train_window = df_train.copy()
        else:
            df_past_test = df_test.iloc[:start].copy()
            df_train_window = pd.concat(
                [df_train, df_past_test], ignore_index=True
            )

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

    # -------- remainder --------
    if remainder > 0:
        start = num_windows * window_length
        df_past_test = df_test.iloc[:start].copy()
        df_train_window = pd.concat(
            [df_train, df_past_test], ignore_index=True
        )

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



def compute_feature_score(y_actual, group_df, method="normalized_deviation"):

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


def aggregate_scores(anomaly_df, method="topk_mean"):

    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1))

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:
        # k = max(1, anomaly_df.shape[1] // 4)
        k=5
        # print(f"using top-k mean with k={k}")
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


def plot_forecast_with_anomaly(feature_groups, df_test, y_score, feature_idx=1, plot_from=None, plot_until=None, smooth_window=100):

    feature_list = [col for col in df_test.columns if col != 'is_anomaly']
    feature_col  = feature_list[feature_idx]

    feat_df    = feature_groups[feature_col]
    predicted  = feat_df['0.5'].values
    actual     = df_test[feature_col].values[-len(predicted):]
    is_anomaly = df_test['is_anomaly'].values[-len(predicted):]
    q_low      = feat_df['0.1'].values
    q_high     = feat_df['0.9'].values
    time       = np.arange(len(predicted))

    # ── Normalize y_score to [0, 1] ───────────────────────────────
    y_score_scaled = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    y_score_scaled = y_score_scaled.values[-len(predicted):]  # align length

    # ── Slice interval ─────────────────────────────────────────────
    start = plot_from  if plot_from  is not None else 0
    end   = plot_until if plot_until is not None else len(predicted)

    if smooth_window > 1:
        predicted      = pd.Series(predicted).rolling(smooth_window, center=True, min_periods=1).mean().values
        q_low          = pd.Series(q_low).rolling(smooth_window, center=True, min_periods=1).mean().values
        q_high         = pd.Series(q_high).rolling(smooth_window, center=True, min_periods=1).mean().values
        y_score_scaled = pd.Series(y_score_scaled).rolling(30, center=True, min_periods=1).mean().values

    predicted      = predicted[start:end]
    actual         = actual[start:end]
    is_anomaly     = is_anomaly[start:end]
    q_low          = q_low[start:end]
    q_high         = q_high[start:end]
    y_score_scaled = y_score_scaled[start:end]
    time           = time[start:end]

    # ── Two stacked panels sharing x-axis ─────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]}
    )
    # fig.suptitle(f' {feature_col} [{start} : {end}]', fontsize=14, fontweight='bold')
    fig.suptitle(f' {feature_col} ', fontsize=10, fontweight='bold')

    # ── Top: Actual vs Predicted ───────────────────────────────────
    ax1.plot(time, actual,    label='Actual',    color='steelblue',  linewidth=1.2)
    ax1.plot(time, predicted, label='Predicted', color='darkorange', linewidth=1.2, linestyle='--')
    ax1.fill_between(time, q_low, q_high, alpha=0.3, color='xkcd:light lavender', label='prediction interval (0.1–0.9)')
    ax1.set_ylabel('Value')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── Bottom: y_score + is_anomaly ──────────────────────────────
    ax2.plot(time, y_score_scaled, color='steelblue', linewidth=1.0, label='anomaly score')
    ax2.fill_between(time, y_score_scaled, alpha=0.15, color='steelblue')
    ax2.fill_between(time, is_anomaly, step='mid', color='red', alpha=0.3, label='is_anomaly (ground truth)')
    ax2.plot(time, is_anomaly, color='red', linewidth=0.8, drawstyle='steps-mid', alpha=0.6)
    ax2.set_ylabel('Score / Anomaly')
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_xlabel('timestamp')
    # ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


# Usage
# plot_forecast_with_anomaly(feature_groups, df_test, y_score_scaled, feature_idx=1, plot_from=1440, plot_until=2040)

def plot_error(df,smooth_window=200):
    y_score = df
    y_score_scaled = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    y_score_scaled = pd.Series(y_score_scaled).rolling(smooth_window, center=True, min_periods=1).mean().values
    plt.figure(figsize=(16, 4))
    plt.plot(y_score_scaled.index, y_score_scaled.values, color='steelblue', linewidth=1.2)
    # plt.fill_between(y_score.index, y_score.values, alpha=0.15, color='steelblue')
    plt.title('Anomaly Score (y_score)')
    plt.xlabel('Time Step')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def reverse_context(df, feature_list):

    df = df.reset_index(drop=True)
    rev = df[feature_list].iloc[::-1].reset_index(drop=True)
    rev["timestamp"] = df["timestamp"].values
    rev["id"] = df["id"].iloc[0]
    return rev


def flip_predictions(pred_df):

    q_cols = ["0.1", "0.5", "0.9"]
    parts = []
    for _, grp in pred_df.groupby("target_name", sort=False):
        grp = grp.reset_index(drop=True)
        flipped = grp.copy()
        flipped[q_cols] = grp[q_cols].iloc[::-1].reset_index(drop=True).values
        parts.append(flipped)
    return pd.concat(parts, ignore_index=True)


def generate_backward_prediction(pipeline, df_train, df_test, feature_list,
                                 prediction_length, context_length):

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


def fuse_anomaly_dfs(fwd_df, bwd_df, method="max"):

    if method == "fwd_only":
        return fwd_df.copy()
    if method == "bwd_only":
        out = bwd_df.copy()
        missing = out.isna().any(axis=1)
        out.loc[missing] = fwd_df.loc[missing]
        return out

    fwd_arr = fwd_df.values
    bwd_arr = bwd_df.values

    if method == "max":
        fused = np.fmax(fwd_arr, bwd_arr)
    elif method == "mean":
        fused = np.nanmean(np.stack([fwd_arr, bwd_arr], axis=0), axis=0)
    else:  # min
        fused = np.fmin(fwd_arr, bwd_arr)

    return pd.DataFrame(fused, columns=fwd_df.columns)


def plot_pca_bidirectional_anomaly(feature_groups, df_test, y_score_bidirectional,
                                   y_score_pca, feature_idx=1, plot_from=None,
                                   plot_until=None, smooth_window=100):

    feature_list = [col for col in df_test.columns if col != 'is_anomaly' and col != 'timestamp']
    feature_col  = feature_list[feature_idx]

    feat_df    = feature_groups[feature_col]
    predicted  = feat_df['0.5'].values
    is_anomaly = df_test['is_anomaly'].values[-len(predicted):]

    time = np.arange(len(predicted))

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    y_bi_scaled  = minmax(y_score_bidirectional)
    y_bi_scaled  = y_bi_scaled[-len(predicted):]

    y_pca_scaled = minmax(y_score_pca)
    y_pca_scaled = y_pca_scaled[-len(predicted):]

    start = plot_from  if plot_from  is not None else 0
    end   = plot_until if plot_until is not None else len(predicted)

    if smooth_window > 1:
        y_bi_scaled  = pd.Series(y_bi_scaled).rolling(30, center=True, min_periods=1).mean().values
        y_pca_scaled = pd.Series(y_pca_scaled).rolling(30, center=True, min_periods=1).mean().values

    is_anomaly   = is_anomaly[start:end]
    y_bi_scaled  = y_bi_scaled[start:end]
    y_pca_scaled = y_pca_scaled[start:end]
    time         = time[start:end]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.suptitle(f'{feature_col} — Anomaly Scores', fontsize=10, fontweight='bold')

    ax.plot(time, y_bi_scaled, color='darkviolet', linewidth=1.0, label='Bidirectional anomaly score')
    ax.fill_between(time, y_bi_scaled, alpha=0.15, color='darkviolet')

    ax.plot(time, y_pca_scaled, color='green', linewidth=1.0, label='PCA anomaly score')
    ax.fill_between(time, y_pca_scaled, alpha=0.15, color='darkorange')

    ax.fill_between(time, is_anomaly, step='mid', color='red', alpha=0.3, label='is_anomaly (ground truth)')
    ax.plot(time, is_anomaly, color='red', linewidth=0.8, drawstyle='steps-mid', alpha=0.6)

    ax.set_ylabel('Score / Anomaly')
    ax.set_xlabel('Timestamp')
    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_pca_chronos_anomaly(feature_groups, df_test, y_score_chronos, y_score_pca, feature_idx=1, plot_from=None, plot_until=None, smooth_window=100):

    feature_list = [col for col in df_test.columns if col != 'is_anomaly' and col != 'timestamp']
    feature_col  = feature_list[feature_idx]

    feat_df    = feature_groups[feature_col]
    predicted  = feat_df['0.5'].values
    is_anomaly = df_test['is_anomaly'].values[-len(predicted):]

    time = np.arange(len(predicted))

    # ── Normalize both scores to [0, 1] ───────────────────────────
    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    y_chronos_scaled = minmax(y_score_chronos)
    y_chronos_scaled = y_chronos_scaled[-len(predicted):]

    y_pca_scaled     = minmax(y_score_pca)
    y_pca_scaled     = y_pca_scaled[-len(predicted):]

    # ── Slice interval ─────────────────────────────────────────────
    start = plot_from  if plot_from  is not None else 0
    end   = plot_until if plot_until is not None else len(predicted)

    # ── Smoothing ─────────────────────────────────────────────────
    if smooth_window > 1:
        y_chronos_scaled = pd.Series(y_chronos_scaled).rolling(30, center=True, min_periods=1).mean().values
        y_pca_scaled     = pd.Series(y_pca_scaled).rolling(30, center=True, min_periods=1).mean().values

    is_anomaly       = is_anomaly[start:end]
    y_chronos_scaled = y_chronos_scaled[start:end]
    y_pca_scaled     = y_pca_scaled[start:end]
    time             = time[start:end]

    # ── Single anomaly score panel ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.suptitle(f'{feature_col} — Anomaly Scores', fontsize=10, fontweight='bold')

    # Chronos score
    ax.plot(time, y_chronos_scaled, color='steelblue',  linewidth=1.0, label='Chronos anomaly score')
    ax.fill_between(time, y_chronos_scaled, alpha=0.15, color='steelblue')

    # PCA score
    ax.plot(time, y_pca_scaled,     color='green', linewidth=1.0, label='PCA anomaly score')
    ax.fill_between(time, y_pca_scaled,     alpha=0.15, color='darkorange')

    # Ground truth
    ax.fill_between(time, is_anomaly, step='mid', color='red', alpha=0.3, label='is_anomaly (ground truth)')
    ax.plot(time, is_anomaly, color='red', linewidth=0.8, drawstyle='steps-mid', alpha=0.6)

    ax.set_ylabel('Score / Anomaly')
    ax.set_xlabel('Timestamp')
    ax.set_ylim(-0.1, 1.5)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
# SMD Anomaly Detection with Chronos-2

This folder contains an end-to-end pipeline for **time-series anomaly detection on
the SMD dataset** using Chronos-2, plus an anomaly-aware fine-tune of it. The whole
workflow lives here and runs in three stages:

1. **Data preparation** — build train/test windows from the raw SMD CSVs.
2. **Fine-tuning** — train an anomaly-aware Chronos-2 variant that uses a per-series
   *normal signal* as an instruction.
3. **Forward evaluation** — score either model on the held-out test split and report
   VUS / AUC / F1 metrics.

Each stage has a `run_*.sh` wrapper (sets paths + `PYTHONPATH`) around a Python
entrypoint. The `chronos` and `VUS_ROC_VUS_PR` packages live one level up, in
`rajib_work_space/`.

---

## The two models we compare

Everything here exists to compare **two models on the same held-out SMD test data**:

| Model | What it is | Input it receives per window |
|-------|------------|------------------------------|
| **Original Chronos-2** (zero-shot) | `amazon/chronos-2`, unmodified | `context` only (512 steps) |
| **Our fine-tuned model** | Chronos-2 + LoRA, anomaly-aware, with a `[SEP]` token | `[normal signal \| context]` (256 + 512 = 768 steps) |

Both models forecast the next **64 steps**, the forecast error is turned into a
per-timestamp anomaly score, and both are scored over the **exact same region of
each test series** (see *Fair-comparison note* below). The difference is the model
**and** the extra information our model is designed to use: a per-series **normal
signal** that tells it what "healthy" behaviour looks like.

---

## Sequence layout (the core idea)

Every prepared window's `target` is laid out as a single fixed-length array:

```
[ normal signal (256) | context (512) | future (64) ]   total length = 832
            \_______________________/   \__________/
                  model input (768)      forecast target
```

- **normal signal** — a 256-step reference of *normal* behaviour, taken once per
  series from that series' normal (non-anomalous) regions. The same reference is
  attached to every window of a series.
- **context** — the 512 real steps immediately before the forecast point.
- **future** — the 64 steps the model must predict; compared against the model's
  forecast to produce the anomaly score.

Our fine-tuned model inserts a learned **`[SEP]` token between the normal signal and
the context** (at patch index `256 / 16 = 16`), so it can tell the reference apart
from the live context. The original Chronos-2 has no `[SEP]` token, so by default it
is fed the `context` only.

---

## Files in this folder

### Scripts / code
| File | Role |
|------|------|
| `run_prepare_smd.sh` → `prepare_smd_split.py` | Stage 1: data preparation |
| `run_finetune_smd.sh` → `../finetune_anomaly_simple.py` | Stage 2: fine-tuning |
| `run_forward_smd.sh` → `forward.py` | Stage 3: forward evaluation |

### Data artifacts (produced by Stage 1)
| File | Contents |
|------|----------|
| `train_model_inputs.pkl` | Training windows: `{target (F, 832), future_labels (64,)}` |
| `test_model_inputs.pkl`  | Test windows, ordered, with positional metadata |
| `test_series_meta.pkl`   | Per-series ground truth: full per-timestamp labels, length, #features |

### Model (produced by Stage 2)
| Path | Contents |
|------|----------|
| `chronos2-single-stage_SMD/finetuned-ckpt/` | LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`) on top of `amazon/chronos-2` |

### Results (produced by Stage 3)
| File | Model | Input | Meaning |
|------|-------|-------|---------|
| `eval_results_zs.csv` | **Original Chronos-2 (zero-shot)** | `context` only | Baseline: how well the off-the-shelf model detects anomalies via forecast error. |
| `eval_results_ft_2000_steps.csv` | **Our fine-tuned model** | `[normal \| context]` | Our method: fine-tuned weights + per-series normal signal. (Suffix = #training steps.) |

Each CSV has one row per test series with: `VUS-PR, VUS-ROC, AUC-PR, AUC-ROC,
Standard-F1, PA-F1, Event-based-F1, R-based-F1, Affiliation-F`. **VUS-PR** and
**VUS-ROC** are the headline (threshold-robust) metrics.

---

## Stage 1 — Data preparation (`run_prepare_smd.sh`)

Builds a **file-based 70/30 train/test split** of the SMD machines (whole machines go
entirely to train or test — no window leakage across the split), then slides windows
over each series.

- Raw data: `/home/rajib/mTSBench/Datasets/mTSBench/SMD`
- Split: 70% train / 30% test, `seed=42`, no validation set by default
- Windowing: `context_length=512`, `prediction_length=64`, `stride=64`
- For each series, `extract_normal_signal` builds the 256-step normal reference from
  that series' label-defined normal zones and attaches it to every window.
- Each future window also gets per-step labels; the test split additionally stores
  positional metadata so the 64 per-step scores can be scattered back onto the
  original series timeline for series-level metrics.

```bash
./run_prepare_smd.sh
```

---

## Stage 2 — Fine-tuning (`run_finetune_smd.sh`)

Single-stage, anomaly-aware LoRA fine-tune of `amazon/chronos-2`.

**Margin (hinge) objective**, per window, driven by `future_type`:

```
L_total = L_good + lambda * max(0, tau - L_bad)
  future_type = 0 (normal future)  -> L_good : forecast the normal future well (minimise)
  future_type = 1 (anomaly future) -> L_bad  : push the forecast error UP toward margin tau
```

A future window is labelled anomalous when it contains ≥ `anomaly_threshold` (=10)
anomalous timesteps. The intent: the model forecasts *normal* continuations
accurately but is *not* rewarded for fitting anomalies, so its forecast error spikes
on anomalies at inference — which is what we score.

Key settings (defaults in the script):

| Parameter | Value |
|-----------|-------|
| Base model | `amazon/chronos-2` |
| Fine-tune mode | LoRA (`r=16`, `alpha=32`, `dropout=0.01`, ~2.4M trainable params) |
| Context length | 768 (= normal 256 + context 512) |
| Prediction length | 64 |
| `[SEP]` token | enabled, `sep_patch_index = 256 / 16 = 16` |
| `input_patch_size` | 16 |
| Margin | `tau=6.0`, `lambda=1.0` |
| Optim | `lr=1e-5`, `batch=160`, `grad_accum=2`, fp16, cosine, warmup 0.03 |
| Validation | disabled (`NO_VALIDATION=1`) → keeps the final-step checkpoint |
| Steps | 2000 (initial run); see file suffixes for the run being evaluated |

```bash
./run_finetune_smd.sh
# e.g. NUM_STEPS=4000 ./run_finetune_smd.sh
```

The checkpoint is saved to `chronos2-single-stage_SMD/finetuned-ckpt/` as a **LoRA
adapter only** (no full `config.json`).

> **Note on the checkpoint.** Because only the adapter is saved, the `[SEP]` position
> (`sep_patch_index=16`) is **not** persisted in the checkpoint. `forward.py` restores
> it on load from `normal_signal_length / input_patch_size`; this is why
> `run_forward_smd.sh` carries an `INPUT_PATCH_SIZE` that must match the value used in
> `run_finetune_smd.sh`.

---

## Stage 3 — Forward evaluation (`run_forward_smd.sh`)

Loads a model, runs forecasting over the ordered test windows, converts forecast
error to per-timestamp anomaly scores, reassembles them onto each series' timeline,
and computes VUS / AUC / F1 metrics per series (and the mean across series).

How the model is chosen and what it is fed:

- **`CHECKPOINT` blank → Original Chronos-2 (zero-shot)**, fed `context` only.
- **`CHECKPOINT` = path → our fine-tuned model**, fed `[normal | context]`, with the
  `[SEP]` position restored.

(The normal-signal input switches automatically with the checkpoint, so the baseline
and our model are each run in the configuration they're designed for.)

Scoring defaults: `score=mse`, `agg=l2`, `smooth=5`; VUS with `window=100`,
`version=opt`, `thre=250`.

```bash
# Baseline: original Chronos-2 (zero-shot) -> eval_results_zs.csv
OUT_CSV=eval_results_zs.csv ./run_forward_smd.sh

# Our model: fine-tuned + normal signal -> eval_results_ft_<steps>.csv
CHECKPOINT=chronos2-single-stage_SMD/finetuned-ckpt \
OUT_CSV=eval_results_ft_4000_steps.csv ./run_forward_smd.sh
```

---

## Fair-comparison note

Both models are scored over the **identical region** of each test series. The scored
positions come from the test pkl (`future_start`/`future_end`) and do **not** depend
on which model is run; in both cases the leading `context_length` (512) steps of each
series are warm-up and are never scored. So the two CSVs are computed on exactly the
same ground-truth timestamps — the only differences are the **model** and the
**normal-signal input** that our model is built to use.

---

## Why this is the right comparison (and what the base model is *not* fed)

We compare each model in **the configuration it is designed to be used in**:

- **Original Chronos-2** is fed the `context` only. This is the standard,
  off-the-shelf way to do forecasting-based anomaly detection with Chronos: give it
  the recent history, forecast, score the error. It is the honest, realistic
  baseline.
- **Our model** is fed `[normal | context]`, because it was *trained* (with a `[SEP]`
  token) to read the first 256 steps as a separate **reference of normal behaviour**
  and the rest as the live context.

**We deliberately do *not* feed the normal signal to the base model**, for a concrete
reason: the same input array means two different things to the two models. Our model
knows — via the `[SEP]` token and training — that the prefix is a *reference*, not
recent history. The base model has no `[SEP]` and no such training, so it can only
read the prefix as extra contiguous history. The normal signal is not temporally
continuous with the context, so the base model sees a **seam/discontinuity at step
256** and interprets it as a real regime change, which biases its forecast. Prepending
the normal signal to the base model therefore does **not** represent normal usage and
would, if anything, *penalise* the baseline for the wrong reason. Under normal
circumstances no one feeds a raw reference snippet to vanilla Chronos, because it has
no mechanism to treat it as a reference.

So the current setup — **base = context only, ours = normal + context** — compares
each model fairly, on identical scored timestamps, in its intended mode of operation.

**On using it as an ablation.** Feeding `[normal | context]` to the base model is
still *mechanically* valid (Chronos accepts any context length) and can be run as a
**purely mechanical ablation** — e.g. to ask "how much of the gain is the normal-signal
*information* vs. the fine-tuning *itself*?". If reported, it must be labelled
explicitly as *"base model + normal signal (no `[SEP]`, untrained)"* and understood as
a diagnostic only — **not** as the baseline, and not as a meaningful way to deploy the
base model.

---

## End-to-end reproduction

```bash
# 1. Prepare data
./run_prepare_smd.sh

# 2. Fine-tune
NUM_STEPS=4000 ./run_finetune_smd.sh

# 3. Evaluate both models
OUT_CSV=eval_results_zs.csv ./run_forward_smd.sh
CHECKPOINT=chronos2-single-stage_SMD/finetuned-ckpt \
  OUT_CSV=eval_results_ft_4000_steps.csv ./run_forward_smd.sh
```

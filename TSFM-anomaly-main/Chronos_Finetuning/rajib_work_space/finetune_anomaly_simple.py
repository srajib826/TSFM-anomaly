"""
Single-stage anomaly-aware fine-tuning for Chronos-2.

Uses a SINGLE combined dataset (normal + anomaly future pairs mixed together).
The loss is a margin (hinge) objective conditioned on the future type of each sample:

    L_total = L_good + lambda * max(0, tau - L_bad)

    future_type == 0  (normal)  →  L_good : minimise (predict normal well)
    future_type == 1  (anomaly) →  L_bad  : push UP toward margin tau, then stop

The hinge self-saturates: once L_bad >= tau it adds no gradient, so training can't
diverge (this replaces the old clamp+negate gradient-ascent ceiling).

At inference: high prediction error ⟹ high anomaly score.

Usage:
    python finetune_anomaly_simple.py
    python finetune_anomaly_simple.py --margin_tau 12 --margin_lambda 1.0
    python finetune_anomaly_simple.py --finetune_mode full --lr 1e-6
"""

import argparse
import functools
import json
import logging
import os
import pickle

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.anomaly_trainer import Chronos2AnomalyTrainer
from chronos.chronos2.dataset import Chronos2Dataset
import chronos.chronos2.pipeline as chronos2_pipeline

log_path = os.path.join("./finetuning_log/log", "finetune_simple.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — carries future_type through to the trainer
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyChronos2Dataset(Chronos2Dataset):
    """
    Chronos2Dataset that preserves a per-sample `future_type` (0=normal, 1=anomaly).

    The stock Chronos2Dataset validates inputs against {target, past_covariates,
    future_covariates} and drops everything else, so `future_type` would be lost.
    Here we strip it before the parent's validation/preparation, then re-attach it
    onto every batch the dataset yields — expanded to one entry per row so it lines
    up with `context`/`future_target`/`group_ids` (each input series contributes
    `group_size` rows). The single-stage trainer pops it back off in compute_loss.
    """

    def __init__(self, inputs, *args, **kwargs):
        future_types = [int(d.get("future_type", 0)) for d in inputs]
        # Strip BOTH our extra keys before the parent's validation: `future_type`
        # (the window-level label) and `future_labels` (the per-timestep array the
        # labeled data prep emits, already collapsed into future_type upstream).
        cleaned = [
            {k: v for k, v in d.items() if k not in ("future_type", "future_labels")}
            for d in inputs
        ]
        super().__init__(cleaned, *args, **kwargs)
        # Parent filters series shorter than min_past + prediction_length. Our fixed
        # 832-step targets are never filtered, so prepared inputs align 1:1 with
        # future_types. Guard loudly in case lengths change and some get dropped.
        if len(self.inputs) != len(future_types):
            raise RuntimeError(
                f"future_type alignment broke: {len(future_types)} inputs given but "
                f"{len(self.inputs)} survived length filtering. Ensure every target is "
                "at least min_past + prediction_length steps long."
            )
        

        for prepared, ft in zip(self.inputs, future_types):
            prepared["future_type"] = ft
        
        # print("---------------Monkey patching worked-----------------")

    def _build_batch(self, input_indices):
        # print("-----------Called -build_batch ---")
        batch = super()._build_batch(input_indices)

        row_future_types = []
        for input_idx in input_indices:
            group_size = self.inputs[input_idx]["context"].shape[0]
            row_future_types.extend([self.inputs[input_idx]["future_type"]] * group_size)
        batch["future_type"] = torch.tensor(row_future_types, dtype=torch.long)


        # print(f"---Batch {batch.keys()}")
        # print(batch['context'].shape)
        # print(batch['future_target'].shape)
        # print(batch['future_type'].shape)

        return batch


# ─────────────────────────────────────────────────────────────────────────────
#  Single-Stage Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Chronos2SingleStageTrainer(Chronos2AnomalyTrainer):
    """
    Single-stage trainer using a margin (hinge) objective on future_type per sample.

        L_total = L_good + lambda * max(0, tau - L_bad)

      - L_good : mean loss on normal-future samples  (future_type==0) -> minimised
      - L_bad  : mean loss on anomaly-future samples (future_type==1) -> pushed UP,
                 but only until it reaches the margin `tau`. Once L_bad >= tau the
                 hinge is 0 and stops contributing gradient (self-saturating, so no
                 divergence — this replaces the old loss_ceiling clamp+negate hack).

    The two sub-batches are forwarded separately so L_good and L_bad are independent.

    Parameters
    ----------
    margin_tau : float
        The margin the anomaly loss is pushed toward. Must sit ABOVE the normal-point
        loss to do anything. The Chronos-2 loss sums pinball loss over 9 quantiles, so
        even well-predicted normal points score ~3-4 on the normalized scale — set tau
        to ~2-4x that (e.g. 10-15), NOT 2.
    margin_lambda : float
        Weight on the anomaly (bad) term. 1.0 is a sensible default.
    """

    def __init__(
        self,
        *args,
        margin_tau: float = 12.0,
        margin_lambda: float = 1.0,
        loss_ceiling: float | None = None,  # accepted for back-compat; unused by hinge
        **kwargs,
    ):
        # print("---------------------Chronos2SingleStageTrainer------------------")
        super().__init__(*args, loss_ceiling=loss_ceiling, **kwargs)
        self.margin_tau = margin_tau
        self.margin_lambda = margin_lambda
        # Running (sample-weighted) sums of the RAW per-future-type losses, so we can
        # log normal_loss / anomaly_loss (pinball) and mse_normal_window /
        # mse_anomalous_window (MSE) separately. Train and eval are kept apart.
        # *_sum  -> pinball loss, *_mse_sum -> mse, both weighted by the row count.
        self._acc = {
            "train": {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0,
                      "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0},
            "eval":  {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0,
                      "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0},
        }


    @staticmethod
    def _select(inputs: dict, mask: torch.Tensor) -> dict:
        """Return a subset of the batch using a boolean mask."""
        return {
            k: v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == mask.shape[0] else v
            for k, v in inputs.items()
        }

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # print("---Compute_loss() ours'-----------")
        # print(f'inputs context shape is {inputs["context"].shape}')
        # print(f'inputs future_target shape is {inputs["future_target"].shape}')
        # print(f'inputs future_type shape is {inputs["future_type"].shape}')
        future_type = inputs.pop("future_type")   # (B,) — not passed to model
        normal_mask  = future_type == 0
        anomaly_mask = future_type == 1
        # print(f"Normal mask shape: {normal_mask.shape}, Anomaly mask shape: {anomaly_mask.shape}")
        n_normal  = normal_mask.sum().item()
        n_anomaly = anomaly_mask.sum().item()
        # print(f"Batch: {n_normal} normal, {n_anomaly} anomaly")
       
        outputs = None
        acc = self._acc["train" if model.training else "eval"]

        # ── Normal sub-batch → L_good (minimise) ─────────────────────────────
        L_good = torch.zeros((), device=future_type.device)
        if n_normal > 0:
            normal_out = model(**self._select(inputs, normal_mask))
            L_good = normal_out.loss
            outputs = normal_out
            acc["n_sum"] += L_good.detach().item() * n_normal
            if normal_out.mse is not None:
                acc["n_mse_sum"] += normal_out.mse.detach().item() * n_normal
            acc["n_cnt"] += n_normal

        # ── Anomaly sub-batch → L_bad (push UP toward the margin tau) ─────────
        L_bad = None
        if n_anomaly > 0:
            anom_out = model(**self._select(inputs, anomaly_mask))
            L_bad = anom_out.loss
            # Log the RAW prediction error on anomaly futures — we want this to go UP.
            acc["a_sum"] += L_bad.detach().item() * n_anomaly
            if anom_out.mse is not None:
                acc["a_mse_sum"] += anom_out.mse.detach().item() * n_anomaly
            acc["a_cnt"] += n_anomaly
            if outputs is None:
                outputs = anom_out

        # ── Combine: L_total = L_good + lambda * max(0, tau - L_bad) ──────────
        # The hinge is active only while L_bad < tau; once the anomaly loss clears
        # the margin it contributes no gradient, so training cannot diverge.
        total_loss = L_good
        if L_bad is not None:
            hinge = torch.clamp(self.margin_tau - L_bad, min=0.0)
            total_loss = total_loss + self.margin_lambda * hinge

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, *args, **kwargs):
        """Inject separate normal/anomaly loss means into trainer_state.json log_history.

        With a dict eval_dataset, HF calls log() once per dataset, with the dataset
        name baked into the standard loss key: "eval_loss" (single set), or
        "eval_val_loss" / "eval_test_loss" (named sets). We mirror that prefix onto
        our custom normal/anomaly keys so each dataset is logged independently.
        A training log instead carries the bare "loss".
        """
        # print("--------------------LOG function()-----------------")
        # Find the standard eval loss key (excluding our own derived keys) to learn
        # both the phase and the per-dataset prefix.
        eval_loss_keys = [
            k for k in logs
            if k.startswith("eval") and k.endswith("loss")
            and not (k.endswith("normal_loss") or k.endswith("anomaly_loss"))
        ]
        if eval_loss_keys:
            phase = "eval"
            prefix = eval_loss_keys[0][: -len("loss")]  # "eval_" / "eval_val_" / "eval_test_"
        else:
            phase = "train"
            prefix = ""
        acc = self._acc[phase]

        if acc["n_cnt"] > 0:
            logs[f"{prefix}normal_loss"] = round(acc["n_sum"] / acc["n_cnt"], 4)
            logs[f"{prefix}mse_normal_window"] = round(acc["n_mse_sum"] / acc["n_cnt"], 4)
        if acc["a_cnt"] > 0:
            logs[f"{prefix}anomaly_loss"] = round(acc["a_sum"] / acc["a_cnt"], 4)
            logs[f"{prefix}mse_anomalous_window"] = round(acc["a_mse_sum"] / acc["a_cnt"], 4)

        if phase == "eval":
            tag = prefix.rstrip("_").upper()  # "EVAL" / "EVAL_VAL" / "EVAL_TEST"
            print(f"\n======================================================")
            print(f" [{tag}] Normal Loss: {logs.get(f'{prefix}normal_loss', 'N/A')} | Anomaly Loss: {logs.get(f'{prefix}anomaly_loss', 'N/A')}")
            print(f"              MSE Normal:  {logs.get(f'{prefix}mse_normal_window', 'N/A')} | MSE Anomalous: {logs.get(f'{prefix}mse_anomalous_window', 'N/A')}")
            print(f"======================================================\n", flush=True)

        self._acc[phase] = {"n_sum": 0.0, "n_mse_sum": 0.0, "n_cnt": 0,
                            "a_sum": 0.0, "a_mse_sum": 0.0, "a_cnt": 0}
        out = super().log(logs, *args, **kwargs)
        # Persist the loss history every logging step. Without this, runs with
        # save_strategy="no" (i.e. --no_validation) never write trainer_state.json
        # and the loss curve is lost when training ends.
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "trainer_state.json"), "w") as fh:
                json.dump({"log_history": self.state.log_history}, fh, indent=2)
        except Exception as exc:  # never let logging kill training
            logger.warning(f"Could not write trainer_state.json: {exc}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-stage anomaly-aware fine-tuning for Chronos-2"
    )

    # Model
    p.add_argument("--model_id", default="amazon/chronos-2",
                   help="Pretrained model ID or local path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Data
    p.add_argument("--data_dir", default="./prepared_data_labeled",
                   help="Output dir from inst_data_prepare_labeled.py "
                        "(must contain train_model_inputs.pkl / val_model_inputs.pkl)")
    p.add_argument("--anomaly_threshold", type=int, default=10,
                   help="A future window is labeled anomalous (future_type=1) iff it "
                        "contains at least this many anomalous timesteps; else normal.")
    p.add_argument("--no_validation", action="store_true")
    p.add_argument("--test_data", default=None,
                   help="Optional path to a third dataset pkl (e.g. test_model_inputs.pkl). "
                        "It is evaluated and logged every eval step exactly like the validation "
                        "set (forward + loss only, NO weight updates). Logged as eval_test_*.")
    p.add_argument("--debug", action="store_true",
                   help="Truncate train/val to the first 50 samples for a quick smoke test.")

    # Fine-tuning mode
    p.add_argument("--finetune_mode", default="lora", choices=["full", "lora"])
    p.add_argument("--lora_r",        type=int,   default=32)
    p.add_argument("--lora_alpha",    type=int,   default=16)
    p.add_argument("--lora_dropout",  type=float, default=0.0)

    # Training hyperparameters
    p.add_argument("--prediction_length", type=int,   default=64)
    p.add_argument("--context_length",    type=int,   default=768,
                   help="Must equal normal_signal_length + actual context length (256+512=768)")
    p.add_argument("--enable_sep_token",  action="store_true")
    p.add_argument("--normal_signal_length", type=int, default=256)
    p.add_argument("--input_patch_size",  type=int,   default=16)
    p.add_argument("--num_steps",         type=int,   default=5000)
    p.add_argument("--lr",                type=float, default=None,
                   help="Learning rate (default: 2e-5 for LoRA, 1e-6 for full)")
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--fp16",              action="store_true", default=True)
    p.add_argument("--no_fp16",           dest="fp16", action="store_false")
    p.add_argument("--logging_steps",     type=int,   default=100)
    p.add_argument("--eval_steps",        type=int,   default=100,
                   help="Run validation (and log eval_loss) every N steps. Ignored when "
                        "--no_validation. Must divide save_steps (100) for best-model selection.")
    p.add_argument("--warmup_ratio",      type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "constant"])

    # Margin (hinge) loss: L_good + lambda * max(0, tau - L_bad)
    p.add_argument("--margin_tau", type=float, default=12.0,
                   help="Margin the anomaly loss is pushed toward. Must sit ABOVE the "
                        "normal-point loss (~3-4 here) to matter — use ~10-15, NOT 2.")
    p.add_argument("--margin_lambda", type=float, default=1.0,
                   help="Weight on the anomaly (bad) term. Default 1.0.")

    # Output
    p.add_argument("--output_dir", default="./chronos2-single-stage")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str, label: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} data not found at {path}. "
            "Run inst_data_prepare_labeled.py first."
        )
    logger.info(f"Loading {label} from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"  {len(data)} samples loaded")
    return data


def derive_future_type(data: list[dict], threshold: int, label: str) -> list[dict]:
    """
    Collapse the per-timestep `future_labels` array (length = prediction_length) into a
    single window-level `future_type` via a count threshold:

        future_type = 1 (anomaly)  if  (#anomalous steps in the window) >= threshold
        future_type = 0 (normal)   otherwise

    Mutates each dict in place, adding `future_type`. Samples that already carry a
    `future_type` and have no `future_labels` (e.g. old-format data) are left as-is.
    """
    n_anom = 0
    for d in data:
        labels = d.get("future_labels")
        if labels is not None:
            d["future_type"] = int(int(np.sum(labels)) >= threshold)
        n_anom += int(d.get("future_type", 0))
    logger.info(
        f"  {label}: threshold={threshold} ones/window -> "
        f"anomaly={n_anom}, normal={len(data) - n_anom}"
    )
    return data


def build_lora_config(args):
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("pip install peft")
    modules_to_save = ["shared"] if args.enable_sep_token else None
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "self_attention.q", "self_attention.v",
            "self_attention.k", "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
        modules_to_save=modules_to_save,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.lr is None:
        args.lr = 5e-6 if args.finetune_mode == "lora" else 1e-6
    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()

    # ── Load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "train_model_inputs.pkl")
    val_path   = os.path.join(args.data_dir, "val_model_inputs.pkl")

    train_data = load_data(train_path, "train")
    # print(train_data[0]['target'].shape)
    # print(train_data[0]['future_labels'].shape)
    
    val_data   = load_data(val_path, "val") if not args.no_validation else None
    # Optional third dataset, evaluated exactly like val (no weight updates).
    test_data = (
        load_data(args.test_data, "test")
        if (args.test_data and not args.no_validation)
        else None
    )
    if args.debug:
        logger.info("DEBUG mode: truncating train/val/test to the first 50 samples.")
        train_data = train_data[:50]
        if val_data is not None:
            val_data = val_data[:50]
        if test_data is not None:
            test_data = test_data[:50]

    # Collapse per-timestep future_labels -> window-level future_type via the threshold.
    derive_future_type(train_data, args.anomaly_threshold, "train")
    if val_data is not None:
        derive_future_type(val_data, args.anomaly_threshold, "val")
    if test_data is not None:
        derive_future_type(test_data, args.anomaly_threshold, "test")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading {args.model_id} on {args.device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=args.device
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = build_lora_config(args) if args.finetune_mode == "lora" else None
    if lora_config:
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # ── Build fit kwargs ──────────────────────────────────────────────────────
    fit_kwargs = dict(
        inputs=train_data,
        prediction_length=args.prediction_length,
        min_past=args.context_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        learning_rate=args.lr,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        trainer_cls=functools.partial(
            Chronos2SingleStageTrainer,
            margin_tau=args.margin_tau,
            margin_lambda=args.margin_lambda,
        ),
    )
    if val_data is not None:
        # When a test set is supplied, hand HF a dict of eval datasets. It then
        # evaluates each one every eval step and logs eval_val_* / eval_test_*
        # separately. Best-model selection uses the first key ("val").
        if test_data is not None:
            fit_kwargs["validation_inputs"] = {"val": val_data, "test": test_data}
        else:
            fit_kwargs["validation_inputs"] = val_data
        # Override the eval cadence hardcoded inside pipeline.fit (default 100).
        # save_steps stays at 100; HF requires save_steps % eval_steps == 0 when
        # load_best_model_at_end=True, so keep eval_steps a divisor of 100.
        fit_kwargs["eval_steps"] = args.eval_steps
    if args.enable_sep_token:
        if args.normal_signal_length % args.input_patch_size != 0:
            raise ValueError(
                f"--normal_signal_length ({args.normal_signal_length}) must be "
                f"a multiple of --input_patch_size ({args.input_patch_size})"
            )
        fit_kwargs["enable_sep_token"] = True
        fit_kwargs["sep_patch_index"] = args.normal_signal_length // args.input_patch_size

    # ── Train ─────────────────────────────────────────────────────────────────
    # pipeline.fit builds its dataset internally; swap in our subclass so that
    # per-sample future_type survives into the trainer's compute_loss.
    chronos2_pipeline.Chronos2Dataset = AnomalyChronos2Dataset

    logger.info(
        f"Single-stage training: lr={args.lr}, steps={args.num_steps}, "
        f"batch={args.batch_size}, margin_tau={args.margin_tau}, "
        f"margin_lambda={args.margin_lambda}, fp16={use_fp16}"
    )
    # print("----------------Calling pipeline_fit from main()---------------")
    pipeline.fit(**fit_kwargs)


    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Done. Checkpoint saved to {ckpt_path}")
    logger.info(f"Load with: BaseChronosPipeline.from_pretrained('{ckpt_path}')")


if __name__ == "__main__":
    main()

"""
Two-stage anomaly-aware fine-tuning script for Chronos-2.

Stage 1 — Normal future training (loss minimized):
    Context: normal or anomaly signal
    Future:  normal signal
    Goal:    model learns to always predict normal future

Stage 2 — Anomaly future training (loss maximized via gradient ascent):
    Context: normal or anomaly signal
    Future:  anomaly signal
    Goal:    model learns to predict badly on anomaly futures

At inference time:
    High prediction error on a region => high anomaly score.

Usage:
    # LoRA two-stage fine-tuning (recommended):
    python finetune_anomaly.py --finetune_mode lora

    # Full two-stage fine-tuning:
    python finetune_anomaly.py --finetune_mode full --stage1_lr 1e-6 --stage2_lr 1e-6

    # Skip stage 1 if already have a stage-1 checkpoint:
    python finetune_anomaly.py --skip_stage1 --stage1_ckpt ./chronos2-stage1/finetuned-ckpt

    # Custom LoRA hypers:
    python finetune_anomaly.py --finetune_mode lora --lora_r 16 --lora_alpha 32
"""

import argparse
import logging
import os
import pickle

# Reduce CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from chronos.chronos2.anomaly_trainer import Chronos2AnomalyTrainer

log_path = os.path.join("./prepared_data/log", "finetune_anomaly.log")
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


def parse_args():
    p = argparse.ArgumentParser(description="Two-stage anomaly fine-tuning for Chronos-2")

    # Model
    p.add_argument("--model_id", default="amazon/chronos-2",
                   help="Pretrained model ID (HuggingFace hub) or local path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: 'cuda' or 'cpu'")

    # Data
    p.add_argument("--data_dir", default="./prepared_data",
                   help="Root directory containing stage1/ and stage2/ subdirectories")
    p.add_argument("--no_validation", action="store_true",
                   help="Skip validation during both stages")

    # Stage 1 control
    p.add_argument("--skip_stage1", action="store_true",
                   help="Skip Stage 1 and load an existing Stage 1 checkpoint for Stage 2")
    p.add_argument("--stage1_ckpt", default=None,
                   help="Path to an existing Stage 1 checkpoint (required when --skip_stage1)")

    # Fine-tuning mode
    p.add_argument("--finetune_mode", default="lora", choices=["full", "lora"],
                   help="'lora' for LoRA fine-tuning, 'full' for full fine-tuning")

    # LoRA hyperparameters (only used when --finetune_mode lora)
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout rate")

    # Shared training hyperparameters
    p.add_argument("--prediction_length", type=int, default=64,
                   help="Forecast horizon the model is fine-tuned for")
    p.add_argument("--context_length", type=int, default=768,
                   help="Maximum context length (default 512 to save VRAM; model native is 2048). "
                        "When --enable_sep_token is set, this must equal "
                        "normal_signal_length + actual_context_length.")
    p.add_argument("--enable_sep_token", action="store_true",
                   help="Enable [SEP] token between normal signal and context. "
                        "Requires prepared data to have normal signal prepended to each target. "
                        "Sequence becomes [normal][SEP][context][REG][future].")
    p.add_argument("--normal_signal_length", type=int, default=256,
                   help="Length of the normal reference signal prepended to each target. "
                        "Required when --enable_sep_token is set. Must be a multiple of input_patch_size (16). "
                        "Example: 512.")
    p.add_argument("--input_patch_size", type=int, default=16,
                   help="Model's input patch size (used to compute sep_patch_index). Default: 16.")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-device train batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Accumulate gradients over N steps (effective batch = batch_size * N)")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use FP16 mixed precision (default on for CUDA)")
    p.add_argument("--no_fp16", dest="fp16", action="store_false",
                   help="Disable FP16 mixed precision")
    p.add_argument("--logging_steps", type=int, default=100,
                   help="Log training loss every N steps")

    # Stage 1 hyperparameters
    p.add_argument("--stage1_steps", type=int, default=5000,
                   help="Number of gradient steps for Stage 1")
    p.add_argument("--stage1_lr", type=float, default=None,
                   help="Learning rate for Stage 1 (default: 1e-5 for LoRA, 1e-6 for full)")

    # Stage 2 hyperparameters
    p.add_argument("--stage2_steps", type=int, default=1500,
                   help="Number of gradient steps for Stage 2 (gradient ascent)")
    p.add_argument("--stage2_lr", type=float, default=None,
                   help="Learning rate for Stage 2 (default: 1e-5 for LoRA, 1e-6 for full)")

    # Output
    p.add_argument("--stage1_output_dir", default="./chronos2-stage1-[SEP]",
                   help="Directory where Stage 1 checkpoint is saved")
    p.add_argument("--stage2_output_dir", default="./chronos2-stage2-[SEP]",
                   help="Directory where Stage 2 (final) checkpoint is saved")

    return p.parse_args()


def build_lora_config(args):
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("peft is required for LoRA fine-tuning. Install with: pip install peft")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    )


def load_data(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} data not found at {path}. "
            "Run your data preparation script first."
        )
    logger.info(f"Loading {label} inputs from {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"  {len(data)} {label} series loaded")
    return data


def build_fit_kwargs(args, inputs, val_inputs, lr, num_steps, output_dir, lora_config, use_fp16):
    fit_kwargs = dict(
        inputs=inputs,
        prediction_length=args.prediction_length,
        min_past=args.context_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        learning_rate=lr,
        num_steps=num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=output_dir,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
    )
    if args.enable_sep_token:
        if args.normal_signal_length <= 0:
            raise ValueError("--normal_signal_length must be > 0 when --enable_sep_token is set")
        if args.normal_signal_length % args.input_patch_size != 0:
            raise ValueError(
                f"--normal_signal_length ({args.normal_signal_length}) must be a multiple of "
                f"--input_patch_size ({args.input_patch_size})"
            )
        fit_kwargs["enable_sep_token"] = True
        fit_kwargs["sep_patch_index"] = args.normal_signal_length // args.input_patch_size
    if val_inputs is not None:
        fit_kwargs["validation_inputs"] = val_inputs
    return fit_kwargs


def main():
    args = parse_args()

    if args.skip_stage1 and args.stage1_ckpt is None:
        raise ValueError("--stage1_ckpt must be specified when --skip_stage1 is set")

    # Default learning rates
    default_lr = 1e-5 if args.finetune_mode == "lora" else 1e-6
    if args.stage1_lr is None:
        args.stage1_lr = default_lr
    if args.stage2_lr is None:
        args.stage2_lr = default_lr

    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Load pretrained pipeline
    # ------------------------------------------------------------------
    logger.info(f"Loading {args.model_id} on {args.device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=args.device
    )

    # ------------------------------------------------------------------
    # Build LoRA config (if applicable)
    # ------------------------------------------------------------------
    lora_config = None
    if args.finetune_mode == "lora":
        lora_config = build_lora_config(args)
        logger.info(
            f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )

    # ------------------------------------------------------------------
    # Stage 1 — Minimize loss on normal-future dataset
    # ------------------------------------------------------------------
    if args.skip_stage1:
        logger.info(f"Skipping Stage 1 — loading checkpoint from {args.stage1_ckpt}")
        stage1_pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
            args.stage1_ckpt, device_map=args.device
        )
    else:
        stage1_train_path = os.path.join(args.data_dir, "stage1", "train_model_inputs.pkl")
        stage1_val_path   = os.path.join(args.data_dir, "stage1", "val_model_inputs.pkl")

        stage1_train = load_data(stage1_train_path, "Stage 1 train")

        stage1_val = None
        if not args.no_validation and os.path.exists(stage1_val_path):
            stage1_val = load_data(stage1_val_path, "Stage 1 val")

        logger.info(
            f"Stage 1 — MINIMIZE loss (normal future): "
            f"lr={args.stage1_lr}, steps={args.stage1_steps}, "
            f"batch_size={args.batch_size}, fp16={use_fp16}"
        )

        stage1_fit_kwargs = build_fit_kwargs(
            args=args,
            inputs=stage1_train,
            val_inputs=stage1_val,
            lr=args.stage1_lr,
            num_steps=args.stage1_steps,
            output_dir=args.stage1_output_dir,
            lora_config=lora_config,
            use_fp16=use_fp16,
        )

        stage1_pipeline = pipeline.fit(**stage1_fit_kwargs)

        stage1_ckpt_path = os.path.join(args.stage1_output_dir, "finetuned-ckpt")
        logger.info(f"Stage 1 complete. Checkpoint saved to {stage1_ckpt_path}")

        # Reload Stage 1 checkpoint from disk so Stage 2 always starts from
        # the saved weights — not just the in-memory object. This ensures
        # Stage 2 is reproducible and safe even if run separately.
        logger.info(f"Reloading Stage 1 checkpoint from {stage1_ckpt_path} for Stage 2")
        stage1_pipeline = BaseChronosPipeline.from_pretrained(
            stage1_ckpt_path, device_map=args.device
        )

    # ------------------------------------------------------------------
    # Stage 2 — Maximize loss on anomaly-future dataset (gradient ascent)
    # ------------------------------------------------------------------
    stage2_train_path = os.path.join(args.data_dir, "stage2", "train_model_inputs.pkl")
    stage2_val_path   = os.path.join(args.data_dir, "stage2", "val_model_inputs.pkl")

    stage2_train = load_data(stage2_train_path, "Stage 2 train")

    stage2_val = None
    if not args.no_validation and os.path.exists(stage2_val_path):
        stage2_val = load_data(stage2_val_path, "Stage 2 val")

    logger.info(
        f"Stage 2 — MAXIMIZE loss (anomaly future, gradient ascent): "
        f"lr={args.stage2_lr}, steps={args.stage2_steps}, "
        f"batch_size={args.batch_size}, fp16={use_fp16}"
    )

    stage2_fit_kwargs = build_fit_kwargs(
        args=args,
        inputs=stage2_train,
        val_inputs=stage2_val,
        lr=args.stage2_lr,
        num_steps=args.stage2_steps,
        output_dir=args.stage2_output_dir,
        lora_config=lora_config,
        use_fp16=use_fp16,
    )
    # Inject anomaly trainer — this is the only difference from Stage 1
    stage2_fit_kwargs["trainer_cls"] = Chronos2AnomalyTrainer
    print("Stage 2 finetuning....")
    finetuned_pipeline = stage1_pipeline.fit(**stage2_fit_kwargs)

    stage2_ckpt_path = os.path.join(args.stage2_output_dir, "finetuned-ckpt")
    logger.info(f"Stage 2 complete. Final checkpoint saved to {stage2_ckpt_path}")
    logger.info(
        f"Load with: BaseChronosPipeline.from_pretrained('{stage2_ckpt_path}', device_map='cuda')"
    )


if __name__ == "__main__":
    main()

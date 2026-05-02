"""
Fine-tuning script for Chronos-2 on mTSBench data.

Supports both full fine-tuning and LoRA fine-tuning via --finetune_mode.
Reads pre-processed data produced by prepare_data.py.

Usage:
    # LoRA fine-tuning (recommended, fewer trainable params):
    python finetune.py --finetune_mode lora

    # Full fine-tuning:
    python finetune.py --finetune_mode full --learning_rate 1e-6

    # Custom LoRA hypers:
    python finetune.py --finetune_mode lora --lora_r 16 --lora_alpha 32
"""

import argparse
import logging
import os
import pickle

# Reduce CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

log_path = os.path.join("./prepared_data/log", "fine_tune.log")
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
    p = argparse.ArgumentParser(description="Fine-tune Chronos-2 on mTSBench data")

    # Model
    p.add_argument("--model_id", default="amazon/chronos-2",
                   help="Pretrained model ID (HuggingFace hub) or local path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: 'cuda' or 'cpu'")

    # Data
    p.add_argument("--data_dir", default="./prepared_data",
                   help="Directory containing train_inputs.pkl and (optionally) val_inputs.pkl")
    p.add_argument("--no_validation", action="store_true",
                   help="Skip validation during fine-tuning")

    # Fine-tuning mode
    p.add_argument("--finetune_mode", default="lora", choices=["full", "lora"],
                   help="'lora' for LoRA fine-tuning, 'full' for full fine-tuning")

    # LoRA hyperparameters (only used when --finetune_mode lora)
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    p.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout rate")

    # Training hyperparameters
    p.add_argument("--prediction_length", type=int, default=24,
                   help="Forecast horizon the model is fine-tuned for")
    p.add_argument("--context_length", type=int, default=512,
                   help="Maximum context length (default 512 to save VRAM; model native is 2048)")
    p.add_argument("--num_steps", type=int, default=1000, help="Number of gradient steps")
    p.add_argument("--learning_rate", type=float, default=None,
                   help="Learning rate (default: 1e-5 for LoRA, 1e-6 for full fine-tuning)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-device train batch size (keep low to fit in VRAM; use --gradient_accumulation_steps to scale up effective batch)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Accumulate gradients over N steps (effective batch = batch_size * N, default gives effective=32)")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use FP16 mixed precision (halves VRAM usage; default on for CUDA)")
    p.add_argument("--no_fp16", dest="fp16", action="store_false",
                   help="Disable FP16 mixed precision")
    p.add_argument("--logging_steps", type=int, default=100,
                   help="Log training loss every N steps")

    # Output
    p.add_argument("--output_dir", default="./chronos2-finetuned",
                   help="Directory where checkpoints are saved")

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


def main():
    args = parse_args()

    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.finetune_mode == "lora" else 1e-6

    # ------------------------------------------------------------------
    # Load prepared data
    # ------------------------------------------------------------------
    train_path = os.path.join(args.data_dir, "train_pairs.pkl")
    val_path = os.path.join(args.data_dir, "val_pairs.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run prepare_instrcution_data.py first."
        )

    logger.info(f"Loading training pairs from {train_path}")
    with open(train_path, "rb") as f:
        train_pairs = pickle.load(f)
    logger.info(f"  {len(train_pairs)} training pairs loaded")

    # Each pair {'context': {'target': (F, ctx)}, 'future': {'target': (F, pred)}}
    # is concatenated into {'target': (F, ctx+pred)} so that Chronos2's internal
    # sampler sees a series of exactly one window — preserving the pre-defined pairs.
    def pairs_to_inputs(pairs):
        return [
            {"target": np.concatenate(
                [p["context"]["target"], p["future"]["target"]], axis=1
            )}
            for p in pairs
        ]

    train_inputs = pairs_to_inputs(train_pairs)
    logger.info(f"  Converted to {len(train_inputs)} fixed-window series for instruction tuning")

    val_inputs = None
    if not args.no_validation and os.path.exists(val_path):
        logger.info(f"Loading validation pairs from {val_path}")
        with open(val_path, "rb") as f:
            val_pairs = pickle.load(f)
        val_inputs = pairs_to_inputs(val_pairs)
        logger.info(f"  {len(val_inputs)} validation series loaded")

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
    # Fine-tune
    # ------------------------------------------------------------------
    logger.info(
        f"Starting {args.finetune_mode} fine-tuning: "
        f"prediction_length={args.prediction_length}, lr={args.learning_rate}, "
        f"steps={args.num_steps}, batch_size={args.batch_size}"
    )

    use_fp16 = args.fp16 and args.device != "cpu" and torch.cuda.is_available()

    logger.info(
        f"Memory config: batch_size={args.batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"(effective_batch={args.batch_size * args.gradient_accumulation_steps}), "
        f"fp16={use_fp16}, context_length={args.context_length}"
    )

    fit_kwargs = dict(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        finetune_mode=args.finetune_mode,
        lora_config=lora_config,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        # extra TrainingArguments kwargs for memory efficiency
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=use_fp16,
    )
    if val_inputs is not None:
        fit_kwargs["validation_inputs"] = val_inputs

    finetuned_pipeline = pipeline.fit(**fit_kwargs)

    ckpt_path = os.path.join(args.output_dir, "finetuned-ckpt")
    logger.info(f"Fine-tuning complete. Checkpoint saved to {ckpt_path}")
    logger.info(
        f"Load with: BaseChronosPipeline.from_pretrained('{ckpt_path}', device_map='cuda')"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare Training Configuration for Stable Audio Open

Generates a YAML config file for fine-tuning SAO with your dataset.
Defaults to the BASE model (1.1B parameters, ~47 seconds).

Usage:
    python 02_prepare_training_config.py  # Uses base model by default
    python 02_prepare_training_config.py --model-size small  # SAO Small
"""

import yaml
import argparse
from pathlib import Path
import json

def calculate_training_specs(
    n_samples=6000,
    batch_size=1,
    grad_accumulation_steps=8,
    max_steps=2000,
    checkpoint_interval=500
):
    """Calculate training specifications."""
    effective_batch_size = batch_size * grad_accumulation_steps
    steps_per_epoch = n_samples / effective_batch_size
    epochs = max_steps / steps_per_epoch if steps_per_epoch > 0 else 0
    
    # Estimate time: ~1 step per 30-40 seconds on RTX 5090 for base model
    time_per_step_seconds = 35  # Base model is slower
    total_time_hours = (max_steps * time_per_step_seconds) / 3600
    
    return {
        "effective_batch_size": effective_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_time_hours": total_time_hours,
        "total_samples_seen": max_steps * effective_batch_size,
    }

def generate_config(args):
    """Generate training config YAML."""
    
    # Model configuration
    if args.model_size == "small":
        model_id = "stabilityai/stable-audio-open-1.0-small"
        seconds = 10  # Crop to 10s for speed
    elif args.model_size == "base":
        model_id = "stabilityai/stable-audio-open-1.0"
        seconds = 30  # Full 30s for better quality
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

    config = {
        # Model configuration
        "model_id": model_id,
        "model_half": False,  # Use full precision
        "pretrained": True,
        
        # Audio configuration
        "sample_rate": 44100,
        "channels": 2,  # Stereo
        "audio_channels": 2,
        "seconds": seconds,
        "use_audio_conditioning": True,
        
        # Data configuration
        "data": {
            "audio_dir": str(Path("~/Documents/restorative_embeddings/data/generation/audio").expanduser().absolute()),
            "metadata_csv": str(Path("~/Documents/restorative_embeddings/data/generation/export_metadata.csv").expanduser().absolute()),
            "caption_column": "caption",  # Column name in CSV
            "split_train": 0.8,
            "split_val": 0.1,
            "split_test": 0.1,
        },
        
        # Training configuration
        "training": {
            "batch_size": args.batch_size,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_steps": 100,
            "max_steps": args.max_steps,
            "seed": 42,
            "fp16": False,  # Use full precision
            "gradient_checkpointing": True,  # Save memory
        },
        
        # Checkpoint configuration
        "checkpoint": {
            "save_dir": str(Path("~/Documents/restorative_embeddings/generation/train/checkpoints").expanduser().absolute()),
            "save_interval": args.checkpoint_interval,
            "keep_last_n": 3,  # Keep last 3 checkpoints
        },
        
        # Logging configuration
        "logging": {
            "log_interval": 50,
            "tensorboard_dir": str(Path("~/Documents/restorative_embeddings/generation/train/logs").expanduser().absolute()),
            "wandb_project": None,  # Set if using W&B
        },
        
        # Conditioning configuration
        "conditioning": {
            "text_conditioning": True,
            "use_classifiers": False,  # Don't use CLAP during training
            "target_length": 77,  # CLIP token length
        },
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training configuration for SAO fine-tuning"
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "base"],
        default="base",
        help="Model size: small (341M, ~11s) or base (1.1B, ~47s, default)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1 for RTX 5090 with base model)"
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5.0e-5,
        help="Learning rate for fine-tuning (default: 5.0e-5)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum training steps (default: 2000)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Preparing Training Configuration (Base Model)            ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Calculate training specs
    specs = calculate_training_specs(
        n_samples=6000,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_steps=args.max_steps,
        checkpoint_interval=args.checkpoint_interval,
    )

    print(f"\nTraining Specifications:")
    print(f"  Model: SAO {args.model_size.upper()} (base = 1.1B, better quality)")
    print(f"  Audio length: {'30 seconds' if args.model_size == 'base' else '10 seconds'}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accumulation_steps} grad accum = {specs['effective_batch_size']} effective")
    print(f"  Steps per epoch: {specs['steps_per_epoch']:.1f}")
    print(f"  Total epochs: {specs['epochs']:.2f}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Estimated training time: {specs['total_time_hours']:.1f} hours")
    print(f"  Estimated memory: ~20-28 GB on RTX 5090")
    print(f"  Total samples seen: {specs['total_samples_seen']:.0f} / 6000")

    # Generate config
    config = generate_config(args)

    # Save config
    config_path = Path(f"generation/train/config_sao_{args.model_size}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Config saved to: {config_path}")
    print(f"""
Data paths set to:
  Audio: ~/Documents/restorative_embeddings/data/generation/audio/
  Captions: ~/Documents/restorative_embeddings/data/generation/export_metadata.csv

Checkpoints will be saved to:
  ~/Documents/restorative_embeddings/generation/train/checkpoints/

Before training, verify these paths exist on the server:
  $ ls ~/Documents/restorative_embeddings/data/generation/audio/ | wc -l
  $ head ~/Documents/restorative_embeddings/data/generation/export_metadata.csv

Next steps:
  1. On server: source ~/stableaudio/bin/activate
  2. Review config: cat {config_path}
  3. Run training: python generation/train/03_train_model.py --config {config_path}

For detailed instructions, see: generation/train/README_FINETUNE.md (Step 4)
    """)

if __name__ == "__main__":
    main()

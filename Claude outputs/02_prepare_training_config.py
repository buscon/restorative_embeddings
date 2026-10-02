#!/usr/bin/env python3
"""
Prepare Training Configuration for Stable Audio Open Fine-Tuning

Purpose: Generate YAML training configuration for Stable Audio Open base model.
Configured for the ARAUS restorative soundscapes dataset.

Usage:
    python 02_prepare_training_config.py [--model-size base] [--batch-size 1]

Output:
    config_sao_base.yaml (ready for training)
"""

import yaml
from pathlib import Path
import argparse
from datetime import datetime


def estimate_training_time(num_samples, batch_size, grad_accum_steps, time_per_step_seconds, num_epochs):
    """Estimate training time in hours."""
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * time_per_step_seconds
    total_hours = total_seconds / 3600
    return total_hours, total_steps


def create_config(model_size="base", batch_size=1, grad_accum_steps=8,
                  num_epochs=3, learning_rate=1e-4, warmup_steps=500):
    """
    Create training configuration for Stable Audio Open.

    Args:
        model_size: "base" (1.1B, 30s) or "small" (341M, 11s)
        batch_size: Batch size per GPU
        grad_accum_steps: Gradient accumulation steps
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        warmup_steps: Warmup steps for learning rate scheduler
    """

    # Audio specifications
    if model_size == "base":
        sample_rate = 44100
        audio_duration = 30.0  # seconds
        time_per_step = 35  # seconds per training step (empirical for base + RTX 5090)
    elif model_size == "small":
        sample_rate = 44100
        audio_duration = 11.0
        time_per_step = 12
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Data paths - using expanduser() for server portability
    project_base = Path("~/Documents/restorative_embeddings").expanduser().absolute()
    data_audio_dir = project_base / "data" / "generation" / "audio"
    metadata_csv = project_base / "data" / "generation" / "export_metadata.csv"
    checkpoint_dir = project_base / "generation" / "train" / "checkpoints"

    # Verify data paths exist
    print("Checking data paths...")
    if not data_audio_dir.exists():
        print(f"⚠ Warning: Audio directory not found: {data_audio_dir}")
        print(f"  Please verify data is at this location before training")
    else:
        audio_files = list(data_audio_dir.glob("*.wav")) + list(data_audio_dir.glob("*.mp3"))
        print(f"✓ Audio directory found: {len(audio_files)} audio files")

    if not metadata_csv.exists():
        print(f"⚠ Warning: Metadata CSV not found: {metadata_csv}")
        print(f"  Please verify metadata is at this location before training")
    else:
        print(f"✓ Metadata CSV found: {metadata_csv}")

    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    num_samples = 6000  # ARAUS dataset size
    total_hours, total_steps = estimate_training_time(
        num_samples, batch_size, grad_accum_steps, time_per_step, num_epochs
    )

    # Stable Audio model checkpoint
    if model_size == "base":
        model_checkpoint = "stabilityai/stable-audio-open-1.0"
    else:
        model_checkpoint = "stabilityai/stable-audio-open-1.0-small"

    # Build configuration
    config = {
        # Model configuration
        "model": {
            "name": f"stable-audio-open-{model_size}",
            "checkpoint": model_checkpoint,
            "sample_rate": sample_rate,
            "audio_duration": audio_duration,  # seconds
        },

        # Dataset configuration
        "dataset": {
            "audio_dir": str(data_audio_dir),
            "metadata_csv": str(metadata_csv),
            "num_samples": num_samples,
            "caption_column": "caption_pleasantness",  # Column with pleasantness-conditioned captions
            "sample_rate": sample_rate,
            "stereo": True,
            "normalize_loudness": True,  # Normalize to -23 LUFS
            "target_loudness": -23.0,  # LUFS
        },

        # Training configuration
        "training": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "mixed_precision": "fp16",  # Use mixed precision for faster training
            "optimizer": "adamw",
            "scheduler": "cosine",
            "num_training_steps": total_steps,
        },

        # Evaluation configuration
        "evaluation": {
            "eval_steps": 500,
            "save_steps": 500,
            "checkpoint_dir": str(checkpoint_dir),
            "keep_best_n_checkpoints": 3,
        },

        # Logging
        "logging": {
            "use_tensorboard": True,
            "log_dir": str(checkpoint_dir.parent / "logs"),
            "log_interval": 50,
            "use_wandb": False,  # Set to True if using Weights & Biases
        },

        # Metadata for reference
        "metadata": {
            "created": datetime.now().isoformat(),
            "model_size": model_size,
            "dataset": "ARAUS-6k-pleasantness",
            "estimated_training_hours": round(total_hours, 1),
            "estimated_total_steps": total_steps,
            "time_per_step_seconds": time_per_step,
        }
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Prepare Stable Audio Open training config")
    parser.add_argument("--model-size", type=str, default="base",
                       choices=["base", "small"],
                       help="Model size: base (1.1B, 30s) or small (341M, 11s)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU (default: 1 for RTX 5090 + base model)")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                       help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup-steps", type=int, default=500,
                       help="Warmup steps (default: 500)")
    parser.add_argument("--output", type=str, default="config_sao_base.yaml",
                       help="Output config filename (default: config_sao_base.yaml)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Prepare Training Configuration: Stable Audio Open (Base Model)")
    print("="*70 + "\n")

    # Create configuration
    config = create_config(
        model_size=args.model_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps
    )

    # Display configuration summary
    print("\n--- Configuration Summary ---")
    print(f"Model Size:              {config['model']['name']}")
    print(f"Sample Rate:             {config['dataset']['sample_rate']} Hz")
    print(f"Audio Duration:          {config['model']['audio_duration']:.1f} seconds")
    print(f"Dataset Size:            {config['dataset']['num_samples']} samples")
    print(f"Epochs:                  {config['training']['num_epochs']}")
    print(f"Batch Size:              {config['training']['batch_size']}")
    print(f"Gradient Accum Steps:    {config['training']['gradient_accumulation_steps']}")
    print(f"Effective Batch Size:    {config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Learning Rate:           {config['training']['learning_rate']}")
    print(f"\nEstimated Training Time: {config['metadata']['estimated_training_hours']} hours")
    print(f"Estimated Total Steps:   {config['metadata']['estimated_total_steps']}")

    # Data paths
    print(f"\n--- Data Paths ---")
    print(f"Audio Directory:         {config['dataset']['audio_dir']}")
    print(f"Metadata CSV:            {config['dataset']['metadata_csv']}")
    print(f"Checkpoint Directory:    {config['evaluation']['checkpoint_dir']}")

    # Save configuration
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Configuration saved to: {output_path}")
    print("\nNext step:")
    print(f"  python 03_train_model.py --config {output_path}")
    print()


if __name__ == "__main__":
    main()

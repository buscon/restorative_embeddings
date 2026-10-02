#!/usr/bin/env python3
"""
Fine-Tune Stable Audio 3 LoRA Model on ARAUS Dataset

Purpose: Wrapper script to run SA3 LoRA training with optimal settings for:
  - ARAUS restorative soundscapes dataset (6,000 samples)
  - RTX 5090 GPU (32GB VRAM)
  - Pleasantness conditioning via captions
  - DoRA (Dimension-wise Output-Rank Adaptation) fine-tuning

Usage:
    python 03_train_lora_sa3.py --data-dir ./araus_for_sa3
    python 03_train_lora_sa3.py --data-dir ./araus_for_sa3 --rank 32 --steps 10000
    python 03_train_lora_sa3.py --data-dir ./araus_for_sa3 --resume-from outputs/lightning_logs/version_0

Output:
    outputs/ (checkpoints and logs for LoRA training)

Model Configuration:
  - Base model: stabilityai/stable-audio-3-medium-base (1.9B parameters)
  - Adapter type: dora-rows (DoRA with row-wise adaptation)
  - LoRA rank: 16 (default, adjustable)
  - Precision: bfloat16 (memory efficient)
  - Batch size: 2 (configurable for RTX 5090)
"""

import argparse
import sys
import subprocess
from pathlib import Path


def get_sa3_directory():
    """Find Stable Audio 3 installation directory."""
    # Try common locations
    candidates = [
        Path.home() / "stable-audio-3",
        Path.home() / "projects" / "stable-audio-3",
        Path("/opt/stable-audio-3"),
        Path.cwd() / "stable-audio-3",
    ]

    for sa3_dir in candidates:
        train_script = sa3_dir / "scripts" / "train_lora.py"
        if train_script.exists():
            return sa3_dir

    return None


def run_training(args):
    """Execute SA3 LoRA training with specified parameters."""

    # Find SA3 directory
    sa3_dir = Path(args.sa3_dir) if args.sa3_dir else get_sa3_directory()

    if not sa3_dir or not (sa3_dir / "scripts" / "train_lora.py").exists():
        print("✗ Stable Audio 3 not found")
        print("\nSetup SA3 with:")
        print("  python 01_setup_environment_sa3.py")
        return False

    # Verify data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False

    # Count audio files
    wav_files = list(data_dir.rglob("*.wav"))
    if not wav_files:
        print(f"✗ No audio files found in: {data_dir}")
        return False

    print("\n" + "="*70)
    print("Fine-Tune: Stable Audio 3 LoRA on ARAUS Dataset")
    print("="*70 + "\n")

    print(f"Stable Audio 3:  {sa3_dir}")
    print(f"Data directory:  {data_dir}")
    print(f"Audio files:     {len(wav_files)}")
    print(f"LoRA rank:       {args.rank}")
    print(f"Adapter type:    {args.adapter_type}")
    print(f"Training steps:  {args.steps}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Base precision:  {args.base_precision}")
    print(f"Dropout:         {args.dropout}")

    # Build training command
    cmd = [
        "uv", "run", "python", "scripts/train_lora.py",
        "--model", args.model,
        "--data_dir", str(data_dir),
        "--rank", str(args.rank),
        "--adapter_type", args.adapter_type,
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--base_precision", args.base_precision,
        "--dropout", str(args.dropout),
        "--checkpoint_every", str(args.checkpoint_every),
    ]

    # Add optional arguments
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])

    if args.warmup_steps:
        cmd.extend(["--warmup_steps", str(args.warmup_steps)])

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            print(f"\n✗ Checkpoint not found: {args.resume_from}")
            return False
        cmd.extend(["--lora_checkpoint", str(args.resume_from)])
        print(f"Resuming from:   {args.resume_from}")

    if args.logger:
        cmd.extend(["--logger", args.logger])

    print(f"\n--- Training Configuration ---")
    print(f"Command: {' '.join(cmd[3:])}")  # Skip 'uv run python'
    print("\n--- Starting Training ---\n")

    try:
        # Change to SA3 directory and run training
        result = subprocess.run(cmd, cwd=sa3_dir, check=False)

        if result.returncode == 0:
            print("\n" + "="*70)
            print("✓ Training completed successfully!")
            print("="*70 + "\n")
            print("Next steps:")
            print("  1. Checkpoints saved in: outputs/lightning_logs/")
            print("  2. Load trained LoRA during inference:")
            print("     model.load_lora('./outputs/...../final.safetensors')")
            print("  3. Adjust LoRA strength during generation: lora_scale=0.5")
            print()
            return True
        else:
            print("\n" + "="*70)
            print(f"✗ Training failed with return code {result.returncode}")
            print("="*70 + "\n")
            return False

    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error running training: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Audio 3 LoRA on ARAUS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training on ARAUS data
  python 03_train_lora_sa3.py --data-dir ./araus_for_sa3

  # Higher rank (more parameters) for longer training
  python 03_train_lora_sa3.py --data-dir ./araus_for_sa3 --rank 32 --steps 10000

  # Resume from checkpoint
  python 03_train_lora_sa3.py --data-dir ./araus_for_sa3 --resume-from outputs/lightning_logs/version_0/checkpoints/last.ckpt

  # Custom learning rate and dropout
  python 03_train_lora_sa3.py --data-dir ./araus_for_sa3 --learning-rate 0.0001 --dropout 0.1

GPU Memory (RTX 5090 - 32GB):
  - rank=16,  batch_size=2: ~24GB ✓ (recommended)
  - rank=32,  batch_size=1: ~28GB ✓
  - rank=64,  batch_size=1: ~30GB ✓ (max)
        """
    )

    # Data and model configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training dataset (audio + caption pairs)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium-base",
        choices=["medium-base", "medium", "small"],
        help="SA3 model size (default: medium-base = 1.9B)"
    )

    # LoRA configuration
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16, higher=more capacity)"
    )
    parser.add_argument(
        "--adapter-type",
        type=str,
        default="dora-rows",
        choices=["lora", "dora", "dora-rows", "bora", "bora-xs"],
        help="Adapter type (default: dora-rows for best quality)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
        help="LoRA dropout (default: 0.05)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU (default: 2 for RTX 5090)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of training steps (default: 5000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (optional, uses SA3 default if not specified)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps (optional)"
    )
    parser.add_argument(
        "--base-precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Base model precision (default: bf16 for efficiency)"
    )

    # Checkpointing and logging
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="csv",
        choices=["csv", "wandb", "comet"],
        help="Logger backend (default: csv)"
    )

    # Resume and environment
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--sa3-dir",
        type=str,
        default=None,
        help="Path to Stable Audio 3 installation (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Run training
    success = run_training(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

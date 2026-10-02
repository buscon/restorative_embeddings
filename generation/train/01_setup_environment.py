#!/usr/bin/env python3
"""
Setup: Stable Audio Open Fine-Tuning Environment (Server Edition)

Purpose: Verify and configure the stable-audio environment for fine-tuning.
This version detects an existing venv installation at ~/stableaudio/venv/

Usage:
    python 01_setup_environment.py --stableaudio-path ~/stableaudio
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def check_installation(python_path):
    """Check if stable-audio-tools is installed in the venv."""
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import stable_audio_tools; print(stable_audio_tools.__version__)"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking installation: {e}")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Setup Stable Audio Open fine-tuning environment"
    )
    parser.add_argument(
        "--stableaudio-path",
        type=str,
        default="~/stableaudio",
        help="Path to stableaudio directory (default: ~/stableaudio)"
    )
    parser.add_argument(
        "--install-stable-audio",
        action="store_true",
        help="Install stable-audio-tools if not found"
    )

    args = parser.parse_args()

    # Expand user path
    stableaudio_path = Path(args.stableaudio_path).expanduser().absolute()
    venv_path = stableaudio_path / "venv"
    python_path = venv_path / "bin" / "python"

    print("\n" + "="*60)
    print("Setup: Stable Audio Open Fine-Tuning (Server Edition)")
    print("="*60 + "\n")

    # Check if stableaudio directory exists
    if not stableaudio_path.exists():
        print(f"❌ Stableaudio directory not found at: {stableaudio_path}")
        sys.exit(1)

    print(f"✓ Stable Audio environment found at: {stableaudio_path}")

    # Check if venv exists
    if not venv_path.exists():
        print(f"❌ Virtual environment not found at: {venv_path}")
        print(f"   Expected location: {stableaudio_path}/venv/")
        sys.exit(1)

    print(f"✓ Virtual environment found at: {venv_path}")

    # Check if Python exists in venv
    if not python_path.exists():
        print(f"❌ Python not found at {python_path}")
        sys.exit(1)

    print(f"✓ Python found at: {python_path}")

    # Check stable-audio-tools installation
    if check_installation(python_path):
        print("✓ stable-audio-tools is installed")
    else:
        print("❌ stable-audio-tools not found in venv")
        if args.install_stable_audio:
            print("   Installing stable-audio-tools...")
            try:
                subprocess.run(
                    [str(python_path), "-m", "pip", "install", "stable-audio-tools"],
                    check=True
                )
                print("✓ stable-audio-tools installed successfully")
            except subprocess.CalledProcessError:
                print("❌ Failed to install stable-audio-tools")
                sys.exit(1)
        else:
            print("   Use --install-stable-audio flag to install, or:")
            print(f"   source {venv_path}/bin/activate && pip install stable-audio-tools")
            sys.exit(1)

    # Check CUDA
    print("\n--- GPU/CUDA Status ---")
    gpu_info = check_cuda()
    if gpu_info:
        print(f"✓ CUDA detected:")
        for line in gpu_info.split("\n"):
            print(f"  {line}")
    else:
        print("⚠ CUDA not detected (training will be slow on CPU)")

    # Summary
    print("\n" + "="*60)
    print("✓ Environment is ready!")
    print("="*60)
    print(f"\nTo activate the environment, run:")
    print(f"  source {venv_path}/bin/activate")
    print(f"\nThen navigate to training scripts:")
    print(f"  cd ~/Documents/restorative_embeddings/generation/train/")
    print(f"\nAnd run:")
    print(f"  python 02_prepare_training_config.py")
    print()


if __name__ == "__main__":
    main()

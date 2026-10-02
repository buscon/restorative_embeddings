#!/usr/bin/env python3
"""
Setup Environment for Stable Audio Open Fine-Tuning

IMPORTANT: This script is designed for a server environment where:
  - stable-audio-tools is already installed in ~/stableaudio/
  - You have an existing Python environment for the project
  - The base (large) model will be used for fine-tuning

If stable-audio-tools is NOT installed, this script can help install it.
Otherwise, you can skip this step.

Usage:
    python 01_setup_environment.py
    
Or with options:
    python 01_setup_environment.py --stableaudio-path ~/stableaudio
    python 01_setup_environment.py --skip-install  # Skip installation
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(cmd, description=None, check=True):
    """Run a shell command and report status."""
    if description:
        print(f"\n{'='*60}")
        print(f"▶ {description}")
        print(f"{'='*60}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=os.getcwd())
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        return False
    return result.returncode == 0

def check_installation(python_path):
    """Check if stable-audio-tools is already installed."""
    check_script = """
import sys
try:
    from stable_audio_tools import get_pretrained_model
    print("✓ stable-audio-tools is installed")
    sys.exit(0)
except ImportError:
    print("✗ stable-audio-tools NOT found")
    sys.exit(1)
"""
    
    result = subprocess.run(
        f"{python_path} -c '{check_script}'",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(
        description="Setup environment for SAO fine-tuning (Server Edition)"
    )
    parser.add_argument(
        "--stableaudio-path",
        default=os.path.expanduser("~/stableaudio"),
        help="Path to stable-audio-tools environment (default: ~/stableaudio)"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installation check (if already installed)"
    )
    parser.add_argument(
        "--install-stable-audio",
        action="store_true",
        help="Install stable-audio-tools from GitHub (if not already installed)"
    )
    args = parser.parse_args()

    print("""
╔════════════════════════════════════════════════════════════╗
║  Setup: Stable Audio Open Fine-Tuning (Server Edition)    ║
╚════════════════════════════════════════════════════════════╝
    """)

    stableaudio_path = Path(args.stableaudio_path).expanduser()
    
    # Check if stable-audio environment exists
    if not stableaudio_path.exists():
        print(f"\n⚠  Stable Audio environment not found at: {stableaudio_path}")
        print(f"   Please ensure stable-audio-tools is installed there.")
        print(f"\n   If you need to install it, run:")
        print(f"   $ python 01_setup_environment.py --install-stable-audio")
        sys.exit(1)
    
    print(f"\n✓ Stable Audio environment found at: {stableaudio_path}")

    # Find Python in stable audio environment
    python_path = stableaudio_path / "bin" / "python"
    if not python_path.exists():
        print(f"❌ Python not found at {python_path}")
        sys.exit(1)
    
    print(f"✓ Python found at: {python_path}")

    # Check if stable-audio-tools is installed
    if not args.skip_install:
        print(f"\nChecking stable-audio-tools installation...")
        if check_installation(str(python_path)):
            print("✓ stable-audio-tools is already installed")
        else:
            print("✗ stable-audio-tools NOT found in environment")
            if args.install_stable_audio:
                print("\nInstalling stable-audio-tools...")
                run_command(
                    f"cd /tmp && git clone https://github.com/Stability-AI/stable-audio-tools.git "
                    f"&& cd stable-audio-tools && {python_path} -m pip install -e . "
                    f"&& cd / && rm -rf /tmp/stable-audio-tools",
                    description="Installing stable-audio-tools"
                )
            else:
                print("To install, run: python 01_setup_environment.py --install-stable-audio")
                sys.exit(1)

    # Verify CUDA
    print(f"\nVerifying CUDA/GPU availability...")
    test_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""
    
    with open("/tmp/gpu_check.py", "w") as f:
        f.write(test_script)
    
    run_command(f"{python_path} /tmp/gpu_check.py", check=False)

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ✓ Setup Complete!                                        ║
╚════════════════════════════════════════════════════════════╝

Environment:
  Location:   {stableaudio_path}
  Python:     {python_path}

To activate before running training:
  source {stableaudio_path}/bin/activate

Next steps:
  1. Activate environment: source {stableaudio_path}/bin/activate
  2. Configure training: python generation/train/02_prepare_training_config.py
  3. Fine-tune model: python generation/train/03_train_model.py

For detailed instructions, see: generation/train/README_FINETUNE.md
    """)

if __name__ == "__main__":
    main()

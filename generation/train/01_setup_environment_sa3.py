#!/usr/bin/env python3
"""
Setup Environment for Stable Audio 3 Fine-Tuning

Purpose: Fresh install of Stable Audio 3 with LoRA training dependencies.
Uses UV package manager for streamlined environment setup.

Usage:
    python 01_setup_environment_sa3.py

NOTE: Assumes uv is installed. If not, install with:
    curl -LsSf https://astral.sh/uv/install.sh | sh
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"$ {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ Error running command: {e}")
        return False


def check_uv_installed():
    """Check if uv package manager is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ uv found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    return False


def main():
    print("\n" + "="*70)
    print("Setup: Stable Audio 3 Fresh Install")
    print("="*70 + "\n")

    # Check prerequisites
    print("Checking prerequisites...\n")

    if not check_uv_installed():
        print("✗ uv package manager not found")
        print("\nInstall uv with:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("\nThen run this script again.")
        return False

    # Check Python version
    py_version = sys.version_info
    if py_version.major == 3 and py_version.minor >= 10:
        print(f"✓ Python {py_version.major}.{py_version.minor} (required: 3.10+)")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (required: 3.10+)")
        return False

    # Step 1: Clone SA3 repository
    sa3_dir = Path.home() / "stable-audio-3"

    if sa3_dir.exists():
        print(f"\n✓ SA3 repository already exists at: {sa3_dir}")
        use_existing = input("  Use existing installation? (y/n): ").strip().lower()
        if use_existing != 'y':
            print("  Cloning fresh copy...")
            run_command(
                ["rm", "-rf", str(sa3_dir)],
                "Removing old SA3 directory"
            )
            success = run_command(
                ["git", "clone", "https://github.com/Stability-AI/stable-audio-3.git", str(sa3_dir)],
                "Clone Stable Audio 3 repository"
            )
            if not success:
                return False
    else:
        success = run_command(
            ["git", "clone", "https://github.com/Stability-AI/stable-audio-3.git", str(sa3_dir)],
            "Clone Stable Audio 3 repository"
        )
        if not success:
            return False

    # Step 2: Setup environment with uv
    print(f"\nChanging to SA3 directory: {sa3_dir}")

    # Install with LoRA support
    success = run_command(
        ["uv", "sync", "--extra", "lora"],
        "Install Stable Audio 3 with LoRA training dependencies"
    )
    if not success:
        return False

    # Step 3: Verify installation
    print("\n" + "="*70)
    print("Verifying Installation")
    print("="*70)

    verify_cmd = [
        "uv", "run", "python", "-c",
        "from stable_audio_3 import StableAudioModel; print('✓ SA3 imports successfully')"
    ]

    try:
        result = subprocess.run(
            verify_cmd,
            cwd=sa3_dir,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"\n{result.stdout}")
            print("✓ Stable Audio 3 is ready for fine-tuning\n")
        else:
            print(f"\n✗ Verification failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Verification error: {e}")
        return False

    # Step 4: Display next steps
    print("="*70)
    print("Next Steps")
    print("="*70 + "\n")
    print(f"SA3 installed at: {sa3_dir}")
    print(f"\nTo run training:")
    print(f"  cd {sa3_dir}")
    print(f"  uv run python scripts/train_lora.py \\")
    print(f"    --model medium-base \\")
    print(f"    --data_dir /path/to/your/dataset \\")
    print(f"    --rank 16 \\")
    print(f"    --adapter_type dora-rows \\")
    print(f"    --steps 1000")
    print()

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

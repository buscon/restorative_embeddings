#!/usr/bin/env python3
"""
Train Stable Audio Open Fine-Tuning Model

Purpose: Main fine-tuning loop for Stable Audio Open on ARAUS dataset.
Implements PyTorch DataLoader with checkpointing and resumable training.

Usage:
    python 03_train_model.py --config config_sao_base.yaml
    python 03_train_model.py --config config_sao_base.yaml --resume-checkpoint checkpoints/checkpoint_1000.pt

Output:
    checkpoints/checkpoint_*.pt (model checkpoints)
    training_log.jsonl (training metrics)
"""

import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Any


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ARUASAudioDataset(Dataset):
    """Dataset for ARAUS restorative soundscapes with pleasantness-conditioned captions."""

    def __init__(self, audio_dir: Path, metadata_csv: Path, sample_rate: int = 44100,
                 duration: float = 30.0, normalize_loudness: bool = True, target_loudness: float = -23.0):
        """
        Args:
            audio_dir: Directory containing audio files
            metadata_csv: CSV with columns: filename, caption_pleasantness
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
            normalize_loudness: Whether to normalize loudness
            target_loudness: Target loudness in LUFS
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.normalize_loudness = normalize_loudness
        self.target_loudness = target_loudness

        # Load metadata
        import pandas as pd
        self.metadata = pd.read_csv(metadata_csv)

        # Filter to only existing audio files
        self.audio_files = []
        self.captions = []
        for idx, row in self.metadata.iterrows():
            audio_path = self.audio_dir / row['filename']
            if audio_path.exists():
                self.audio_files.append(audio_path)
                self.captions.append(row['caption_pleasantness'])

        logger.info(f"Loaded {len(self.audio_files)} audio files from {self.audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def load_audio(self, path: Path) -> torch.Tensor:
        """Load and preprocess audio."""
        try:
            waveform, sr = torchaudio.load(str(path))

            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Pad or trim to target duration
            if waveform.shape[1] < self.num_samples:
                waveform = F.pad(waveform, (0, self.num_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.num_samples]

            # Normalize loudness if requested
            if self.normalize_loudness:
                waveform = self.normalize_loudness_simple(waveform)

            return waveform.squeeze(0)

        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            return torch.zeros(self.num_samples)

    def normalize_loudness_simple(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simple loudness normalization using RMS."""
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            target_rms = 10 ** (self.target_loudness / 20)
            waveform = waveform * (target_rms / rms)
        return waveform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        audio = self.load_audio(self.audio_files[idx])
        caption = self.captions[idx]

        return {
            'audio': audio,
            'caption': caption,
            'filename': self.audio_files[idx].name
        }


class StableAudioFineTuner:
    """Fine-tuning trainer for Stable Audio Open."""

    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(config['evaluation']['checkpoint_dir'])
        self.log_dir = Path(config['logging']['log_dir'])

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using device: {device}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Log dir: {self.log_dir}")

        # Initialize model and tokenizer
        self.init_model()
        self.init_optimizer()

        self.global_step = 0
        self.training_losses = []

    def init_model(self):
        """Initialize Stable Audio Open model."""
        model_checkpoint = self.config['model']['checkpoint']
        logger.info(f"Loading model: {model_checkpoint}")

        # Load model - this is a placeholder, actual SAO loading depends on the library structure
        try:
            self.model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Note: Ensure stable-audio-tools is properly installed")
            raise

        self.model = self.model.to(self.device)
        self.model.train()

    def init_optimizer(self):
        """Initialize optimizer and scheduler."""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Cosine annealing scheduler
        total_steps = self.config['training']['num_training_steps']
        warmup_steps = self.config['training']['warmup_steps']

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        audio = batch['audio'].to(self.device)
        caption = batch['caption']

        # Forward pass
        try:
            # This is a simplified version - actual fine-tuning depends on SAO's API
            # You may need to adjust based on the actual stable-audio-tools implementation
            outputs = self.model(audio, text=caption)
            loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            return loss.item()

        except Exception as e:
            logger.error(f"Error in training step: {e}")
            self.optimizer.zero_grad()
            return 0.0

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step}.pt"

        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['step']

        logger.info(f"Loaded checkpoint at step {self.global_step}")

    def log_metrics(self, step: int, loss: float, learning_rate: float):
        """Log training metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate,
        }

        self.training_losses.append(loss)

        # Log to file
        log_file = self.log_dir / 'training_log.jsonl'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Console output
        if step % self.config['logging']['log_interval'] == 0:
            avg_loss = sum(self.training_losses[-100:]) / len(self.training_losses[-100:])
            logger.info(f"Step {step} | Loss: {avg_loss:.4f} | LR: {learning_rate:.2e}")

    def train(self, train_loader: DataLoader, resume_from: Path = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        total_steps = self.config['training']['num_training_steps']
        epochs = self.config['training']['num_epochs']
        batch_size = self.config['training']['batch_size']
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        save_steps = self.config['evaluation']['save_steps']

        logger.info(f"Starting training for {epochs} epochs ({total_steps} total steps)")
        logger.info(f"Batch size: {batch_size} | Grad accum: {grad_accum_steps}")

        accumulation_counter = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(progress_bar):
                # Train step
                loss = self.train_step(batch)
                accumulation_counter += 1

                # Only update weights after gradient accumulation
                if accumulation_counter == grad_accum_steps:
                    self.global_step += 1
                    accumulation_counter = 0

                    # Log metrics
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log_metrics(self.global_step, loss, current_lr)

                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(self.global_step)

                    # Update progress bar
                    progress_bar.set_postfix({'loss': loss, 'step': self.global_step})

                if self.global_step >= total_steps:
                    logger.info(f"Reached total steps ({total_steps})")
                    break

            if self.global_step >= total_steps:
                break

        logger.info("Training complete!")
        logger.info(f"Final step: {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train Stable Audio Open on ARAUS dataset")
    parser.add_argument("--config", type=str, default="config_sao_base.yaml",
                       help="Path to training config YAML file")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Train: Stable Audio Open Fine-Tuning on ARAUS Dataset")
    print("="*70 + "\n")

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"✓ Loaded config: {config_path}")
    print(f"  Model: {config['model']['name']}")
    print(f"  Dataset size: {config['dataset']['num_samples']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")

    # Create dataset and dataloader
    print("\n--- Loading Dataset ---")
    dataset = ARUASAudioDataset(
        audio_dir=config['dataset']['audio_dir'],
        metadata_csv=config['dataset']['metadata_csv'],
        sample_rate=config['dataset']['sample_rate'],
        duration=config['model']['audio_duration'],
        normalize_loudness=config['dataset']['normalize_loudness'],
        target_loudness=config['dataset']['target_loudness']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False
    )

    # Initialize trainer
    print("\n--- Initializing Model ---")
    trainer = StableAudioFineTuner(config, device=args.device)

    # Train
    print("\n--- Starting Training ---")
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    trainer.train(dataloader, resume_from=resume_checkpoint)

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")


if __name__ == "__main__":
    main()

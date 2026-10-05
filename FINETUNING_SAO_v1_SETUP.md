# Fine-Tuning Stable Audio Open v1 with RestorationModelling Data

**Status:** Configuration files created. Ready for data preparation and training.

**Model:** Stable Audio Open v1 (base: 1.1B params, ~30s audio)

**Dataset:** Restorative soundscapes with pleasantness-conditioned captions (30-second files, 44.1 kHz stereo)

---

## Prerequisites Checklist

- [ ] `stable-audio-tools` installed (from official Stability AI repo)
- [ ] PyTorch 2.1+ with CUDA support
- [ ] Python 3.10+
- [ ] RTX 5090 or equivalent GPU (32GB VRAM recommended for base model)

---

## Step 1: Verify Your Audio Data

Your audio files should be:
- **Duration:** 30 seconds (1,323,000 samples at 44.1 kHz)
- **Sample rate:** 44.1 kHz
- **Channels:** 2 (stereo)
- **Bit depth:** 16-bit PCM
- **Loudness:** Normalized to -1 dBFS (peak)

**Find your audio files:**
```bash
find ~/Documents/restorative_embeddings -name "*.wav" -type f | head -10
du -sh ~/Documents/restorative_embeddings/generation/train/
```

---

## Step 2: Organize Audio Files by Pleasantness Bins

Create the folder structure for fine-tuning data:

```bash
mkdir -p ~/Documents/restorative_embeddings/generation/train/diverse_audio_with_captions/{very_unpleasant,unpleasant,neutral,pleasant,very_pleasant}
```

**File structure needed:**
```
generation/train/diverse_audio_with_captions/
├── very_unpleasant/
│   ├── audio_1.wav
│   ├── audio_1.txt
│   ├── audio_2.wav
│   ├── audio_2.txt
│   └── ...
├── unpleasant/
│   ├── audio_100.wav
│   ├── audio_100.txt
│   └── ...
├── neutral/
├── pleasant/
└── very_pleasant/
```

**For each audio file, create a corresponding `.txt` caption file with the same basename.**

Caption format (from plan_generation.md):
```
{scene} with {source}. Loudness: {loudness} (LA50: {LA50} dB). Pleasantness: {pleasantness_label} (ISOPleasant: {ISOPleasant}).
```

Example:
```
Park with birdsong. Loudness: moderately loud (LA50: 60.0 dB). Pleasantness: pleasant (ISOPleasant: 0.43).
```

**Distribute files across bins:**
- very_unpleasant: ISOPleasant ≤ -0.50
- unpleasant: -0.50 < ISOPleasant ≤ -0.15
- neutral: -0.15 < ISOPleasant < +0.15
- pleasant: +0.15 ≤ ISOPleasant < +0.50
- very_pleasant: ISOPleasant ≥ +0.50

---

## Step 3: Prepare Training Data

Run the data preparation script:

```bash
cd ~/Documents/restorative_embeddings
python 09_prepare_finetuning_data.py
```

**What this does:**
1. Validates all audio files (44.1 kHz, stereo, 16-bit)
2. Creates `metadata.csv` with columns: `file`, `caption`
3. Normalizes audio to -1 dBFS if needed
4. Outputs ready-to-train dataset to `generation/train/sa3_finetuning_data/`

**Expected output:**
```
Found 6000 audio files with captions

[  1] audio_1 ✓
[  2] audio_2 ✓
...
[6000] audio_6000 ✓

Successfully processed: 6000/6000
Metadata saved to: generation/train/sa3_finetuning_data/metadata.csv
```

---

## Step 4: Configuration Files (Already Created)

Two configuration files have been created for you:

### A. `sao_model_config.json` - Model Configuration

```json
{
  "model_type": "diffusion_cond",
  "sample_size": 1323000,
  "sample_rate": 44100,
  "audio_channels": 2,
  "model": {
    "pretrained_model_name_or_path": "stabilityai/stable-audio-open-1.0",
    "type": "autoencoder_kl",
    "model_class": "StableAudioOpenAutoencoder"
  },
  "training": {
    "learning_rate": 5e-5,
    "use_ema": false,
    "lora_config": null
  }
}
```

**Key fields:**
- `model_type`: `diffusion_cond` (diffusion model with text conditioning)
- `sample_size`: 1,323,000 samples = 30 seconds at 44.1 kHz
- `pretrained_model_name_or_path`: Official Stability AI model checkpoint
- `learning_rate`: 5e-5 (recommended for fine-tuning)

### B. `dataset_config.json` - Dataset Configuration

```json
{
  "dataset_type": "audio_dir",
  "datasets": [
    {
      "id": "restorative_soundscapes",
      "path": "~/Documents/restorative_embeddings/generation/train/diverse_audio_with_captions"
    }
  ],
  "random_crop": true,
  "drop_last": true
}
```

**Key fields:**
- `dataset_type`: `audio_dir` (loads audio files with caption metadata)
- `path`: Points to your organized audio + captions folder
- `random_crop`: Randomly crops audio during training (data augmentation)
- `drop_last`: Drops incomplete batches

---

## Step 5: Install `stable-audio-tools`

If not already installed:

```bash
# Clone the official repository
git clone https://github.com/Stability-AI/stable-audio-tools.git
cd stable-audio-tools

# Install with training dependencies
pip install -e ".[train]"

# Verify installation
python -c "from stable_audio_tools.training import train; print('✓ stable-audio-tools installed')"
```

---

## Step 6: Run Training

From the `stable-audio-tools` directory:

```bash
python -m train \
  --dataset-config ~/Documents/restorative_embeddings/dataset_config.json \
  --model-config ~/Documents/restorative_embeddings/sao_model_config.json \
  --output-dir ~/Documents/restorative_embeddings/generation/train/checkpoints \
  --num-gpus 1 \
  --batch-size 2 \
  --accum-batches 4 \
  --max-steps 2000 \
  --save-interval 500
```

**Parameters:**
- `--dataset-config`: Path to dataset configuration
- `--model-config`: Path to model configuration
- `--output-dir`: Where to save checkpoints
- `--batch-size`: 2 for RTX 5090 with 32GB VRAM (base model)
- `--accum-batches`: Gradient accumulation steps (effective batch = 2 × 4 = 8)
- `--max-steps`: 2000 ≈ 1 epoch on 6000 samples with batch 2
- `--save-interval`: Save checkpoint every 500 steps

**Alternative (smaller model for faster iteration):**

Modify `sao_model_config.json` to use the small model:
```json
"pretrained_model_name_or_path": "stabilityai/stable-audio-open-1.0-small"
```

Then train with:
```bash
python -m train \
  --dataset-config ~/Documents/restorative_embeddings/dataset_config.json \
  --model-config ~/Documents/restorative_embeddings/sao_model_config.json \
  --output-dir ~/Documents/restorative_embeddings/generation/train/checkpoints_small \
  --batch-size 4 \
  --accum-batches 2 \
  --max-steps 2000
```

---

## Step 7: Monitor Training

Watch training progress in real-time:

```bash
# In a separate terminal, watch the checkpoint directory
watch -n 10 'ls -lh ~/Documents/restorative_embeddings/generation/train/checkpoints/ | tail -10'

# Or check loss logs (if available)
tail -f ~/Documents/restorative_embeddings/generation/train/checkpoints/training.log
```

**Expected loss curve:**
- Step 0–200: Rapid drop (model learns basic audio reconstruction)
- Step 200–1000: Slower decrease (learns caption details + pleasantness)
- Step 1000+: Plateau (saturation; consider stopping early)

---

## Step 8: Test Fine-Tuned Model

After training reaches a checkpoint (e.g., step 1000):

```bash
# Generate test audio with fine-tuned model
python -c "
from stable_audio_tools.inference import generate
import torchaudio

# Load checkpoint
ckpt_path = 'generation/train/checkpoints/checkpoint_step_1000.pt'

# Generate with pleasantness prompts
for pleasantness in ['pleasant', 'neutral', 'unpleasant']:
    prompt = f'Park with birds. Pleasantness: {pleasantness}.'
    audio = generate(prompt, duration=30, checkpoint=ckpt_path)
    
    # Save
    output_path = f'generation/train/test_{pleasantness}.wav'
    torchaudio.save(output_path, audio, sample_rate=44100)
    print(f'Saved: {output_path}')
"
```

---

## Troubleshooting

### Issue: "Model not found: stabilityai/stable-audio-open-1.0"

**Solution:** Ensure you can download from Hugging Face Hub:
```bash
huggingface-cli login  # Use your Hugging Face token
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size: `--batch-size 1 --accum-batches 8`
2. Use the small model instead of base
3. Check GPU memory: `nvidia-smi`

### Issue: "Dataset loading error"

**Check:**
1. Audio folder exists: `ls ~/Documents/restorative_embeddings/generation/train/diverse_audio_with_captions/`
2. Files are organized: Each `.wav` has matching `.txt` caption
3. Run data prep script first: `python 09_prepare_finetuning_data.py`

---

## References

- **Official Repo:** https://github.com/Stability-AI/stable-audio-tools
- **Docs:** https://github.com/Stability-AI/stable-audio-tools/tree/main/docs
- **Model Card:** https://huggingface.co/stabilityai/stable-audio-open-1.0
- **Plan:** See `claude/plan_generation.md` in project docs

---

## Next Steps

1. ✓ Configurations created (`sao_model_config.json`, `dataset_config.json`)
2. → Organize audio files by pleasantness bins
3. → Create caption `.txt` files for each audio
4. → Run data preparation script
5. → Start training
6. → Evaluate on test prompts
7. → Iterate and refine


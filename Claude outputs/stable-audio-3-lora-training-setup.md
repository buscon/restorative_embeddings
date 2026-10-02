# Stable Audio 3 LoRA Fine-Tuning Setup on RTX 5090

Complete reproducible guide for setting up and running LoRA fine-tuning of Stable Audio 3 on ARAUS restorative soundscapes dataset.

## System Requirements

- **GPU:** RTX 5090 (Blackwell, sm_120 architecture, 32GB VRAM)
- **CUDA:** 13.3 (driver 595.84+)
- **Python:** 3.10+
- **Package Manager:** uv (PEP 723 compliant)

## 1. Clone Stable Audio 3

```bash
cd ~
git clone https://github.com/stability-ai/stable-audio-3.git
cd stable-audio-3
```

## 2. PyTorch Installation (Critical: cu128 Index)

**Do not use cu126 or cu133 indices.** RTX 5090 support requires cu128 wheels specifically.

Create/replace `pyproject.toml` with:

```toml
[project]
name = "stable-audio-3"
version = "0.1.0"
description = "Stable Audio 3: A state-of-the-art open platform for fast, high-quality generated audio and music"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "einops>=0.8.2",
    "einops-exts>=0.0.4",
    "numpy>=2.2.6",
    "packaging>=26.0",
    "safetensors>=0.7.0",
    "torch==2.7.1",
    "torchaudio==2.7.1",
    "tqdm>=4.67.3",
    "huggingface-hub>=1.7.1",
    "transformers>=5.8.0",
    "soundfile>=0.13.1",
]

[project.optional-dependencies]
ui = [
  "gradio==6.3.0",
  "matplotlib>=3.10.8",
  "accelerate>=1.13.0",
]

lora = [
  "pytorch_lightning==2.5.5",
  "dill>=0.4.1",
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }
]
torchaudio = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" }
]

[project.scripts]
stable-audio = "stable_audio_3.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=9.0.3",
    "ruff>=0.15.9",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
exclude = [
    "stable_audio_3/models",
    "stable_audio_3/inference",
    "stable_audio_3/interface",
    "stable_audio_3/data",
    "stable_audio_3/training",
    "optimized",
]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
```

Sync with uv:

```bash
uv sync --reinstall
```

Verify PyTorch installation:

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

Expected output: `PyTorch: 2.7.1+cu128` and `CUDA: 12.8`

## 3. Flash Attention Installation

Install flash-attn using uv with no-build-isolation flag:

```bash
uv pip install flash-attn --no-build-isolation
```

This builds flash-attn==2.8.3.post1 with RTX 5090 sm_120 support.

Verify:

```bash
uv run python -c "from flash_attn import flash_attn_func; print('Flash Attention OK')"
```

## 4. Install LoRA Dependencies

```bash
cd ~/stable-audio-3
uv sync --extra lora
```

This installs pytorch_lightning==2.5.5 and dill>=0.4.1 required by the training script.

Verify:

```bash
uv run python -c "import pytorch_lightning; print('pytorch_lightning OK')"
```

## 5. Dataset Preparation

Organize audio and caption files in:

```
~/Documents/restorative_embeddings/generation/train/araus_for_sa3/
├── neutral/
│   ├── clip_name_1.wav
│   ├── clip_name_1.txt
│   ├── clip_name_2.wav
│   ├── clip_name_2.txt
│   └── ...
├── pleasant/
├── unpleasant/
├── very_pleasant/
└── very_unpleasant/
```

**Format requirements:**
- Audio: 44.1 kHz WAV files
- Duration: 10 seconds (matches `--duration 10` parameter)
- Captions: Plain text files with same basename as audio, one description per line
- Total: ~6000 files across all pleasantness bins

## 6. Training Configuration

Run training from the stable-audio-3 root directory:

```bash
cd ~/stable-audio-3

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir ~/Documents/restorative_embeddings/generation/train/araus_for_sa3 \
  --rank 16 \
  --adapter_type dora-rows \
  --steps 5000 \
  --batch_size 1 \
  --lr 1e-4 \
  --duration 10 \
  --base_precision bf16 \
  --name araus_lora_rank16_dora \
  --save_dir ./lora_checkpoints \
  --checkpoint_every 500 \
  --log_every 100 \
  --demo_every 500 \
  --logger csv
```

### Parameter Explanation

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model` | medium-base | 1.9B parameters; efficient for RTX 5090 32GB VRAM |
| `rank` | 16 | LoRA rank; ~1-5% trainable parameters |
| `adapter_type` | dora-rows | Distributed DoRA variant for better generalization |
| `steps` | 5000 | ~0.83 epochs of 6000-sample ARAUS dataset |
| `batch_size` | 1 | Fits in VRAM with bf16 precision |
| `lr` | 1e-4 | Standard LoRA learning rate |
| `duration` | 10 | Audio clip length in seconds |
| `base_precision` | bf16 | bfloat16 mixed precision; faster, memory-efficient |
| `checkpoint_every` | 500 | Save 10 checkpoints (at steps 500, 1000, ..., 5000) |

## 7. Training Performance

On RTX 5090 with this configuration:

- **Speed:** ~1.65 iterations/second
- **Total time:** ~50 minutes for 5000 steps
- **Checkpoints:** 10 total (500-step intervals), ~50-100MB each
- **Epochs:** 0.83 (does not complete full epoch of dataset)

## 8. Monitoring Training

### Real-time progress bar

The terminal shows live progress:

```
Epoch 0:   2%|█                  | 83/5000 [00:50<49:44,  1.65it/s, v_num=2]
```

### CSV logs

Check training metrics in CSV format:

```bash
tail -f lora_checkpoints/*/metrics.csv
```

### Checkpoint directory

Monitor checkpoint saves:

```bash
watch -n 5 'ls -lh lora_checkpoints/ | tail -10'
```

## 9. Output

After training completes:

- **LoRA adapter:** `lora_checkpoints/araus_lora_rank16_dora/` (step 5000)
- **Checkpoint size:** ~50-100MB
- **Training logs:** CSV format in checkpoint directory

## Troubleshooting

### PyTorch downgrade to cu126 after `uv sync`

If `uv sync` downgrades to torch 2.7.1+cu126:
- Verify `pyproject.toml` has `pytorch-cu128` index with `explicit = true`
- Delete `uv.lock`
- Run `uv sync --reinstall`

### Flash Attention build fails

If `uv pip install flash-attn` fails:
- Ensure CUDA 13.3 is set: `export CUDA_HOME=/usr/local/cuda-13.3`
- Set compute capability: `export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;12.0"`
- Try: `uv pip install flash-attn --no-build-isolation -v`

### PyTorch not found on cu128 index

Only cu128 wheels work with RTX 5090. Indices cu126, cu133, cu139 do not provide compatible wheels for this GPU.

### pytorch_lightning not found during training

Ensure you ran: `uv sync --extra lora`

## Notes

- The medium-base model is frozen; only the LoRA adapter is trainable
- ARAUS dataset should have matching audio-caption pairs (same filename, different extension)
- Loss should decrease monotonically over steps; if not, consider reducing learning rate
- Checkpoints save every 500 steps; use final checkpoint (step 5000) unless earlier checkpoints show better validation metrics
- Training does not reach full epoch (5000 < 6000 samples); this prevents overfitting

## References

- Stable Audio 3: https://github.com/stability-ai/stable-audio-3
- PyTorch wheels: https://download.pytorch.org/whl/cu128
- Flash Attention: https://github.com/Dao-AILab/flash-attention

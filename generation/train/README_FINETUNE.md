# Fine-Tuning Stable Audio Open on Restorative Soundscapes

**Objective:** Fine-tune Stable Audio Open (SAO) on 6,000 stratified ARAUS recordings with pleasantness-conditioned captions to test if SAO can generate audio at different perceived pleasantness levels.

**Status:** Phase 1 complete. Dataset ready. Phase 2 begins here.

---

## Quick Start (TL;DR)

```bash
cd ~/Documents/Bamberg/restorative_embeddings

# 1. Setup environment (one-time)
python generation/train/01_setup_environment.py

# 2. Configure training (edit config as needed)
python generation/train/02_prepare_training_config.py --model-size small --batch-size 2

# 3. Run training (on orpheus)
python generation/train/03_train_model.py --config generation/train/config_sao_small.yaml

# 4. Generate test audio and evaluate
python generation/train/04_generate_and_evaluate.py --checkpoint [path_to_checkpoint]
```

---

## Detailed Workflow: 7 Steps to Phase 2 Complete

### Step 0: Pre-Flight Check
Before starting, ensure:
- [ ] You have 6,000 audio files in `data/generation/audio/`
- [ ] You have `data/generation/export_metadata.csv` with captions
- [ ] You're on `orpheus` (RTX 5090, 32 GB VRAM)
- [ ] You have Python 3.10+ in a fresh `.venv`
- [ ] You understand the dataset: stereo, 44.1 kHz, ~30s, -23 LUFS normalized

**Validation:**
```bash
ls data/generation/audio/ | wc -l  # Should be ~6000
head -3 data/generation/export_metadata.csv
```

---

### Step 1: Environment Setup (30 min)

Run the automated setup script:
```bash
python generation/train/01_setup_environment.py
```

**What it does:**
1. Creates a fresh `.venv` for training (isolated from milestone 1)
2. Installs `stable-audio-tools` from GitHub
3. Installs PyTorch + CUDA dependencies for RTX 5090
4. Installs evaluation tools: `pyloudnorm`, `librosa`, `transformers`, `clap`
5. Validates installation with a smoke test

**Manual fallback** (if script fails):
```bash
python3.10 -m venv venv_sao_training
source venv_sao_training/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/Stability-AI/stable-audio-tools.git
cd stable-audio-tools && pip install -e . && cd ..
pip install pyloudnorm librosa transformers laion-clap tqdm
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Step 2: Prepare Training Configuration (15 min)

Run the config generator:
```bash
python generation/train/02_prepare_training_config.py \
  --model-size small \
  --batch-size 2 \
  --learning-rate 5.0e-5 \
  --max-steps 2000 \
  --checkpoint-interval 500
```

**Options:**
- `--model-size`: `small` (341M, ~11s) or `large` (1.1B, ~47s). **Start with `small` for 10s crops.**
- `--batch-size`: 1–2 for 32GB VRAM. Reduce if OOM.
- `--learning-rate`: Typically 5.0e-5 for fine-tuning. Don't change unless you know why.
- `--max-steps`: 2000 recommended (one epoch on 6k samples with batch size 2).
- `--checkpoint-interval`: Save every 500 steps (~1 hour).

**Output:**
- `generation/train/config_sao_small.yaml` (or `config_sao_large.yaml`)
- A detailed report showing:
  - Expected training time
  - Total gradient updates
  - Checkpoint schedule
  - Memory usage estimate

**What you CAN change in the generated config:**
- `data_audio_dir`: Path to audio files (defaults to `data/generation/audio/`)
- `metadata_csv`: Path to captions (defaults to `data/generation/export_metadata.csv`)
- `checkpoint_dir`: Where to save model checkpoints (defaults to `generation/train/checkpoints/`)

**What you MUST NOT change:**
- `sample_rate`, `channels`, `audio_channels`, `seconds` (SAO's fixed format)
- `model_id` (must match `model_size`)

---

### Step 3: Establish Baseline (Unmodified SAO) (45 min – 2 hours)

Before fine-tuning, test the **unmodified SAO** on your data to understand what it can already do:

```bash
python generation/train/03_generate_baseline.py \
  --n-prompts 10 \
  --n-seeds 4 \
  --output-dir generation/train/baseline_generations
```

**What it generates:**
- 50 audio clips (10 held-out scene prompts × 5 pleasantness levels × 1 seed, base SAO)
- A results CSV with:
  - Prompt text
  - Pleasantness level word
  - Generated audio path
  - CLAP text–audio similarity (how well does it follow the prompt?)

**Interpret the baseline:**
- Does base SAO respond to "pleasant" vs. "unpleasant" at all?
- How similar are generations at different pleasantness levels?
- This becomes your control for phase 2. If fine-tuning does nothing, baseline is your fallback story.

**Example output:**
```
generation/train/baseline_generations/
├── results.csv
└── audio/
    ├── baseline_prompt_00_pleasant_seed001.wav
    ├── baseline_prompt_00_unpleasant_seed001.wav
    ├── ...
    └── baseline_prompt_09_very_pleasant_seed001.wav
```

---

### Step 4: Fine-Tune SAO (2–6 hours on RTX 5090)

Run the training script:
```bash
python generation/train/03_train_model.py \
  --config generation/train/config_sao_small.yaml \
  --resume-checkpoint none \
  --grad-accumulation-steps 4
```

**What happens:**
1. Loads your 6,000 audio files + captions into a PyTorch DataLoader
2. Initializes SAO from the pretrained checkpoint
3. Freezes the encoder, fine-tunes the decoder + diffusion head
4. Trains with text conditioning (caption) + timing metadata
5. Saves checkpoints every 500 steps
6. Logs loss curves to TensorBoard (optional)

**Monitoring:**
- Check GPU memory: `nvidia-smi` (should stay < 30 GB on RTX 5090)
- Monitor loss in console output. Typical pattern:
  - Step 0–200: rapid loss drop (model learns to reconstruct)
  - Step 200–1000: slower loss decrease (learns caption details)
  - Step 1000+: loss plateau (saturation)

**If you run out of memory:**
```bash
# Reduce batch size in config, or:
python generation/train/03_train_model.py \
  --config generation/train/config_sao_small.yaml \
  --override-batch-size 1 \
  --override-grad-accum 8
```

**Checkpoints saved to:** `generation/train/checkpoints/sao_small_*_step_*.pt`

**Typical output:**
```
Step 0: loss=5.234
Step 100: loss=1.842
Step 500: loss=0.654 [CHECKPOINT SAVED]
Step 1000: loss=0.521 [CHECKPOINT SAVED]
...
Step 2000: loss=0.412 [CHECKPOINT SAVED]
Training complete. Best checkpoint: checkpoints/sao_small_best_step_1800.pt
```

---

### Step 5: Generate with Fine-Tuned Model (1–2 hours)

Generate audio using the trained checkpoint:

```bash
python generation/train/04_generate_and_evaluate.py \
  --checkpoint generation/train/checkpoints/sao_small_best_step_1800.pt \
  --model-size small \
  --output-dir generation/train/finetuned_generations \
  --n-held-out-prompts 20 \
  --n-pleasantness-levels 5 \
  --seeds 1 2 3 4
```

**Options:**
- `--checkpoint`: Path to your trained `.pt` file. Try the **last** checkpoint first, then best by validation loss if available.
- `--n-held-out-prompts`: Use scenes from held-out ARAUS soundscapes (section 5 of plan_generation.md). Script extracts them automatically.
- `--n-pleasantness-levels`: 5 (very_unpleasant, unpleasant, neutral, pleasant, very_pleasant)
- `--seeds`: Different random seeds for diversity. At least 4 recommended.

**Output:**
```
generation/train/finetuned_generations/
├── results.csv              # All generations + metadata
├── pleasantness_analysis.json  # Within-prompt correlations
└── audio/
    ├── finetuned_prompt_00_very_unpleasant_seed1.wav
    ├── finetuned_prompt_00_unpleasant_seed1.wav
    ├── finetuned_prompt_00_neutral_seed1.wav
    ├── finetuned_prompt_00_pleasant_seed1.wav
    ├── finetuned_prompt_00_very_pleasant_seed1.wav
    └── ... (100 total: 20 prompts × 5 levels)
```

---

### Step 6: Automatic Evaluation (30 min)

The script already runs evaluation during generation. Check the results:

```bash
# View summary statistics
cat generation/train/finetuned_generations/pleasantness_analysis.json | python -m json.tool

# Compare fine-tuned vs. baseline (if you have both)
python generation/train/05_compare_baseline_finetuned.py \
  --baseline-results generation/train/baseline_generations/results.csv \
  --finetuned-results generation/train/finetuned_generations/results.csv \
  --output-dir generation/train/comparison
```

**Metrics the script computes:**

1. **Pleasantness Effect (Main Test):**
   - Spearman correlation between pleasantness level (1–5) and CLAP-predicted pleasantness score
   - Per prompt: r ≈ 0.3–0.8 is good
   - Across prompts (mean r): target > 0.4

2. **Source Shortcut Check:**
   - CLAP relative source scores (bird, traffic, human, natural) across pleasantness levels
   - Do "pleasant" samples just have more birds/nature? Scores should be independent of level.
   - Report via partial correlation: pleasantness effect controlling for sources

3. **Prompt Adherence:**
   - CLAP text–audio similarity (prompt vs. generated audio)
   - Target: ≥ 0.2 (low bar; SAO is not a captioner)

4. **Realism (FAD):**
   - Fréchet Audio Distance against held-out real recordings
   - Trend: fine-tuned < baseline is good (closer to real data)

**Interpretation matrix:**

| Main Effect (r > 0.4) | Sources Indep. | Outcome |
|---|---|---|
| ✅ Yes | ✅ Yes | **SUCCESS**: Real pleasantness effect, not just sources. Proceed to phase 2. |
| ✅ Yes | ❌ No | **Shortcut found**: "Pleasant" = more natural. Important finding, but slider idea needs rethinking. |
| ❌ No | — | **No effect**: Label noise or SAO can't learn from these captions. Try label variant (b) or coarser bins. |

---

### Step 7: Listening Test (Internal Pilot) (2–3 hours prep + 30 min listening)

If automatic evaluation shows a clear effect (r > 0.4), run a blinded listening test:

```bash
python generation/train/06_listening_set.py \
  --finetuned-results generation/train/finetuned_generations/results.csv \
  --n-clips 20 \
  --output-dir generation/train/listening_test
```

**Output:**
- `listening_test/clips/` — 20 anonymized audio clips
- `listening_test/rating_sheet.html` — Interactive web form with 8 PAQ items per clip
- `listening_test/answer_key.csv` — Maps clips back to conditions (for later analysis)

**Procedure:**
1. Open `rating_sheet.html` in a browser
2. Listen to each clip (headphones recommended)
3. Rate on 5-point scales: eventfulness, familiarity, pleasantness, eventfulness, loudness, relaxation, etc. (reuse PAQ from milestone 1, section 6)
4. Save responses to `listening_test/responses.csv`
5. Run analysis:
   ```bash
   python generation/train/07_analyze_listening_test.py \
     --responses listening_test/responses.csv \
     --answer-key listening_test/answer_key.csv
   ```

---

## Decision Tree: What to Do After Step 6

```
Clear within-prompt pleasantness effect (r > 0.4)?
│
├─ YES → Sources independent (partial r > 0.3)?
│        │
│        ├─ YES → ✅ PROCEED TO PHASE 2 (numeric slider)
│        │
│        └─ NO  → 🔍 IMPORTANT FINDING (effect is sourced)
│                   |→ Document & decide: continue anyway, or reframe?
│
└─ NO  → 🧪 TRY THESE:
         |→ Use label variant (b) [stimulus mean across SMRs]
         |→ Coarsen bins to 3 words (very unpleasant, neutral, very pleasant)
         |→ Add more field recordings (ISD training data)
         |→ Try SAO Large instead of Small
```

---

## File Structure After Phase 2

```
generation/train/
├── README_FINETUNE.md (this file)
├── 01_setup_environment.py
├── 02_prepare_training_config.py
├── 03_train_model.py
├── 03_generate_baseline.py
├── 04_generate_and_evaluate.py
├── 05_compare_baseline_finetuned.py
├── 06_listening_set.py
├── 07_analyze_listening_test.py
│
├── config_sao_small.yaml [AUTO-GENERATED]
├── config_sao_large.yaml [AUTO-GENERATED]
│
├── checkpoints/
│   ├── sao_small_warmup_step_100.pt
│   ├── sao_small_best_step_500.pt
│   └── sao_small_final_step_2000.pt
│
├── baseline_generations/
│   ├── results.csv
│   └── audio/ [50 base SAO clips]
│
├── finetuned_generations/
│   ├── results.csv
│   ├── pleasantness_analysis.json
│   └── audio/ [100 fine-tuned clips]
│
├── comparison/
│   ├── effect_size_comparison.csv
│   └── figure_pleasantness_vs_level.png
│
└── listening_test/ [if run]
    ├── rating_sheet.html
    ├── answer_key.csv
    └── responses.csv
```

---

## Important Notes & Gotchas

### 1. License Check
Before running step 3, verify licenses:
- **SAO weights:** Stability AI Community License (free for non-commercial research)
- **ARAUS:** Need to check sources (USotW, maskers)
- **ISD:** CC BY 4.0

Document in your final report.

### 2. Seed & Reproducibility
All scripts accept `--seed` argument. Use `seed=42` for reproducible results.

### 3. GPU Memory
If training OOMs:
- Reduce batch size to 1 (increase grad_accumulation to 16)
- Use SAO Small instead of Large
- Crop to 10 s instead of 30 s (edit config)

### 4. Checkpoint Selection
After training, you'll have checkpoints at steps 500, 1000, 1500, 2000. The "best" is not always the last:
- Use the **last** checkpoint for baseline comparison (highest training step)
- If overfitting: use the **early** checkpoint (e.g., step 500)

### 5. Generation is Stochastic
Even with the same seed, SAO's diffusion sampling has variability. Run at least 2–4 seeds per prompt for stable estimates.

### 6. Loudness in Captions
Your captions include LA50 decibels (e.g., "quiet (LA50: 48 dB)"). SAO won't learn to *control* loudness; it learns to generate audio matching the caption's semantic content. Check if level information helps or hurts (ablation in phase 3).

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: stable_audio_tools` | Installation failed | Re-run `01_setup_environment.py` or manual install |
| OOM (out of memory) | Batch size too large | Reduce `batch_size` to 1, increase `grad_accum` |
| Loss NaN | Learning rate too high or bad data | Reduce LR to 1e-5, check audio files for silence/clipping |
| Generations sound like base SAO | Insufficient fine-tuning | Increase `max_steps` to 5000, lower LR to 1e-5 |
| No pleasantness effect | Data or captions weak | Try label variant (b), add ISD training, or reframe |
| Clips sound very similar | Low temperature or bad seeds | Increase seeds, lower guidance scale in config |

---

## Computing Time Estimate

On **orpheus** (RTX 5090, 32 GB):

| Step | Time | Notes |
|---|---|---|
| Step 1 (Setup) | 30 min | One-time |
| Step 2 (Config) | 5 min | One-time |
| Step 3 (Baseline) | 1 hour | Can skip if confident |
| Step 4 (Train) | 3–6 hours | Depends on `max_steps` and batch size |
| Step 5 (Generate) | 1–2 hours | 20 prompts × 5 levels × 4 seeds |
| Step 6 (Evaluate) | 10 min | Auto, happens in step 5 |
| Step 7 (Listening) | 2–3 hours | Optional; 30 min to run, 30 min to listen |
| **Total** | **~8–12 hours** | Spread over 2–3 days |

---

## References

- Stable Audio Open: [Evans et al., 2024](https://arxiv.org/abs/2407.14358)
- fine-tuning tutorial: [inkuele/stableaudio wiki](https://github.com/inkuele/stableaudio/wiki/Fine-Tuning-with-DCASE-Augmented-Dataset)
- Phase plan: `docs/plan_generation.md` (section 6: fine-tuning)
- Data: `docs/PHASE1_SUMMARY.md`

---

**Questions or issues?** Check the troubleshooting section above, or refer to the individual script docstrings.

**Next phase (phase 2):** Numeric pleasantness slider (continuous conditioning), if effect is clear.


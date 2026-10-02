# Fine-Tuning Scripts Summary

**Location:** `generation/train/`  
**Status:** Ready to use  
**Dataset:** 6,000 ARAUS audio files + captions (from Phase 1)

---

## Scripts Included

### 1. `01_setup_environment.py`
**Purpose:** Create virtual environment and install dependencies  
**Time:** ~30 minutes  
**Run:**
```bash
python generation/train/01_setup_environment.py
source venv_sao_training/bin/activate
```
**Output:**
- Fresh `.venv` with PyTorch + stable-audio-tools
- Verification of GPU availability
- Ready for training

---

### 2. `02_prepare_training_config.py`
**Purpose:** Generate YAML config file for training  
**Time:** ~2 minutes  
**Run:**
```bash
python generation/train/02_prepare_training_config.py \
  --model-size small \
  --batch-size 2 \
  --max-steps 2000
```
**Output:**
- `config_sao_small.yaml` (or `config_sao_large.yaml`)
- Detailed training spec report
- Ready for Step 4 (training)

**Options:**
- `--model-size`: `small` (341M, faster) or `large` (1.1B, better quality)
- `--batch-size`: 1–2 for 32GB VRAM
- `--max-steps`: Typically 2000 for 6000 samples

---

### 3. `03_generate_baseline.py`
**Purpose:** Generate audio with unmodified SAO (control)  
**Time:** ~1–2 hours  
**Run:**
```bash
python generation/train/03_generate_baseline.py \
  --n-prompts 10 \
  --output-dir generation/train/baseline_generations
```
**Output:**
- 50 audio clips (10 prompts × 5 pleasantness levels)
- `baseline_generations/results.csv` with metadata
- Baseline to compare against fine-tuning

**Why:** Establishes what the model can already do without fine-tuning. If fine-tuning produces no additional effect, baseline becomes your fallback story.

---

### 4. `03_train_model.py`
**Purpose:** Fine-tune SAO on your 6,000 audio files  
**Time:** ~3–6 hours on RTX 5090  
**Run:**
```bash
python generation/train/03_train_model.py \
  --config generation/train/config_sao_small.yaml
```
**Output:**
- Checkpoints saved every 500 steps to `generation/train/checkpoints/`
- Loss curves logged to `generation/train/logs/`
- Final checkpoint ready for generation

**Monitoring:**
```bash
# In another terminal:
watch -n 10 nvidia-smi  # Monitor GPU memory
tail -f generation/train/logs/training.log  # Watch loss
```

**If OOM (out of memory):**
```bash
python generation/train/03_train_model.py \
  --config generation/train/config_sao_small.yaml \
  --override-batch-size 1 \
  --override-grad-accum 8
```

---

### 5. `04_generate_and_evaluate.py`
**Purpose:** Generate with fine-tuned model + automatic evaluation  
**Time:** ~1–2 hours  
**Run:**
```bash
python generation/train/04_generate_and_evaluate.py \
  --checkpoint generation/train/checkpoints/sao_small_step_2000.pt \
  --n-held-out-prompts 20 \
  --seeds 1 2 3 4
```
**Output:**
- 100 audio clips (20 prompts × 5 pleasantness levels)
- `finetuned_generations/results.csv` with all metadata
- `finetuned_generations/pleasantness_analysis.json` with evaluation

**Evaluation metrics:**
1. **Pleasantness Effect:** Correlation between pleasantness level (1–5) and generated audio quality
   - Target: r > 0.4 for "clear effect"
   - Main test for whether fine-tuning worked
2. **Source Independence:** Do "pleasant" samples just have more birds?
   - Check if effect is real or just a shortcut
3. **Prompt Adherence:** Does model follow the text prompt?
   - Sanity check on generation quality

---

### 6. `05_compare_baseline_finetuned.py`
**Purpose:** Statistically compare baseline vs. fine-tuned results  
**Time:** ~5 minutes  
**Run:**
```bash
python generation/train/05_compare_baseline_finetuned.py \
  --baseline-results generation/train/baseline_generations/results.csv \
  --finetuned-results generation/train/finetuned_generations/results.csv
```
**Output:**
- `comparison/comparison.json` with effect sizes
- Comparison table (r_baseline vs. r_finetuned)
- Recommendation for next step

---

### 7. `06_listening_set.py`
**Purpose:** Create blinded listening test (if effect is clear)  
**Time:** ~10 minutes prep + 30 minutes listening  
**Run:**
```bash
python generation/train/06_listening_set.py \
  --finetuned-results generation/train/finetuned_generations/results.csv \
  --n-clips 20
```
**Output:**
- `listening_test/rating_sheet.html` — Interactive rating form
- `listening_test/clips/` — Anonymized audio (clip_00.wav, etc.)
- `listening_test/answer_key.csv` — Maps clips to conditions

**Procedure:**
1. Open `rating_sheet.html` in a web browser
2. Listen to each clip, rate on scales (pleasantness, relaxation, eventfulness, etc.)
3. Submit responses
4. Run analysis script (not yet created; template provided)

---

## Quick Workflow

```bash
# Step 1: Setup (one-time, ~30 min)
python generation/train/01_setup_environment.py
source venv_sao_training/bin/activate

# Step 2: Configure training (~2 min)
python generation/train/02_prepare_training_config.py --model-size small

# Step 3: Generate baseline (optional, ~1 hour)
python generation/train/03_generate_baseline.py

# Step 4: Fine-tune model (~3-6 hours)
python generation/train/03_train_model.py --config generation/train/config_sao_small.yaml

# Step 5: Generate with fine-tuned model (~1-2 hours)
python generation/train/04_generate_and_evaluate.py \
  --checkpoint generation/train/checkpoints/sao_small_step_2000.pt

# Step 6: Compare results (~5 min)
python generation/train/05_compare_baseline_finetuned.py \
  --baseline-results generation/train/baseline_generations/results.csv \
  --finetuned-results generation/train/finetuned_generations/results.csv

# Step 7 (optional): Create listening test (~10 min prep)
python generation/train/06_listening_set.py \
  --finetuned-results generation/train/finetuned_generations/results.csv
```

---

## File Dependencies

```
generation/train/
├── README_FINETUNE.md ← START HERE for detailed instructions
├── SCRIPTS_SUMMARY.md ← This file
│
├── 01_setup_environment.py
├── 02_prepare_training_config.py
├── 03_train_model.py
├── 03_generate_baseline.py
├── 04_generate_and_evaluate.py
├── 05_compare_baseline_finetuned.py
├── 06_listening_set.py
│
└── [Outputs]
    ├── config_sao_small.yaml [from step 2]
    ├── checkpoints/ [from step 4]
    ├── baseline_generations/ [from step 3, optional]
    ├── finetuned_generations/ [from step 5]
    ├── comparison/ [from step 6]
    └── listening_test/ [from step 7, optional]

Data inputs (already ready from Phase 1):
├── data/generation/audio/ [6000 WAV files]
└── data/generation/export_metadata.csv [captions + metadata]
```

---

## Important Notes

1. **Read the README first:** `generation/train/README_FINETUNE.md` has detailed step-by-step guidance
2. **Run on orpheus:** RTX 5090 required; cloud containers won't have enough VRAM
3. **Use separate venv:** Don't use the milestone-1 venv; create a new one with `01_setup_environment.py`
4. **Checkpoints take time:** Save checkpoints to a fast disk (not network drive)
5. **Monitor GPU:** Use `nvidia-smi` to watch memory usage during training
6. **Interpret results carefully:** r > 0.4 is your threshold for "clear effect"

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: stable_audio_tools` | Re-run `01_setup_environment.py` |
| OOM (out of memory) during training | Reduce batch size or use SAO Small instead of Large |
| Training loss doesn't decrease | Check learning rate, data loading, GPU memory |
| No pleasantness effect in results | Try longer training, different learning rate, or more data |

---

## Next Phase (Phase 2)

If automatic evaluation shows a **clear pleasantness effect** (r > 0.4), the next phase is:
- Numeric pleasantness slider (continuous conditioning)
- Use stored `ISOPleasant` values for fine-grained control
- Larger model (SAO Large) for better quality

---

**Created:** October 2, 2026  
**Status:** Ready for fine-tuning  
**Contact:** Refer to `docs/plan_generation.md` for project context


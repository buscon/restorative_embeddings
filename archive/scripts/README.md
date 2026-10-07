# ARAUS LoRA Training - Diverse Dataset Pipeline

## Problem
The current LoRA checkpoint generates birds in all traffic scenarios because the training data is too homogeneous (58% identical captions: "urban park with birds and ambient activity"). The model has learned to associate all inputs with the "park → birds" pattern rather than learning pleasantness conditioning.

## Solution
Create a diverse training subset stratified by pleasantness levels with diverse captions that describe different acoustic characteristics at different pleasantness ratings.

## Scripts

### 1. `01_inspect_prompts.py`
**Purpose**: Analyze current training dataset for caption diversity

**What it does**:
- Loads all .json files from `araus_for_sa3_encoded/`
- Counts total vs unique prompts
- Identifies most common phrases and repetition patterns
- Outputs: `all_prompts.txt`, `prompt_stats.txt`

**Run on**: Remote server (where data exists)
```bash
python scripts/01_inspect_prompts.py
```

**Expected output**: ~58% of prompts identical, very low diversity (4.7%)

---

### 2. `03_analyze_araus_diversity.py`
**Purpose**: Understand ARAUS dataset structure and available diversity

**What it does**:
- Loads raw ARAUS CSVs: soundscapes.csv, responses.csv, maskers.csv
- Analyzes scene distribution (which scenes appear frequently)
- Analyzes masker type distribution (birds, water, wind, traffic, etc.)
- Analyzes pleasantness distribution (ISOPleasant ratings 0.0-1.0)
- Identifies available scene-masker combinations
- Outputs: `araus_diversity_report.txt`

**Run on**: Remote server
```bash
python scripts/03_analyze_araus_diversity.py
```

**What to look for**:
- Which scenes dominate? (aim to reduce dominance)
- What masker types are available?
- How are pleasantness ratings distributed? (should be relatively uniform)
- How many unique scene-masker combinations exist? (aim to sample from this space)

---

### 3. `04_select_diverse_subset.py`
**Purpose**: Select balanced subset and generate diverse captions

**What it does**:
- Stratifies ARAUS responses into 5 pleasantness bins
- Selects ~20 samples per bin (adjust `SAMPLES_PER_BIN` to control size)
- Ensures variety of scenes and masker types within each bin
- Generates diverse captions that reflect pleasantness level:
  - Low pleasantness: "harsh", "disturbing", "unpleasant"
  - Mid pleasantness: "moderate", "neutral", "acceptable"
  - High pleasantness: "pleasant", "enjoyable", "delightful"
- Outputs: `diverse_subset.csv`, `caption_diversity_report.txt`

**Run on**: Remote server
```bash
python scripts/04_select_diverse_subset.py
```

**Output CSV columns**:
- `stimulus_id`: ARAUS stimulus ID (maps to audio file)
- `masker_id`: ARAUS masker ID
- `scene`: Scene description
- `masker_type`: Type of masker sound
- `pleasantness_bin`: Bin 0-4 (low to high)
- `pleasantness_rating`: Average rating for this combo
- `new_caption`: Generated diverse caption with pleasantness language

---

## Complete Workflow

### Step 1: Understand Current Problem
```bash
python scripts/01_inspect_prompts.py
cat all_prompts.txt
cat prompt_stats.txt
```

Look for: Very low diversity (%), high repetition of identical prompts

### Step 2: Analyze Available Data
```bash
python scripts/03_analyze_araus_diversity.py
cat araus_diversity_report.txt
```

Look for: Distribution of scenes, maskers, pleasantness; available combinations

### Step 3: Select Diverse Subset
```bash
python scripts/04_select_diverse_subset.py
cat caption_diversity_report.txt
head -20 diverse_subset.csv
```

This creates the mapping of which files to use and what captions to assign.

### Step 4: Extract Audio Files (MANUAL)
Use `diverse_subset.csv` to identify which stimulus_id + masker_id combinations to extract from ARAUS:

```bash
# For each row in diverse_subset.csv:
# Extract audio file for stimulus_id and masker_id combination
# The ARAUS dataset structure stores these at specific paths
```

### Step 5: Pre-encode Selected Audio
Create training dataset with new diverse captions:
- Load audio for each selected stimulus_id + masker_id
- Pre-encode to latents using SA3 model
- Use `new_caption` from diverse_subset.csv instead of original caption
- Save as .json files in new training directory

### Step 6: Re-train LoRA
Train with the diverse dataset:
```bash
python train_lora.py --dataset diverse_training_data/ --epochs 10
```

Expected results:
- Model learns that pleasantness affects acoustic characteristics
- Birds should NOT appear in traffic scenarios
- Different pleasantness levels should produce distinctly different audio

## Tuning Parameters

Edit these in `04_select_diverse_subset.py`:

```python
PLEASANTNESS_BINS = 5      # Number of bins (5-6 is typical)
SAMPLES_PER_BIN = 20        # Samples per bin (increase for larger subset)
MIN_SCENES_PER_BIN = 3      # Minimum scene variety per bin
MIN_MASKERS_PER_BIN = 3     # Minimum masker type variety per bin
```

Recommended starting point: ~100 total samples (5 bins × 20 samples)

## Expected Improvements

### Before (homogeneous data):
- All prompts similar: "urban park with birds..." → model learns scene pattern
- Loss doesn't decrease (overfitting to scene)
- Conditioning on pleasantness has no effect
- Output: birds in all scenarios

### After (diverse data):
- Captions vary by pleasantness level
- Model learns pleasantness affects acoustic characteristics
- Loss decreases and converges
- Conditioning on pleasantness produces audible differences
- Output: birds only in high-pleasantness park scenarios; traffic sounds in traffic scenarios

## Troubleshooting

**Problem**: Still getting birds in traffic after retraining
- Check caption diversity report: are pleasantness levels clearly different?
- Increase `SAMPLES_PER_BIN` to get more training data
- Ensure traffic scenes are well-represented (not dominated by park scenes)

**Problem**: Training loss not decreasing
- Check that new captions are actually different from old ones
- Verify audio extraction worked correctly
- Check learning rate in training config

**Problem**: Script can't find ARAUS data
- Verify CSV files exist at `/home/marcello/Documents/restorative_embeddings/data/raw/araus/data/`
- Check column names in CSVs (scripts use flexible detection but may need adjustment)

## Files Generated

After running all scripts, you'll have:

```
/home/marcello/Documents/restorative_embeddings/
├── all_prompts.txt              # All current training prompts
├── prompt_stats.txt             # Prompt diversity statistics
├── araus_diversity_report.txt   # ARAUS dataset analysis
├── diverse_subset.csv           # Selected samples & captions
└── caption_diversity_report.txt # Subset statistics
```

Then:
1. Extract audio for selected stimulus_ids from ARAUS
2. Pre-encode with new captions
3. Train LoRA with new dataset

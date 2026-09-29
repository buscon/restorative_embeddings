# Quickstart: Data Prep & Pipeline Testing

## Step 1: Prepare ISD Data

### 1.1 Download ISD Dataset

Visit: https://zenodo.org/records/10672568

Download all audio files (WAV archives). Total: ~13 GB.

Extract to a directory, e.g., `~/data/isd_raw/`

### 1.2 Organize Audio Files

```bash
python prepare_isd_data.py
```

This will:
- Find all extracted WAV files
- Flatten directory structure for easy batch loading
- Create a manifest (metadata about each file)
- Output: `./isd_audio/` directory with organized files

**If you've already extracted:** Point the script to your raw directory, or manually organize audio into a flat folder (`./isd_audio/`) with names like:
```
location_sessionid_groupid_1.wav
location_sessionid_groupid_2.wav
...
```

## Step 2: Test Pipeline on Small Subset

Testing on ~100 files takes **3–5 minutes** on RTX 5090 (full dataset: ~30–45 minutes).

### 2.1 Run Test

```bash
python test_pipeline_subset.py ./isd_audio --subset-size 100 --n-clusters 6
```

This will:
1. Sample 100 random audio files
2. Extract CLAP embeddings
3. Run k-means clustering (6 clusters)
4. Extract acoustic features
5. Characterize clusters
6. Generate stratified listening samples
7. Clean up test files (use `--keep-subset` to retain)

### 2.2 Review Results

Results saved to `./test_results/`:

- **`cluster_characterization.json`** — Acoustic profile per cluster
  - Spectral properties (centroid, flatness, contrast)
  - Soundscape indices (NDSI, bioacoustic index)
  - Sample file paths for each cluster

- **`listening_samples.csv`** — Stratified sample for perceptual evaluation
  - ~15 files (varies by n_clusters)
  - Balanced representation across clusters
  - Ready for listening/annotation

- **`embeddings.npy`** — Raw CLAP embeddings (100, 512)
- **`labels_kmeans.npy`** — Cluster assignments
- **`acoustic_features.csv`** — Full feature matrix

### 2.3 Check Cluster Quality

Open `cluster_characterization.json` and examine:

```json
{
  "0": {
    "size": 18,
    "spectral_centroid_mean": 2840.5,
    "spectral_flatness_mean": 0.42,
    "ndsi_mean": 0.15,
    "sample_filepaths": ["location_1_1_1.wav", ...]
  },
  ...
}
```

- **size**: Is cluster distribution reasonable? (Not all samples in 1 cluster?)
- **NDSI**: Higher values = more natural (typically more restorative)
- **spectral_centroid**: Lower = bassier, higher = more treble

Listen to a sample from each cluster:

```bash
# macOS
open test_results/listening_samples.csv
# Then navigate to the audio files and open samples
```

## Step 3: Full Pipeline on Complete Dataset

Once testing validates the approach:

```python
from restorative_soundscape_pipeline import main

results = main(
    audio_dir='./isd_audio',
    output_dir='./full_results',
    n_clusters=10,  # Increase for more granularity
    batch_size=16
)
```

Expect:
- Runtime: **30–60 minutes** on RTX 5090
- Output: Same as test, but with ~1000 samples & better cluster separation

## Step 4: Listening & Annotation

Use `full_results/listening_samples.csv`:

1. **Extract listening samples** to a dedicated folder
2. **Listen systematically** to each sample (30 seconds)
3. **Rate restorativeness**:
   - 1–7 scale (1 = not restorative, 7 = highly restorative)
   - Or binary (restorative/not)
4. **Note acoustic observations**:
   - "Mostly wind + birds" → likely natural
   - "Traffic + construction" → anthropogenic
   - etc.

**Create annotation CSV**:

```csv
filepath,cluster,restorativeness_rating,notes
location_1_1_1.wav,0,6,Strong bird calls; natural soundscape
location_2_3_2.wav,1,2,Heavy traffic noise
...
```

This becomes your ground truth for:
- Validating cluster assignments
- Training a supervised model (next phase)
- Designing the formal study

## Troubleshooting

### CLAP model download fails
```
FileNotFoundError: transformers model not cached
```

First run of CLAP will download ~500MB from Hugging Face. Ensure internet connection & ~2GB free space.

### Out of memory error
```
torch.cuda.OutOfMemoryError
```

Reduce batch_size in `main()`:
```python
results = main(audio_dir, batch_size=8)  # Was 16
```

### No WAV files found
```
Found 0 audio files
```

Check `./isd_audio/` exists and contains `.wav` files (not `.mp3` or other formats). Verify extraction was complete.

### Slow embedding extraction
- GPU not detected? Check `torch.cuda.is_available()`
- Consider: are you running on CPU? (Much slower)

## Next Steps

After listening & annotation:

1. **Analyze patterns**
   - Which clusters are most restorative?
   - Acoustic features that correlate with restoration?

2. **HDBSCAN refinement** (optional)
   - Replace k-means with automatic clustering
   - Better for finding natural cluster boundaries

3. **Validation study design**
   - Sample size & recruitment
   - Study measures (subjective + physiological)
   - Timeline & analysis plan

---

For details on pipeline stages and parameters, see **README.md**.

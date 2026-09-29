# Restorative Soundscape Discovery

Unsupervised learning pipeline to discover restorative soundscape acoustic patterns using CLAP embeddings and acoustic feature analysis.

## Overview

This project uses a data-driven approach to identify soundscapes with restorative potential:

1. **CLAP Embeddings**: Extract high-level audio representations from raw soundscapes
2. **k-means Clustering**: Group soundscapes by acoustic similarity
3. **Acoustic Characterization**: Summarize spectral, temporal, and soundscape indices per cluster
4. **Listening & Validation**: Stratified sampling for perceptual evaluation and ground-truth studies

## Data

The pipeline uses the **Inspiring Soundscape Dataset (ISD)** from Zenodo:
https://zenodo.org/records/10672568

Download and place audio files in your data directory.

## Requirements

```bash
pip install librosa soundfile torch transformers scikit-learn soundscapy tqdm pandas numpy
```

**GPU Recommended**: CLAP embedding extraction is much faster on GPU (RTX 5090+ recommended for large datasets).

## Usage

### Basic Run

```python
from restorative_soundscape_pipeline import main

# Run the full pipeline
results = main(
    audio_dir='/path/to/isd/audio',
    output_dir='./results',
    n_clusters=8,
    batch_size=16
)
```

### Output Files

The pipeline saves:
- `embeddings.npy`: CLAP embeddings (N, 512)
- `labels_kmeans.npy`: k-means cluster assignments
- `cluster_characterization.json`: Acoustic summaries per cluster
- `listening_samples.csv`: Stratified sample for perceptual evaluation
- `acoustic_features.csv`: Full acoustic feature matrix

## Pipeline Stages

### 1. CLAP Embedding Extraction
- Loads pretrained CLAP model from Hugging Face
- Extracts 512-dim embeddings per audio file
- GPU-accelerated batch processing

### 2. k-means Clustering
- Fits k-means on embeddings (default k=8, adjustable)
- Quick exploratory clustering for initial inspection
- Later refined with HDBSCAN if needed

### 3. Acoustic Features
Uses librosa + soundscapy:
- **Spectral**: Centroid, flatness, contrast, MFCCs
- **Temporal**: Zero-crossing rate, onset density
- **Soundscape indices**: NDSI, bioacoustic index, acoustic complexity
  - NDSI is especially valuable: high values indicate natural environments (typically more restorative)
- **Loudness**: RMS approximation

### 4. Cluster Characterization
Summarizes each cluster's acoustic properties:
- Mean/std spectral features
- NDSI and naturalness indicators
- Sample file paths for listening

### 5. Stratified Sampling
Selects representative soundscapes from each cluster:
- ~8 samples per cluster (adjustable)
- Ensures balanced representation across clusters
- Output: `listening_samples.csv` for perceptual annotation

## Workflow

1. **Prepare data**: Download ISD, organize audio files
2. **Run pipeline**: Execute `main()` with your audio directory
3. **Listen & annotate**: Review samples from `listening_samples.csv`
   - Rate restorativeness (e.g., 1–7 scale)
   - Note acoustic observations
4. **Analyze listening data**: Cross-cluster patterns and outliers
5. **Run validation study**: Quantitative testing of promising clusters

## Next Steps

- HDBSCAN refinement: Automatic cluster detection on embeddings
- Validation study design: User studies or physiological measurement
- Acoustic feature importance: Identify which features predict restorativeness

## References

- CLAP: https://github.com/LAION-AI/CLAP
- soundscapy: https://github.com/marl/soundscapy
- librosa: https://librosa.org/
- ISD Dataset: https://zenodo.org/records/10672568

---

**Contact**: Marcello Lussana, Computational Humanities Group, University of Bamberg

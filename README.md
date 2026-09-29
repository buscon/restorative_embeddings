# Restorative soundscape embeddings

Can a general-purpose audio embedding (CLAP) predict how people perceive a
soundscape, and does that prediction carry over from the lab to the field?

The model is **trained on ARAUS** (lab ratings of ~25k augmented urban
soundscapes) and **tested on ISD** (in-situ ratings from 18 locations in
London, Venice, Granada and Groningen). The targets are the ISO 12913-3
circumplex coordinates ISOPleasant and ISOEventful. Restorativeness itself
is not modelled here: ISOPleasant is the step in the chain
acoustics → pleasantness → restorativeness that can be trained at scale.
The link to measured PRSS scores is a later stage (AMSS data; see
`docs/session_handoff.md`).

## Why this design

The reasoning and the numbers behind it are in `docs/session_handoff.md`.
In short:

- No open dataset pairs restorativeness (PRSS) ratings with audio at scale.
- On ISD alone, psychoacoustic predictors reach R² ≈ 0 or below under
  location-grouped cross-validation (reproduced from Versümer et al., 2025).
  About 70 % of the ISOPleasant variance lies between people at the same
  place, each of whom gave one rating.
- ARAUS has many acoustically independent stimuli and official folds that are
  disjoint in soundscapes, maskers and participants. Honest R² there is
  0.17–0.28 with psychoacoustic predictors.
- So ARAUS carries the training, and ISD tests whether a lab-trained model
  generalises to real places.

**Expected ceiling.** Acoustics usually explain about a third of perceptual
variance. Modest ISD numbers are therefore the realistic outcome, not a sign
of failure. The question is whether CLAP improves on the psychoacoustic
baseline, and by how much.

## Data

| | ARAUS v1 (train) | ISD v1.0 (test) |
|---|---|---|
| Source | DR-NTU `doi:10.21979/N9/9OTEVX`, via the authors' `download.py` | Zenodo 10672568 (CC BY 4.0) |
| Audio | 30 s binaural stimuli: USotW soundscape + masker at SMR −6…+6 dB | ~30–35 s binaural field recordings, 48 kHz, 32-bit float, calibrated in Pa |
| Ratings | lab, headphones; 600 participants in folds 1–5, 5 in test fold 0 | in situ; one rating per person |
| Size used | responses in folds 0–5 without practice/attention stimuli | 1,452 WAVs; 823 of them rated (1,444 ratings, 18 locations). The rest are mostly 2020 lockdown recordings without surveys. |

The ISD figures were checked against the Zenodo archives and
`ISD v1.0 Data.csv` (September 2026).

## Pipeline

```
scripts/download_isd.sh          ISD CSV, metadata, audio  (-> data/raw/isd)
scripts/download_araus.sh        ARAUS via the authors' downloader (-> data/raw/araus)
scripts/01_prepare.py            tables: ratings, ISO coordinates, harmonised psychoacoustics
scripts/02_embed.py              CLAP embeddings + ecoacoustic indices, per dataset
scripts/03_train_evaluate.py     train on ARAUS, test on ARAUS fold 0 and on ISD
scripts/04_clusters.py           unsupervised: k-means on ARAUS, ISD assigned; HDBSCAN on ISD
scripts/05_listening_sample.py   stratified, blinded listening set per cluster
tests/smoke_test.py              runs everything on synthetic data (no downloads)
```

### Setup

```bash
# 1. PyTorch for your CUDA version first (RTX 5090 needs a CUDA 12.8+ build)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. the rest
pip install -r requirements.txt
# 3. check that everything runs (synthetic data, ~5 min on CPU)
python tests/smoke_test.py
```

### Run

```bash
bash scripts/download_isd.sh           # ~16 GB; SKIP_LOCKDOWN=1 to skip unrated archives
bash scripts/download_araus.sh         # ~3 GB
python scripts/01_prepare.py
python scripts/02_embed.py --dataset isd --limit 50    # quick check first
python scripts/02_embed.py --dataset isd
python scripts/02_embed.py --dataset araus --workers 12 --fp16
python scripts/03_train_evaluate.py
python scripts/04_clusters.py          # or --k 10
python scripts/05_listening_sample.py --per-cluster 6 --araus-per-cluster 2
```

`02_embed.py` resumes where it stopped. For ARAUS the stimuli are rebuilt in
memory from soundscape + masker + SMR, exactly as in the authors'
`make_augmented_soundscapes` (checked sample by sample against their code).
This avoids writing ~132 GB of WAVs. CPU work (loading, mixing, indices) is
usually the bottleneck, so give it as many `--workers` as you have cores.
`--no-indices` skips the ecoacoustic indices.

## Method details

**Audio preparation (both datasets).** Binaural → mono (channel mean), 48 kHz,
first 30 s, RMS-normalised to −26 dBFS. Normalisation is needed because ISD
files are calibrated in pascal while ARAUS mixes are digital playback signals.
Absolute level is removed from the audio and supplied separately as calibrated
LA50 (`clap+level` feature set).

**Embeddings.** `laion/clap-htsat-unfused`. There are three non-overlapping 10 s
windows per recording, and their embeddings are averaged. Using exactly 10 s
windows avoids CLAP's random cropping, so the embeddings are deterministic.

**Ecoacoustic indices** (scikit-maad): NDSI, BI, ACI, ADI, H, plus spectral
centroid, spectral flatness and onset rate (librosa).

**Targets.** ISOPleasant and ISOEventful in [−1, 1], computed with
`soundscapy.surveys.calculate_iso_coords` from the eight PAQ items. Both
datasets use the same items and scales.

**Harmonised psychoacoustic baseline** (`rsd/data.py`): LA50, LA10−LA90,
LC50−LA50, N5, roughness, fluctuation strength and tonality. Relative Approach
is missing in ARAUS. Sharpness is left out because the methods differ
(ARAUS DIN 45692, ISD Aures). LA50 is used instead of LAeq because ARAUS
reports a fast-averaged mean level, not LAeq.

**Models.** Ridge regression for every feature set, plus gradient boosting for
the low-dimensional sets. Hyper-parameters are tuned on ARAUS folds 1–5
(official split). The model is then refitted on folds 1–5 and applied
unchanged to ARAUS fold 0 and to ISD.

**Evaluation on ISD** at three levels: individual ratings, recording means and
location means. R² is reported with Pearson r and the mean bias, because a
lab-to-field shift in the mean lowers R² even when the ranking is right. The
recording-level r has a 95 % CI from a bootstrap over locations. For
ISOPleasant, the correlation with two restoration-adjacent ISD items is also
reported: overall soundscape quality (`sss01`) and wish to revisit (`sss05`).

**Unsupervised part.** k-means is fitted on ARAUS embeddings (PCA 50) and ISD
recordings are assigned to the nearest cluster. Clusters are profiled by lab
and field ratings, masker types, locations and indices. A cluster that is
pleasant in both datasets is evidence that the embedding captures something
transferable, whatever the regressor does. HDBSCAN on ISD alone is
exploratory.

## Outputs

```
results/metrics.csv | metrics.md       all scores, one row per features × model × target
results/isd_predictions.csv            per-recording predictions for every model
results/model_*.joblib                 fitted models
results/clusters/kmeans_profiles.csv   cluster descriptions (lab + field)
results/clusters/pca_isopleasant.png   ARAUS vs ISD in the first two PCs
results/listening/                     blinded clips, listening sheet, key
```

## Known limitations

- ARAUS is a lab dataset (headphones, video, augmented stimuli). ISD is
  in situ, so participants also see the place and have their own reasons for
  being there. Part of the gap is context, not acoustics.
- Mono downmixing discards binaural cues. CLAP is a mono model.
- Channel aggregation of the psychoacoustic indicators may differ between the
  two datasets. Tonality uses ECMA-74 in both, but the implementations are
  not identical.
- ISD has one rating per person and only WHO-5 as a person variable. The
  person-versus-sound question needs ARAUS or AMSS (see the handoff).
- ISOPleasant is a proxy. In AMSS, ISOPleasant explains about 40–56 % of the
  PRSS composite, so an acoustics → PRSS ceiling of R² ≈ 0.07–0.10 is the
  pre-registered expectation for the next stage.

## References

- Ooi et al. (2023). ARAUS: A large-scale dataset and baseline models of affective responses to augmented urban soundscapes. *IEEE Trans. Affective Computing*. arXiv:2207.01078
- Mitchell et al. International Soundscape Database v1.0. Zenodo 10672568
- Versümer, Blättermann, Rosenthal & Weinzierl (2025). *JASA* 157(1), 234–255. doi:10.1121/10.0034849
- Elizalde et al. (2023). CLAP: Learning audio concepts from natural language supervision. / LAION-CLAP, Wu et al. (2023), ICASSP
- Soundscapy: https://github.com/MitchellAcoustics/Soundscapy
- scikit-maad: https://github.com/scikit-maad/scikit-maad
- Payne (2013). The production of a Perceived Restorativeness Soundscape Scale. *Applied Acoustics*

Marcello Lussana, Computational Humanities Group, University of Bamberg

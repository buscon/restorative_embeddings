# Project report: restorative soundscape modelling, milestone 1

**Repository:** `github.com/buscon/restorative_embeddings` · **Author:** Marcello Lussana (University of Bamberg) · **Status as of:** 2026-10-01
**Purpose of this file:** a self-contained record of the work so far, so that another session can continue without re-deriving anything. All numbers below come from runs of the scripts in this repository on the real data.

---

## 1. Aim and how it evolved

The long-term aim is to model the **restorativeness** of soundscapes from audio. No open dataset pairs restorativeness ratings (PRSS) with audio at scale, so this milestone models **ISOPleasant** (and ISOEventful), the ISO 12913-3 circumplex coordinates, as the trainable step towards restorativeness.

Course of the work:

1. **Original idea (Marcello):** unsupervised clustering of soundscape embeddings to see whether pleasant/restorative soundscapes form their own groups, then listen to the clusters and run a study.
2. **Design from a parallel session** (`docs/session_handoff.md`): ISD alone cannot support training (R² ≈ 0 or negative under location-grouped CV, reproducing Versümer et al., 2025). Decision: **train on ARAUS, test on ISD**.
3. **Milestone 1 (this report):** supervised lab→field transfer with CLAP embeddings, then a source-recognition analysis and a hybrid model.
4. **Next phase (planned, not started):** return to the unsupervised clustering question in a separate `clustering/` folder (section 9).

Background from the handoff that still matters:
- Versümer et al. (2025, JASA 157(1), 234–255) report honest out-of-sample R² on ARAUS of 0.17–0.19 (ISOPleasant) and 0.22–0.28 (ISOEventful) with psychoacoustic predictors; ISD gives negative R².
- AMSS (`github.com/ntudsp/amss-insitu-replication`, the only open data with PRSS + acoustics, **no audio**): ISOPleasant explains about 40–56 % of the PRSS composite; the expected acoustics→PRSS ceiling is R² ≈ 0.07–0.10. Person-level psychometrics explain about as much as the circumplex.

## 2. Data

| | ARAUS v1 (training) | ISD v1.0 (test) |
|---|---|---|
| Source | DR-NTU `doi:10.21979/N9/9OTEVX`, downloaded with the authors' `download.py` | Zenodo 10672568, CC BY 4.0 |
| Audio | 30 s binaural stimuli: USotW soundscape + masker (bird, water, wind, traffic, construction, silence) at SMR −6…+6 dB; rebuilt in memory, never written to disk | binaural field recordings, 48 kHz, 32-bit float, **calibrated in Pa**, mostly 30–35 s |
| Ratings | lab, headphones + video | in situ, one rating per person |
| Size used | 25,440 responses (folds 0–5 without practice/consistency/attention stimuli), 605 participants, 22,218 unique stimuli, 240 base soundscapes; training = folds 1–5 = 25,200 responses; test fold 0 = 48 stimuli × the same 5 raters | 1,452 WAVs with audio; 823 rated recordings, 1,444 ratings, 18 locations in 4 cities; ~630 unrated recordings (mostly 2020 lockdown); 2,108 survey ratings have no published audio |
| ISOPleasant mean (SD) | +0.025 (0.394) | +0.303 (0.395) |
| ISOEventful mean (SD) | −0.037 (0.404) | +0.128 (0.329) |

ISD locations and cities (rated set): **London 11** (CamdenTown, EustonTap, MarchmontGarden, PancrasLock, RegentsParkFields, RegentsParkJapan, RussellSq, StPaulsCross, StPaulsRow, TateModern, TorringtonSq), **Granada 4** (CampoPrincipe, CarloV, MiradorSanNicolas, PlazaBibRambla), **Venice 2** (SanMarco, MonumentoGaribaldi), **Groningen 1** (Noorderplantsoen).

Data quirks handled in code: macOS `__MACOSX/._*.wav` files in the ISD zips (not audio, skipped); odd names `NP125.hdf.wav` and duplicate `NP102.1.wav` (shortest name per GroupID kept); ~70 ISD recordings shorter than 30 s (shortest 7.5 s WAV); ISD audio files are named by GroupID; ARAUS responses CSVs are only on DR-NTU (blocked for cloud fetch tools; Marcello downloaded them locally).

## 3. Pipeline (milestone 1 scripts; keep unchanged)

Marcello wants these scripts kept as they are: they are the first milestone and may serve other purposes.

```
scripts/download_isd.sh          ISD CSV, metadata, audio zips (wget -c, unzip without macOS junk)
scripts/download_araus.sh        ARAUS via the authors' downloader (~3 GB; response CSVs from DR-NTU)
scripts/01_prepare.py            tables: ratings, ISO coordinates (soundscapy), harmonised psychoacoustics
scripts/02_embed.py              CLAP embeddings + ecoacoustic indices (resumable; ARAUS mixed on the fly)
scripts/03_train_evaluate.py     train on ARAUS folds 1-5, test on ARAUS fold 0 and ISD; paired bootstrap
scripts/04_clusters.py           k-means on ARAUS embeddings, ISD assigned; HDBSCAN on ISD (NOT yet run)
scripts/05_listening_sample.py   blinded, stratified listening set per cluster (NOT yet run)
scripts/06_source_recognition.py CLAP zero-shot source scores vs ISD source ratings; naturalness; partial r
scripts/07_hybrid_isd.py         ARAUS predictions + zero-shot source scores, grouped CV on ISD + robustness
tests/smoke_test.py              end-to-end on synthetic data (RSD_FAKE_EMBED=0 to use real CLAP)
rsd/                             audio.py, embed.py, indices.py, data.py (shared helpers)
docs/session_handoff.md          handoff from the parallel session (Versümer, PRSS, AMSS analyses)
```

Key method choices:
- **Audio:** binaural → mono (channel mean), 48 kHz, at most first 30 s, RMS-normalised to −26 dBFS (ISD in Pa vs ARAUS digital scale). Absolute level is therefore absent from CLAP and enters only via calibrated LA50. Short recordings are not padded: three exactly-10 s windows overlap to cover them; under 10 s the clip is tiled. 30 s input gives non-overlapping windows (ARAUS embeddings unaffected).
- **CLAP:** `laion/clap-htsat-unfused`, projected 512-d embedding (`pooler_output` in transformers 5.x), mean of the three window embeddings; deterministic (no random crop).
- **Ecoacoustic indices** (scikit-maad, on normalised mono): NDSI, BI, ACI, ADI, H, plus spectral centroid, flatness, onset rate (librosa).
- **Targets:** `soundscapy.surveys.calculate_iso_coords` from the 8 PAQ items, range [−1, 1].
- **Harmonised psychoacoustics (7):** LA50, LA10−LA90, LC50−LA50, N5, roughness, fluctuation strength, tonality. ARAUS columns `LA50_r, LA10_r, LA90_r, LC50_r, N05_r, Ravg_r, Favg_r, Tavg_r`; ISD columns `LAeq_L50(A), LAeq_L10(A), LAeq_L90(A), LCeq_L50(C), N_N5, R_R, FS_F, T_TonalityHMS`. Sharpness excluded (ARAUS DIN 45692 vs ISD Aures); Relative Approach missing in ARAUS; LA50 used because ARAUS reports fast-averaged LAavg, not LAeq.
- **Models:** ridge (alpha grid 1…1e6) for all feature sets; HistGradientBoosting for low-dimensional sets; tuned with the official ARAUS folds 1–5 (disjoint in soundscapes, maskers, participants).
- **Evaluation on ISD:** individual, recording-mean and location-mean levels; Pearson r with 95 % CI from a bootstrap over locations; R²; centred R² (offset removed; descriptive); mean bias. Paired bootstrap for model differences (participants in ARAUS, locations in ISD).

## 4. Results: lab training and transfer to ISD (script 03)

| Feature set | Model | ARAUS CV R² Pl | ISD rec r Pl [95 % CI] | ISD loc r Pl | ARAUS CV R² Ev | ISD rec r Ev [95 % CI] |
|---|---|---|---|---|---|---|
| psycho | ridge | 0.151 | 0.41 [0.12, 0.56] | 0.74 | 0.211 | 0.30 [0.14, 0.43] |
| psycho | hgb | 0.149 | 0.38 [0.06, 0.55] | 0.76 | 0.244 | 0.29 [0.12, 0.41] |
| handcrafted | ridge | 0.154 | 0.41 [0.12, 0.55] | 0.73 | 0.218 | 0.31 [0.14, 0.43] |
| handcrafted | hgb | 0.168 | 0.38 [0.09, 0.54] | 0.77 | 0.250 | 0.28 [0.12, 0.41] |
| clap | ridge | 0.219 | 0.37 [0.14, 0.51] | 0.71 | 0.239 | 0.26 [0.09, 0.37] |
| clap+level | ridge | 0.281 | 0.41 [0.14, 0.56] | 0.77 | 0.309 | 0.32 [0.14, 0.43] |
| clap+psycho | ridge | 0.284 | 0.43 [0.18, 0.56] | 0.82 | 0.315 | 0.33 [0.15, 0.44] |

Paired differences (95 % CI):
- **ARAUS:** clap+psycho vs psycho ΔR² = +0.132 [0.122, 0.143] (Pl), +0.104 [0.094, 0.113] (Ev). CLAP alone vs psycho +0.067 (Pl), +0.028 (Ev). Level/psycho added to CLAP: +0.06–0.08.
- **ISD:** clap+psycho vs psycho Δr = +0.013 [−0.063, +0.086] (Pl), +0.027 [−0.030, +0.083] (Ev). Only significant ISD difference: adding level/psycho to plain CLAP for eventfulness (Δr ≈ +0.07, p ≈ 0.01).
- **Ecoacoustic indices** add ΔR² +0.003 to +0.007 in ARAUS and nothing on ISD.

Other observations:
- Plain R² on ISD is negative for all models (bias −0.09 to −0.29: field rated more pleasant). Centred R² for pleasantness 0.13–0.18 ≈ r² (scale transfers); for eventfulness negative (ranking transfers, scale does not).
- ARAUS fold 0 (48 stimuli × 5 raters) gives unstable scores; not interpreted.
- Short-recording sensitivity: excluding recordings < 20 s (12) or < 30 s (58) changes recording-level r by ≤ 0.015.
- Restoration proxies (recording level, ISOPleasant predictions): r with overall quality `sss01` 0.31–0.39; with wish to revisit `sss05` 0.11–0.20. Observed ISOPleasant (same raters) correlates 0.62 and 0.28. Across 18 locations predicted pleasantness vs revisit ≈ 0.5 (observed 0.67). sss01 and sss05 correlate only 0.20.

**Interpretation:** CLAP adds clearly in the lab but nothing in the field compared with psychoacoustics. A lab-trained psychoacoustic model transfers reasonably (r = 0.41 recording level, 0.74 location level) even though ISD-internal training fails.

## 5. Results: does CLAP recognise sources in the field? (script 06)

Zero-shot source scores: cosine similarity between audio embedding and CLAP text embeddings ("the sound of …"), 5 phrases per category taken from the ISD questionnaire's examples; relative score = score minus mean of the four categories. ISD items: ssi01 traffic, ssi02 other noise, ssi03 human sounds, ssi04 natural sounds (1 = not at all … 5 = dominates completely).

- **Recognition (822 rated recordings):** relative score vs rated dominance r = 0.46 traffic, 0.30 other noise, 0.44 human, 0.52 natural (all CIs exclude 0); location level 0.69–0.87. Diagonal highest in every row of the 4×4 matrix; traffic vs other noise least separated. Raw scores are clearly worse than relative scores.
- **Lab check:** ROC AUC for detecting the ARAUS masker class 0.85–0.99 (falls as the masker gets quieter).
- **Source → pleasantness (r with ISOPleasant):** natural +0.24 ARAUS / +0.37 ISD (rated item +0.48); other noise −0.37 / −0.31 (−0.43); traffic −0.13 / −0.26 (−0.47); human +0.16 / +0.01 (−0.09). Same signs, field effects similar or larger.
- **Source-only model trained on ARAUS:** ARAUS CV R² 0.133, ISD r = 0.26 [0.02, 0.45]; coefficients (relative to human) traffic −0.12, other noise −2.75, natural −0.45 — the lab gives natural sounds the wrong weight.
- **Naturalness measures vs rated natural dominance (ssi04):** CLAP natural 0.52 [0.32, 0.63]; ADI 0.25 [−0.03, 0.41]; NDSI 0.19 [−0.02, 0.32]; H 0.11; ACI −0.18; BI −0.29 [−0.40, −0.11]. Ecoacoustic indices fail as naturalness markers in these urban recordings (computed on level-normalised mono).
- **Overlap with level:** CLAP source scores vs LA50 |r| ≤ 0.10 (by design: normalised audio). Rated ssi04 vs LA50 −0.32.
- **Beyond psychoacoustics (partial r with ISOPleasant, controlling the psycho prediction):** natural +0.34 [0.11, 0.44], other noise −0.27 [−0.39, −0.08], traffic −0.18 [−0.33, 0.01], human −0.04.
- **Phrase sets (robustness):** natural-sound recognition 0.52 (questionnaire), 0.54 (alternative wording), 0.51 (category names only).

**Interpretation:** the hypotheses "CLAP does not hear field sources" and "source information is redundant with level" were both rejected. CLAP carries field-relevant source information, but ARAUS teaches poor weights for it (natural sounds added as maskers at controlled, often high levels).

## 6. Results: hybrid model on ISD (script 07)

Small linear models, weights fitted on ISD with grouped CV; inputs not fitted to ISD. Baselines recalibrated the same way. Models: M0 psy (ARAUS psychoacoustic prediction), M0b clapsy (ARAUS CLAP+psycho prediction), M1 psy + natural + other noise, **M2 psy + natural + other noise + traffic (primary, no selection among sources)**, M3 sources only. Human sounds = reference category (relative scores sum to 0).

Primary (questionnaire phrases, leave-one-location-out, 823 recordings), ISOPleasant:

| Model | r [95 % CI] | R² | loc r | r with sss01 | r with sss05 |
|---|---|---|---|---|---|
| M0 psy | 0.353 [0.07, 0.52] | 0.123 | 0.634 | 0.289 | 0.178 |
| M0b clapsy | 0.378 [0.13, 0.52] | 0.142 | 0.738 | 0.360 | 0.173 |
| M1 | 0.485 [0.28, 0.60] | 0.235 | 0.846 | 0.407 | 0.194 |
| M2 | 0.495 [0.28, 0.62] | 0.245 | 0.871 | 0.446 | 0.190 |
| M3 sources only | 0.358 [0.17, 0.49] | 0.123 | 0.704 | 0.373 | 0.095 |

- M2 vs M0: Δr = +0.142 [0.014, 0.272], p = 0.017; ΔR² = +0.122 [0.024, 0.229]. M1: Δr +0.132, p = 0.039 (inputs chosen after step 6 → optimistic). M0b: +0.025 n.s. M3 ≈ M0.
- Standardised weights (M2, full fit): psy 0.33, natural 0.31, other noise 0.01, traffic −0.14.
- ISOEventful: no gain (M2 Δr +0.036 n.s.).

Robustness (M2 vs M0, ISOPleasant, Δr [95 % CI], p):

| Validation | questionnaire | alternative | labels |
|---|---|---|---|
| leave-one-location-out | +0.142 [0.014, 0.272], 0.017 | +0.135 [0.005, 0.266], 0.032 | +0.123 [0.013, 0.237], 0.017 |
| leave-one-city-out | +0.084 [−0.045, 0.227], 0.21 | +0.049 [−0.100, 0.201], 0.52 | +0.107 [0.017, 0.202], 0.008 |

- ISOEventful under leave-one-city-out: adding sources makes predictions worse (Δr −0.05 to −0.17). Do not use source scores for eventfulness.
- M0b (ARAUS CLAP model) never gains (Δr −0.05 to +0.03).

**Defensible claim:** zero-shot natural-sound detection improves pleasantness predictions for new locations within the studied cities, independent of phrase wording. Generalisation to new cities is not established; confirmation on independent field data is needed.

## 7. Caveats to keep in any write-up

- 18 locations, 11 in London; the idea of adding source scores came from descriptive analysis of the same ISD data.
- Bootstrap on out-of-fold predictions without refitting probably understates uncertainty slightly.
- Lab vs field differ in context (headphones + video vs being at the place).
- Mono downmix; partly different psychoacoustic implementations between datasets; indices computed on normalised audio.
- ISOPleasant is a proxy; ISD has no restorativeness measure; AMSS (with PRSS) has no audio.
- **Marcello's point (agreed):** source recognition cannot replace questionnaires. A source label says *what* is heard, not *how it is experienced* (context, expectation, person). Even the best model explains about a quarter of recording-level variance. Frame automatic estimates as the sound-attributable part of pleasantness (screening, mapping, covariates), not as a substitute for asking people.

## 8. Findings document

A findings document (Claude Docs) summarises sections 4–7 in plain academic prose with three charts (naturalness measures, hybrid models, robustness): https://claude.ai/code/artifact/e0551c44-9805-4b0e-b068-9b1eefba9b15 — it does not yet include the "not a substitute for questionnaires" framing from section 7.

## 9. Next phase: unsupervised clustering (planned, not started)

Marcello's original question: **can an unsupervised method group soundscapes so that the groups differ in perceived pleasantness?** To be done in a new `clustering/` folder in the same repo, reusing `data/processed`, `data/features` and `rsd/` read-only, leaving milestone-1 scripts untouched.

Agreed reasoning so far:
- There is no representation-free clustering: the question becomes *in which representation does the natural grouping align with ratings?*
- CLAP's space is organised by sources/scenes (caption training); our CLAP input lacks loudness (normalised), temporal structure (window mean) and spatial cues (mono). Expect clusters to split by source composition, aligning with pleasantness only moderately.
- Proposed comparison of representations, all clustered and evaluated identically:
  1. CLAP (existing embeddings) — semantic view;
  2. AST (transformers, AudioSet-supervised) or PANNs — event categories;
  3. BEATs (self-supervised, extra install) — acoustic texture without labels;
  4. psychoacoustic + ecoacoustic features — interpretable, includes level;
  5. combined (e.g. CLAP + level + temporal features).
- Procedure: cluster all 1,452 ISD recordings (incl. unrated) with k-means (k = 2–12) and HDBSCAN; only afterwards evaluate with ratings: η² of ISOPleasant/ISOEventful between clusters vs random partitions of equal sizes; cluster means with CIs; overlap of clusters with locations (AMI) and whether clusters explain pleasantness *within* locations (location ICC ≈ 0.30 is a trivial route); cluster profiles (source scores, level, locations); blinded listening set per cluster for Marcello's own listening and a later study. Supervised hybrid R² ≈ 0.25 is a rough upper benchmark. ARAUS clustering only as a secondary check (would mostly recover masker types).
- **Open decision:** all five representations, or a first round with CLAP, psychoacoustic features and BEATs.

## 10. Working setup and conventions

- **Machines:** code is written and smoke-tested by Claude in a cloud sandbox; delivered as a git patch into Marcello's Mac clone (`~/Documents/Bamberg/restorative_embeddings`, connected folder); Marcello commits and pushes from the Mac (Claude cannot push: no SSH keys) and runs on his server `orpheus` (`~/Documents/restorative_embeddings`, Python 3.12 `.venv`, RTX 5090 32 GB). Data and results live only on the server (`data/raw/{araus,isd}`, `data/processed`, `data/features`, `results/`, all git-ignored).
- Commit messages end with a `Co-Authored-By: Claude …` line; commits are made under Marcello's own git identity (an early mistake set the repo's local identity to "Claude"; removed).
- Deleting files in the connected folder needs Marcello's approval each session; leftover `*.patch` files are removed by him before committing.
- Marcello's preferences: balanced, fact-based answers, no flattery; academic text in clear, plain language (no ornate wording); Zotero is read-only for Claude.

## 11. Key references

- Versümer, Blättermann, Rosenthal & Weinzierl (2025). A comparison of methods for modeling soundscape dimensions based on different datasets. *JASA* 157(1), 234–255. doi:10.1121/10.0034849
- Ooi et al. (2023). ARAUS: a large-scale dataset and baseline models of affective responses to augmented urban soundscapes. *IEEE Trans. Affective Computing*. arXiv:2207.01078
- Mitchell et al. International Soundscape Database v1.0. Zenodo 10672568
- Payne (2013). The production of a Perceived Restorativeness Soundscape Scale. *Applied Acoustics*; Payne & Guastavino (2018). *Front. Psychol.* 9:2224
- Wu et al. (2023). Large-scale contrastive language-audio pretraining (LAION-CLAP). ICASSP
- Soundscapy (MitchellAcoustics/Soundscapy); scikit-maad

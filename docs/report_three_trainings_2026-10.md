# Fine-tuning Stable Audio Open on pleasantness-rated soundscapes: condensed report (October 2026)

Base material for a meeting presentation. Status: proof of concept. Every result below comes from descriptive audio metrics on generated files. The listening tests are still open, so no claim about audible effects is made.

## 1. Aim and setup

- Question: can the number in a caption, `<description> [ISOPleasant: x.xx]`, steer what a text-to-audio model generates?
- Model: Stable Audio Open 1.0, full fine-tune with stable-audio-tools (learning rate 5e-5, batch 1 with 8 accumulation steps, AdamW, T5-base text conditioning). One checkpoint is 17 GB.
- Three fine-tunings are reported: ARAUS, ISD and Clotho. A fourth (ISD + ARAUS together) is prepared.
- Generation is reproducible with a seed. A deterministic sampler (dpmpp-2m) is used, so that the same seed gives the same starting noise at every p level.

## 2. Datasets and training sets

| | ARAUS | ISD | Clotho v2.1 |
|---|---|---|---|
| Content | park recordings with added maskers (bird, water, wind, traffic, construction) | real soundscapes recorded in London | Freesound clips with five human captions |
| Rating | human ISOPleasant | human ISOPleasant | predicted by SoundAQnet, not human |
| Source size | 5,681 stimuli | 823 rated recordings (median about 34 s, 12–122 s) | 5,929 clips, 15–30 s (37.6 h) |
| Caption | `park soundscape with <masker> in the background [ISOPleasant: x]` | `soundscape with <sources> [ISOPleasant: x]` (no city) | human caption + `[ISOPleasant: x]` |
| Selection | 160 stimuli per pleasantness bin (800) | all rated recordings; very unpleasant repeated 4×, unpleasant 2× | soundscape-like clips (keyword and tag filter, CC0 and CC-BY), stability filter on the SoundAQnet predictions: 777 clips, 737 for training, 40 held out |
| Whole-file set | 800 files, 6.67 h | 1,029 files (867 different), 7.93 h unique, 9.43 h seen per epoch | 737 files, 4.72 h |

Bins use the ISD thresholds (very unpleasant ≤ −0.5, unpleasant ≤ −0.15, neutral ≤ 0.15, pleasant ≤ 0.5, very pleasant above).

Distribution problems that matter for the interpretation:
- ISD training values are mostly positive. Very unpleasant recordings give only 0.18 h of different audio.
- Clotho ratings span only −0.56 to +0.47 (SD 0.20). The bins hold 2, 271, 362, 102 and 0 clips (very unpleasant to very pleasant). 30% of the kept clips have a music score above 0.5 and 39% a speech score above 0.5; neither was filtered.
- In ARAUS the rating follows the masker and its level. The generated files from ARAUS models contain music-like backgrounds, and I heard the same music and out-of-context birds in ARAUS training chunks. They come from the ARAUS audio, not from the model.

## 3. A methodological problem found along the way: clip length

- The first runs of all three datasets used 10 s chunks. Stable Audio Open is trained on windows of about 47.5 s, and the ISD recordings are longer than I first assumed (median about 34 s, not 30 s).
- All three training sets were rebuilt as whole files (ISD recordings longer than 47 s are split into equal pieces). Clotho clips are 15–30 s, so 30 s clips were not possible there.
- Padding to the full window does not enter the loss. In the installed code the padding weight is 0.0, and `train/mse_signal` is the loss on real audio. (`train/mse_masked_loss` is the loss in the padded region and is not trained.) An earlier explanation of a weak effect by padding dilution was checked in the code and rejected.

## 4. Training runs (30 s whole-file data)

| Run | Files | Steps per epoch | Steps run | Epochs | Loss on real audio |
|---|---|---|---|---|---|
| ISD whole | 1,029 | about 129 | stopped at about 5.2k, checkpoints to 5,000 | about 39 at 5,000 | 0.652 to 0.632 (−3%), flat curve |
| ARAUS whole | 800 | about 100 | 6,000 | 60 | 0.61 to 0.54 (−11%), steeper after 3k |
| Clotho whole | 737 | about 92 | planned: about 4,000 | about 43 | prepared, rated run being started; a no-rating control follows |

The loss curves cannot tell whether the number is used. With random noise levels per step they are noisy and they do not show where to stop. Whether the steeper ARAUS decline is learning or memorisation is not decided. Checkpoints (every 1,000 steps) are compared by generated audio instead. A held-out loss with the correct, shuffled and missing number is proposed for the future; the held-out stimuli would need base soundscapes not in the training set.

## 5. Evaluation method

- Grids of seeds × prompts × p levels. Current grids: seeds 1–3, p = −0.7, 0.0, +0.7, 30 s, 100 steps, deterministic sampler. For ISD 7 prompts (5 with training vocabulary, 2 free text), for ARAUS 5 maskers.
- Guidance scale 7 (default), 12 and 15 were compared for ISD (step 5,000) and ARAUS (step 6,000).
- Metrics per file (descriptive only): share of A-weighted power above 1 kHz and 4 kHz (A_gt1k, A_gt4k), stereo correlation, level variation over time (cv), rms, and two tonality measures (flat_db, peak_db).
- For each prompt: effect = mean at p = +0.7 minus mean at p = −0.7, compared with the seed-to-seed SD. A new script (`generation/compare_grids.py`) puts all grids in one table. It also shows whether the response is ordered (monotone over −0.7, 0.0, +0.7) and how far the middle level departs from the straight line between the extremes.

## 6. Results so far (metrics only)

**ISD (30 s model, step 5,000)**
- The response is strong and consistent across seeds (seed SD of A_gt4k about 0.5–0.7), and it grows with guidance. Example: natural + traffic, A_gt1k effect +9.8 (guidance 7), +19.7 (12), +23.4 (15). The checkpoint of the guidance-7 table was not recorded.
- It is not ordered: p = 0.0 differs from both extremes. Stereo correlation is 0.83–0.91 at ±0.7 and 0.2–0.75 at 0.0 in every prompt. Most clear tonality effects are not monotone (flat_db 1 of 5, peak_db 0 of 5 ordered).
- Prompts without natural sounds ("other noise", "no dominant source") barely respond. Traffic + human sounds at +0.7 loses almost all energy above 1 kHz (A_gt1k 4–6 against about 30 at −0.7), which needs listening.
- Possible causes, not tested: the training values are sparse near 0 and below −0.3; or the text encoder reacts to the written string and not to the value.

**ARAUS (30 s model, step 6,000)**
- More ordered in some metrics: stereo correlation rises with p for 4 of 5 maskers (e.g. water 0.36 to 0.82); `peak_db` rises and `flat_db` falls (more tonal at higher p) in about 80% of seed groups. Birdsong brightness (A_gt4k) falls strongly (−32, seed SD 6.3).
- The direction depends on the masker (wind goes the opposite way). Traffic changes in temporal variation (cv −0.13 to −0.24) but not in spectrum.
- Higher guidance does not strengthen the effects. Seed SD is 5–6 for the brightness metrics, so effects are less clear than in ISD.

**Comparison.** ISD has the larger and more consistent effect, but it is not ordered. ARAUS is more ordered in some metrics but has a smaller effect compared with its noise. The low seed variance in ISD could mean strong conditioning or low diversity; this is not decided. Earlier grids with the 10 s models gave the same kinds of patterns (ARAUS step 1,500: tonality rising with p in 20 of 20 groups; ISD step 1,000: non-monotone with 0.0 as the odd level).

## 7. Limitations to state in the talk

- Metrics describe the signal. They are not a measure of what a listener hears, and listening is still pending.
- Three seeds and three levels per group; the seed SD itself is uncertain.
- Pleasantness is confounded with the source: in ISD natural sounds are mostly pleasant and traffic mostly unpleasant; in ARAUS the rating follows the masker and its level; the Clotho ratings are predicted, not human, and the clips probably overlap with the pretraining data of the model.
- Memorisation is not ruled out (60 epochs for ARAUS). The ARAUS-only run has not been checked for the number of different base soundscapes; the 400 ARAUS files selected for the joint set come from 195 bases.
- Silenzi in Quota (Zenodo) was looked at as an external evaluation set, but it provides questionnaires and no audio files.

## 8. Next steps

1. Listening tests: blind comparison of the three p levels (same seed and prompt) for ISD and ARAUS, at guidance 7 and 12/15.
2. Clotho: rated run on the 30 s clip set, then the no-rating control; the generation grid uses levels ±0.4, the range of the data.
3. Joint ISD + ARAUS training. Raw set built: 1,429 files, 12.77 h seen per epoch (ISD 9.43 h, ARAUS 3.33 h), about 179 steps per epoch. 400 ARAUS files, 80 per bin, balanced over the five maskers, with captions in the ISD vocabulary (my assumed mapping: bird, water and wind to natural sounds, construction to other noise). Cleaning of ARAUS bases with music is under test (CLAP score is saturated: 43 of 200 files above 0.99; a listening check of 15 files is in progress; alternative: AudioSet tagger on the 400 selected files).
4. Held-out loss with correct, shuffled and missing number as a measurement that does not depend on listening.
5. Continuity grid (levels −0.7, −0.3, −0.05, 0.05, 0.3, 0.7) if the ISD non-monotone response needs an explanation.
6. SoundAQnet check against human ratings on ISD before using its ratings in the paper (evaluation files exist; their outcome is not summarised here).

## 9. Suggested slide outline (about 12 slides)

1. Question and approach (caption format, model, three datasets).
2. Datasets: one table (section 2).
3. Problem found: 10 s chunks against 30–47 s training windows; rebuild.
4. Training runs and loss curves; why the loss cannot decide.
5. Evaluation design (grids, metrics, seed noise).
6. ISD: effect and guidance; 0.0 as the odd level (stereo correlation figure).
7. ARAUS: ordered metrics, masker dependence, guidance has no effect.
8. Side-by-side comparison (table from `compare_grids.py`).
9. Artifacts: music and out-of-context birds come from the ARAUS audio.
10. Limitations (section 7).
11. Joint ISD + ARAUS and Clotho: plan and status.
12. Next steps, including listening tests.

## 10. Points to check before presenting

- Which checkpoint the ISD guidance-7 table was (3,000 or 5,000).
- Whether the Clotho rated run has started and its step count at the time of the talk.
- The exact selection used in the first 10 s ARAUS run (script default 360 stimuli) if the first runs are shown.
- The Clotho 10 s run was trained earlier on the server; no results from it were reviewed in this work.
- Replace "metrics" by listening results as soon as the notes are in.

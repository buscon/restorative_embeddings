# Clotho variant (selection only)

Goal: test the caption + rating pipeline on clips that have proper human-written captions
and are long enough for Stable Audio Open (FSD50K clips are too short). The ratings would
come from SoundAQnet, not from people, so a result here shows that the model can follow a
predicted rating, not that the output is more pleasant for listeners.

Source: Clotho v2.1, Zenodo record 4783391. Development + validation + evaluation =
5929 clips (the test captions are withheld). Freesound clips of 15 to 30 s (mean 22.8 s),
five captions each (mean 11 words), about 37.6 h in total.

## Steps

| # | What | Script | Output |
|---|------|--------|--------|
| 1 | Download captions + metadata CSVs (no audio) | `01_download_metadata.sh` | `data/raw/clotho/` |
| 2 | Count and select soundscape-like clips | `02_select_candidates.py` | `selected_clotho.csv` (git-ignored) |

## Result of step 2 (7 Oct 2026, default settings)

Licences in Clotho: CC0 2444, BY 2425, BY-NC 835, Sampling+ 225. Default keeps CC0 and BY
(4869 clips, 30.9 h). Durations are known for 70% of the clips (`start_end_samples`); the
rest use the mean, so hours are estimates.

| Selection | Clips (CC0+BY) | Hours |
|---|---|---|
| At least 2 captions with a soundscape keyword, or tag + 1 caption (default) | 2883 | 18.3 |
| Soundscape tag and at least 2 captions with a keyword (strict) | 1588 | 10.1 |

For comparison, the 360-file ARAUS set is about 5.4 h.

## Caveats

- The keyword and tag lists are a first pass and were not tuned. In a sample of 8 selected
  clips about half were plausibly soundscapes; others were single events or speech (a radio
  dispatcher, a train horn). Read about 50 (`--show 50`) before using the selection.
- Crowd-written captions sometimes disagree with each other and with the Freesound tags
  (one fountain recording had a car caption). Options: keep only clips where several
  captions agree, or train on all five captions per clip.
- The clips come from Freesound, so they probably overlap with SAO's pretraining data. A
  control run on the same clips and captions without the rating word is needed to separate
  the effect of the word from seeing the data again.
- SoundAQnet was trained on calibrated ARAUS audio and takes a loudness input; Freesound
  clips are uncalibrated. Validate it on ISD (human ratings) before rating anything. See
  plan B in the project docs (`plan_generation_fsd50k.md`).
- Check that the caption text does not already explain the predicted rating (for example a
  ridge regression from text or CLAP embeddings to the rating). If it does, the rating word
  adds little.

## Not done yet

Audio download for the selected clips, rating, captions with the rating word, training
chunks, fine-tuning, evaluation.

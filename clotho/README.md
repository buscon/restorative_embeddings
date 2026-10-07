# Clotho variant (selection and caption check; no training yet)

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
| 3 | Download the audio archives (about 7 GB, resumable) | `03_download_audio.sh` | `data/raw/clotho/archives/` |
| 4 | Unpack only the selected clips (needs `pip install py7zr`) | `04_extract_selected_audio.py` | `data/raw/clotho/audio_selected/` |
| 5 | Score the five captions of each clip with CLAP, keep the best | `05_clap_caption_check.py` | `clotho/checked/` (git-ignored) |

```bash
bash clotho/01_download_metadata.sh
python clotho/02_select_candidates.py --show 50              # add --require-tag for the strict set
bash clotho/03_download_audio.sh                             # or one split: ... evaluation
python clotho/04_extract_selected_audio.py
# env B (torch + transformers; add soundfile scipy pandas tqdm if missing):
python clotho/05_clap_caption_check.py --min-z 1.0
```

## Step 2: selection (7 Oct 2026)

Licences in Clotho: CC0 2444, BY 2425, BY-NC 835, Sampling+ 225. The default keeps CC0 and BY
(4869 clips, 30.9 h). Durations are known for 70% of the clips (`start_end_samples`); the
rest use the mean, so hours are estimates. The keyword lists are a first pass; "station",
"train" and "bus" were dropped after a first sample let single events through.

| Selection | Clips (CC0+BY) | Hours |
|---|---|---|
| At least 2 captions with a soundscape keyword, or tag + 1 caption (default) | 2779 | 17.7 |
| `--require-tag`: soundscape tag and at least 2 captions with a keyword | 1545 | 9.8 |

For comparison, the 360-file ARAUS set is about 5.4 h.

## Step 5: caption check with CLAP

Each clip's audio (laion/clap-htsat-unfused, up to three 10 s windows, averaged, same
preparation as `rsd/audio.py`) is compared with its five captions. Each caption gets a
z-score against the captions of the other selected clips for the same audio. The script
keeps the best caption per clip as `caption`, lists all captions above `--min-z` in
`good_captions`, and drops clips whose best z is below `--min-z`. It prints how many clips
survive at several thresholds; pick `--min-z` from that table.

Test run (100 evaluation clips, CPU, small comparison pool, so the numbers will shift on the
full set):

- Mean percentile of the true captions: 0.96 (0.50 would mean CLAP carries no information),
  so CLAP separates matching from non-matching captions well.
- Best z per clip: 10th percentile 2.4, median 3.3. All 100 clips kept at `--min-z 1.0`;
  99 at 2.0; 62 at 3.0. Only 4% of individual captions had z below 1.
- So the check mostly removes single bad captions, not whole clips. Example: a stream clip
  where two of five annotators wrote "taking a shower" (z about 0.1) while the other three
  describe running water (z about 1.9 to 2.0).
- It does not judge whether a clip is a soundscape: a clean walkie-talkie clip scored high.
  That depends on step 2.
- Correction to the first caption-quality sample: the "fountain recording with car captions"
  was not a mismatch. All five captions agree (cars, motorcycles, rain; z 1.8 to 3.1) and
  its tags include motorbike, so the audio really contains traffic. The tags are not a
  reliable check of the captions either.

## Caveats

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

Run steps 3 to 5 on the full selection, rating with SoundAQnet, captions with the rating
word, training chunks, fine-tuning, evaluation.

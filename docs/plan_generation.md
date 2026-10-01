# Plan: fine-tuning Stable Audio Open on soundscapes with pleasantness captions

**Repository:** `github.com/buscon/restorative_embeddings` · **Author:** Marcello Lussana (University of Bamberg) · **Written:** 2026-10-01 · **Status:** planned, no code yet

**Purpose of this file:** a self-contained plan for the generation phase, so that another session can write the code without re-deriving the decisions. Read `docs/project_report_milestone1.md` first for the data, the milestone-1 results and the working conventions.

---

## 1. Aim

Fine-tune Stable Audio Open (SAO) on ARAUS and ISD recordings with text captions that describe the scene, the sound sources, the loudness and the **perceived pleasantness**. Pleasantness is given as one of five fixed words. The question for this phase is:

> Does a fine-tuned SAO, prompted with the same scene at different pleasantness levels, produce audio that is judged more or less pleasant in the intended direction, and is this effect more than "add birds, remove traffic"?

A continuous pleasantness slider (a numeric conditioner) is **not** part of this phase. It becomes phase 2 only if the text-based effect works (section 8). The exact ISOPleasant value is stored with every training item so that phase 2 needs no new data preparation.

## 2. Decisions taken so far and why

**Human ratings, not generated ratings.** We use ARAUS and ISD, which have human PAQ ratings, and generate only the content part of the captions. The alternative was AudioSet or FSD50K with pleasantness predicted by SoundAQnet (Hou et al., 2024, SoundSCaper). We rejected it for three reasons:

- SoundAQnet is trained on ARAUS. Its predictions can't carry more affect information than ARAUS itself.
- Its accuracy is modest (MSE ≈ 0.9–1.1 per PAQ item on the 1–5 scale), and its vocabulary is narrow: 3 scenes and 15 event classes, the events being PANNs pseudo-labels.
- Milestone 1 showed that ARAUS-trained pleasantness predictions reach only r ≈ 0.4 on field recordings (ISD). AudioSet and FSD50K are much further from ARAUS than ISD is (speech, music, single events).

**Captions from templates, not from a captioning model.** DRCap (zero-shot captioning from CLAP latents) or similar models are not needed for the first round. The datasets already contain better content information:

- ARAUS: masker type and SMR are known.
- ISD: people rated source dominance (ssi01–04).
- CLAP zero-shot source scores fill the gaps. They were validated in milestone 1 (script 06): r ≈ 0.45–0.52 against human source ratings.

A captioning model can be added later if template captions turn out too repetitive.

**Five fixed pleasantness words** in a fixed position in every caption. This fits how SAO is conditioned (text plus timing), and a simple interface can map a slider to the five words without changing the model.

## 3. Repository layout and code reuse

New work goes into a `generation/` folder. Milestone-1 scripts (`scripts/01`–`07`) stay unchanged. `data/processed`, `data/features` and `rsd/` are read-only inputs; new shared helpers go into new modules.

```
generation/
  01_captions.py        captions + metadata table
  02_export.py          splits, audio export or latent pre-encoding
  train/                SAO model + dataset configs, custom metadata module
  03_generate.py        prompts x pleasantness levels x seeds
  04_evaluate.py        automatic evaluation
  05_listening_set.py   blinded listening set
rsd/sources.py          CLAP text embeddings + source scores (copied from script 06)
tests/smoke_generation.py   end-to-end on synthetic data, as tests/smoke_test.py
```

**Reused as is:**

- `rsd/data.py`: `load_araus()`, `load_isd()`, `index_isd_wavs()`, `add_iso()`, `PAQ_LABELS`. These give ratings, ISO coordinates (soundscapy), stimulus and recording tables, and handle the ISD file-name quirks (`__MACOSX`, `NP125.hdf.wav`, duplicate `NP102.1.wav`).
- `rsd/audio.py`: `ArausMixer` rebuilds ARAUS stimuli in memory (soundscape + masker at SMR), exactly as the authors' code, so the ~132 GB of mixes are never written. `read()` too.
- `data/features/{araus,isd}_clap.npz`: existing CLAP embeddings (`laion/clap-htsat-unfused`, 512-d, mean of three 10 s windows). Source scores for all items are one matrix product.
- `data/processed/araus_stimuli.csv`, `isd_recordings.csv`: stimulus and recording tables.
- `results/model_*.joblib`: the ARAUS-trained CLAP ridge model, used as a screening judge in evaluation (see the caveat in section 7).

**Copied into `rsd/sources.py`** (do not import from the script): from `scripts/06_source_recognition.py`, the phrase sets `CATEGORIES` and `ALT_PHRASES` and the functions `text_embeddings()` and `scores()`. Use relative scores (score minus mean of the four categories), as in milestone 1.

**Not reused:** `rsd/audio.prepare()`. It produces mono, 48 kHz, RMS-normalised audio for CLAP. SAO needs stereo at 44.1 kHz, so the binaural recordings stay two-channel.

## 4. Step 1: captions (`generation/01_captions.py`)

One caption per ARAUS stimulus (folds 0–5, without practice and attention stimuli, as in `load_araus`) and per rated ISD recording (823).

**Template** (fixed order; wording of the parts may vary slightly, the pleasantness phrase must not):

> `{scene}, {sources}, {loudness}. {pleasantness} soundscape.`

Example: *"urban park, birdsong in the foreground, distant traffic, a few voices, quiet. Pleasant soundscape."*

**Pleasantness** from ISOPleasant in [−1, 1], fixed thresholds (not per-dataset quantiles: ISD is rated about 0.3 higher than ARAUS, and that difference is real):

| ISOPleasant | word |
|---|---|
| ≤ −0.50 | very unpleasant |
| −0.50 … −0.15 | unpleasant |
| −0.15 … +0.15 | neutral |
| +0.15 … +0.50 | pleasant |
| > +0.50 | very pleasant |

Thresholds are a starting point. Report the counts per bin and dataset. If an extreme bin is very thin, widen it before training, not after seeing results.

**Label noise.** ARAUS has about 1.15 ratings per unique stimulus, and around 70 % of pleasantness variance is between people. Write two label variants:

- (a) the stimulus's own mean rating (primary);
- (b) the mean over all stimuli with the same soundscape + masker across SMRs (more ratings, less noise, but SMR effects averaged out).

Train on (a). Keep (b) for a second run if the effect is weak.

**Sources:**

- ISD: human ratings ssi01 traffic, ssi02 other noise, ssi03 human, ssi04 natural (1–5, recording mean). ≥ 4 → "dominant/in the foreground"; 3 → "present/in the background"; < 3 → omitted.
- ARAUS: the masker type is known (bird, water, wind, traffic, construction, silence). Describe it by SMR: masker louder than soundscape → "in the foreground", otherwise "in the background". The base soundscape's sources come from CLAP relative source scores, thresholded. Choose the thresholds on ISD, where human ratings exist, and apply them unchanged to ARAUS.
- Optional finer phrases (e.g. "church bells", "fountain", "music"): score with CLAP and keep only high-scoring ones. Treat as less certain.

**Loudness** from calibrated LA50 (ARAUS `LA50_r`, ISD `LAeq_L50(A)`, harmonised as `LA50` in `rsd/data.py`): "quiet / moderate / loud", thresholds to be set from the pooled distribution (e.g. tertiles). This is needed because training audio is loudness-normalised (step 2), and level is one of the strongest pleasantness predictors.

**Scene:**

- ISD: assign a type to each of the 18 locations once by hand (park, square, street, waterside, …) and store the mapping in a small CSV in the repo.
- ARAUS: check whether `soundscapes.csv` has a scene or location column for the USotW base soundscapes. If not, either annotate the 240 base soundscapes by hand (the SoundAQnet authors did this with 3 classes) or use a generic "urban soundscape".

**Output:** `data/generation/captions.csv` with `id, dataset, caption, caption_b, ISOPleasant, ISOPleasant_b, bin, bin_b, n_ratings, LA50, source scores, split`. Read 50 random captions before continuing.

## 5. Step 2: splits and audio (`generation/02_export.py`)

**Splits:**

- ARAUS: hold out by **base soundscape** (e.g. 20 % of the 240), not by stimulus. Otherwise held-out scenes leak into training through other masker versions. Stratify by masker type if possible.
- ISD: open decision (section 9). Either some locations for training and 3–4 held out for testing, or all of ISD kept as the field test set.

**Audio format:**

- Stereo (binaural kept as two channels), 44.1 kHz, 30 s.
- Loudness-normalised to one fixed target (e.g. −23 LUFS integrated, `pyloudnorm`), because ISD is calibrated in Pa and ARAUS is digital. Level information lives in the caption.
- About 70 ISD recordings are shorter than 30 s. Do not pad with silence. Use the true duration in the timing metadata (`seconds_total`).

**Storage:** all ARAUS stimuli as stereo WAV would be roughly 100 GB. Two options:

1. **Subsample** about 5–8k ARAUS stimuli, balanced over pleasantness bins and masker types. With only 240 base soundscapes, more stimuli add little acoustic variety.
2. **Pre-encode** to SAO VAE latents while mixing on the fly with `ArausMixer`. Latents for 30 s are a few hundred KB. `stable-audio-tools` has a pre-encoding path; check how the current version handles it and how pre-encoded datasets are configured.

Start with option 1, which is simpler. Output to `data/generation/` (git-ignored like `data/`).

## 6. Step 3: fine-tuning (`generation/train/`)

**Baseline first.** Generate with unmodified SAO using the same prompts, including the five pleasantness words. Base SAO may already respond to "pleasant" vs "unpleasant" from its general training. Every result is compared with this baseline.

**Model choice** (to check before starting):

- Stable Audio Open 1.0: about 1.1B parameters, up to ~47 s. Full fine-tuning on one RTX 5090 (32 GB) is tight; gradient checkpointing or LoRA would be needed.
- Stable Audio Open Small: about 341M parameters, up to ~11 s. Cheaper and faster to iterate.
- Suggestion: pilot on Small with 10 s crops, move to 1.0 if the effect is there.
- Check the memory needs and the Stability AI Community License for both.

**Configuration:** `stable-audio-tools` with the standard conditioning (text prompt plus `seconds_start` and `seconds_total`). A custom metadata module returns the caption from `captions.csv`. Keep `ISOPleasant` in the metadata dictionary even though it's unused now. Fix seeds and log configs with every run.

**Server:** `orpheus`, Python 3.12 `.venv`, RTX 5090. SAO and `stable-audio-tools` may pin different `torch` / `transformers` versions from milestone 1. Use a separate virtual environment for training.

## 7. Step 4: generation and evaluation

**Generation (`03_generate.py`):** about 20 held-out scene prompts (from held-out ARAUS soundscapes and ISD test locations) × 5 pleasantness levels × several seeds (e.g. 4), for the fine-tuned model and base SAO. Only the pleasantness word changes between levels of the same prompt.

**Automatic evaluation (`04_evaluate.py`):**

1. **Main test:** within each prompt, does predicted pleasantness rise with the level word? Spearman correlation per prompt, then the mean across prompts with a bootstrap CI over prompts. Compare fine-tuned vs base SAO.
2. **Source-shortcut check:** CLAP relative source scores of the generated audio across levels. If "pleasant" only means more natural and less traffic, it shows here. This is the most likely failure mode; report it in either case. Also report the within-prompt effect after controlling for source scores (partial correlation, as in milestone-1 section 5).
3. **Prompt adherence:** CLAP text–audio similarity between the caption without the pleasantness word and the generated audio.
4. **Realism:** FAD (Fréchet Audio Distance) against held-out real recordings, e.g. with CLAP or VGGish embeddings.

**Caveat on the automatic judge.** The ARAUS-trained CLAP ridge model mostly reads sources, so it rewards exactly the shortcut that check 2 looks for. It can't use psychoacoustic or level features either, because generated audio has no calibration. Use it for screening only.

**Listening test (`05_listening_set.py`):** a blinded set of generated clips across levels and prompts, rated on the 8 PAQ items. A small internal pilot first. Reuse the pattern of `scripts/05_listening_sample.py` (blinded clips, rating sheet, separate key).

## 8. Decision point after phase 1

- **Clear within-prompt pleasantness effect beyond base SAO, and not only through sources:** phase 2, a numeric pleasantness conditioner (slider), using the stored ISOPleasant values.
- **Effect only through sources:** a reportable finding about what "pleasant" means in these data. The slider idea needs rethinking.
- **No effect:** probable causes are label noise and limited ARAUS variety. Next tries are label variant (b), coarser bins (3 words), or more field data.

## 9. Open decisions

1. ISD: partly training data, or test set only?
2. Pilot on SAO Small, or go directly to SAO 1.0?
3. Order relative to the planned clustering phase (milestone-1 report, section 9): before, after or in parallel?
4. ARAUS scene labels: hand annotation of the 240 base soundscapes, or generic "urban soundscape"?
5. Licences for generative training: ARAUS and its sources (USotW soundscapes, masker recordings) need checking. ISD is CC BY 4.0. SAO weights are under the Stability AI Community License.

## 10. Caveats to keep in any write-up

- Pleasantness labels are mostly single ratings. The model learns from noisy targets, and the bins are coarse on purpose.
- ARAUS consists of artificial mixes of 240 base soundscapes with added maskers. SAO may learn the overlay pattern rather than real soundscapes.
- Pleasantness and sources are strongly linked in both datasets (milestone 1, section 5), so within-scene effects are the real test.
- ISOPleasant is a proxy for restorativeness, as in milestone 1.
- Automatic pleasantness judges are trained on ARAUS and are partly circular. The listening test decides.

## 11. Working conventions (unchanged from milestone 1)

- Code is written and smoke-tested in a cloud sandbox and delivered as a git patch into the Mac clone (`~/Documents/Bamberg/restorative_embeddings`). Marcello commits under his own identity and pushes, then runs on `orpheus`.
- Commit messages end with a `Co-Authored-By: Claude …` line.
- Data and results stay on the server and are git-ignored.
- Every script gets a synthetic-data path in a smoke test, so it runs without downloads.

## 12. References

- Evans et al. (2024). Stable Audio Open. arXiv:2407.14358
- Ooi et al. (2023). ARAUS. *IEEE Trans. Affective Computing*. arXiv:2207.01078
- Mitchell et al. International Soundscape Database v1.0. Zenodo 10672568
- Hou et al. (2024). Soundscape captioning using Sound Affective Quality Network and large language model (SoundSCaper / SoundAQnet). arXiv:2406.05914
- Li et al. (2024). DRCap: decoding CLAP latents with retrieval-augmented generation for zero-shot audio captioning. arXiv:2410.09472
- Wu et al. (2023). LAION-CLAP. ICASSP
- `stable-audio-tools`: github.com/Stability-AI/stable-audio-tools

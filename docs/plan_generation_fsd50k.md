# Plan B: fine-tuning Stable Audio Open on FSD50K / AudioSet with SoundAQnet pleasantness labels

**Repository:** `github.com/buscon/restorative_embeddings` · **Author:** Marcello Lussana (University of Bamberg) · **Written:** 2026-10-01 · **Status:** planned, no code yet

**Purpose of this file:** a self-contained plan for an alternative route to the generation phase. Plan A (`docs/plan_generation.md`) uses ARAUS and ISD with human pleasantness ratings. Plan B uses large general sound datasets with human *content* labels and *predicted* pleasantness. Read `docs/project_report_milestone1.md` for the data, the milestone-1 results and the working conventions. Sections of plan A that apply unchanged are referenced rather than repeated.

---

## 1. Aim

Same question as plan A:

> Does a fine-tuned Stable Audio Open (SAO), prompted with the same scene at different pleasantness levels, produce audio that people judge more or less pleasant in the intended direction, and is this more than "add birds, remove traffic"?

The difference is where the labels come from. Pleasantness is predicted by SoundAQnet (Hou et al., 2024), not rated by people. Sources come from the datasets' human labels.

## 2. Why consider this route, and what speaks against it

**For:**

- **Acoustic variety.** FSD50K has 51,197 real recordings across 200 sound classes. ARAUS has 240 base soundscapes with added maskers. SAO would see many more kinds of places and sources.
- **Human content labels.** FSD50K labels are human-verified. AudioSet labels are human-assigned but weak (clip-level, often incomplete).
- **No artificial mixes.** The model learns from real recordings rather than overlays at set SMRs.
- **Pretrained tools exist.** The SoundSCaper repository (github.com/Yuanbo2020/SoundSCaper) provides pretrained SoundAQnet models and inference code. They output 8 PAQ values, ISOPleasant and ISOEventful, 3 scene classes and audio-event probabilities.

**Against (to be tested, not assumed):**

- **SoundAQnet is trained on ARAUS.** Its pleasantness labels can't hold more information about how people feel than ARAUS does. They're ARAUS's mapping from sound to pleasantness, applied to new audio.
- **Its accuracy is modest.** The reported MSE is about 0.9–1.1 per PAQ item on the 1–5 scale. In milestone 1, ARAUS-trained models reached r ≈ 0.4 on field recordings (ISD), and FSD50K/AudioSet are further from ARAUS than ISD is.
- **Calibrated level is needed and missing.** SoundAQnet takes ISO 532-1 loudness as an input. This needs calibrated audio, and FSD50K and AudioSet clips are not calibrated. Their digital level is arbitrary.
- **Clip length.** SoundAQnet was trained on 30 s clips. FSD50K clips are 0.3–30 s (mean ~7 s in development, ~10 s in evaluation); AudioSet clips are 10 s.
- **The source shortcut is built in.** If pleasantness labels come from a model that mainly reads sources, SAO learns "pleasant = these sources". The within-scene effect that plan A tests for is then hard to get by construction.
- **Probable overlap with SAO's pretraining.** SAO was trained largely on Freesound recordings, which is where FSD50K comes from. Fine-tuning would mostly add new captions, not new sounds. Check the overlap in the SAO paper's data description.

Because of these points, plan B starts with a **validation gate** (step 0). If SoundAQnet fails it, plan B either switches to a different rater or stops.

## 3. Repository layout and code reuse

New work in `generation_b/` (or a `--labels soundaqnet` option inside `generation/` if plan A exists by then). Milestone-1 scripts stay unchanged.

```
generation_b/
  00_validate_rater.py   SoundAQnet (and alternatives) vs human ratings on ISD and ARAUS
  01_select.py           FSD50K / AudioSet filtering to soundscape-like clips
  02_rate.py             run the chosen rater on the selected clips
  03_captions.py         captions + metadata table
  04_export.py           splits, audio export or latent pre-encoding
  train/                 SAO configs (shared with plan A where possible)
  05_generate.py, 06_evaluate.py, 07_listening_set.py   as in plan A
third_party/SoundSCaper/ pinned checkout, not modified
```

**Reused from the repo:**

- `rsd/data.py` (`load_isd`, `load_araus`, `add_iso`) and the processed ISD/ARAUS tables, for the validation gate and as human-rated test sets.
- `rsd/audio.py` (`read`, `ArausMixer`) for feeding ISD/ARAUS audio to SoundAQnet.
- `rsd/embed.py` (`ClapEmbedder`, `windows`) to compute CLAP embeddings of FSD50K clips.
- The CLAP source scoring from `scripts/06_source_recognition.py`, copied into `rsd/sources.py` (as in plan A).
- `results/model_*.joblib`: the milestone-1 ARAUS CLAP ridge model, as an alternative rater (step 0) and as a screening judge.

## 4. Step 0: validation gate (`00_validate_rater.py`)

Before rating any FSD50K clip, test how well the rater agrees with people on audio it wasn't trained on.

1. **Run SoundAQnet on the 823 rated ISD recordings.** Use its own feature extraction (log-mel, ISO 532-1 loudness). Check the expected sample rate and clip length in the repository. ISD is calibrated in Pa, so the loudness input is meaningful here.
2. **Compare with human ratings** exactly as in milestone 1: Pearson r with ISOPleasant at recording and location level, with bootstrap CIs over locations, and bias.
   Reference points from milestone 1 (recording level): psychoacoustic ridge r = 0.41, CLAP ridge r = 0.37, CLAP + psycho r = 0.43, hybrid M2 r = 0.50 (fitted on ISD, not a fair comparison).
3. **Level sensitivity.** Rescale ISD audio by −20, −10, +10 dB and rerun. If predictions move strongly with digital gain, SoundAQnet can't be used on uncalibrated clips as is. Options are then a fixed assumed level for all clips (state it) or a rater without level input.
4. **Length sensitivity.** Cut ISD recordings to 5 s and 10 s and rerun. This shows how much is lost on FSD50K-length clips.
5. **ARAUS fold 0.** Check which ARAUS folds SoundAQnet was trained on. Score fold 0 only if it was held out; otherwise skip.
6. **Alternative rater.** Score the same ISD audio, rescaled and cut the same way, with the milestone-1 CLAP ridge model. It has no level input and so isn't sensitive to gain, but it reads mostly sources.

**Decision rule** (fix before running):

- Use SoundAQnet if, at full length and calibrated level, its recording-level r on ISD is at least that of the CLAP ridge model (0.37), *and* its predictions are stable under the 10 s cut and the gain changes.
- Otherwise use the CLAP ridge model as rater and say so.
- If neither reaches r ≈ 0.3 at 10 s length, stop plan B. Labels would be mostly noise.

## 5. Step 1: selecting clips (`01_select.py`)

**FSD50K (primary):**

- **Licences:** keep CC0 and CC-BY (about 43k of 51k clips). Exclude CC-BY-NC and CC Sampling+ unless you decide otherwise. Keep each clip's licence and uploader in the metadata for attribution.
- **Length:** at least 5 s (pilot on SAO Small, which generates up to ~11 s) or at least 10 s. Report how many clips survive. Probably a few thousand, which needs checking.
- **Content:** soundscape-like clips. Use the AudioSet ontology to keep environmental classes and drop the rest:
  - keep: natural sounds (wind, water, rain, thunder), animals in the wild and birds, vehicles and traffic, human crowds and chatter, footsteps, bells, construction and tools, domestic and urban ambiences;
  - drop: music, musical instruments, speech-dominant clips, isolated studio foley (single knocks, clicks).
  Write the class list into a CSV in the repo so the selection is transparent.
- **Splits:** use FSD50K's own development / evaluation split for train / test. Check whether the split separates uploaders; if not, group by uploader to avoid near-duplicates across splits.

**AudioSet (optional, second step):**

- 10 s YouTube clips, about 2M; labels weaker than FSD50K.
- Audio isn't distributed with the dataset and must be downloaded from YouTube. Many videos are no longer available, and YouTube's terms and copyright make training a generative model on it legally unclear. Get advice from the university before using it.
- If used: same ontology filter, prefer the strongly labelled subset, and keep it as a separate data source in all analyses.

**ISD and ARAUS stay as test sets** with human ratings. They're not used for training in plan B. This keeps one evaluation with real pleasantness ratings that the rater never produced.

## 6. Step 2: rating (`02_rate.py`)

- Run the rater chosen in step 0 on all selected clips. Store ISOPleasant, ISOEventful and the 8 PAQ values (SoundAQnet), or ISOPleasant only (CLAP ridge).
- **Level:** if SoundAQnet is used, apply the level choice made in step 0 (e.g. normalise all clips to one assumed playback level) and record it.
- **Check the distribution of predicted ISOPleasant.** Regression models usually compress predictions towards the mean. If most clips fall in "neutral", the five fixed thresholds of plan A produce empty extreme bins. In that case use quantile bins (quintiles) on the predicted values, and say that the words then mean "relative pleasantness within this dataset".
- Spot-check by listening: 20 clips from each extreme bin.

## 7. Step 3: captions (`03_captions.py`)

Same template idea as plan A, minus the parts that need calibration:

> `{scene}, {sources}. {pleasantness} soundscape.`

- **Sources:** FSD50K's human labels, mapped from ontology class names to plain phrases ("Bird vocalization, bird call, bird song" → "birdsong"). Keep the mapping in a CSV. For multi-label clips, list up to three labels, leaf classes first.
- **Scene:** SoundAQnet's scene output has only three classes (public square, park, street traffic). Many FSD50K clips (forest, beach, kitchen, train station) fit none of them. Either omit the scene or derive a coarse one from the source labels (e.g. water + wind + no traffic → "outdoors, natural setting"). Do not use SoundAQnet's scene label unless its confidence is high.
- **Loudness:** omitted. FSD50K and AudioSet levels are uncalibrated. This is a real loss compared with plan A, since level is one of the strongest pleasantness predictors.
- **Pleasantness:** the five fixed words of plan A, or quintile bins (step 2).
- **Metadata:** keep the exact predicted ISOPleasant, the rater name and version, licence, uploader and FSD50K labels.

Read 50 random captions before continuing.

## 8. Steps 4–5: export and fine-tuning

As plan A, sections 5–6, with these differences:

- Clips are short. Pilot on **SAO Small** with clips of at least 5 s. `seconds_total` is the true clip length. Do not loop or pad.
- Loudness-normalise all clips to one target (e.g. −23 LUFS) for training, as in plan A.
- FSD50K clips are mono or stereo at various sample rates. Convert to 44.1 kHz stereo (mono duplicated to both channels).
- Base SAO with the same prompts is the baseline. Given the probable overlap with SAO's training data, also compare against a fine-tune on the *same clips* with captions *without* the pleasantness word. This separates the effect of the pleasantness word from the effect of simply seeing the data again.

## 9. Step 6: evaluation

As plan A, section 7 (within-prompt Spearman across the five levels, source-shortcut check, prompt adherence, FAD, listening test), with these differences:

- **Circularity.** An automatic judge trained on ARAUS (SoundAQnet or the CLAP ridge model) shares its biases with the labels. If the fine-tuned model learned the rater's mapping, the rater will confirm it whether or not people agree. Automatic scores in plan B show only that SAO learned the labels.
- **Human evaluation is therefore the main result.** A blinded listening test with PAQ ratings on generated clips across the five levels. Without it, plan B can't answer the research question.
- **Real-data check.** On held-out ISD recordings, compare the rater's predictions with human ratings (step 0 numbers). This bounds how good the labels could be.

## 10. Comparing plans A and B

The two plans can be run on the same evaluation:

- same held-out prompts, same five pleasantness levels, same seeds;
- same automatic checks and one shared listening test, with clips from base SAO, plan A and plan B mixed and blinded.

This gives a direct answer to "human affect labels on limited audio vs. predicted labels on varied audio". A combined design is also possible later: FSD50K for content variety (captions without pleasantness) plus ARAUS/ISD for the pleasantness words.

## 11. Open decisions

1. Rater: SoundAQnet or the milestone-1 CLAP ridge model. Decided by step 0, with the rule fixed beforehand.
2. Datasets: FSD50K only, or also AudioSet (licensing and download issues)?
3. Licences: exclude CC-BY-NC and CC Sampling+ clips, or keep them for non-commercial research?
4. Minimum clip length (5 s vs 10 s) and with it SAO Small vs SAO 1.0.
5. Fixed or quantile pleasantness bins (depends on the distribution in step 2).
6. Run plan B alone, or alongside plan A with a shared evaluation (section 10)?

## 12. Caveats to keep in any write-up

- The pleasantness words in the training data are model predictions, not human judgements. They inherit the rater's training data (ARAUS: lab, headphones, urban scenes).
- Uncalibrated audio: level, a main driver of pleasantness, is missing from both the rater input (unless fixed by assumption) and the captions.
- The source shortcut is likely by construction: the rater mostly reads sources, so the generated "pleasant" may mean "more natural sounds".
- Probable overlap between FSD50K and SAO's pretraining data.
- Only a human listening test can show that the generated pleasantness levels are perceived as intended.

## 13. References

- Hou et al. (2024). Soundscape captioning using Sound Affective Quality Network and large language model (SoundSCaper / SoundAQnet). arXiv:2406.05914. Code: github.com/Yuanbo2020/SoundSCaper
- Fonseca et al. (2022). FSD50K: an open dataset of human-labeled sound events. *IEEE/ACM TASLP* 30. arXiv:2010.00475. Data: Zenodo 4060432
- Gemmeke et al. (2017). Audio Set: an ontology and human-labeled dataset for audio events. ICASSP
- Evans et al. (2024). Stable Audio Open. arXiv:2407.14358
- Ooi et al. (2023). ARAUS. arXiv:2207.01078
- Mitchell et al. International Soundscape Database v1.0. Zenodo 10672568

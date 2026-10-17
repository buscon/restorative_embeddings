# Pleasantness conditioning in Stable Audio Open: what was done after the three trainings (October 2026)

Follow-up to `report_three_trainings_2026-10.md`. Status: proof of concept. Every result below comes from descriptive audio metrics on generated files, except where a listening remark is stated. The listening tests are only partly done. The main new finding (section 7) affects some results of the first report.

## 1. Summary

- A sampler problem was found: all earlier grids used the deterministic sampler `dpmpp-2m`, and files generated with it contain a continuous, chirp-like background artifact. The base model shows it too. With `dpmpp-3m-sde` (the Gradio default) the base model is clean. The metric comparisons made with `dpmpp-2m` have to be repeated before they are used (section 7).
- The tag `[ISOPleasant: x.xx]` is read by a frozen T5 encoder. Its tokenisation is irregular, and "0.00" is a single special token. This is a plausible reason for the "p = 0.0 is the odd level" pattern in ISD, but it is not proven (section 4).
- In the ISD training files, pleasantness is only weakly related to simple spectral measures, and almost not at all within files that share the same description text (section 5).
- Three ways of giving the rating to the model were compared on the same ISD set: the number in the caption, five words in the caption, and a separate number input (section 6). All three showed effects in the spectral metrics at 5,000 steps, but none was stable across checkpoints, and in the listening test of the words run no clear audible difference between the levels was noticed. At equal step the formats differ mostly in how ordered the response is, less in its size.
- Clotho and the joint ISD + ARAUS set were completed (sections 2 and 3).
- LoRA is available in the main branch of stable-audio-tools. A setup is prepared, but no LoRA run has been done (section 8).

## 2. Clotho (rated, whole files)

- Run: 737 files, 5,000 steps (epoch 53 at 5,000), checkpoints every 1,000 steps. Grids were made at 2,000, 4,000 and 5,000, with levels ±0.4 (the range of the data), and a base-model control with the tag (`base_with_tag`).
- The base model with the tag is not a null reference. For the change from −0.4 to +0.4 it gave A_gt1k +46 (waves) and +24 (city) and −1 (cicadas), with a seed SD of 6.7. The same applies to ISD and ARAUS, whose grids have no such control.
- Fine-tuned model at step 5,000: a consistent negative A_gt1k effect (−12 to −24) and clear tonality effects in 3 of 3 prompts. The effects are weaker at 2,000 and 4,000. Stereo correlation is constant (the clips are dual-mono).
- Not decided: whether the direction agrees with the training data (the script `data_vs_rating.py` exists but was not run on Clotho), whether the outputs are low-diversity recall of training clips, and why the effect changes between checkpoints.

## 3. Joint ISD + ARAUS set

- ARAUS contains music-like backgrounds that come from the ARAUS audio. A screening was run with CLAP on all 5,681 stimuli (239 base soundscapes). The score is saturated: the median over the stimuli of a base is 0, 0.386, 1 and 1 at the 10, 50, 90 and 99% quantiles.
- Listening checks (user): in the high-score group most files had music (two did not; they contained voices or other sounds that can be heard as music), in the mid group one file, in the low group none.
- Number of bases kept by threshold: 0.1: 69; 0.2: 77; 0.3: 88; 0.4: 122; 0.5: 126; 0.9: 176 (of 239). The set built at 0.3 had 86 bases (54 recordings). The final set uses 0.4: 400 ARAUS files from 115 bases (69 recordings), 80 per bin; the very pleasant bin is split over the maskers 20/12/12/18/18.
- Final training set: 1,429 files (ISD 1,029 including repeats, ARAUS 400), 12.77 h per epoch. An earlier build into the same folder gave 707 ARAUS files because the builder does not clear its output folder (the stale build is kept under `joint/train/_stale_mixed`).
- The joint run was trained (checkpoints to 6,000, epoch 33 at 6,000). It has not been evaluated yet. The small number of different ARAUS recordings behind the 400 files has to be stated as a limitation.

## 4. Does the model read the tag? Code, tokenisation, embeddings, precedents

- In stable-audio-tools the T5 text encoder is frozen (no gradient, fp16, maximum length 128). The prompt tokens go to the diffusion transformer by cross-attention. Only the transformer can adapt to the tag.
- T5-base tokenisation of the number: "-0.70" gives ▁ - 0.7 0, "-0.07" gives ▁ - 0.0 7, "0.00" gives one token (▁ 0.00), and "0.01", "0.07", "0.70" give ▁0. plus a two-digit piece. The number is therefore split in different ways for different values.
- Embedding probe (CPU, T5-base, `generation/probe_t5_tag.py`): the distance between tag embeddings correlates with the difference in rating (Spearman 0.63). The embedding of "0.00" is about 3.5 times further from ±0.01 than typical neighbours, and d(−0.5, 0) = 1.40 is larger than d(−0.5, +0.5) = 1.34, so the middle level does not lie between the extremes. Other formats gave Spearman 0.67 (percent, 0–100) and 0.82 (scale 0–10, one token per value, largest distance between the extremes). ARAUS used the same 0.0 level, so this explanation is not specific to ISD.
- Precedents (Stable Audio Open 1.0 fine-tunes on Hugging Face, those reviewed): fixed templates with BPM, key and bars taken from small closed sets, for example BPM in {100, 110, 120, 128, 130, 140, 150}. Chords appear only as "chord progression", with no chord symbols. One model card reports tempo accuracy 0.876 (base 0.747) and key accuracy 0.26 (base 0.21); another used 420 files per (type, mode, bars) cell. Most cards report no quantitative control metric. My inference, not a finding: a closed vocabulary with many repeats is easier to learn than about 800 different continuous strings.
- A `NumberConditioner` (Fourier embedding of a float) exists in the library. It is used for `seconds_start` and `seconds_total`, and it enters the model as one extra cross-attention token, so a new one needs no change of weight shapes.

## 5. The rating in the ISD training files

Script `generation/data_vs_rating.py` on 867 different ISD files (repeats removed): rating minimum −0.85, median +0.34, maximum +1.00, SD 0.37; 17 different description texts, 13 of them with at least 8 files.

| Metric | Spearman rho (all files) | Expected change −0.7 to +0.7 | Median within-description rho |
|---|---|---|---|
| A_gt1k | +0.10 | +8.9 | +0.04 |
| A_gt4k | +0.20 | +3.5 | +0.05 |
| stereo_corr | 0.00 | −0.03 | +0.05 |
| cv | −0.23 | −0.09 | −0.14 |
| rms_db | −0.01 | −0.20 | +0.09 |
| flat_db | +0.16 | +3.2 dB | +0.05 |
| peak_db | −0.15 | −4.2 dB | −0.04 |

Reading: most of the relation between rating and these metrics comes from the description (traffic against natural sounds), which the prompt already contains. Within the same description, the rating adds almost nothing to these measures. The metrics are crude, and pleasantness may relate to other features, so this does not show that the rating cannot be learned.

## 6. Three ways to give the rating, same ISD whole set (1,029 files)

| Run | Rating given as | Steps / epoch at last checkpoint |
|---|---|---|
| Numbers (first report) | `[ISOPleasant: 0.62]` in the caption | 5,000 / 38 |
| Words | `[ISOPleasant: very pleasant]` etc. (five ISD bins; 80, 211, 162, 294, 282 files; the 80 very unpleasant files are about 20 different recordings) | 6,000 / 46 |
| Score input | no tag in the caption, separate number conditioner `pleasantness` (range −1 to 1, one extra cross-attention token, random start); full fine-tune, because in LoRA mode the new embedder would not be trained | 7,000 / 54 (two earlier starts were stopped after a few hundred steps) |

Evaluation of this section: 7 prompts, seeds 1–3, levels −0.7, 0.0, +0.7, 30 s, **sampler `dpmpp-2m`** (see section 7). Base-model control for the words run: base with the word tags. The effect is the mean at +0.7 minus the mean at −0.7; "clear" means at least twice the seed SD; "ordered" means the middle level lies between the extremes.

Number of prompts (of 7) with a clear effect, and in brackets how many of them are also ordered:

| Grid | flat_db | peak_db | A_gt1k | A_gt4k | seed SD flat_db |
|---|---|---|---|---|---|
| base with words | 2 (1) | 2 (1) | 1 (1) | 1 (1) | 3.43 |
| numbers 5,000 | 5 (1) | 6 (0) | 6 (4) | 4 (2) | 0.64 |
| words 5,000 | 5 (5) | 3 (3) | 2 (2) | 5 (1) | 1.26 |
| words 6,000 | 5 (3) | 5 (3) | 3 (1) | 5 (2) | 1.37 |
| score 3,000 | 4 (4) | 4 (2) | 2 (1) | 4 (4) | 0.97 |
| score 5,000 | 7 (6) | 6 (5) | 5 (4) | 5 (5) | 0.75 |
| score 7,000 | 4 (3) | 5 (4) | 5 (4) | 3 (1) | 0.89 |

Effect on flat_db in dB (very pleasant minus very unpleasant):

| Prompt | numbers 5,000 | words 5,000 | words 6,000 | score 3,000 | score 5,000 | score 7,000 |
|---|---|---|---|---|---|---|
| birdsong and wind | +5.1 | +4.8 | +0.6 | +4.3 | +8.5 | +0.6 |
| natural + human | +4.3 | +4.5 | +0.5 | +4.4 | +9.7 | +1.5 |
| natural + traffic | +5.7 | +6.1 | +3.8 | +5.7 | +9.8 | +1.1 |
| no dominant source | +0.2 | +3.6 | −5.6 | +1.8 | −5.0 | −3.3 |
| other noise | −0.8 | +1.8 | −5.5 | +0.1 | −3.1 | −5.3 |
| traffic + human | −6.3 | +0.4 | −5.2 | +0.9 | −1.6 | −4.2 |
| waves on a beach | +8.2 | +3.5 | +2.9 | +7.4 | +5.6 | +2.8 |

Observations:
- Base model with words: the seed SD is large and most prompts show no clear effect, but "other noise" and "no dominant source" react strongly (for example flat_db −58 and −38, peak_db +76 and +46 at the very pleasant word). The base model reads the words, and for these prompts it moves to near-silent or tonal output.
- At 5,000 steps all three formats show effects in the same direction for the nature-containing prompts (flat_db rises with pleasantness by about 4 to 10 dB). This is larger than the +3.2 dB the training data show over all files, and the data show almost no relation within a description. I read this as a content cue: the label pushes the output towards a more nature-like, noisier sound. This is an interpretation.
- The score input at 5,000 steps is the most ordered and the most consistent (flat_db clear in 7 of 7 prompts, 6 ordered). Its size on the nature prompts is about twice that of the other formats. The guidance scale of 7 amplifies differences, and the small seed SD of the fine-tuned models inflates the ratios; the absolute effects are more comparable than the ratios.
- Avoiding the "0.00" token seems to be what restores the order. The words run was also ordered at 5,000 steps, and the numbers run mostly was not (flat_db 1 of 5 clear prompts ordered, peak_db 0 of 6).
- The effects are not stable over training. Score at 7,000 and words at 6,000 show nearly the same pattern: the nature effects fall to about 0 to 4 dB, and the three generic prompts move negative. A possible explanation, not tested: late in training, with 46 to 54 passes over 1,029 files, the labels select memorised clips, and for generic prompts the "very pleasant" clips are more tonal.
- Listening (words run, step 5,000): background music was heard often and no clear difference between the levels was noticed. The score grids have not been judged by listening, because the artifacts (section 7) were noticed first.

## 7. Sampler artifact and its consequences

- Observation: files from all grids of this period, and from the base model, contain repeating artifacts, described as a continuous, chirp-like background.
- Checks: the base model with the plain prompt (no tag) still showed it, so the tag syntax is not the cause. The Gradio interface with its default settings (sigma minimum 0.01, sigma maximum 100, which are the library defaults our scripts also used) gave no artifacts with `dpmpp-3m-sde`, and with `dpmpp-2m` it produced them. Generation with `dpmpp-3m-sde` in our own script (base model) was clean.
- The sigma range is therefore not the cause. The sampler type is. `dpmpp-2m` had been chosen for determinism and had been written into the grid scripts.
- Consequences:
  - All grids of the first report (`..._range_det`) and of this report (sections 2 and 6) were made with `dpmpp-2m`. The artifact may have affected the spectral metrics, in particular A_gt4k, flat_db and peak_db. The conclusions are provisional until the grids are regenerated with `dpmpp-3m-sde`.
  - `dpmpp-3m-sde` is stochastic. With a fixed seed the noise draws are the same at every level, so comparisons within a seed remain fair, but the seed-to-seed spread may be larger than before, and the "clear" counts will not be directly comparable.
- Status: short grids with `dpmpp-3m-sde` (3 prompts, seed 1, three levels) for the score, words and numbers models at 5,000 steps are prepared for a listening check. They were not yet evaluated at the time of writing. Full grids for all runs follow if the files are clean.

## 8. LoRA

- LoRA is in the main branch of stable-audio-tools (documentation `docs/lora.md`). It is not in the PyPI release 0.0.19. An earlier statement in this work that the library has no LoRA was wrong for the main branch.
- Code facts: a `lora_config` block in the `training` section enables it; the base weights are frozen; the optimiser settings come from `optimizer_configs`; EMA is switched off; checkpoints contain only the adapters; generation needs the base model plus the adapter. The code also attaches adapters to the conditioner layers, T5 included; T5 gets no gradient, so these stay at their start value.
- A LoRA run cannot be combined with the score input as it stands: the new embedder would stay at its random start values.
- Prepared: `generation/make_lora_config.py` (rank 16, lr 1e-4, adapters only in the diffusion transformer; these values are untuned guesses) and the option `--lora-ckpt` in the generation script. No LoRA run has been done and the code has not been run on a model.
- Reason to try it later: less drift from the base model with about 1,000 files. Risk: less capacity to learn a new conditioning signal.

## 9. Scripts added or changed (repository)

- `generation/probe_t5_tag.py`: tokenisation and embedding probe. Run on CPU.
- `generation/data_vs_rating.py`: metrics of the training files against the rating. Run on ISD.
- `generation/heldout_loss.py`: paired loss with the correct, flipped, mirrored, shuffled and missing label. Only syntax-checked, never run on a model; it reads numeric tags only, so it needs an adaptation for words.
- `isd/08_make_words_set.py`, `isd/09_words_grid.sh`: words dataset and grids (default levels −0.7, 0, +0.7).
- `isd/10_make_score_set.py`, `generation/train/sa3_finetuning_data/score_metadata_adapter.py`, `generation/make_score_config.py`: score-input dataset, adapter and config.
- `generation/make_lora_config.py`, `generation/strip_ckpt.py` (weights-only copy of a checkpoint; tested only on a synthetic file).
- `generation/generate_prompt_range.py`: new options `--labels`, `--score-cond`, `--lora-ckpt`, `--sigma-min`, `--sigma-max`.
- `generation/compare_grids.py`: option `--levels`; levels not shared by all prompts are ignored (it failed before when a folder held a stray level).
- `generation/train/03_train.sh`: the config can be set with `MODEL_CONFIG`; the header names the correct environment (`~/Documents/stable-audio-tools/.venv`).

## 10. Housekeeping

- Disk on the server: 253 GB free of 2.3 TB (89% used). One checkpoint is 17.5 GB; the training state (optimiser and EMA) makes up about half of it. A list of 14 checkpoints (all 500- and 1,000-step ones plus a few unused intermediate steps, about 242 GB) was proposed for deletion. Whether it was executed is not recorded here. Checkpoints to keep for the report: ISD whole 5,000, ARAUS whole 6,000, Clotho 5,000, words 5,000 and 6,000, score 5,000, 6,000 and 7,000 (7,000 is the one to resume from), and the latest of the joint run.

## 11. Limitations

- All grid metrics rest on three seeds, three levels and one checkpoint per run, and on the sampler that produced artifacts (section 7).
- The spectral measures are crude proxies of pleasantness. The listening evidence is limited to one listener and the words run.
- Pleasantness is confounded with the source in ISD and ARAUS (traffic against natural sounds, masker and level). The label acts at least partly as a content cue.
- Memorisation is not ruled out: 38 to 60 epochs on 800 to 1,029 files. The very unpleasant bin of ISD holds about 20 different recordings.
- There is no base-model control for ISD and ARAUS with the tag. The base model reads the tag text (Clotho, words).
- The comparison between formats uses one training run per format.

## 12. Next steps

1. Listen to the short `dpmpp-3m-sde` grids (score, words and numbers at 5,000). If they are clean, regenerate the full grids for all runs (including the joint run and Clotho) with identical settings, and redo the comparison.
2. Listening test (blind, same prompt and seed, three levels) on the regenerated files, starting with the score input at 5,000 steps.
3. Music check on the generated clips (CLAP score per level) to see whether the music heard in the words run depends on the label.
4. A better measuring tool than spectral statistics: a regressor from CLAP embeddings to the ISD rating, cross-validated by recording, evaluated within the same description; it would also show whether the rating can be predicted from the audio at all. Not built.
5. Paired loss test (heldout_loss.py), adapted to words and to the score input; needs the GPU.
6. LoRA run on the number tags (ISD whole), as a separate comparison, after the sampler question is settled.
7. Clotho: run `data_vs_rating.py` and compare the direction of the effect with the data.

## 13. Corrections to earlier statements in this work

- LoRA: "stable-audio-tools has no LoRA" was true only for the PyPI release.
- Noise range: the idea that the model-card sigma values (0.3 and 500) were the cause of the artifacts was wrong; Gradio uses 0.01 and 100.
- Score range of the ISD set: about −0.74 to +0.74 was assumed; the real range is −0.85 to +1.00 (median +0.25 with repeats).
- Two helper counts given during the work (the number of `score_cond` matches in the generation script) were miscounted; the script itself was correct.

## 14. Points to check before presenting

- Replace the section 6 tables by the regenerated `dpmpp-3m-sde` results, and state the sampler on the slide.
- Mark the first report's ISD, ARAUS and Clotho metric results as obtained with `dpmpp-2m` until they are regenerated.
- Whether the checkpoint deletion list was carried out, and which checkpoints remain.
- Joint run: no evaluation yet.
- The score run's effect at 5,000 steps is the strongest result so far, and it did not hold at 7,000. Show both.

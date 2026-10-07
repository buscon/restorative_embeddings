# Session handoff — Modelling soundscape restorativeness

**Exported from:** Cowork session `claude-4c` (soundscape restorativeness feasibility work)
**Destination:** session `cse_01Hhz5ApkDjrLZF9oLAXtJ8y`
**Date:** 2026-09-29
**Purpose:** self-contained record of everything established so far, so the receiving session can continue without re-deriving anything.

---

## 1. The project

Marcello is running a small test on **predicting the mental restorativeness of soundscapes**, guided by a reference paper that models the ISO 12913 circumplex dimensions. Two questions opened the work: which labels to use, and which dataset.

**Decisions taken:**

- **Label:** PRSS composite (single averaged score), not subscales — see §4 for the empirical justification.
- **Data:** existing datasets only. No listening test is possible right now.
- **Design:** a two-stage transfer test — train acoustics → circumplex at scale, then test how far that transfers to measured PRSS.

---

## 2. Reference paper — and an important calibration

**Versümer, S., Blättermann, P., Rosenthal, F., & Weinzierl, S. (2025).** *A comparison of methods for modeling soundscape dimensions based on different datasets.* J. Acoust. Soc. Am. 157(1), 234–255. https://doi.org/10.1121/10.0034849

It is a **methods-comparison paper, not a successful-prediction paper**. This matters — the project was initially framed around it having "reliably predicted" pleasantness and eventfulness. It did not.

- **Datasets:** ISD, HSDD, ARAUS v1. **Targets:** ISOPleasant / ISOEventful (ISO 12913-3 trigonometric projection of the 8 PAQ items, rescaled to [0,4]).
- **Predictors (10 only):** L<sub>Aeq</sub>, L<sub>A10</sub>−L<sub>A90</sub>, Relative Approach, L<sub>Ceq</sub>−L<sub>Aeq</sub>, Sharpness (DIN 45631/A1), Tonality (ECMA-418-2), Roughness (ECMA-418-2), plus Age, Gender, WHO-5.
- **Methods:** LR, RF, XGBoost, MARS, SVR, each in fixed- and mixed-effects form (10 model types × 3 datasets × 2 targets = 60 models). Nested CV, custom objective function penalising both over- and underfitting.

**Honest out-of-sample R² (group-preserving splitting, sgkf-gkf):**

| Dataset | ISOPleasant | ISOEventful |
|---|---|---|
| ARAUS | 0.168 – 0.191 | 0.221 – 0.278 |
| HSDD | **0.00 – 0.03** | 0.169 – 0.237 |
| ISD | **negative** (not reportable) | negative |

The attractive numbers in the paper (0.35–0.47) come only from **deliberately leaky** splitting conditions included to demonstrate inflation. Other findings: nonlinear beat linear in 93% of cases (ΔR² up to 0.13); tree-based best in 70%; ICC 0.12–0.35; and the standing result that acoustics explain only ~⅓ of perceptual variance (Guski 1999).

**Implication:** design around a low ceiling. Restorativeness is more person-dependent than pleasantness, so expect worse.

---

## 3. The PRSS and the dataset landscape

**Scale:** Payne (2013), *The production of a Perceived Restorativeness Soundscape Scale*, Applied Acoustics. 19 items → 14 retained, across five ART components (Fascination 5, Being-Away-From 2, Being-Away-To 2, Compatibility 2, Coherence 2, Extent/Scope 2). Seven-point agreement scale, *not at all* (0) → *completely* (6). Factor analyses did **not** recover five clean subscales: lab study gave 2 factors (10-item general + 4-item Being-Away-To/Coherence), in-situ gave **one 9-item general factor, α = .88**.

**Known validity issues** (Payne & Guastavino, 2018, Front. Psychol.): the Compatibility "fits" item has poor face validity; Extent items underperform with short exposure; Fascination items need explicit *desirability* framing; participants drift to visual impressions without repeated sound anchoring.

**Datasets carrying PRSS/restorativeness labels — exhaustive search result:**

| Dataset | Measure | Size | Acoustics | Open |
|---|---|---|---|---|
| Xixi Wetland (PLOS One 2021) | PRSS 16-item, 4 dims, 5-pt | **903 respondents** | **none** | yes (S1 Data.xlsx) |
| NTU AMSS in-situ (2024) | modified PRSS 18-item, 5 dims + I-PANAS-SF | 272 designed / **136 usable** evaluations, 68 participants | L<sub>Aeq</sub>, L<sub>Ceq</sub>, N95 + environmental | yes (CC BY) |
| Fuzhou campus (Front. Psychol. 2025) | PRS-11 | 114 students | perceptual categories only | not stated |
| Payne (2013) original | PRSS 14-item | small | lab stimuli | no |

A review of 21 soundscape datasets (ISD, ARAUS, SATP, HSDD, Emo-Soundscapes, IADS…) found **none** carrying PRS/PRSS labels. **No large dataset pairs PRSS ratings with audio.** That is the governing constraint on this project.

---

## 4. Analysis 1 — AMSS ceiling test (COMPLETE)

**Source:** `github.com/ntudsp/amss-insitu-replication` — the repo **ships the data**, in `data/fullData.RData`. No download barrier.

Contents: 205 rows = 68 participants × 3 locations (the third is a meeting-point baseline with no ratings) → **136 real soundscape evaluations**. Each carries all five PRSS subscales, ISOPL, ISOEV, the 8 raw PAQ items, overall quality, loudness, source dominance; plus per participant, item-level PANAS, WHO-5, PSS-10 and the 21-item Weinstein noise sensitivity scale. A second table gives 88 site-sessions with L<sub>Aeq</sub>, L<sub>Ceq</sub>, N95.

### The composite is empirically justified

- Inter-subscale correlations **0.66 – 0.85**
- **Cronbach's α = 0.924** across the five subscales
- PRSS composite: mean −0.260, SD 0.382, **total variance 0.1463**

### Individual level (n = 136, 68 participants, random intercept per participant)

| Model | marginal R² | conditional R² | AIC |
|---|---|---|---|
| M0 null | — | — | 123.3 |
| M1 circumplex (ISOPL + ISOEV) | 0.3965 | 0.5854 | 54.6 |
| M2 person-level only (WHO-5, PSS, WNSS, PANAS±) | 0.3732 | 0.5292 | 73.5 |
| M3 both | **0.5712** | 0.6860 | 23.7 |

- **ICC ≈ 0.30** — a third of variance between people.
- In M1, **ISOPL carries everything**: β = 0.609, p < 0.001. **ISOEV is irrelevant**: β = 0.066, p = 0.519, raw r = −0.029.
- Raw correlations with ISOPL: composite 0.641; Compatibility 0.697, Being-Away 0.668, Extent-Coherence 0.626, Fascination 0.462, Extent-Scope 0.382.
- Marginal R² of `subscale ~ ISOPL + ISOEV`: Com 0.467, BA 0.441, EC 0.373, Fas 0.223, **ES 0.168** — Extent-Scope and Fascination track the circumplex worst, consistent with Payne & Guastavino's critique.

### Session level (n = 88)

| Model | R² | adj R² | p |
|---|---|---|---|
| acoustics → PRSS composite | **0.021** | −0.014 | 0.626 |
| acoustics → ISOPL | 0.099 | 0.067 | **0.031** |
| acoustics → OSQ | 0.063 | 0.030 | 0.138 |
| ISOPL → PRSS composite | **0.559** | 0.554 | 5.6e-17 |
| ISOPL + acoustics → PRSS | 0.570 | 0.549 | — |

With site controls: PRSS 0.048; with site + condition 0.133 (adj 0.080); ISOPL + site 0.135 (adj 0.094).

### Interpretation

The transfer chain is measurable end to end: **acoustics → pleasantness (~0.10–0.19) × pleasantness → restorativeness (~0.40–0.56) ⇒ expected acoustics → PRSS ceiling of R² ≈ 0.07–0.10.** Treat this as the pre-registered expectation for Stage 2.

**Caveat:** only two sites, L<sub>Aeq</sub> IQR just 58.2–64.0 dB (SD 4.95). That range restriction attenuates every acoustic correlation. The correct conclusion is "this dataset cannot demonstrate an acoustic effect", not "there is none".

**The headline finding:** person-level psychometrics explain about as much as the measured circumplex, and the two are complementary (0.57 combined). This reframes the project from *"predict restorativeness from sound"* to **"how much of soundscape restorativeness is the listener rather than the sound?"**

---

## 5. Analysis 2 — ISD benchmark (COMPLETE)

**Source:** International Soundscape Database v1.0.1-alpha.1, Zenodo record 10672568, CC BY 4.0. `ISD v1.0 Data.csv` is 2.1 MB and downloads without obstruction.

**Structure:** 3,589 rows × 142 columns; 26 locations, 66 sessions, 2,706 GroupIDs across London, Granada, Venice, Groningen, Shenyang, Shenzhen. **Usable sample: 1,759 rows** complete on the 7 acoustic predictors + WHO-5 + age/gender + PAQ (24 locations, 46 sessions). 1,911 complete on PAQ + acoustics alone.

**Strengths:**
- **All seven Versümer predictors present natively** (RA_cp, S_S, R_R, T_TonalityHMS, L<sub>Aeq</sub> percentiles, L<sub>Ceq</sub>) plus fluctuation strength, impulsiveness, speech interference level, THD, and full percentile sets. ARAUS required the authors to compute Relative Approach with ArtemiS on request.
- Real field data, 6 cities. L<sub>Aeq</sub> 44.7–88.7 dB, IQR 8.6 dB (vs AMSS's 5.8 dB).
- Binaural 30 s recordings available (16 GB total), CC BY.
- Item-level WHO-5, plus age, gender, education, occupation.

**Fatal weaknesses for this project:**
- **One rating per participant.** Clustering is by location/session, not person. The person-vs-sound question cannot be asked here at all.
- **WHO-5 is the only psychometric** — no PANAS, PSS or noise sensitivity. Telling: WHO-5 correlates only **r = 0.158** with ISOPleasant, versus marginal R² = 0.373 for the full battery in AMSS.

### Benchmark: 5-fold GroupKFold grouped by LOCATION

| Target | Features | LR | RF |
|---|---|---|---|
| ISOPleasant | acoustics (7) | 0.0124 | −0.0150 |
| ISOPleasant | person only | −0.0366 | −0.0895 |
| ISOPleasant | acoustics + person | 0.0461 | 0.0200 |
| ISOEventful | acoustics (7) | −0.0593 | −0.0702 |
| ISOEventful | acoustics + person | −0.0576 | −0.0528 |

Session-level aggregation (46 units) does not rescue it: RF 0.068 (Pl) / 0.151 (Ev); LR collapses to −0.95 overfitting 7 predictors on 46 units.

**This reproduces Versümer et al.'s reported negative R² for ISD.**

### Why

- ICC by location: ISOPleasant **0.300**, ISOEventful 0.152 → ~70% of variance is *within* location, i.e. between people in the same place hearing the same thing, with one rating each and no way to average it out.
- Acoustics *do* vary within location (1,082 distinct acoustic vectors; within-location L<sub>Aeq</sub> SD ≈ 4.30 dB; L<sub>Aeq</sub> ICC by location 0.626), so the problem is not fixed profiles.
- In-sample session-level correlations are real: L<sub>Aeq</sub> r = −0.474 and Roughness r = −0.447 with ISOPleasant; L<sub>Aeq</sub> 0.488 and Roughness 0.531 with ISOEventful. The signal exists — 24 locations is simply too few independent units to generalise to a held-out one.

### Restoration-adjacent items already in ISD (n = 3,547)

| Item | Wording | r with ISOPleasant |
|---|---|---|
| sss01 | overall soundscape quality | 0.567 |
| sss02 | appropriateness to the place | 0.338 |
| **sss05** | **how often would you like to visit this place again** | **0.242** |
| sss04 | visit frequency | −0.029 |
| WHO_Sum | WHO-5 wellbeing | 0.158 |

`sss05` is a usable revisit-preference proxy on a large sample.

---

## 6. Current verdict

**ISD is not a substitute for ARAUS. It is a validation set.**

| Role | Dataset | Why |
|---|---|---|
| Stage 1 — train acoustics → circumplex | **ARAUS** | measured: ISD cannot carry it (R² ≈ 0). ARAUS reaches 0.17–0.28 honestly, because it has orders of magnitude more acoustically independent stimuli plus ~42 repeated measures per participant to average out person noise |
| Person vs. sound question | **ARAUS** | the only dataset with the same battery as AMSS (PANAS, WHO-5, PSS-10, WNSS-10) *and* repeated measures per person |
| Stage 2 — transfer validation | **AMSS** | the only open data pairing real PRSS with acoustics |
| External field validation | **ISD** | tests lab→field generalisation of an ARAUS-trained model — arguably the most publishable use |
| Cheap secondary analysis | **ISD** (sss05) and **Xixi** (n=903 PRSS psychometrics) | revisit-preference proxy at scale; composite structure check |

**Bonus contribution available with no new data:** Versümer et al. used only age, gender and WHO-5 as person predictors. ARAUS also carries PANAS, PSS-10 and WNSS-10. Adding those to their exact pipeline is self-contained and fully reproducible, and given ICCs of 0.12–0.35 it is a plausible place to find real gains.

---

## 7. Blocker — needs the user

**The ARAUS response CSVs cannot be downloaded from the cloud session.** They are hosted only at `researchdata.ntu.edu.sg` (DR-NTU, `doi:10.21979/N9/9OTEVX`), which the session's fetch tooling is refused access to (robots-disallowed). The *audio* is on Zenodo and reachable; the response data is not.

**Resolution:** Marcello downloads `data.zip` from the DR-NTU record, or runs `python download.py manifest.csv` from a local clone of `github.com/ntudsp/araus-dataset-baseline-models`, and attaches the CSVs. It is a few hundred MB at most if the audio is skipped.

---

## 8. Next steps (agreed direction, not yet started)

1. Build and test the full Versümer-style pipeline against ISD so it runs unchanged the moment ARAUS CSVs arrive (nested CV, group-preserving splits, custom objective function, FE + ME variants, R² reported alongside target variance).
2. On ARAUS: replicate ISOPleasant/ISOEventful, then extend the person-predictor set to PANAS + PSS-10 + WNSS-10.
3. Apply the ARAUS model to the 136 AMSS evaluations; test PRSS composite variance explained by predicted circumplex coordinates (mixed effects, participant random intercept). Pre-registered expectation R² ≈ 0.07–0.10.
4. Optionally validate externally on ISD, and run the sss05 proxy analysis.
5. Open offer: write the three analyses up as a findings document.

---

## 9. Reproduction

```bash
git clone --depth 1 https://github.com/ntudsp/amss-insitu-replication.git      # data included
git clone --depth 1 https://github.com/ntudsp/araus-dataset-baseline-models.git  # code only
curl -sSL -o "ISD_v1.0_Data.csv" \
  "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Data.csv?download=1"
curl -sSL -o "ISD_metadata.xlsx" \
  "https://zenodo.org/records/10672568/files/ISD%20v1.0%20Metadata.xlsx?download=1"
pip install pyreadr statsmodels scikit-learn openpyxl pandas
```

ISO 12913-3 projections used (5-point items → [−1, 1]):

```
ISOPleasant = [(pleasant − annoying) + cos45°·(calm − chaotic) + cos45°·(vibrant − monotonous)] / (4 + √32)
ISOEventful = [(eventful − uneventful) + cos45°·(chaotic − calm) + cos45°·(vibrant − monotonous)] / (4 + √32)
```

**Artefacts produced this session** (delivered to Marcello in the originating conversation):

| File | Contents |
|---|---|
| `amss_individual.csv` | 136 individual PRSS evaluations + circumplex + person psychometrics |
| `amss_session.csv` | 88 site-sessions with acoustics and aggregated attributes |
| `analysis_amss.py` | reproduces every AMSS number in §4 |
| `isd_modelling_sample.csv` | 1,759 complete ISD rows with ISO projections and the 7 predictors |
| `analysis_isd.py` | reproduces every ISD number in §5 |

---

## 10. Key references

- Versümer et al. (2025), JASA 157(1), 234–255. https://doi.org/10.1121/10.0034849
- Versümer et al. (2023), *Day-to-day loudness assessments of indoor soundscapes*, JASA 153(5), 2956 — HSDD dataset, https://doi.org/10.5281/zenodo.7858848 (audio excluded for privacy)
- Payne (2013), *The production of a Perceived Restorativeness Soundscape Scale*, Applied Acoustics
- Payne & Guastavino (2018), Front. Psychol. 9:2224
- Ooi et al. (2023), *ARAUS*, IEEE Trans. Affective Computing — arXiv:2207.01078
- Lam et al. (2024), *Automating urban soundscape enhancements with AI*, Building & Environment — arXiv:2407.05744; data `doi:10.21979/N9/NEH5TR`
- Mitchell et al., International Soundscape Database v1.0, Zenodo 10672568
- Mitchell et al. (2020), *The SSID Protocol*, Applied Sciences 10(7), 2397
- Zhang et al. (2021), *Perceived restorativeness of natural soundscapes under COVID-19*, PLOS One 16(9)

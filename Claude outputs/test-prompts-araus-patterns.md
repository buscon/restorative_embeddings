# Test Prompts for SA3 LoRA at Step 3500
## Using ARAUS Caption Language Patterns

**Checkpoint:** Step 3500 (mid-training)  
**Training Progress:** ~0.583 epochs of 6000 samples  
**Purpose:** Validate that the fine-tuned model learned to generate acoustically different versions based on pleasantness rating and loudness metadata using exact caption language from training data.

---

## Prompt Structure (ARAUS Format)

Each prompt follows the exact structure learned during training:

```
{Scene description}. Loudness: {loudness_descriptor} (LA50: {db_value} dB). Pleasantness: {pleasantness_descriptor} (ISOPleasant: {iso_score}). [pleasantness: {tag}]
```

---

## Test Set 1: Park Scene - Pleasantness Variation
**Base scene:** "quiet park with natural ambience and birdsong"

### Prompt 1a: Unpleasant
```
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.8 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.25). [pleasantness: unpleasant]
```
**Expected:** Park sounds become harsher, birdsong may seem intrusive, wind noise more prominent

### Prompt 1b: Neutral
```
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]
```
**Expected:** Balanced park ambience, natural birdsong, moderate sound texture

### Prompt 1c: Pleasant
```
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.4 dB). Pleasantness: pleasant (ISOPleasant: 0.25). [pleasantness: pleasant]
```
**Expected:** Softer birdsong, more serene ambience, gentle breeze

### Prompt 1d: Very Pleasant
```
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.1 dB). Pleasantness: very pleasant (ISOPleasant: 0.63). [pleasantness: very_pleasant]
```
**Expected:** Highly refined park sounds, gentle birdsong prominent, minimal harsh frequencies

---

## Test Set 2: Urban Environment - Loudness Variation
**Base scene:** "urban street with traffic and pedestrian sounds"

### Prompt 2a: Loud + Unpleasant
```
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 67.2 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.35). [pleasantness: unpleasant]
```
**Expected:** Harsh traffic noise, honking, chaotic pedestrian ambient, high-frequency stress sounds

### Prompt 2b: Loud + Neutral
```
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 66.5 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]
```
**Expected:** Balanced urban soundscape, typical city traffic rhythm, pedestrian murmur

### Prompt 2c: Loud + Pleasant
```
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 65.8 dB). Pleasantness: pleasant (ISOPleasant: 0.32). [pleasantness: pleasant]
```
**Expected:** Softened traffic, clearer pedestrian conversations, more musical rhythm in traffic

### Prompt 2d: Loud + Very Pleasant
```
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 65.2 dB). Pleasantness: very pleasant (ISOPleasant: 0.68). [pleasantness: very_pleasant]
```
**Expected:** Traffic flows with musicality, pedestrian sounds become social ambient, minimized harshness

---

## Test Set 3: Water Ambience - Acoustic Transformation
**Base scene:** "flowing water with ripples and splashing"

### Prompt 3a: Quiet + Unpleasant
```
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.1 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.42). [pleasantness: unpleasant]
```
**Expected:** Water sounds become harsh, splashing irregular and abrupt, dripping metallic

### Prompt 3b: Quiet + Neutral
```
flowing water with ripples and splashing. Loudness: quiet (LA50: 49.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]
```
**Expected:** Natural water flow, consistent ripples, balanced splash patterns

### Prompt 3c: Quiet + Pleasant
```
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.5 dB). Pleasantness: pleasant (ISOPleasant: 0.28). [pleasantness: pleasant]
```
**Expected:** Gentle water flow, soft ripples, calming splash rhythm

### Prompt 3d: Quiet + Very Pleasant
```
flowing water with ripples and splashing. Loudness: quiet (LA50: 47.9 dB). Pleasantness: very pleasant (ISOPleasant: 0.72). [pleasantness: very_pleasant]
```
**Expected:** Serene water ambience, minimal splashing, gentle flowing meditation sounds

---

## Test Set 4: Nature Ambience - Complete Variation
**Base scene:** "natural forest ambience with wind rustling through leaves"

### Prompt 4a: Unpleasant
```
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 54.2 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.30). [pleasantness: unpleasant]
```
**Expected:** Wind becomes harsh and threatening, leaf rustle chaotic, possible creaking branches

### Prompt 4b: Neutral
```
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 53.5 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]
```
**Expected:** Natural wind patterns, consistent leaf rustle, typical forest ambience

### Prompt 4c: Pleasant
```
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 52.8 dB). Pleasantness: pleasant (ISOPleasant: 0.35). [pleasantness: pleasant]
```
**Expected:** Gentle wind, soothing leaf rustle, peaceful forest setting

### Prompt 4d: Very Pleasant
```
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 52.1 dB). Pleasantness: very pleasant (ISOPleasant: 0.65). [pleasantness: very_pleasant]
```
**Expected:** Sublime wind sounds, delicate leaf rustle, deeply calming forest meditation

---

## Evaluation Framework

For each prompt pair (e.g., 1a vs 1d with same scene), listen for:

### Acoustic Differences (Pleasantness Axis)
- **Frequency content:** Do pleasant versions emphasize mid/high pleasant frequencies? Do unpleasant versions have sharper high-frequency components?
- **Temporal texture:** Does very pleasant have slower, more predictable patterns? Does unpleasant have more chaotic timing?
- **Harmonic content:** Are pleasant versions more harmonic? Do unpleasant versions have more noise/dissonance?
- **Loudness dynamics:** Does ISOPleasant correlate with reduced dynamic range in pleasant versions?

### Loudness Fidelity
- Verify that LA50 dB values correspond to recognizable loudness levels
- Compare Prompt 2a (67.2 dB) vs 4a (54.2 dB) for loudness difference perception

### Scene Consistency
- Verify scene description content (park/urban/water/forest) is preserved
- The model should understand "park" ≠ "traffic" across all pleasantness levels

---

## Testing Procedure

### Step 1: Generate Audio
Use Gradio UI with checkpoint at `lora_checkpoints/` (step 3500):
```bash
cd ~/stable-audio-3
uv run python -m stable_audio_3.interface.gradio --model medium-base --lora ./lora_checkpoints/araus_lora_rank16_dora --device cuda
```

### Step 2: Systematic Comparison
- Generate 1a, 1b, 1c, 1d (same scene, different pleasantness)
- Listen for acoustic transformations across pleasantness scale
- Document which transformations are evident vs. missing

### Step 3: A/B Comparison
- Generate base model output (no LoRA) for same prompts
- Compare: base model vs. fine-tuned at step 3500
- Assess whether fine-tuned model shows ARAUS training patterns

### Step 4: Loudness Validation
- Generate 2a-2d (same scene, fixed pleasantness, varying loudness)
- Verify loudness differences are perceptually aligned with LA50 values

---

## Interpretation Guide

### Strong Signal (Model Learned Well)
- Prompt 1a (unpleasant park) noticeably harsher than 1d (very pleasant park)
- Same scene consistently recognized across all pleasantness levels
- Loudness values (LA50) correlate with perceived volume

### Weak Signal (Model Still Training)
- Pleasantness variations subtle or absent
- Scene confusion across prompts
- Loudness inconsistent with LA50 values

### At Step 3500
Expect moderate signal—model has seen ~0.583 epochs. Pleasantness variations should be emerging but may not be as pronounced as final checkpoint (step 5000). This is a good checkpoint to evaluate mid-training learning progress.

---

## Notes

- All prompts use exact language from ARAUS training captions
- Pleasantness scale: -1 (very unpleasant) to +1 (very pleasant), with 0 = neutral
- ISOPleasant scores mirror the training data distribution
- Audio duration: Use consistent duration (e.g., 10 seconds) across all tests for fair comparison
- Sample rate: 44.1 kHz to match training data

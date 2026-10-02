# Focused Test Strategy: Pleasantness vs. Scene Effects

Two complementary test approaches to isolate what your fine-tuned model learned at step 3500.

---

## Test A: Same Scene, Different Pleasantness

**Goal:** Isolate pleasantness effect. Verify the model generates acoustically different audio for the same scene at different pleasantness levels.

**Method:** Keep scene description constant, vary only the pleasantness metadata.

### Test A1: Park Scene Across Pleasantness Spectrum

**Base scene:** `quiet park with natural ambience and birdsong`

```
A1-a (Unpleasant):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.8 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.25). [pleasantness: unpleasant]

A1-b (Neutral):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

A1-c (Pleasant):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.4 dB). Pleasantness: pleasant (ISOPleasant: 0.25). [pleasantness: pleasant]

A1-d (Very Pleasant):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.1 dB). Pleasantness: very pleasant (ISOPleasant: 0.63). [pleasantness: very_pleasant]
```

**What to listen for:**
- A1-a: Harsh, intrusive park sounds—harsh wind, sharp birdsong, creaking branches
- A1-b: Balanced park soundscape
- A1-c: Softer, more serene birdsong
- A1-d: Refined, delicate park ambience—subtle birdsong, gentle wind

**Acoustic indicators:**
- Pleasantness progression should show increasing smoothness and reduced harsh frequencies
- Birdsong should become less aggressive as pleasantness increases
- Overall noise floor should decrease from A1-a to A1-d

---

### Test A2: Water Scene Across Pleasantness Spectrum

**Base scene:** `flowing water with ripples and splashing`

```
A2-a (Unpleasant):
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.1 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.42). [pleasantness: unpleasant]

A2-b (Neutral):
flowing water with ripples and splashing. Loudness: quiet (LA50: 49.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

A2-c (Pleasant):
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.5 dB). Pleasantness: pleasant (ISOPleasant: 0.28). [pleasantness: pleasant]

A2-d (Very Pleasant):
flowing water with ripples and splashing. Loudness: quiet (LA50: 47.9 dB). Pleasantness: very pleasant (ISOPleasant: 0.72). [pleasantness: very_pleasant]
```

**What to listen for:**
- A2-a: Jarring water sounds, irregular splashing, metallic edges
- A2-b: Natural water flow
- A2-c: Gentle, flowing water, predictable ripple patterns
- A2-d: Meditative water ambience, minimal splashing, smooth flow

**Acoustic indicators:**
- Water becomes progressively more fluid and less turbulent
- Splash impact sounds soften from A2-a to A2-d
- Frequency characteristics shift from harsh/bright to warm/smooth

---

### Test A3: Urban Street Across Pleasantness Spectrum

**Base scene:** `urban street with traffic and pedestrian sounds`

```
A3-a (Unpleasant):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 67.2 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.35). [pleasantness: unpleasant]

A3-b (Neutral):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 66.5 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

A3-c (Pleasant):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 65.8 dB). Pleasantness: pleasant (ISOPleasant: 0.32). [pleasantness: pleasant]

A3-d (Very Pleasant):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 65.2 dB). Pleasantness: very pleasant (ISOPleasant: 0.68). [pleasantness: very_pleasant]
```

**What to listen for:**
- A3-a: Aggressive traffic, harsh honking, chaotic pedestrian noise
- A3-b: Standard urban soundscape
- A3-c: Traffic flows with rhythm, clearer conversations, less harsh
- A3-d: Musicality in traffic, social pedestrian ambience, refined urban soundscape

**Acoustic indicators:**
- Traffic transitions from aggressive to rhythmic
- Honking becomes less frequent or less harsh
- Pedestrian sounds evolve from chaotic to conversational

---

## Test B: Same Pleasantness, Different Scenes

**Goal:** Verify scene-specific knowledge. Confirm the model generates contextually appropriate audio for different environments at the same pleasantness level.

**Method:** Keep pleasantness constant, use completely different scene descriptions.

### Test B1: Pleasant Pleasantness (+0.25 to +0.35) Across Four Scenes

All prompts maintain pleasantness in the pleasant range; only scene changes.

```
B1-Park (Pleasant):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 52.4 dB). Pleasantness: pleasant (ISOPleasant: 0.25). [pleasantness: pleasant]

B1-Water (Pleasant):
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.5 dB). Pleasantness: pleasant (ISOPleasant: 0.28). [pleasantness: pleasant]

B1-Forest (Pleasant):
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 52.8 dB). Pleasantness: pleasant (ISOPleasant: 0.35). [pleasantness: pleasant]

B1-Urban (Pleasant):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 65.8 dB). Pleasantness: pleasant (ISOPleasant: 0.32). [pleasantness: pleasant]
```

**What to listen for:**
- B1-Park: Serene park with gentle birdsong (identifiably a park)
- B1-Water: Flowing water with calm splashing (distinct from park)
- B1-Forest: Gentle wind and rustling leaves (clearly forest, not park)
- B1-Urban: Traffic with musicality but clearly urban (not nature)

**Acoustic indicators:**
- Each scene should be acoustically recognizable despite same pleasantness level
- Loudness differences should correlate with LA50 values (urban B1-Urban should be noticeably louder than B1-Water)
- Scene-specific characteristics preserved across all pleasant-level prompts

---

### Test B2: Neutral Pleasantness (0.0) Across Four Scenes

All prompts maintain neutral pleasantness; only scene changes.

```
B2-Park (Neutral):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

B2-Water (Neutral):
flowing water with ripples and splashing. Loudness: quiet (LA50: 49.0 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

B2-Forest (Neutral):
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 53.5 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]

B2-Urban (Neutral):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 66.5 dB). Pleasantness: neutral (ISOPleasant: 0.0). [pleasantness: neutral]
```

**What to listen for:**
- All at neutral pleasantness (0.0), but each should retain scene identity
- Scene characteristics should be more prominent than pleasantness variations
- Urban should still sound distinctly urban despite neutral rating
- Park should retain birdsong despite neutral rating

---

### Test B3: Unpleasant Pleasantness (-0.25 to -0.35) Across Four Scenes

All prompts maintain unpleasant pleasantness; only scene changes.

```
B3-Park (Unpleasant):
quiet park with natural ambience and birdsong. Loudness: moderately quiet (LA50: 53.8 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.25). [pleasantness: unpleasant]

B3-Water (Unpleasant):
flowing water with ripples and splashing. Loudness: quiet (LA50: 48.1 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.42). [pleasantness: unpleasant]

B3-Forest (Unpleasant):
natural forest ambience with wind rustling through leaves. Loudness: moderately quiet (LA50: 54.2 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.30). [pleasantness: unpleasant]

B3-Urban (Unpleasant):
urban street with traffic and pedestrian sounds. Loudness: moderately loud (LA50: 67.2 dB). Pleasantness: unpleasant pleasantness (ISOPleasant: -0.35). [pleasantness: unpleasant]
```

**What to listen for:**
- All unpleasant (negative ISOPleasant), but each scene should have characteristic harshness
- Park becomes harsh/unpleasant in a park-specific way (intrusive birdsong, chaotic ambience)
- Urban becomes harsh/unpleasant in an urban-specific way (aggressive traffic, honking)
- Forest becomes harsh/unpleasant in a nature-specific way (threatening wind, creaking)

---

## Analysis Matrix

### Test A Results (Same Scene, Different Pleasantness)

| Scene | Unpleasant | Neutral | Pleasant | Very Pleasant | Signal? |
|-------|-----------|---------|----------|---------------|---------|
| Park (A1) | Harsh/intrusive | Balanced | Serene | Refined | ✓/✗ |
| Water (A2) | Jarring | Natural | Gentle | Meditative | ✓/✗ |
| Urban (A3) | Aggressive | Standard | Rhythmic | Musicality | ✓/✗ |

**Scoring:** ✓ = Clear acoustic difference, ✗ = Minimal/no difference, ? = Ambiguous

### Test B Results (Same Pleasantness, Different Scenes)

| Pleasantness | Park | Water | Forest | Urban | Consistency? |
|--------------|------|-------|--------|-------|--------------|
| Unpleasant (B3) | Park-harsh | Water-jarring | Forest-threatening | Urban-aggressive | ✓/✗ |
| Neutral (B2) | Park-normal | Water-normal | Forest-normal | Urban-normal | ✓/✗ |
| Pleasant (B1) | Park-serene | Water-gentle | Forest-calming | Urban-refined | ✓/✗ |

**Scoring:** ✓ = Scene identity preserved, ✗ = Scenes sound similar despite description

---

## Interpretation Guide

### Strong Learning Signal (Model Well-Tuned at Step 3500)

- **Test A:** Clear acoustic progression from unpleasant to very pleasant within same scene
- **Test B:** Scene identity clearly preserved despite constant pleasantness
- **Combined:** Both effects present and non-interfering

**Conclusion:** Model learned both scene content AND pleasantness modulation

### Moderate Signal (Model Still Training)

- **Test A:** Some pleasantness variation audible but subtle
- **Test B:** Scene identity present but pleasantness effects weak
- **Combined:** One axis stronger than the other

**Conclusion:** Step 3500 shows progress; final model (step 5000) likely stronger

### Weak Signal (Model Needs More Training)

- **Test A:** Minimal pleasantness differences; mostly same audio across 4 prompts
- **Test B:** Scenes sound similar; pleasantness barely varied
- **Combined:** Both effects absent or very subtle

**Conclusion:** Continue training; model not yet capturing ARAUS patterns

---

## Testing Workflow

1. **Generate all Test A prompts** (3 scenes × 4 pleasantness levels = 12 audio files)
2. **Listen for pleasantness progression** within each scene
3. **Generate all Test B prompts** (4 scenes × 3 pleasantness levels = 12 audio files)
4. **Listen for scene consistency** within each pleasantness level
5. **Fill matrix above** with ✓/✗/? scores
6. **Compare to base model** (no LoRA) for the same prompts to see improvement

---

## Next Steps After Testing

### If Test A is strong, Test B is weak:
- Model learned pleasantness modulation well
- Recommend: Increase scene description variety in future training or add scene-specific fine-tuning

### If Test A is weak, Test B is strong:
- Model retained scene knowledge well
- Recommend: Increase pleasantness variation in training data or adjust learning rate

### If both are strong:
- Model learned both dimensions effectively
- Recommendation: Step 3500 checkpoint is good for intermediate testing; step 5000 should be excellent

### If both are weak:
- Model not yet capturing training patterns
- Recommendation: Continue training to step 5000; revisit testing at step 2500 and 4000 for debugging

"""Step 4 - Generate audio captions for ARAUS soundscapes using sound analysis.

Analyzes the acoustic content of each soundscape using librosa feature
extraction and spectral analysis to generate more accurate, detailed
descriptions that capture what's actually in the recording.

For each soundscape:
1. Load audio file
2. Extract acoustic features (spectral centroid, zero crossing rate, MFCC)
3. Classify into audio concepts based on feature patterns
4. Generate natural language caption

Output:
    data/generation/soundscape_captions.csv
    [soundscape, caption, confidence]

Dependencies: librosa, numpy, pandas

    python generation/04_soundscape_captions.py \
      --araus data/raw/araus \
      --out data/generation
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import librosa
except ImportError:
    print("ERROR: librosa not installed")
    print("Install with: pip install --break-system-packages librosa")
    sys.exit(1)


def load_soundscape_audio(path, sr=44100, duration=30):
    """Load soundscape audio and return as numpy array."""
    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, duration=duration, mono=True)
        return y, sr
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None, None


def extract_audio_features(y, sr):
    """Extract acoustic features from audio.
    
    Returns:
        Dict with feature statistics
    """
    features = {}
    
    try:
        # Spectral features
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
        features['spectral_rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
        
        # Temporal features
        features['spectral_bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfcc))
        features['mfcc_std'] = float(np.std(mfcc))
        
        # Tempogram (for rhythm detection)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_strength'] = float(np.mean(onset_env))
        
    except Exception as e:
        print(f"ERROR extracting features: {e}")
        return None
    
    return features


def classify_audio_concepts(features):
    """Classify audio into concepts based on extracted features.
    
    Returns:
        List of (concept, confidence) tuples
    """
    if features is None:
        return []
    
    concepts = []
    
    # Feature thresholds for different audio concepts
    # (concept, feature_checks, base_confidence)
    
    sc = features['spectral_centroid']  # Hz
    zcr = features['zero_crossing_rate']
    rms = features['rms_energy']
    onset = features['onset_strength']
    bandwidth = features['spectral_bandwidth']
    
    # Bird song: high frequency, moderate onset, periodic
    if 3000 < sc < 8000 and zcr > 0.1 and onset > 0.05:
        concepts.append(("birdsong", 0.85))
    
    # Water: low frequency, smooth spectral content, low ZCR
    if sc < 2000 and zcr < 0.05 and rms > 0.01:
        concepts.append(("water", 0.8))
    
    # Traffic/vehicle: broad spectrum, high RMS, consistent
    if 500 < sc < 4000 and rms > 0.02 and bandwidth > 2000:
        concepts.append(("traffic", 0.8))
    
    # Wind/noise: very low frequency, very low ZCR, low RMS
    if sc < 1000 and zcr < 0.03 and rms < 0.01:
        concepts.append(("wind", 0.75))
    
    # Construction/machinery: high energy, broad spectrum, sharp onsets
    if rms > 0.03 and onset > 0.1 and bandwidth > 3000:
        concepts.append(("construction", 0.8))
    
    # Speech/voices: high ZCR, midrange frequencies, dynamic
    if 0.08 < zcr < 0.15 and 1500 < sc < 4000:
        concepts.append(("speech", 0.75))
    
    # Music/melodic: periodic, mid-high frequency, structured
    if 1000 < sc < 5000 and 0.05 < onset < 0.2:
        concepts.append(("music", 0.7))
    
    # Generic ambient/nature sounds
    if not concepts:
        if sc < 2000:
            concepts.append(("low_frequency_ambience", 0.6))
        elif sc < 4000:
            concepts.append(("midrange_sounds", 0.6))
        else:
            concepts.append(("high_frequency_ambience", 0.6))
    
    return sorted(concepts, key=lambda x: x[1], reverse=True)


def generate_caption_from_concepts(concepts):
    """Generate natural language caption from classified concepts.
    
    Returns:
        (caption_text, average_confidence)
    """
    if not concepts:
        return "environment with varied acoustic activity", 0.0
    
    # Filter by confidence threshold
    high_conf = [c for c, s in concepts if s > 0.6]
    
    if not high_conf:
        return "environment with ambient sounds", concepts[0][1]
    
    # Map concepts to natural descriptions
    concept_map = {
        "birdsong": "birds singing",
        "water": "flowing water",
        "traffic": "traffic noise",
        "construction": "construction machinery",
        "wind": "wind",
        "speech": "human voices",
        "music": "musical sounds",
        "low_frequency_ambience": "low frequency rumble",
        "midrange_sounds": "ambient midrange sounds",
        "high_frequency_ambience": "high frequency activity",
    }
    
    # Build description
    if len(high_conf) == 1:
        main_concept = high_conf[0]
        desc_map = {
            "birdsong": "natural soundscape with birdsong",
            "water": "waterside environment with flowing water",
            "traffic": "urban area with traffic noise",
            "construction": "construction site with heavy machinery",
            "wind": "outdoor environment with wind",
            "speech": "public space with human voices and activity",
            "music": "environment with musical sounds",
        }
        desc = desc_map.get(main_concept, f"soundscape with {main_concept}")
    else:
        # Multiple concepts
        descriptions = [concept_map.get(c, c) for c in high_conf[:3]]
        desc = "soundscape with " + " and ".join(descriptions)
    
    avg_confidence = np.mean([s for c, s in concepts])
    return desc, avg_confidence


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--araus", required=True, help="Path to ARAUS dataset root")
    ap.add_argument("--out", default="data/generation")
    a = ap.parse_args()
    
    araus_root = Path(a.araus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    
    soundscape_dir = araus_root / "soundscapes"
    
    if not soundscape_dir.exists():
        print(f"ERROR: Soundscape directory not found at {soundscape_dir}")
        sys.exit(1)
    
    # Get all soundscape files
    soundscape_files = sorted(soundscape_dir.glob("*.wav"))
    print(f"Found {len(soundscape_files)} soundscape files")
    print("Analyzing soundscape audio content...")
    
    results = []
    for idx, wav_file in enumerate(soundscape_files):
        if idx % 50 == 0 and idx > 0:
            print(f"  Processed {idx} / {len(soundscape_files)}...")
        
        soundscape_id = wav_file.stem
        
        # Load audio
        y, sr = load_soundscape_audio(wav_file)
        if y is None:
            continue
        
        # Extract features
        features = extract_audio_features(y, sr)
        if features is None:
            continue
        
        # Classify concepts
        concepts = classify_audio_concepts(features)
        
        # Generate caption
        caption, confidence = generate_caption_from_concepts(concepts)
        
        results.append({
            "soundscape": soundscape_id,
            "caption": caption,
            "confidence": confidence,
            "top_concepts": "|".join([c for c, s in concepts[:3]])
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_path = out / "soundscape_captions.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nGenerated captions for {len(df)} soundscapes")
    print(f"Saved to {output_path}")
    print(f"\nSample captions:")
    print(df.head(10).to_string())

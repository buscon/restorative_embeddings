"""Step 4 - Generate audio captions for ARAUS soundscapes using CLAP embeddings.

Analyzes the acoustic content of each soundscape using CLAP (Contrastive
Language-Audio Pre-training) to generate more accurate, detailed descriptions
that capture what's actually in the recording.

For each soundscape:
1. Load audio file
2. Extract CLAP embedding
3. Match against known audio concepts and descriptions
4. Generate natural language caption

Output:
    data/generation/soundscape_captions.csv
    [soundscape, caption, concept_confidence]

Dependencies: laion-clap, librosa, torch

    python generation/04_soundscape_captions.py \
      --araus data/raw/araus \
      --out data/generation
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import librosa
    import torch
    from laion_clap import CLAP_Module
except ImportError as e:
    print(f"WARNING: Missing dependency: {e}")
    print("Install with: pip install laion-clap librosa torch")


# Audio concept categories for describing soundscapes
AUDIO_CONCEPTS = {
    # Nature sounds
    "bird": {"synonyms": ["birdsong", "birds chirping", "avian"], "category": "animal"},
    "water": {"synonyms": ["flowing water", "stream", "water sounds", "aquatic"], "category": "nature"},
    "wind": {"synonyms": ["wind noise", "breeze", "air"], "category": "nature"},
    "rain": {"synonyms": ["rainfall", "precipitation"], "category": "nature"},
    "thunder": {"synonyms": ["thunder", "storm"], "category": "nature"},

    # Traffic/urban
    "traffic": {"synonyms": ["traffic noise", "cars", "vehicle", "road"], "category": "traffic"},
    "horn": {"synonyms": ["horn", "siren", "alarm"], "category": "traffic"},
    "motorcycle": {"synonyms": ["motorcycle", "bike engine"], "category": "traffic"},

    # Human/speech
    "speech": {"synonyms": ["voice", "speaking", "conversation", "human voice"], "category": "human"},
    "music": {"synonyms": ["music", "musical"], "category": "human"},
    "laughter": {"synonyms": ["laughter", "laughing"], "category": "human"},
    "applause": {"synonyms": ["applause", "clapping"], "category": "human"},

    # Construction/mechanical
    "construction": {"synonyms": ["construction", "machinery", "drilling", "jackhammer"], "category": "mechanical"},
    "saw": {"synonyms": ["saw", "sawing"], "category": "mechanical"},
    "power tools": {"synonyms": ["power tools", "electric drill"], "category": "mechanical"},

    # Animals
    "dog": {"synonyms": ["dog", "barking", "canine"], "category": "animal"},
    "cat": {"synonyms": ["cat", "meow", "feline"], "category": "animal"},
    "cow": {"synonyms": ["cow", "moo", "cattle"], "category": "animal"},

    # Ambience
    "crowd": {"synonyms": ["crowd", "crowd noise", "people"], "category": "human"},
    "footsteps": {"synonyms": ["footsteps", "walking", "steps"], "category": "human"},
    "ambient": {"synonyms": ["ambient", "background", "environment"], "category": "ambience"},
}

CONCEPT_DESCRIPTIONS = {
    # Templates for different concept combinations
    "bird_only": "natural soundscape with birdsong",
    "bird_traffic": "urban environment with birdsong and traffic",
    "bird_speech": "public space with birds and human voices",
    "water_only": "waterside environment with water sounds",
    "water_traffic": "waterfront with traffic and water sounds",
    "traffic_only": "urban street with traffic noise",
    "traffic_speech": "busy urban area with traffic and voices",
    "construction_only": "construction site with machinery",
    "wind_only": "outdoor environment with wind",
    "nature": "natural soundscape",
    "urban": "urban soundscape",
    "indoor": "indoor environment",
}


def load_soundscape_audio(path, sr=44100, duration=30):
    """Load soundscape audio and return as numpy array.

    Args:
        path: Path to soundscape WAV file
        sr: Sample rate
        duration: Duration in seconds to load (30s is standard)

    Returns:
        (y, sr) audio array and sample rate
    """
    try:
        y, sr_loaded = librosa.load(str(path), sr=sr, duration=duration, mono=True)
        return y, sr
    except Exception as e:
        print(f"ERROR loading {path}: {e}")
        return None, None


def get_clap_embedding(y, sr, clap_model):
    """Extract CLAP embedding from audio.

    Args:
        y: Audio array
        sr: Sample rate
        clap_model: CLAP model instance

    Returns:
        Embedding vector (shape: [512] typically)
    """
    if y is None:
        return None

    try:
        # Prepare audio for CLAP (expects tensor)
        with torch.no_grad():
            embedding = clap_model.get_audio_embedding_from_data(
                {"audio": torch.from_numpy(y).float().unsqueeze(0)},
                use_tensor=True
            )
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"ERROR computing embedding: {e}")
        return None


def match_concepts(embedding, clap_model, top_k=3):
    """Match audio embedding against known concepts.

    Uses CLAP to score similarity between audio and concept descriptions.

    Returns:
        List of (concept, score) tuples, sorted by score
    """
    if embedding is None:
        return []

    matches = []
    try:
        for concept, info in AUDIO_CONCEPTS.items():
            # Get embedding for concept text
            with torch.no_grad():
                concept_text = f"{concept}. {info['synonyms'][0]}."
                concept_embedding = clap_model.get_text_embedding([concept_text])

            # Compute cosine similarity
            score = np.dot(embedding, concept_embedding.flatten()) / (
                np.linalg.norm(embedding) * np.linalg.norm(concept_embedding.flatten()) + 1e-8
            )
            matches.append((concept, float(score)))
    except Exception as e:
        print(f"ERROR matching concepts: {e}")
        return []

    # Return top matches
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


def generate_caption_from_concepts(concepts, confidence_threshold=0.3):
    """Generate natural language caption from matched concepts.

    Args:
        concepts: List of (concept, score) tuples
        confidence_threshold: Only include concepts above this score

    Returns:
        (caption_text, average_confidence)
    """
    if not concepts:
        return "environment with varied acoustic activity", 0.0

    # Filter by confidence
    high_conf = [c for c, s in concepts if s > confidence_threshold]

    if not high_conf:
        return "environment with ambient sounds", concepts[0][1]

    # Build description based on top concepts
    if len(high_conf) == 1:
        concept = high_conf[0]
        # Single dominant concept
        if concept == "bird":
            desc = "natural soundscape with birds"
        elif concept == "water":
            desc = "waterside environment with water sounds"
        elif concept == "traffic":
            desc = "urban area with traffic noise"
        elif concept == "construction":
            desc = "construction site with machinery"
        elif concept == "wind":
            desc = "outdoor environment with wind"
        elif concept == "crowd" or concept == "speech":
            desc = "public space with human activity"
        else:
            desc = f"environment with {concept}"
    else:
        # Multiple concepts - create compound description
        concepts_str = ", ".join(high_conf[:3])
        desc = f"soundscape with {concepts_str}"

    avg_confidence = np.mean([s for c, s in concepts])
    return desc, avg_confidence


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--araus", required=True, help="Path to ARAUS dataset root")
    ap.add_argument("--out", default="data/generation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    araus_root = Path(a.araus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    soundscape_dir = araus_root / "soundscapes"

    if not soundscape_dir.exists():
        print(f"ERROR: Soundscape directory not found at {soundscape_dir}")
        exit(1)

    print("Loading CLAP model...")
    try:
        clap_model = CLAP_Module(enable_fusion=True, device=a.device)
        clap_model.load_ckpt()
    except Exception as e:
        print(f"ERROR loading CLAP: {e}")
        print("Make sure laion-clap is installed: pip install laion-clap")
        exit(1)

    # Get all soundscape files
    soundscape_files = sorted(soundscape_dir.glob("*.wav"))
    print(f"Found {len(soundscape_files)} soundscape files")

    results = []
    for idx, wav_file in enumerate(soundscape_files):
        if idx % 100 == 0:
            print(f"  Processing {idx} / {len(soundscape_files)}...")

        soundscape_id = wav_file.stem

        # Load audio
        y, sr = load_soundscape_audio(wav_file)
        if y is None:
            continue

        # Get CLAP embedding
        embedding = get_clap_embedding(y, sr, clap_model)
        if embedding is None:
            continue

        # Match concepts
        concepts = match_concepts(embedding, clap_model, top_k=5)

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

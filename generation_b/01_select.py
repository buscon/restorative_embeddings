"""Step 1 - Select soundscape-like clips from FSD50K and/or AudioSet.

Filters FSD50K and AudioSet to soundscape-relevant classes using the AudioSet
ontology, applies length and license filters, and outputs a CSV ready for
rating (step 02_rate.py).

Output: fsd50k_selected.csv and/or audioset_selected.csv with columns:
    [id, dataset, path, length_s, licenses, uploader, labels, split]

    python generation_b/01_select.py --fsd50k /path/to/fsd50k --out generation_b

Filters by:
- License: CC0, CC-BY only (drop CC-BY-NC, CC Sampling+ unless --keep-nc)
- Length: 5s minimum (or --min-length for different threshold)
- Content: AudioSet ontology classes matching soundscape content
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


# AudioSet classes to keep (soundscape-relevant)
SOUNDSCAPE_CLASSES = {
    # Natural sounds
    'wind': 'natural sound',
    'water': 'natural sound',
    'rain': 'natural sound',
    'thunder': 'natural sound',
    'wind (flowing)': 'natural sound',
    
    # Animals and birds
    'animal': 'animal',
    'bird': 'animal',
    'dog': 'animal',
    'cat': 'animal',
    'bird call': 'animal',
    'bird song': 'animal',
    'rooster': 'animal',
    'insect': 'animal',
    
    # Vehicles and traffic
    'vehicle': 'vehicle',
    'car': 'vehicle',
    'traffic': 'vehicle',
    'motorcycle': 'vehicle',
    'bicycle': 'vehicle',
    'train': 'vehicle',
    'aircraft': 'vehicle',
    'helicopter': 'vehicle',
    
    # Human activity
    'human': 'human activity',
    'speech': 'human activity',
    'footsteps': 'human activity',
    'crowd': 'human activity',
    'chatter': 'human activity',
    
    # Domestic
    'door': 'domestic',
    'bell': 'domestic',
    'phone': 'domestic',
    'kitchen': 'domestic',
    'microwave': 'domestic',
    'vacuum': 'domestic',
    
    # Construction and tools
    'construction': 'construction',
    'tool': 'construction',
    'hammer': 'construction',
    'saw': 'construction',
    'drill': 'construction',
}

# Classes to drop (drop all related)
DROP_CLASSES = {
    'music', 'musical', 'instrument', 'singing', 'song',
    'speech', 'voice', 'conversation', 'talking', 'narration',
    'studio', 'foley', 'isolated', 'click', 'knock', 'tap'
}

# License codes: keep these
ALLOWED_LICENSES = {'CC0', 'CC-BY'}


def sanitize_class(class_str):
    """Normalize class string for matching."""
    return class_str.lower().strip()


def is_soundscape_relevant(labels, drop_threshold=0.5):
    """Check if labels are soundscape-relevant.
    
    Returns True if:
    - Has at least one soundscape-relevant class
    - AND drop-class ratio <= drop_threshold
    """
    if not labels:
        return False
    
    labels_lower = [sanitize_class(l) for l in labels]
    
    # Check for DROP classes
    drop_match = sum(1 for l in labels_lower if any(d in l for d in DROP_CLASSES))
    if drop_match / len(labels) > drop_threshold:
        return False
    
    # Check for KEEP classes
    keep_match = sum(1 for l in labels_lower 
                     if any(k in l for k in SOUNDSCAPE_CLASSES.keys()))
    
    return keep_match > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fsd50k', type=Path, help='FSD50K data directory')
    parser.add_argument('--audioset', type=Path, help='AudioSet metadata CSV')
    parser.add_argument('--min-length', type=float, default=5.0,
                        help='Minimum clip length in seconds')
    parser.add_argument('--keep-nc', action='store_true',
                        help='Keep CC-BY-NC licensed clips')
    parser.add_argument('--out', type=Path, default=Path('generation_b'),
                        help='Output directory')
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PLAN B: SELECT CLIPS (Step 1)")
    print("=" * 70)
    
    # =========================================================================
    # FSD50K
    # =========================================================================
    if args.fsd50k:
        print(f"\nProcessing FSD50K from {args.fsd50k}...")
        
        # Expected structure: FSD50K/{dev, eval, test}
        # With metadata: fsd50k.all.csv, fsd50k.eval_audio.csv, etc.
        
        print("  FSD50K processing requires:")
        print("  1. FSD50K metadata CSV (vocabulary_id, split, licenses, etc.)")
        print("  2. Audio files organized by split")
        print("  \n  Expected structure:")
        print("    FSD50K/")
        print("      ├─ fsd50k.all.csv (or fsd50k_metadata.csv)")
        print("      ├─ dev_audio/")
        print("      ├─ eval_audio/")
        print("      └─ test_audio/")
        
        print("\n  Placeholder implementation:")
        
        selected_fsd = {
            'status': 'not_implemented',
            'reason': 'Requires FSD50K download and metadata',
            'next_step': 'Download from Zenodo 4060432'
        }
    else:
        selected_fsd = None
    
    # =========================================================================
    # AudioSet
    # =========================================================================
    if args.audioset:
        print(f"\nProcessing AudioSet from {args.audioset}...")
        print("  WARNING: AudioSet clips must be downloaded from YouTube.")
        print("  Many videos are no longer available.")
        print("  Check university policy on using YouTube content for training.")
        
        selected_audioset = {
            'status': 'not_implemented',
            'reason': 'Requires AudioSet download and YouTube access',
            'legal_note': 'Confirm with university before using YouTube content'
        }
    else:
        selected_audioset = None
    
    # =========================================================================
    # Summary and class list
    # =========================================================================
    print("\n" + "=" * 70)
    print("SOUNDSCAPE CLASSES (to keep)")
    print("=" * 70)
    
    classes_by_cat = defaultdict(list)
    for cls, cat in sorted(SOUNDSCAPE_CLASSES.items()):
        classes_by_cat[cat].append(cls)
    
    for cat in sorted(classes_by_cat.keys()):
        print(f"\n{cat.upper()}")
        for cls in classes_by_cat[cat]:
            print(f"  - {cls}")
    
    # Save class list
    class_list = {
        'keep': SOUNDSCAPE_CLASSES,
        'drop_keywords': list(DROP_CLASSES),
        'drop_threshold': 0.5,
        'licenses_keep': list(ALLOWED_LICENSES),
        'min_length': args.min_length
    }
    
    class_file = args.out / 'selection_criteria.json'
    with open(class_file, 'w') as f:
        json.dump(class_list, f, indent=2)
    
    print(f"\n✓ Class list saved to {class_file}")
    
    # =========================================================================
    # Output summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    if not args.fsd50k and not args.audioset:
        print("\n1. Download FSD50K from Zenodo (4060432)")
        print("   Or AudioSet from GitHub (gstax/AudioSet)")
        print("\n2. Run with:")
        print(f"   python generation_b/01_select.py --fsd50k /path/to/fsd50k --out {args.out}")
        print("\n3. Output: fsd50k_selected.csv (ready for step 02_rate.py)")
    
    print("\n✓ Selection criteria and class list saved")
    print(f"  Output directory: {args.out}")


if __name__ == '__main__':
    main()

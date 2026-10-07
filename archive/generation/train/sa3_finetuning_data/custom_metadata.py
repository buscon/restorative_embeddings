import csv
from pathlib import Path

def get_custom_metadata(info, audio):
    """
    Extract metadata for each audio file from metadata.csv
    
    Args:
        info: dict with file info (includes 'relpath')
        audio: audio data (not used here)
    
    Returns:
        dict with 'prompt' key containing the caption from metadata.csv
    """
    
    # Get the audio filename from the path
    audio_filename = Path(info["relpath"]).stem  # Just the filename without extension
    
    # Path to metadata CSV in the same directory
    metadata_csv = Path(info["relpath"]).parent / "metadata.csv"
    
    # Read metadata.csv and find the matching caption
    if metadata_csv.exists():
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['file'].replace('.wav', '') == audio_filename:
                    return {"prompt": row['caption']}
    
    # Fallback if file not found in CSV
    return {"prompt": "Soundscape audio"}

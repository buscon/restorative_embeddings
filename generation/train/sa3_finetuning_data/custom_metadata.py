import os
from pathlib import Path

def get_metadata(filepath):
    """
    Load metadata for an audio file from a corresponding .txt caption file.
    
    Args:
        filepath: Path to the audio file (.wav)
    
    Returns:
        Dictionary with 'caption' key containing the text from the .txt file
    """
    # Get the base name without extension
    base_path = os.path.splitext(filepath)[0]
    
    # Look for corresponding .txt file
    txt_path = base_path + '.txt'
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        return {'caption': caption}
    else:
        # Fallback: use filename as caption if no .txt file
        return {'caption': Path(filepath).stem}

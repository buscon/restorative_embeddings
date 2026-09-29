"""
Prepare ISD Dataset for Restorative Soundscape Pipeline

Downloads audio files from Zenodo and organizes them for embedding extraction.
"""

import os
import json
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm

# Zenodo record ID for ISD
ZENODO_RECORD_ID = 10672568

# Map of file keys to Zenodo file IDs (you'll need to check these on Zenodo)
# For now, this is a guide — you may need to download manually or use zenodo API
ZENODO_FILES = {
    'audio_granada': None,  # Replace with actual file ID
    'audio_groningen': None,
    'audio_london': None,
    'audio_venice': None,
    'survey_data': 'ISD v1.0 Data.csv',
    'metadata': 'ISD v1.0 Metadata.xlsx',
}


def download_from_zenodo(zenodo_id, output_dir='./isd_raw'):
    """
    Download ISD files from Zenodo.
    
    Note: This requires either:
    - Manual download from https://zenodo.org/records/{zenodo_id}
    - Or use zenodo API with curl/requests
    
    For now, this is a template — adjust based on Zenodo's actual file structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"ISD Dataset: https://zenodo.org/records/{zenodo_id}")
    print("\nTo download:")
    print("  Option 1: Visit Zenodo link above, download files manually")
    print("  Option 2: Use zenodo-client or curl")
    print(f"\nExtract audio files into: {output_dir}/")
    
    return output_dir


def organize_audio_files(raw_audio_dir, organized_dir='./isd_audio'):
    """
    Organize downloaded ISD audio files into flat directory structure.
    
    ISD structure: LocationID/SessionID/GroupID/audio.wav
    → Flatten to: audio_locationid_sessionid_groupid.wav
    
    This makes batch loading simpler and preserves metadata in filename.
    """
    os.makedirs(organized_dir, exist_ok=True)
    
    raw_path = Path(raw_audio_dir)
    organized_path = Path(organized_dir)
    
    audio_files = list(raw_path.glob('**/*.wav'))
    print(f"Found {len(audio_files)} WAV files")
    
    if len(audio_files) == 0:
        print(f"No WAV files found in {raw_audio_dir}")
        print("Make sure ISD audio archives are extracted.")
        return None
    
    # Create manifest
    manifest = {
        'total_files': len(audio_files),
        'files': []
    }
    
    for wav_file in tqdm(audio_files, desc="Organizing audio files"):
        # Extract hierarchical structure from path
        parts = wav_file.parts
        
        # Try to extract LocationID, SessionID, GroupID from path
        # This depends on actual ISD structure — adjust as needed
        rel_path = wav_file.relative_to(raw_path)
        new_name = '_'.join(rel_path.parts[:-1] + (wav_file.stem,)) + '.wav'
        
        new_path = organized_path / new_name
        
        # Copy file (or symlink if on same filesystem)
        try:
            import shutil
            shutil.copy2(wav_file, new_path)
            
            manifest['files'].append({
                'original_path': str(wav_file),
                'organized_path': str(new_path),
                'filename': new_name
            })
        except Exception as e:
            print(f"Error copying {wav_file}: {e}")
    
    # Save manifest
    manifest_path = organized_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nOrganized {len(manifest['files'])} files into {organized_dir}/")
    print(f"Manifest saved to {manifest_path}")
    
    return organized_path


def extract_isd_zip_archives(raw_dir, extract_to='./isd_raw'):
    """
    Extract ZIP archives from Zenodo download.
    
    Usage:
        extract_isd_zip_archives('~/Downloads')  # Where Zenodo files are
    """
    raw_path = Path(raw_dir)
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(raw_path.glob('*.zip')) + list(raw_path.glob('**/*.zip'))
    
    if not zip_files:
        print(f"No ZIP files found in {raw_dir}")
        return None
    
    for zip_file in zip_files:
        print(f"\nExtracting {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(extract_path)
            print(f"✓ Extracted to {extract_path}")
        except Exception as e:
            print(f"✗ Error extracting {zip_file}: {e}")
    
    return extract_path


def main():
    """
    Full preparation workflow
    """
    print("=" * 60)
    print("ISD Dataset Preparation")
    print("=" * 60)
    
    # Step 1: Download
    print("\n[1] Downloading ISD from Zenodo...")
    raw_dir = download_from_zenodo(ZENODO_RECORD_ID)
    
    # Step 2: Extract (if using ZIP archives)
    print("\n[2] Extracting archive files...")
    print("   If you've already extracted, you can skip this.")
    
    # Step 3: Organize
    print("\n[3] Organizing audio files...")
    if os.path.exists(raw_dir):
        organized_dir = organize_audio_files(raw_dir)
        
        if organized_dir:
            print(f"\n✓ Ready for pipeline!")
            print(f"  Audio directory: {organized_dir}/")
            print(f"  Use in pipeline: audio_dir='{organized_dir}'")
        else:
            print("\n✗ No audio files found. Check download.")
    else:
        print(f"\n✗ Raw directory not found: {raw_dir}")
        print("   Download files manually from:")
        print(f"   https://zenodo.org/records/{ZENODO_RECORD_ID}")


if __name__ == '__main__':
    main()

#!/bin/bash

# Download ISD Audio Files from Zenodo
# Total: ~13 GB across 9 ZIP archives

# Output directory
OUTPUT_DIR="${1:-.}"  # Default to current dir, or use $1
RAW_DIR="$OUTPUT_DIR/isd_raw"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "Downloading ISD dataset to: $RAW_DIR"
echo "Total size: ~13 GB"
echo ""

# Audio file archives
declare -a FILES=(
    "WAV_Granada_1.zip"
    "WAV_Groningen_1.zip"
    "WAV_London_1.zip"
    "WAV_London_2.zip"
    "WAV_London_3.zip"
    "WAV_London_4.zip"
    "WAV_Venice_1.zip"
    "WAV_Lockdown_London.zip"
    "WAV_Lockdown_Venice.zip"
)

# Base URL
BASE_URL="https://zenodo.org/records/10672568/files"

# Download each file
for FILE in "${FILES[@]}"; do
    URL="$BASE_URL/$FILE?download=1"
    
    if [ -f "$FILE" ]; then
        echo "✓ Already downloaded: $FILE"
    else
        echo "↓ Downloading: $FILE..."
        wget --progress=bar:force "$URL" -O "$FILE" || {
            echo "✗ Failed to download $FILE"
            exit 1
        }
        echo "✓ Downloaded: $FILE"
    fi
done

echo ""
echo "Download complete!"
echo "Files downloaded to: $RAW_DIR"
echo ""
echo "Next: Extract all ZIP files"
echo "  for f in *.zip; do unzip -q \"\$f\" && echo \"✓ Extracted: \$f\"; done"

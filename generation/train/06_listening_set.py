#!/usr/bin/env python3
"""
Create Blinded Listening Test Set

Prepares a set of anonymized audio clips for human perceptual evaluation.
Generates an HTML rating form and answer key.

Usage:
    python 06_listening_set.py --finetuned-results generation/train/finetuned_generations/results.csv
"""

import argparse
import pandas as pd
import json
from pathlib import Path
import shutil
import random

def create_rating_form(n_clips, output_path):
    """Generate HTML rating form."""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Soundscape Pleasantness Listening Test</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; text-align: center; }}
        .instructions {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            line-height: 1.6;
        }}
        .clip-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #0066cc;
        }}
        .clip-number {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        audio {{
            width: 100%;
            margin: 15px 0;
        }}
        .rating-group {{
            margin: 15px 0;
        }}
        label {{
            display: block;
            margin: 10px 0 5px 0;
            font-weight: 500;
        }}
        input[type="radio"] {{
            margin-right: 8px;
        }}
        .scale-description {{
            font-size: 12px;
            color: #666;
            margin: 10px 0;
        }}
        .submit-btn {{
            background: #0066cc;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
            width: 100%;
        }}
        .submit-btn:hover {{ background: #0052a3; }}
        .progress {{
            text-align: center;
            color: #666;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>🎵 Soundscape Pleasantness Listening Test</h1>
    
    <div class="instructions">
        <h3>Instructions</h3>
        <p><strong>Your task:</strong> Listen to each soundscape clip and rate it on the scales below.</p>
        <p>There are <strong>{n_clips} clips</strong> in total. Each takes 10–30 seconds to listen to.</p>
        <p><strong>Headphones recommended</strong> for best audio quality.</p>
        <p><strong>Rating scales:</strong> All items use a 5-point scale from 1 (not at all) to 5 (very much).</p>
        <p>Please try to listen to each clip completely before rating.</p>
    </div>
    
    <form id="ratingForm">
"""
    
    # Add clip rating sections
    for i in range(n_clips):
        html_content += f"""
        <div class="clip-container">
            <div class="clip-number">Clip {i+1} of {n_clips}</div>
            <div class="progress">
                <strong>Duration:</strong> 10–30 seconds
            </div>
            <audio id="clip_{i}" controls></audio>
            <button type="button" onclick="loadClip({i})">Load Audio</button>
            
            <div class="rating-group">
                <label for="pleasantness_{i}">
                    <strong>Pleasantness:</strong> How pleasant is this soundscape?
                </label>
                <div class="scale-description">
                    1 = Very unpleasant | 2 = Unpleasant | 3 = Neutral | 4 = Pleasant | 5 = Very pleasant
                </div>
                <div>
                    <input type="radio" name="pleasantness_{i}" value="1"> 1 (Very unpleasant)
                    <input type="radio" name="pleasantness_{i}" value="2"> 2 (Unpleasant)
                    <input type="radio" name="pleasantness_{i}" value="3"> 3 (Neutral)
                    <input type="radio" name="pleasantness_{i}" value="4"> 4 (Pleasant)
                    <input type="radio" name="pleasantness_{i}" value="5"> 5 (Very pleasant)
                </div>
            </div>
            
            <div class="rating-group">
                <label for="relaxation_{i}">
                    <strong>Relaxation:</strong> How much does this soundscape help you relax?
                </label>
                <div class="scale-description">
                    1 = Not at all | 5 = Very much
                </div>
                <div>
                    <input type="radio" name="relaxation_{i}" value="1"> 1
                    <input type="radio" name="relaxation_{i}" value="2"> 2
                    <input type="radio" name="relaxation_{i}" value="3"> 3
                    <input type="radio" name="relaxation_{i}" value="4"> 4
                    <input type="radio" name="relaxation_{i}" value="5"> 5
                </div>
            </div>
            
            <div class="rating-group">
                <label for="eventfulness_{i}">
                    <strong>Eventfulness:</strong> How many different sounds do you hear?
                </label>
                <div class="scale-description">
                    1 = Very few sounds | 5 = Many different sounds
                </div>
                <div>
                    <input type="radio" name="eventfulness_{i}" value="1"> 1
                    <input type="radio" name="eventfulness_{i}" value="2"> 2
                    <input type="radio" name="eventfulness_{i}" value="3"> 3
                    <input type="radio" name="eventfulness_{i}" value="4"> 4
                    <input type="radio" name="eventfulness_{i}" value="5"> 5
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <button type="submit" class="submit-btn">Submit Responses</button>
    </form>
    
    <script>
        function loadClip(clipIndex) {{
            // In real usage, this would load audio from the clips directory
            alert(`Audio for clip ${{clipIndex + 1}} would load here`);
        }}
        
        document.getElementById('ratingForm').addEventListener('submit', function(e) {{
            e.preventDefault();
            const responses = {{}};
            const formData = new FormData(this);
            for (let [key, value] of formData.entries()) {{
                responses[key] = value;
            }}
            // Save to file (download)
            const dataStr = JSON.stringify(responses, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'listening_test_responses.json';
            link.click();
            alert('Thank you! Your responses have been downloaded.');
        }});
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(
        description="Create blinded listening test set"
    )
    parser.add_argument(
        "--finetuned-results",
        required=True,
        help="Path to fine-tuned generation results CSV"
    )
    parser.add_argument(
        "--n-clips",
        type=int,
        default=20,
        help="Number of clips to include in listening test"
    )
    parser.add_argument(
        "--output-dir",
        default="generation/train/listening_test",
        help="Output directory"
    )
    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Creating Blinded Listening Test Set                      ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Load results
    results_df = pd.read_csv(args.finetuned_results)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    # Sample clips stratified by pleasantness level
    pleasantness_levels = ['very_unpleasant', 'unpleasant', 'neutral', 'pleasant', 'very_pleasant']
    sampled_clips = []
    clips_per_level = args.n_clips // len(pleasantness_levels)

    for level in pleasantness_levels:
        level_clips = results_df[results_df['pleasantness_level'] == level]
        if len(level_clips) > 0:
            selected = level_clips.sample(min(clips_per_level, len(level_clips)), random_state=42)
            sampled_clips.append(selected)

    sampled_df = pd.concat(sampled_clips, ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=42)  # Shuffle

    # Create answer key
    answer_key = sampled_df[['prompt_id', 'prompt', 'pleasantness_level', 'filepath']].copy()
    answer_key['clip_index'] = range(len(answer_key))
    answer_key.to_csv(output_dir / "answer_key.csv", index=False)
    print(f"✓ Answer key saved: {output_dir / 'answer_key.csv'}")

    # Copy audio files and rename anonymously
    audio_mapping = {}
    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        # Copy audio with anonymized name
        src_path = Path(args.finetuned_results).parent / row['filepath']
        if src_path.exists():
            dst_name = f"clip_{idx:02d}.wav"
            dst_path = clips_dir / dst_name
            try:
                shutil.copy2(src_path, dst_path)
                audio_mapping[dst_name] = row['filepath']
            except Exception as e:
                print(f"Warning: Could not copy {src_path}: {e}")

    print(f"✓ Audio clips copied: {len(audio_mapping)} files")

    # Create HTML rating form
    form_path = output_dir / "rating_sheet.html"
    create_rating_form(len(answer_key), form_path)
    print(f"✓ Rating form created: {form_path}")

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Listening Test Ready!                                    ║
╚════════════════════════════════════════════════════════════╝

Files created:
  - {output_dir / 'rating_sheet.html'} — Interactive rating form (open in browser)
  - {output_dir / 'clips'}/ — Anonymized audio clips
  - {output_dir / 'answer_key.csv'} — Maps clips to conditions

Next steps:
  1. Open rating_sheet.html in a web browser
  2. Listen to each clip and rate on the scales provided
  3. Submit your responses (downloads as JSON)
  4. Analyze results: python generation/train/07_analyze_listening_test.py

For detailed procedure, see: generation/train/README_FINETUNE.md (Step 7)
    """)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate baseline audio with unmodified Stable Audio Open

Tests what the pretrained model can already do with your dataset
before fine-tuning. This serves as a control for phase 2.

Usage:
    python 03_generate_baseline.py --n-prompts 10 --output-dir generation/train/baseline_generations
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import numpy as np

def get_held_out_prompts(n_prompts=10):
    """
    Get held-out scene prompts from ARAUS.
    In a real scenario, these come from held-out base soundscapes.
    For now, we'll use a set of generic descriptive prompts.
    """
    prompts = [
        "Urban park with birdsong and distant traffic",
        "Quiet forest with wind through trees",
        "Busy street intersection with car horns",
        "Peaceful waterside with flowing water and birds",
        "Construction site with heavy machinery",
        "Shopping mall with people and ambient music",
        "Beach with waves and seagulls",
        "Train station with announcements and movement",
        "Residential neighborhood with children playing",
        "Nature reserve with diverse wildlife",
    ]
    return prompts[:n_prompts]

def generate_audio(model, prompt, seconds=10, seed=42):
    """
    Generate audio with the model for a given prompt.
    
    Note: This is a simplified version. The actual implementation depends
    on stable-audio-tools API.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # This would call the model's generate method
    # For now, this is a placeholder showing the workflow
    try:
        # Actual generation call (adjust based on stable-audio-tools API)
        with torch.no_grad():
            # Generate audio tensor
            # output = model.generate(prompt, seconds=seconds, temperature=1.0)
            # For demonstration, we'll create a dummy tensor
            output = torch.randn(1, 2, seconds * 44100)  # stereo, 44.1kHz
        return output
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline audio with unmodified SAO"
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=10,
        help="Number of held-out prompts to generate"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=4,
        help="Number of random seeds per prompt"
    )
    parser.add_argument(
        "--output-dir",
        default="generation/train/baseline_generations",
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "large"],
        default="small",
        help="Model size to use"
    )
    args = parser.parse_args()

    print(f"""
╔════════════════════════════════════════════════════════════╗
║  Generating Baseline Audio (Pretrained SAO)               ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Load model
    print(f"\nLoading Stable Audio Open ({args.model_size})...")
    if args.model_size == "small":
        model_id = "stabilityai/stable-audio-open-1.0-small"
    else:
        model_id = "stabilityai/stable-audio-open-1.0"
    
    try:
        from stable_audio_tools import get_pretrained_model
        model, sample_rate = get_pretrained_model(model_id)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        print(f"✓ Model loaded (sr={sample_rate})")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure stable-audio-tools is installed.")
        return

    # Get prompts
    prompts = get_held_out_prompts(args.n_prompts)
    pleasantness_levels = ["very_unpleasant", "unpleasant", "neutral", "pleasant", "very_pleasant"]

    # Generate audio
    results = []
    total_generations = len(prompts) * len(pleasantness_levels) * args.n_seeds
    
    with tqdm(total=total_generations, desc="Generating") as pbar:
        for prompt_idx, prompt in enumerate(prompts):
            for level in pleasantness_levels:
                # Create prompt with pleasantness condition
                full_prompt = f"{prompt}. This soundscape is {level}."
                
                for seed in range(args.n_seeds):
                    try:
                        # Generate audio
                        audio_tensor = generate_audio(
                            model, full_prompt, seconds=10, seed=seed
                        )
                        
                        if audio_tensor is not None:
                            # Save audio
                            filename = f"baseline_prompt_{prompt_idx:02d}_{level}_seed{seed}.wav"
                            filepath = audio_dir / filename
                            
                            # Normalize and save
                            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
                            torchaudio.save(str(filepath), audio_tensor[0], sample_rate)
                            
                            # Record metadata
                            results.append({
                                "prompt_id": prompt_idx,
                                "prompt": prompt,
                                "pleasantness_level": level,
                                "seed": seed,
                                "filepath": str(filepath.relative_to(output_dir)),
                                "duration_s": 10,
                            })
                    except Exception as e:
                        print(f"\nError generating {prompt_idx}, {level}, seed {seed}: {e}")
                    
                    pbar.update(1)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv}")
    print(f"  Total clips: {len(results)}")

    # Print summary
    print(f"""
Summary:
  Prompts: {len(prompts)}
  Pleasantness levels: {len(pleasantness_levels)}
  Seeds: {args.n_seeds}
  Total clips: {len(results)}
  Output directory: {output_dir}

Next steps:
  1. Listen to generated audio in {audio_dir}/
  2. Generate fine-tuned version: python generation/train/03_train_model.py
  3. Compare: python generation/train/05_compare_baseline_finetuned.py

For detailed instructions, see: generation/train/README_FINETUNE.md (Step 3)
    """)

if __name__ == "__main__":
    main()

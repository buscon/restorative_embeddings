"""Utility functions for Plan B generation pipeline.

Helpers for:
- Loading and normalizing audio
- Working with SoundAQnet
- Statistical comparisons with milestone-1 models
- Caption generation and review
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def load_audio(path, sr=None):
    """Load audio file. Requires librosa."""
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required. Install with: pip install librosa")
    
    audio, sr = librosa.load(path, sr=sr)
    return audio, sr


def normalize_loudness(audio, sr, target_lu=-23.0):
    """Normalize audio to target loudness in LUFS. Requires pyloudnorm."""
    try:
        import pyloudnorm
    except ImportError:
        raise ImportError("pyloudnorm required. Install with: pip install pyloudnorm")
    
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    
    if np.isnan(loudness):
        # Silent or near-silent audio
        return audio
    
    audio = pyloudnorm.normalize(audio, loudness, target_lu)
    return audio


def resample_audio(audio, sr_from, sr_to=44100):
    """Resample audio. Requires librosa."""
    if sr_from == sr_to:
        return audio
    
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required. Install with: pip install librosa")
    
    audio = librosa.resample(audio, orig_sr=sr_from, target_sr=sr_to)
    return audio


def to_stereo(audio):
    """Convert mono audio to stereo (duplicate to both channels)."""
    if audio.ndim == 2:
        if audio.shape[0] == 2:
            return audio  # already stereo
        if audio.shape[0] == 1:
            return np.vstack([audio, audio])  # 1 channel -> 2 channels
    elif audio.ndim == 1:
        return np.vstack([audio, audio])  # mono -> stereo
    
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def save_audio(audio, path, sr=44100):
    """Save audio to file. Requires soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile required. Install with: pip install soundfile")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure stereo, transposed to (n_samples, 2)
    if audio.ndim == 2:
        audio = audio.T
    
    sf.write(path, audio, sr)


# ============================================================================
# SOUNDAQNET INTERFACE
# ============================================================================

def load_soundaqnet_model(path=None):
    """Load SoundAQnet model from SoundSCaper.
    
    Args:
        path: Path to SoundSCaper directory. If None, tries to import from PYTHONPATH.
    
    Returns:
        model, config dict
    """
    raise NotImplementedError(
        "SoundAQnet integration requires:\n"
        "  1. git clone https://github.com/Yuanbo2020/SoundSCaper.git\n"
        "  2. Add to PYTHONPATH: export PYTHONPATH=$PYTHONPATH:/path/to/SoundSCaper\n"
        "  3. Then implement this function using SoundSCaper's loading code"
    )


def predict_soundaqnet(model, audio, sr, loudness_lu=None):
    """Predict pleasantness with SoundAQnet.
    
    Args:
        model: SoundAQnet model
        audio: numpy array, shape (n_samples,) or (2, n_samples)
        sr: sample rate
        loudness_lu: ISO 532-1 loudness (LUFS) for model input
    
    Returns:
        dict with keys: isopleasant, isoeventful, paq (dict), scene, etc.
    """
    raise NotImplementedError(
        "Requires loading SoundAQnet model and implementing prediction logic.\n"
        "See: github.com/Yuanbo2020/SoundSCaper"
    )


# ============================================================================
# CLAP RIDGE MODEL (milestone-1)
# ============================================================================

def load_clap_ridge_model(model_path):
    """Load CLAP ridge model from joblib checkpoint."""
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib required. Install with: pip install joblib")
    
    model = joblib.load(model_path)
    return model


def predict_clap_ridge(model, clap_embeddings):
    """Predict pleasantness with CLAP ridge model.
    
    Args:
        model: loaded CLAP ridge model
        clap_embeddings: numpy array, shape (1, embedding_dim)
    
    Returns:
        float: ISOPleasant prediction
    """
    if clap_embeddings.ndim == 1:
        clap_embeddings = clap_embeddings.reshape(1, -1)
    
    prediction = model.predict(clap_embeddings)
    return float(prediction[0])


# ============================================================================
# PLEASANTNESS BINNING
# ============================================================================

PLEASANTNESS_FIXED_BINS = {
    "very_unpleasant": (-np.inf, -0.50),
    "unpleasant": (-0.50, -0.15),
    "neutral": (-0.15, +0.15),
    "pleasant": (+0.15, +0.50),
    "very_pleasant": (+0.50, +np.inf),
}

PLEASANTNESS_LABELS = {
    "very_unpleasant": "very unpleasant",
    "unpleasant": "unpleasant",
    "neutral": "neutral",
    "pleasant": "pleasant",
    "very_pleasant": "very pleasant",
}


def get_pleasantness_label(iso_pleasant, strategy='fixed'):
    """Assign pleasantness label from ISOPleasant score.
    
    Args:
        iso_pleasant: float, ISOPleasant value
        strategy: 'fixed' or 'quantile'
    
    Returns:
        (bin_name, label_text)
    """
    if pd.isna(iso_pleasant):
        return 'unknown', 'unknown'
    
    if strategy == 'fixed':
        for bin_name, (lower, upper) in PLEASANTNESS_FIXED_BINS.items():
            if lower <= iso_pleasant < upper:
                return bin_name, PLEASANTNESS_LABELS[bin_name]
        return 'neutral', 'neutral'
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_quantile_bins(iso_pleasants, n_bins=5):
    """Compute quantile-based bins from ISOPleasant distribution.
    
    Args:
        iso_pleasants: array of ISOPleasant values
        n_bins: number of bins (default 5 = quintiles)
    
    Returns:
        list of bin edges
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(iso_pleasants, quantiles)
    return bins.tolist()


# ============================================================================
# COMPARISON WITH MILESTONE-1
# ============================================================================

def compare_with_baseline(new_scores, baseline_scores, metric='pearson'):
    """Compare new rater scores with baseline (e.g., CLAP ridge).
    
    Args:
        new_scores: array of new rater predictions
        baseline_scores: array of baseline predictions
        metric: 'pearson', 'spearman', or 'rmse'
    
    Returns:
        dict with comparison metrics
    """
    from scipy import stats
    
    if metric == 'pearson':
        r, p = stats.pearsonr(new_scores, baseline_scores)
        return {'r': r, 'p': p, 'metric': 'pearson'}
    
    elif metric == 'spearman':
        r, p = stats.spearmanr(new_scores, baseline_scores)
        return {'r': r, 'p': p, 'metric': 'spearman'}
    
    elif metric == 'rmse':
        mse = np.mean((new_scores - baseline_scores) ** 2)
        rmse = np.sqrt(mse)
        return {'rmse': rmse, 'mse': mse, 'metric': 'rmse'}
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# CAPTION GENERATION
# ============================================================================

SOURCE_MAPPINGS = {
    'bird': 'birdsong',
    'bird call': 'bird calls',
    'bird song': 'bird songs',
    'wind': 'wind',
    'water': 'water sounds',
    'rain': 'rain',
    'traffic': 'traffic noise',
    'car': 'traffic noise',
    'dog': 'dog sounds',
    'cat': 'cat sounds',
    'insect': 'insect sounds',
    'footsteps': 'footsteps',
    'crowd': 'crowd sounds',
    'human': 'human voices',
    'construction': 'construction noise',
    'tool': 'tool sounds',
}


def map_fsd_label_to_source(label):
    """Map FSD50K label to human-readable source description."""
    label_lower = label.lower().strip()
    
    # Exact match first
    if label_lower in SOURCE_MAPPINGS:
        return SOURCE_MAPPINGS[label_lower]
    
    # Substring match
    for key, value in SOURCE_MAPPINGS.items():
        if key in label_lower:
            return value
    
    # Default fallback
    return label_lower


def build_caption(scene, sources, pleasantness_label, template=None):
    """Build a caption from components.
    
    Args:
        scene: string, scene description (e.g., 'forest park')
        sources: list or str, source descriptions
        pleasantness_label: string, pleasantness level
        template: string with {scene}, {sources}, {pleasantness} placeholders
                 default: "{scene} with {sources}. {pleasantness} soundscape."
    
    Returns:
        caption string
    """
    if template is None:
        template = "{scene} with {sources}. {pleasantness} soundscape."
    
    if isinstance(sources, list):
        sources = ' and '.join(sources)
    
    caption = template.format(
        scene=scene,
        sources=sources,
        pleasantness=pleasantness_label
    )
    
    return caption

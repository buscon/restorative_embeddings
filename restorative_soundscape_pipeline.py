"""
Restorative Soundscape Discovery Pipeline

Workflow:
1. Load soundscapes from ISD dataset
2. Extract CLAP embeddings (GPU-accelerated)
3. k-means clustering for initial exploration
4. Extract acoustic features (librosa + soundscapy)
5. Cluster characterization
6. Stratified sampling for listening
7. (Later) HDBSCAN refinement
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
import soundscapy as sp
from tqdm import tqdm
import json
from collections import Counter

# ============================================================================
# 1. LOAD CLAP MODEL & PREPARE FOR EMBEDDINGS
# ============================================================================

def load_clap_model(device='cuda'):
    """Load pretrained CLAP model from HuggingFace"""
    from transformers import ClapModel, ClapProcessor
    
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.eval()
    
    return model, processor, device


# ============================================================================
# 2. AUDIO DATASET & EMBEDDING EXTRACTION
# ============================================================================

class SoundscapeDataset(Dataset):
    """Load soundscapes from ISD dataset"""
    
    def __init__(self, audio_dir, sr=48000, duration=10.0):
        """
        Args:
            audio_dir: Path to ISD audio files
            sr: Sample rate (CLAP expects 48kHz)
            duration: Clip duration in seconds (None = full audio)
        """
        self.audio_dir = Path(audio_dir)
        self.sr = sr
        self.duration = duration
        self.audio_files = list(self.audio_dir.glob('**/*.wav')) + \
                          list(self.audio_dir.glob('**/*.mp3'))
        print(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        filepath = self.audio_files[idx]
        
        try:
            waveform, sr = librosa.load(filepath, sr=self.sr)
            
            # Resample if needed
            if sr != self.sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)
            
            # Truncate to duration
            if self.duration is not None:
                max_samples = int(self.sr * self.duration)
                if len(waveform) > max_samples:
                    waveform = waveform[:max_samples]
                elif len(waveform) < max_samples:
                    waveform = np.pad(waveform, (0, max_samples - len(waveform)))
            
            return {
                'filepath': str(filepath),
                'waveform': torch.FloatTensor(waveform),
                'sr': self.sr
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None


def extract_embeddings_batch(audio_dir, model, processor, device, 
                            batch_size=16, sr=48000, duration=10.0):
    """
    Extract CLAP embeddings for all soundscapes (batched, GPU-efficient)
    
    Returns:
        embeddings: (N, 512) array
        filepaths: list of audio file paths
    """
    dataset = SoundscapeDataset(audio_dir, sr=sr, duration=duration)
    
    embeddings_list = []
    filepaths_list = []
    
    # Custom collate to handle None values
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return {
            'waveform': torch.stack([b['waveform'] for b in batch]),
            'filepath': [b['filepath'] for b in batch],
            'sr': batch[0]['sr']
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CLAP embeddings"):
            if batch is None:
                continue
            
            # Process audio with CLAP processor
            inputs = processor(
                audios=batch['waveform'].numpy(),
                sampling_rate=batch['sr'],
                return_tensors="pt"
            ).to(device)
            
            # Extract embeddings
            audio_embeds = model.get_audio_features(**inputs)
            
            embeddings_list.append(audio_embeds.cpu().numpy())
            filepaths_list.extend(batch['filepath'])
    
    embeddings = np.vstack(embeddings_list)
    return embeddings, filepaths_list


# ============================================================================
# 3. CLUSTERING: K-MEANS (EXPLORATORY)
# ============================================================================

def kmeans_clustering(embeddings, n_clusters=8, random_state=42):
    """Fit k-means to embeddings"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, 
                   n_init=10, verbose=1)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


# ============================================================================
# 4. ACOUSTIC FEATURE EXTRACTION
# ============================================================================

def extract_acoustic_features(audio_path, sr=48000):
    """
    Extract acoustic features using librosa + soundscapy
    
    Returns dict with:
    - Spectral features (centroid, flatness, contrast, MFCCs)
    - Temporal features (ZCR, onset density)
    - Soundscape indices (NDSI, bioacoustic index, etc.)
    - Loudness (LUFS approximation)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        features = {}
        
        # ---- Spectral Features ----
        S = librosa.magphase(librosa.stft(y))[0]
        
        features['spectral_centroid'] = librosa.feature.spectral_centroid(S=S, sr=sr)[0].mean()
        features['spectral_flatness'] = librosa.feature.spectral_flatness(S=S)[0].mean()
        features['spectral_contrast'] = librosa.feature.spectral_contrast(S=S, sr=sr)[0].mean()
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = mfcc.mean(axis=1)  # mean per coefficient
        
        # ---- Temporal Features ----
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y)[0].mean()
        
        # Onset detection (proxy for temporal activity)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features['onset_density'] = len(librosa.onset.onset_frames(onset_env=onset_env)) / len(y)
        
        # ---- Soundscape Indices (soundscapy) ----
        try:
            # NDSI: Normalized Difference Soundscape Index (natural vs anthropogenic)
            ndsi = sp.bioacoustics.bioacoustic_index(y, sr)
            features['ndsi'] = ndsi
            
            # Bioacoustic index
            bi = sp.bioacoustics.bioacoustic_index(y, sr)
            features['bioacoustic_index'] = bi
            
            # Acoustic complexity index
            aci = sp.bioacoustics.acoustic_complexity_index(y, sr)
            features['acoustic_complexity'] = aci
            
        except Exception as e:
            print(f"Soundscapy indices failed for {audio_path}: {e}")
            features['ndsi'] = np.nan
            features['bioacoustic_index'] = np.nan
            features['acoustic_complexity'] = np.nan
        
        # ---- Loudness ----
        # Approximate LUFS (RMS-based, not true loudness meter)
        rms = librosa.feature.rms(y=y)[0].mean()
        features['rms'] = rms
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def extract_features_for_dataset(filepaths, sr=48000):
    """Batch extract acoustic features for all soundscapes"""
    feature_list = []
    valid_filepaths = []
    
    for fp in tqdm(filepaths, desc="Extracting acoustic features"):
        feats = extract_acoustic_features(fp, sr=sr)
        if feats is not None:
            feature_list.append(feats)
            valid_filepaths.append(fp)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)
    features_df['filepath'] = valid_filepaths
    
    return features_df


# ============================================================================
# 5. CLUSTER CHARACTERIZATION
# ============================================================================

def characterize_clusters(embeddings, labels, features_df, filepaths, n_clusters=8):
    """
    Summarize acoustic properties per cluster
    
    Returns dict with per-cluster statistics
    """
    cluster_info = {}
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
        # Get filepaths for this cluster
        cluster_fps = [filepaths[i] for i in cluster_indices]
        
        # Get features for this cluster
        cluster_features = features_df.iloc[cluster_indices]
        
        # Summarize
        info = {
            'size': len(cluster_indices),
            'sample_filepaths': cluster_fps[:3],  # First 3 samples
            'spectral_centroid_mean': cluster_features['spectral_centroid'].mean(),
            'spectral_centroid_std': cluster_features['spectral_centroid'].std(),
            'spectral_flatness_mean': cluster_features['spectral_flatness'].mean(),
            'zero_crossing_rate_mean': cluster_features['zero_crossing_rate'].mean(),
            'onset_density_mean': cluster_features['onset_density'].mean(),
            'rms_mean': cluster_features['rms'].mean(),
        }
        
        # Soundscape indices (handle NaNs)
        if 'ndsi' in cluster_features.columns:
            valid_ndsi = cluster_features['ndsi'].dropna()
            if len(valid_ndsi) > 0:
                info['ndsi_mean'] = valid_ndsi.mean()
                info['ndsi_std'] = valid_ndsi.std()
        
        if 'bioacoustic_index' in cluster_features.columns:
            valid_bi = cluster_features['bioacoustic_index'].dropna()
            if len(valid_bi) > 0:
                info['bioacoustic_index_mean'] = valid_bi.mean()
        
        cluster_info[cluster_id] = info
    
    return cluster_info


# ============================================================================
# 6. STRATIFIED SAMPLING FOR LISTENING
# ============================================================================

def stratified_sample_for_listening(labels, filepaths, n_per_cluster=8, 
                                   random_state=42):
    """
    Stratified sampling: ensure representation from each cluster
    
    Returns list of filepaths to listen to, with cluster labels
    """
    np.random.seed(random_state)
    
    listening_samples = []
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Sample n_per_cluster from this cluster (or all if cluster is smaller)
        sample_size = min(n_per_cluster, len(cluster_indices))
        sampled_indices = np.random.choice(cluster_indices, size=sample_size, 
                                         replace=False)
        
        for idx in sampled_indices:
            listening_samples.append({
                'filepath': filepaths[idx],
                'cluster': cluster_id,
                'embedding_idx': idx
            })
    
    return listening_samples


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(audio_dir, output_dir='./restorative_soundscape_results', 
         n_clusters=8, batch_size=16):
    """
    Main pipeline orchestration
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load CLAP
    print("\n[1/6] Loading CLAP model...")
    model, processor, device = load_clap_model(device)
    
    # 2. Extract embeddings
    print("\n[2/6] Extracting CLAP embeddings...")
    embeddings, filepaths = extract_embeddings_batch(
        audio_dir, model, processor, device, 
        batch_size=batch_size, sr=48000, duration=10.0
    )
    print(f"Extracted {len(filepaths)} embeddings, shape: {embeddings.shape}")
    
    # 3. k-means clustering
    print(f"\n[3/6] k-means clustering (k={n_clusters})...")
    kmeans, labels = kmeans_clustering(embeddings, n_clusters=n_clusters)
    print(f"Cluster sizes: {Counter(labels)}")
    
    # 4. Extract acoustic features
    print("\n[4/6] Extracting acoustic features...")
    features_df = extract_features_for_dataset(filepaths, sr=48000)
    print(f"Features extracted: {features_df.shape[0]} samples, {features_df.shape[1]} features")
    
    # 5. Characterize clusters
    print("\n[5/6] Characterizing clusters...")
    cluster_info = characterize_clusters(
        embeddings, labels, features_df, filepaths, n_clusters=n_clusters
    )
    
    # Save cluster info
    with open(os.path.join(output_dir, 'cluster_characterization.json'), 'w') as f:
        # Convert to serializable format
        info_serializable = {}
        for cid, info in cluster_info.items():
            info_serializable[str(cid)] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in info.items()
            }
        json.dump(info_serializable, f, indent=2)
    
    print("Cluster characterization saved.")
    
    # 6. Stratified sampling for listening
    print("\n[6/6] Stratified sampling for listening...")
    listening_samples = stratified_sample_for_listening(
        labels, filepaths, n_per_cluster=8
    )
    
    # Save listening list
    listening_df = pd.DataFrame(listening_samples)
    listening_df.to_csv(os.path.join(output_dir, 'listening_samples.csv'), index=False)
    print(f"Sampled {len(listening_samples)} files for listening")
    print(f"Listening list saved to: {output_dir}/listening_samples.csv")
    
    # Save embeddings, labels, features for later use
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'labels_kmeans.npy'), labels)
    features_df.to_csv(os.path.join(output_dir, 'acoustic_features.csv'), index=False)
    
    print(f"\nAll results saved to: {output_dir}/")
    print("Next steps:")
    print("  1. Listen to samples in listening_samples.csv")
    print("  2. Annotate restorativeness & acoustic observations")
    print("  3. Run HDBSCAN on embeddings for refinement")
    print("  4. Design validation study based on your listening insights")
    
    return {
        'embeddings': embeddings,
        'labels': labels,
        'filepaths': filepaths,
        'features_df': features_df,
        'cluster_info': cluster_info,
        'listening_samples': listening_samples
    }


if __name__ == '__main__':
    # TODO: Update to your actual ISD data path
    audio_dir = '/path/to/isd/audio'
    results = main(audio_dir, n_clusters=8, batch_size=16)

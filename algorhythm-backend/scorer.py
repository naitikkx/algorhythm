import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import mahalanobis
from dna_builder import DNA_FEATURES

# Number of features determines expected Mahalanobis distance
N_FEATURES = len(DNA_FEATURES)

def score_song(song_features: dict, dna: dict) -> dict:
    """
    Scores a single song against a playlist DNA.
    
    Pipeline:
    1. Normalize the song using the DNA's scaler
    2. Cosine similarity (directional alignment / vibe check)
    3. Mahalanobis distance (statistical distance from cluster center)
    4. Isolation Forest (anomaly detection)
    5. Composite score → Add/Reject signal
    
    Returns a structured score report.
    """
    
    # 1. Extract song features
    song_vector_raw = np.array([song_features.get(f, 0) for f in DNA_FEATURES]).reshape(1, -1)
    
    # Normalize using DNA's scaler (for Mahalanobis + Isolation Forest)
    normalized_song = dna["scaler"].transform(song_vector_raw).flatten()
    
    # Raw vectors (for cosine similarity)
    raw_song = song_vector_raw.flatten()
    raw_mean = dna["raw_mean_vector"]
    
    # Normalized references (for Mahalanobis)
    mean_vector = dna["mean_vector"]
    inv_cov = dna["inv_cov_matrix"]
    
    # 2. Cosine Similarity — uses RAW values (actual feature magnitudes matter)
    cosine_dist = cosine(raw_song, raw_mean)
    cosine_sim = 1 - cosine_dist
    cosine_score = round(max(0, min(1, cosine_sim)), 4)
    
    # 3. Mahalanobis Distance (lower = closer to playlist center)
    maha_dist = mahalanobis(normalized_song, mean_vector, inv_cov)
    # In 9D space, expected distance ≈ √9 = 3.0 for samples from the same distribution
    # Scale so that: dist ≈ 3.0 → score ≈ 0.75, dist ≈ 5.0 → score ≈ 0.35
    # Using exp(-dist² / (2 * n_features)) — chi-squared aware scaling
    maha_score = round(np.exp(-(maha_dist ** 2) / (2 * N_FEATURES)), 4)
    
    # 4. Isolation Forest
    iso_raw_score = dna["isolation_forest"].score_samples(normalized_song.reshape(1, -1))[0]
    # sklearn IF scores: inliers ≈ -0.35 to -0.50, outliers ≈ -0.55 to -0.70+
    # Map [-0.70, -0.30] → [0, 1]  (linear rescale)
    iso_score = round(max(0, min(1, (iso_raw_score + 0.70) / 0.40)), 4)
    
    # 5. Composite Score (weighted blend)
    weights = {
        "cosine": 0.30,       # Vibe alignment
        "mahalanobis": 0.45,  # Statistical fit (most important)
        "isolation": 0.25     # Anomaly check
    }
    
    composite = (
        weights["cosine"] * cosine_score +
        weights["mahalanobis"] * maha_score +
        weights["isolation"] * iso_score
    )
    composite = round(composite, 4)
    
    # 6. Decision
    if composite >= 0.60:
        verdict = "✅ ADD"
    elif composite >= 0.40:
        verdict = "🟡 MAYBE"
    else:
        verdict = "❌ REJECT"
    
    return {
        "song": song_features.get("name", "Unknown"),
        "artist": song_features.get("artist", "Unknown"),
        "playlist": dna["playlist_name"],
        "scores": {
            "cosine_similarity": cosine_score,
            "mahalanobis_fit": maha_score,
            "isolation_forest": iso_score,
            "composite": composite,
        },
        "weights": weights,
        "verdict": verdict,
    }


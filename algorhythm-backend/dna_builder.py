import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# 9 continuous features used for mathematical DNA (key/mode excluded)
DNA_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def build_playlist_dna(df: pd.DataFrame, playlist_name: str = "Unknown") -> dict:
    """
    Takes a cleaned DataFrame and builds the playlist's mathematical DNA.
    
    Returns a structured DNA object containing:
    - Fitted scaler (for normalizing new songs the same way)
    - Mean vector (center of playlist cluster)
    - Covariance matrix (spread/shape)
    - Inverse covariance matrix (for Mahalanobis distance)
    - Fitted Isolation Forest (for anomaly detection)
    - Key/Mode distribution (categorical stats, kept separate)
    """
    
    if len(df) < 3:
        return {"error": f"Need at least 3 tracks to build DNA. Got {len(df)}."}
    
    # 1. Extract continuous features
    feature_data = df[DNA_FEATURES].copy()
    
    # 2. Store raw stats (mean, std, min, max per feature)
    raw_mean_vector = feature_data.mean().values
    raw_std_vector = feature_data.std().values
    raw_min_vector = feature_data.min().values
    raw_max_vector = feature_data.max().values
    
    # 3. Normalize (StandardScaler: zero mean, unit variance)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(feature_data)
    normalized_df = pd.DataFrame(normalized, columns=DNA_FEATURES)
    
    # 4. Compute normalized mean vector (center of playlist cluster)
    mean_vector = normalized_df.mean().values
    
    # 5. Compute covariance matrix with regularization
    # Small epsilon prevents singularity with few tracks
    cov_matrix = normalized_df.cov().values
    epsilon = 1e-6
    cov_matrix += np.eye(len(DNA_FEATURES)) * epsilon
    
    # 6. Compute inverse covariance (needed for Mahalanobis)
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if still singular
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        print("⚠️ Used pseudo-inverse (covariance was near-singular)")
    
    # 7. Fit Isolation Forest (learns what's "normal" for this playlist)
    iso_forest = IsolationForest(
        contamination=0.1,  # expect ~10% outliers
        random_state=42,
        n_estimators=100
    )
    iso_forest.fit(normalized)
    
    # 8. PCA — captures inter-feature correlations
    # If features co-vary (e.g. energy + loudness), fewer PCA components
    # explain the data → playlist is structured/cohesive even if raw ranges are wide
    n_components = min(len(DNA_FEATURES), len(df) - 1)  # Can't have more components than samples-1
    pca = PCA(n_components=n_components)
    pca.fit(normalized)
    
    explained_variance = pca.explained_variance_ratio_
    
    # How many components needed for 90% explained variance?
    cumulative = np.cumsum(explained_variance)
    dims_for_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    
    # Correlation matrix (shows which features co-vary)
    corr_matrix = normalized_df.corr().values
    
    # 9. Key/Mode distribution (categorical, kept separate)
    key_distribution = df['key'].value_counts().to_dict() if 'key' in df.columns else {}
    mode_distribution = df['mode'].value_counts().to_dict() if 'mode' in df.columns else {}
    
    dna = {
        "playlist_name": playlist_name,
        "track_count": len(df),
        "feature_columns": DNA_FEATURES,
        "scaler": scaler,
        "raw_mean_vector": raw_mean_vector,
        "raw_std_vector": raw_std_vector,
        "raw_min_vector": raw_min_vector,
        "raw_max_vector": raw_max_vector,
        "mean_vector": mean_vector,
        "cov_matrix": cov_matrix,
        "inv_cov_matrix": inv_cov_matrix,
        "isolation_forest": iso_forest,
        "pca": pca,
        "explained_variance": explained_variance,
        "dims_for_90": dims_for_90,
        "corr_matrix": corr_matrix,
        "key_distribution": key_distribution,
        "mode_distribution": mode_distribution,
    }
    
    print(f"🧬 DNA built for [{playlist_name}] from {len(df)} tracks")
    print(f"   📐 Feature space: {len(DNA_FEATURES)}D")
    print(f"   📊 Mean vector: {np.round(mean_vector, 3)}")
    
    return dna

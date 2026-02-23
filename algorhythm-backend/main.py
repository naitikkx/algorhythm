from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
from fetcher import fetch_playlist_data, fetch_track_features
from dna_builder import build_playlist_dna
from scorer import score_song
from neighborhood import build_artist_neighborhood, get_discovered_on, find_sonic_twins

app = FastAPI()

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active playlist session (single user, single playlist at a time)
active_session = {
    "playlist_id": None,
    "playlist_name": None,
    "data": None,  # pandas DataFrame
    "dna": None     # computed playlist DNA
}

class PlaylistRequest(BaseModel):
    url: str

class ExcludeRequest(BaseModel):
    positions: list[int]  # 1-indexed positions of tracks to remove

class ScoreRequest(BaseModel):
    url: str  # Spotify track URL

class NeighborhoodRequest(BaseModel):
    url: str  # Spotify artist URL
    max_artists: Optional[int] = 30

class DiscoveredOnRequest(BaseModel):
    url: str  # Spotify artist URL

class SonicTwinRequest(BaseModel):
    track_url: str   # Spotify track URL (the song to match)
    artist_url: str  # Spotify artist URL (neighborhood to search)
    top_n: Optional[int] = 5

@app.get("/")
def read_root():
    return {"status": "Algorhythm Engine is Running 🚀"}

@app.post("/analyze")
def analyze_playlist(request: PlaylistRequest):
    """
    Fetches playlist tracks + audio features, stores as active session.
    """
    result = fetch_playlist_data(request.url)
    
    if "error" in result:
        return result
    
    # Set as active session (reset DNA since playlist changed)
    active_session["playlist_id"] = result["playlist_id"]
    active_session["playlist_name"] = result["playlist_name"]
    active_session["data"] = result["data"]
    active_session["dna"] = None
    
    df = result["data"]
    records = df.to_dict(orient="records")
    for i, rec in enumerate(records):
        rec["position"] = i + 1
    
    return {
        "playlist_name": result["playlist_name"],
        "track_count": len(df),
        "data": records
    }

@app.post("/exclude")
def exclude_tracks(request: ExcludeRequest):
    """
    Removes tracks at specified positions from the active playlist.
    """
    if active_session["data"] is None:
        raise HTTPException(
            status_code=404,
            detail="No active playlist. Run /analyze first."
        )
    
    df = active_session["data"]
    playlist_name = active_session["playlist_name"]
    original_count = len(df)
    
    # Validate positions
    indices_to_drop = []
    for pos in request.positions:
        if pos < 1 or pos > len(df):
            raise HTTPException(
                status_code=400,
                detail=f"Position {pos} is out of range. Valid: 1-{len(df)}"
            )
        indices_to_drop.append(pos - 1)
    
    # Log what's being excluded
    for idx in sorted(indices_to_drop):
        row = df.iloc[idx]
        print(f"   🗑️ Removing #{idx+1}: {row['name']} - {row['artist']}")
    
    # Drop, re-index, and reset DNA (needs rebuild after exclusion)
    filtered_df = df.drop(df.index[indices_to_drop]).reset_index(drop=True)
    active_session["data"] = filtered_df
    active_session["dna"] = None  # Force rebuild
    
    excluded_count = original_count - len(filtered_df)
    print(f"📊 [{playlist_name}] Remaining: {len(filtered_df)} tracks")
    
    records = filtered_df.to_dict(orient="records")
    for i, rec in enumerate(records):
        rec["position"] = i + 1
    
    return {
        "playlist_name": playlist_name,
        "excluded_count": excluded_count,
        "remaining_count": len(filtered_df),
        "data": records
    }

@app.post("/build-dna")
def build_dna():
    """
    Computes playlist DNA from the current (possibly filtered) track data.
    Must run /analyze first, optionally /exclude, then this.
    """
    if active_session["data"] is None:
        raise HTTPException(
            status_code=404,
            detail="No active playlist. Run /analyze first."
        )
    
    df = active_session["data"]
    playlist_name = active_session["playlist_name"]
    
    dna = build_playlist_dna(df, playlist_name)
    
    if "error" in dna:
        return dna
    
    active_session["dna"] = dna
    
    # Build enriched response with full DNA signature
    features = dna["feature_columns"]
    mean = dna["raw_mean_vector"]
    std = dna["raw_std_vector"]
    mins = dna["raw_min_vector"]
    maxs = dna["raw_max_vector"]
    
    # Per-feature profile: mean, std, range
    feature_profile = {}
    for i, f in enumerate(features):
        feature_profile[f] = {
            "mean": round(float(mean[i]), 4),
            "std": round(float(std[i]), 4),
            "min": round(float(mins[i]), 4),
            "max": round(float(maxs[i]), 4),
            "range": round(float(maxs[i] - mins[i]), 4),
        }
    
    # ─── Playlist Cohesion (Conviction-based) ───
    # For each feature, compute how DEFINING it is for this playlist:
    #   Tightness: std relative to feature's natural range (tight = songs agree)
    #   Identity:  mean's distance from baseline "average music" (extreme = distinctive)
    # A feature that's tight AND/OR extreme = strong playlist fingerprint.
    
    # Natural ranges (theoretical span for each feature)
    FEATURE_RANGES = {
        "danceability": 1.0, "energy": 1.0, "loudness": 60.0,
        "speechiness": 1.0, "acousticness": 1.0, "instrumentalness": 1.0,
        "liveness": 1.0, "valence": 1.0, "tempo": 200.0
    }
    
    baselines = {
        "danceability": 0.55, "energy": 0.55, "loudness": -8.0,
        "speechiness": 0.08, "acousticness": 0.30, "instrumentalness": 0.10,
        "liveness": 0.18, "valence": 0.45, "tempo": 120.0
    }
    
    n_features = len(features)
    conviction_details = []
    
    for i, f in enumerate(features):
        natural_range = FEATURE_RANGES[f]
        
        # Tightness: std / natural_range → 0 = perfectly tight, 0.33+ = loose
        # Mapped to 0-1: tight (std=0) → 1.0, loose (std=range/3) → 0.0
        range_ratio = float(std[i]) / natural_range
        tightness = max(0.0, 1.0 - range_ratio * 3)
        
        # Identity: how far from baseline, relative to natural range
        deviation_abs = abs(float(mean[i]) - baselines[f])
        identity = min(deviation_abs / natural_range * 2.5, 1.0)
        
        # Conviction = whichever is stronger (tight cluster OR extreme position)
        conviction = max(tightness, identity)
        
        conviction_details.append({
            "feature": f,
            "tightness": round(tightness, 3),
            "identity": round(identity, 3),
            "conviction": round(conviction, 3),
        })
    
    avg_conviction = float(np.mean([c["conviction"] for c in conviction_details]))
    cohesion_score = round(avg_conviction * 100, 1)
    
    # PCA breakdown (kept for transparency, not used in score)
    explained_variance = dna["explained_variance"]
    dims_for_90 = dna["dims_for_90"]
    corr_matrix = dna["corr_matrix"]
    
    # Find correlated feature pairs
    corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r = corr_matrix[i][j]
            if abs(r) > 0.3:
                corr_pairs.append({
                    "feature_a": features[i],
                    "feature_b": features[j],
                    "correlation": round(float(r), 3),
                    "direction": "positive" if r > 0 else "inverse",
                })
    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    pca_breakdown = {
        "dims_for_90pct": dims_for_90,
        "total_dims": n_features,
        "top3_explained_pct": round(float(np.sum(explained_variance[:3])) * 100, 1),
    }
    
    # Dominant traits (features deviating >15% from baseline)
    dominant_traits = []
    for f in features:
        val = feature_profile[f]["mean"]
        baseline = baselines.get(f, 0.5)
        
        if f == "loudness":
            deviation = (baseline - val) / abs(baseline) if baseline != 0 else 0
        else:
            deviation = (val - baseline) / baseline if baseline != 0 else 0
        
        if abs(deviation) > 0.15:
            direction = "high" if deviation > 0 else "low"
            dominant_traits.append({
                "feature": f,
                "direction": direction,
                "value": val,
                "deviation_pct": round(deviation * 100, 1),
            })
    
    dominant_traits.sort(key=lambda x: abs(x["deviation_pct"]), reverse=True)
    
    return {
        "status": "🧬 DNA Built Successfully",
        "playlist_name": playlist_name,
        "track_count": dna["track_count"],
        "features_used": features,
        "feature_profile": feature_profile,
        "mean_vector": {f: round(v, 4) for f, v in zip(features, mean)},
        "std_vector": {f: round(v, 4) for f, v in zip(features, std)},
        "cohesion_score": cohesion_score,
        "conviction_breakdown": conviction_details,
        "pca_breakdown": pca_breakdown,
        "correlated_features": corr_pairs[:6],
        "dominant_traits": dominant_traits[:5],
        "key_distribution": dna["key_distribution"],
        "mode_distribution": dna["mode_distribution"],
    }

@app.post("/score")
def score_track(request: ScoreRequest):
    """
    Scores a single song against the active playlist DNA.
    Send a Spotify track URL.
    """
    if active_session["dna"] is None:
        raise HTTPException(
            status_code=404,
            detail="No DNA built yet. Run /build-dna first."
        )
    
    # Fetch the song's audio features
    song_data = fetch_track_features(request.url)
    
    if "error" in song_data:
        return song_data
    
    # Score against DNA
    result = score_song(song_data, active_session["dna"])
    return result


# ─────────────────────────────────────────────
# Phase 1: Neighborhood Mapping Endpoints
# ─────────────────────────────────────────────

@app.post("/neighborhood")
def get_neighborhood(request: NeighborhoodRequest):
    """
    Builds an artist neighborhood graph by crawling related artists.
    Returns a graph with artist nodes and their connections.
    """
    result = build_artist_neighborhood(
        artist_url=request.url,
        max_artists=request.max_artists
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/discovered-on")
def discovered_on(request: DiscoveredOnRequest):
    """
    Finds playlists where an artist is featured.
    Reverse-engineers the "Discovered On" section.
    """
    result = get_discovered_on(artist_url=request.url)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@app.post("/sonic-twins")
def sonic_twins(request: SonicTwinRequest):
    """
    Finds artists whose sound DNA is closest to a target track.
    Compares against related artists' top tracks in 9D feature space.
    """
    result = find_sonic_twins(
        track_url=request.track_url,
        artist_url=request.artist_url,
        top_n=request.top_n
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result
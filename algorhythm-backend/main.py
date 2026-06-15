from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
from fetcher import fetch_playlist_data, fetch_track_features
from dna_builder import build_playlist_dna
from scorer import score_song
from neighborhood import build_artist_neighborhood, get_discovered_on, find_sonic_twins
from lyrics_fetcher import fetch_playlist_lyrics
from sentiment import build_semantic_dna
from cache import (
    get_cached_playlist, save_playlist_cache,
    get_cached_dna, save_dna_cache,
    get_cached_semantic, save_semantic_cache,
)
import pandas as pd

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
    "dna": None,    # computed playlist DNA
    "semantic_dna": None,  # Phase 2: lyrics sentiment analysis
}

# Progress tracker for streaming UI updates
semantic_progress = {
    "running": False,
    "current": 0,
    "total": 0,
    "current_track": "",
    "status": "idle",  # idle | fetching_lyrics | analyzing | done | error
    "found": 0,
    "missing": 0,
    "gemini_batch": 0,
    "gemini_total": 0,
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
    Checks SQLite cache first to avoid re-burning API credits.
    """
    # Extract playlist ID for cache lookup
    try:
        playlist_id = request.url.split("/")[-1].split("?")[0]
    except IndexError:
        return {"error": "Invalid Playlist URL format"}
    
    # Check cache first
    cached = get_cached_playlist(playlist_id)
    if cached:
        df = pd.DataFrame(cached["track_data"])
        active_session["playlist_id"] = playlist_id
        active_session["playlist_name"] = cached["playlist_name"]
        active_session["data"] = df
        active_session["dna"] = None
        active_session["semantic_dna"] = None
        
        records = df.to_dict(orient="records")
        for i, rec in enumerate(records):
            rec["position"] = i + 1
        
        return {
            "playlist_name": cached["playlist_name"],
            "track_count": len(df),
            "data": records,
            "cached": True,
        }
    
    # No cache — fetch from APIs
    result = fetch_playlist_data(request.url)
    
    if "error" in result:
        return result
    
    # Set as active session (reset DNA since playlist changed)
    active_session["playlist_id"] = result["playlist_id"]
    active_session["playlist_name"] = result["playlist_name"]
    active_session["data"] = result["data"]
    active_session["dna"] = None
    active_session["semantic_dna"] = None
    
    # Save to cache
    save_playlist_cache(result["playlist_id"], result["playlist_name"], result["data"])
    
    df = result["data"]
    records = df.to_dict(orient="records")
    for i, rec in enumerate(records):
        rec["position"] = i + 1
    
    return {
        "playlist_name": result["playlist_name"],
        "track_count": len(df),
        "data": records,
        "cached": False,
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
    
    # Check cache first
    if active_session["playlist_id"]:
        cached = get_cached_dna(active_session["playlist_id"], df)
        if cached:
            # Still need to build the actual dna object for scorer
            dna = build_playlist_dna(df, playlist_name)
            if "error" not in dna:
                active_session["dna"] = dna
            cached["cached"] = True
            return cached
    
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
    
    dna_response = {
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
    
    # Save to cache
    if active_session["playlist_id"]:
        save_dna_cache(active_session["playlist_id"], df, dna_response)
    
    return dna_response

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


# ─────────────────────────────────────────────
# Phase 2: Semantic / NLP Analysis Endpoints
# ─────────────────────────────────────────────

@app.post("/build-semantic-dna")
def build_semantic():
    """
    Fetches lyrics for the active playlist and analyzes sentiment via Gemini.
    Must run /analyze first (and optionally /exclude).
    
    Pipeline: tracks → LRCLIB (lyrics) → Local Keyword Analyzer → Semantic DNA
    """
    if active_session["data"] is None:
        raise HTTPException(
            status_code=404,
            detail="No active playlist. Run /analyze first."
        )
    
    df = active_session["data"]
    playlist_name = active_session["playlist_name"]
    
    # Check cache first
    if active_session["playlist_id"]:
        cached = get_cached_semantic(active_session["playlist_id"], df)
        if cached:
            active_session["semantic_dna"] = cached.get("_semantic_dna_obj", cached)
            cached_response = {k: v for k, v in cached.items() if not k.startswith("_")}
            cached_response["cached"] = True
            return cached_response
    
    # Set up progress tracker
    semantic_progress["running"] = True
    semantic_progress["current"] = 0
    semantic_progress["total"] = len(df)
    semantic_progress["status"] = "fetching_lyrics"
    semantic_progress["found"] = 0
    semantic_progress["missing"] = 0
    semantic_progress["gemini_batch"] = 0
    semantic_progress["gemini_total"] = 0
    
    def on_lyrics_progress(current, total, track_name, status):
        semantic_progress["current"] = current
        semantic_progress["total"] = total
        semantic_progress["current_track"] = track_name
        if status == "found":
            semantic_progress["found"] += 1
        else:
            semantic_progress["missing"] += 1
    
    # 1. Fetch lyrics for all tracks (with progress)
    lyrics_result = fetch_playlist_lyrics(df, progress_callback=on_lyrics_progress)
    lyrics_dict = lyrics_result["lyrics"]
    
    if not lyrics_dict:
        semantic_progress["running"] = False
        semantic_progress["status"] = "error"
        return {
            "error": "Could not find lyrics for any tracks in this playlist.",
            "missing_lyrics": lyrics_result["missing"],
        }
    
    # 2. Build track info lookup
    semantic_progress["status"] = "analyzing"
    track_info = {}
    for _, row in df.iterrows():
        track_info[row["id"]] = {
            "name": row["name"],
            "artist": row["artist"],
        }
    
    def on_gemini_progress(phase, current, total, detail):
        semantic_progress["status"] = "analyzing"
        semantic_progress["gemini_batch"] = current
        semantic_progress["gemini_total"] = total
        semantic_progress["current_track"] = detail
    
    # 3. Run batch Gemini sentiment analysis
    semantic_dna = build_semantic_dna(lyrics_dict, track_info, playlist_name, progress_callback=on_gemini_progress)
    
    if "error" in semantic_dna:
        semantic_progress["running"] = False
        semantic_progress["status"] = "error"
        return semantic_dna
    
    # Store in session
    active_session["semantic_dna"] = semantic_dna
    
    # Build response
    response = {
        "status": "🧠 Semantic DNA Built Successfully",
        "playlist_name": playlist_name,
        "lyrics_coverage": {
            "total_tracks": lyrics_result["total"],
            "lyrics_found": lyrics_result["found"],
            "hit_rate": lyrics_result["hit_rate"],
        },
        "missing_lyrics": lyrics_result["missing"],
        "tracks_analyzed": semantic_dna["tracks_analyzed"],
        "semantic_cohesion": semantic_dna["semantic_cohesion"],
        "aggregate": semantic_dna["aggregate"],
        "track_sentiments": [
            {
                "track": s["track_name"],
                "artist": s["artist_name"],
                "mood": s["mood"],
                "themes": s["themes"],
                "valence": s["emotional_valence"],
                "energy": s["lyrical_energy"],
                "summary": s.get("summary", ""),
            }
            for s in semantic_dna.get("track_sentiments", [])
        ],
    }
    
    # If audio DNA also exists, compute combined cohesion
    if active_session["dna"] is not None:
        response["combined_available"] = True
        response["note"] = "Run /build-dna to get audio cohesion, then compare with semantic_cohesion"
    
    # Save to cache
    if active_session["playlist_id"]:
        cache_obj = dict(response)
        cache_obj["_semantic_dna_obj"] = semantic_dna
        save_semantic_cache(active_session["playlist_id"], df, cache_obj)
    
    # Mark progress as done
    semantic_progress["running"] = False
    semantic_progress["status"] = "done"
    
    return response

@app.get("/semantic-progress")
def get_semantic_progress():
    """Returns current progress of semantic DNA building (poll this from frontend)."""
    return dict(semantic_progress)

@app.post("/semantic-score")
def semantic_score_track(request: ScoreRequest):
    """
    Scores a single song against the active playlist's Semantic DNA.
    Send a Spotify track URL.
    """
    if active_session.get("semantic_dna") is None:
        raise HTTPException(
            status_code=404,
            detail="No Semantic DNA built yet. Run /build-semantic-dna first."
        )
    
    # Extract track ID and get metadata from Spotify directly (not Reccobeats)
    try:
        track_id = request.url.split("/")[-1].split("?")[0]
    except IndexError:
        return {"error": "Invalid Track URL format"}
    
    from auth import get_spotify_client
    try:
        sp = get_spotify_client()
        track_info = sp.track(track_id)
        track_name = track_info["name"]
        artist_name = track_info["artists"][0]["name"]
    except Exception as e:
        return {"error": f"Could not fetch track metadata: {str(e)}"}
    
    # Fetch lyrics
    from lyrics_fetcher import fetch_lyrics
    lyrics_res = fetch_lyrics(track_name, artist_name)
    
    if not lyrics_res.get("lyrics"):
        return {"error": f"No lyrics found for '{track_name}' by {artist_name}."}
        
    # Analyze sentiment
    from sentiment import analyze_track_sentiment
    sentiment = analyze_track_sentiment(lyrics_res["lyrics"], track_name, artist_name)
    
    if not sentiment:
        return {"error": "Failed to analyze sentiment with Gemini."}
        
    # Compare with semantic DNA
    semantic_dna = active_session["semantic_dna"]
    agg = semantic_dna["aggregate"]
    
    playlist_themes = [t[0].lower() for t in agg["dominant_themes"]]
    playlist_moods = [m[0].lower() for m in agg["dominant_moods"]]
    
    track_themes = [t.lower() for t in sentiment.get("themes", [])]
    track_moods = [m.lower() for m in sentiment.get("mood", [])]
    
    # Calculate overlaps (0 to 1) - how much of the track's themes match top playlist themes
    theme_overlap = len(set(track_themes) & set(playlist_themes)) / max(1, len(track_themes))
    mood_overlap = len(set(track_moods) & set(playlist_moods)) / max(1, len(track_moods))
    
    # Emotional distance
    v_diff = abs(sentiment["emotional_valence"] - agg["avg_valence"])
    e_diff = abs(sentiment["lyrical_energy"] - agg["avg_energy"])
    
    v_score = max(0, 1 - (v_diff / 2.0))  # Range is -1 to 1, max diff is 2
    e_score = max(0, 1 - e_diff)          # Range is 0 to 1, max diff is 1
    
    # Composite semantic score (0-100)
    cohesion_score = (
        theme_overlap * 40 +
        mood_overlap * 30 +
        v_score * 15 +
        e_score * 15
    )
    
    return {
        "track": track_name,
        "artist": artist_name,
        "semantic_score": round(cohesion_score, 1),
        "details": {
            "theme_overlap_pct": round(theme_overlap * 100, 1),
            "mood_overlap_pct": round(mood_overlap * 100, 1),
            "valence_match_pct": round(v_score * 100, 1),
            "energy_match_pct": round(e_score * 100, 1),
        },
        "track_sentiment": {
            "themes": track_themes,
            "moods": track_moods,
            "valence": sentiment["emotional_valence"],
            "energy": sentiment["lyrical_energy"],
            "summary": sentiment.get("summary", "")
        }
    }
"""
lyrics_fetcher.py — Lyrics Fetcher via LRCLIB

Free API, no key required, no rate limiting.
Fetches plain text lyrics for tracks by artist + title.
"""

import requests
import urllib.parse
import time


LRCLIB_BASE = "https://lrclib.net/api"


def fetch_lyrics(track_name: str, artist_name: str) -> dict:
    """
    Fetches lyrics for a single track from LRCLIB.
    
    Tries exact match first, falls back to search.
    
    Returns:
        {
            "track": str,
            "artist": str,
            "lyrics": str or None,
            "source": "lrclib" or None,
        }
    """
    result = {
        "track": track_name,
        "artist": artist_name,
        "lyrics": None,
        "source": None,
    }
    
    # 1. Try exact match
    try:
        params = {
            "artist_name": artist_name,
            "track_name": track_name,
        }
        resp = requests.get(f"{LRCLIB_BASE}/get", params=params, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            lyrics = data.get("plainLyrics") or data.get("syncedLyrics")
            if lyrics and len(lyrics.strip()) > 20:
                result["lyrics"] = lyrics.strip()
                result["source"] = "lrclib"
                return result
    except Exception as e:
        print(f"   ⚠️ LRCLIB exact match error: {e}")
    
    # 2. Fallback: search API
    try:
        query = f"{artist_name} {track_name}"
        resp = requests.get(
            f"{LRCLIB_BASE}/search",
            params={"q": query},
            timeout=10
        )
        
        if resp.status_code == 200:
            results = resp.json()
            if results and len(results) > 0:
                # Take first result with lyrics
                for item in results:
                    lyrics = item.get("plainLyrics") or item.get("syncedLyrics")
                    if lyrics and len(lyrics.strip()) > 20:
                        result["lyrics"] = lyrics.strip()
                        result["source"] = "lrclib"
                        return result
    except Exception as e:
        print(f"   ⚠️ LRCLIB search error: {e}")
    
    return result


def fetch_playlist_lyrics(tracks_df, progress_callback=None) -> dict:
    """
    Batch-fetches lyrics for all tracks in a playlist DataFrame.
    Checks SQLite lyrics cache first — only calls LRCLIB for uncached tracks.
    """
    from cache import get_cached_lyrics_batch, save_lyrics_cache
    
    lyrics_dict = {}
    missing = []
    total = len(tracks_df)
    
    # Check cache first
    track_ids = tracks_df["id"].tolist()
    cached_lyrics = get_cached_lyrics_batch(track_ids)
    
    cached_count = len(cached_lyrics)
    if cached_count > 0:
        print(f"⚡ Lyrics cache HIT: {cached_count}/{total} tracks")
        lyrics_dict.update(cached_lyrics)
    
    # Only fetch uncached tracks from LRCLIB
    uncached_df = tracks_df[~tracks_df["id"].isin(cached_lyrics.keys())]
    uncached_total = len(uncached_df)
    
    if uncached_total > 0:
        print(f"📝 Fetching lyrics for {uncached_total} uncached tracks (skipping {cached_count} cached)...")
    
    for idx, row in uncached_df.iterrows():
        track_name = row["name"]
        artist_name = row["artist"]
        track_id = row["id"]
        
        result = fetch_lyrics(track_name, artist_name)
        
        current = cached_count + len(lyrics_dict) - cached_count + len(missing) + 1
        
        if result["lyrics"]:
            lyrics_dict[track_id] = result["lyrics"]
            # Save to cache immediately
            save_lyrics_cache(track_id, track_name, artist_name, result["lyrics"])
            print(f"   ✅ [{current}/{total}] {track_name} - {artist_name}")
            if progress_callback:
                progress_callback(current, total, f"{track_name} - {artist_name}", "found")
        else:
            missing.append(f"{track_name} - {artist_name}")
            if progress_callback:
                progress_callback(current, total, f"{track_name} - {artist_name}", "missing")
        
        # Small delay to be nice to the API
        time.sleep(0.1)
    
    found = len(lyrics_dict)
    hit_rate = found / total if total > 0 else 0
    
    print(f"📊 Lyrics fetched: {found}/{total} ({hit_rate:.0%} hit rate) — {cached_count} from cache")
    if missing:
        print(f"   ❌ Missing lyrics for {len(missing)} tracks")
    
    return {
        "lyrics": lyrics_dict,
        "hit_rate": round(hit_rate, 3),
        "total": total,
        "found": found,
        "missing": missing,
        "from_cache": cached_count,
    }

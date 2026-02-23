"""
neighborhood.py — Artist Neighborhood Mapping Engine

Phase 1 of the Music Intelligence Framework.
Builds the "Map of the Musical Neighborhood":
  1. Artist Neighborhood (genre-based search — related_artists is deprecated)
  2. Discovered On playlists (reverse-engineer entry points)
  3. Sonic Twins (find artists with closest audio DNA)

Note: Spotify deprecated the related_artists endpoint in Nov 2024 for
Client Credentials apps. We use genre-based search as the alternative.
"""

import numpy as np
import requests
from auth import get_spotify_client
from dna_builder import DNA_FEATURES

RECCOBEATS_URL = "https://api.reccobeats.com/v1/audio-features"

sp = get_spotify_client()


# ─────────────────────────────────────────────
# 1. Artist Neighborhood (Genre-Based Search)
# ─────────────────────────────────────────────

def build_artist_neighborhood(artist_url: str, max_artists: int = 30) -> dict:
    """
    Builds an artist neighborhood using genre-based search.
    
    Since Spotify deprecated related_artists (Nov 2024), we:
    1. Get the target artist's genres
    2. Search for other artists in the same genre(s)
    3. Filter by similar popularity range
    4. Return a neighborhood of comparable artists
    
    Returns:
        {
            "root": { id, name, genres, popularity, followers },
            "neighbors": [ { id, name, genres, popularity, followers } ],
            "total_found": int
        }
    """
    artist_id = artist_url.split("/")[-1].split("?")[0]
    
    print(f"🗺️  Building neighborhood for artist: {artist_id}")
    
    # 1. Get root artist info
    try:
        root_info = sp.artist(artist_id)
    except Exception as e:
        return {"error": f"Failed to fetch artist: {str(e)}"}
    
    root = {
        "id": root_info["id"],
        "name": root_info["name"],
        "genres": root_info.get("genres", []),
        "popularity": root_info.get("popularity", 0),
        "followers": root_info.get("followers", {}).get("total", 0),
        "image": root_info["images"][0]["url"] if root_info.get("images") else None,
    }
    
    print(f"   🎤 Root: {root['name']} | Genres: {root['genres']} | Pop: {root['popularity']}")
    
    # 2. Search for similar artists by genre
    genres = root["genres"]
    if not genres:
        # Fallback: search by artist name to find co-occurring artists
        genres = [root["name"].lower()]
    
    neighbors = []
    seen_ids = {artist_id}  # Don't include self
    pop = root["popularity"]
    
    # Multi-strategy search to find comparable artists
    search_queries = []
    
    # Strategy 1: Genre-based searches
    for genre in genres[:3]:
        search_queries.append(f'genre:"{genre}"')
    
    # Strategy 2: Artist name search (finds co-occurring artists)
    search_queries.append(root["name"])
    
    # Strategy 3: Genre + keywords for discovery
    genre_keywords = {
        "rap": ["hip hop", "trap", "r&b"],
        "pop": ["dance pop", "electropop", "indie pop"],
        "rock": ["alternative rock", "indie rock", "punk"],
        "hip hop": ["rap", "trap", "conscious hip hop"],
        "r&b": ["neo soul", "contemporary r&b", "hip hop"],
        "trap": ["rap", "hip hop", "drill"],
        "indie": ["alternative", "indie rock", "indie pop"],
    }
    for genre in genres[:2]:
        for kw in genre_keywords.get(genre, []):
            search_queries.append(f'genre:"{kw}"')
    
    # Remove duplicates while preserving order
    seen_queries = set()
    unique_queries = []
    for q in search_queries:
        if q not in seen_queries:
            seen_queries.add(q)
            unique_queries.append(q)
    
    all_candidates = []
    
    for query in unique_queries:
        try:
            results = sp.search(q=query, type="artist", limit=50)
            artists = results.get("artists", {}).get("items", [])
            
            for a in artists:
                if a["id"] in seen_ids:
                    continue
                seen_ids.add(a["id"])
                
                a_pop = a.get("popularity", 0)
                a_genres = a.get("genres", [])
                
                # Skip artists with 0 popularity (too small to be useful)
                if a_pop < 10:
                    continue
                
                # Compute genre overlap score
                shared_genres = set(a_genres) & set(genres)
                genre_score = len(shared_genres) / max(len(genres), 1)
                
                # Popularity proximity (0 = identical, 1 = far apart)
                pop_distance = abs(a_pop - pop) / 100
                
                # Combined relevance score (higher = better match)
                relevance = (genre_score * 0.6) + ((1 - pop_distance) * 0.4)
                
                all_candidates.append({
                    "id": a["id"],
                    "name": a["name"],
                    "genres": a_genres,
                    "popularity": a_pop,
                    "followers": a.get("followers", {}).get("total", 0),
                    "image": a["images"][0]["url"] if a.get("images") else None,
                    "matched_genre": query.replace('genre:"', '').replace('"', ''),
                    "relevance_score": round(relevance, 3),
                })
                    
        except Exception as e:
            print(f"   ⚠️ Search error for '{query}': {e}")
            continue
    
    # Sort by relevance score (best matches first), then cap
    all_candidates.sort(key=lambda a: a["relevance_score"], reverse=True)
    neighbors = all_candidates[:max_artists]
    
    # Sort by popularity (closest to target first)
    neighbors.sort(key=lambda a: abs(a["popularity"] - pop))
    
    print(f"✅ Neighborhood complete: {len(neighbors)} artists found")
    for i, n in enumerate(neighbors[:5], 1):
        print(f"   {i}. {n['name']} (pop: {n['popularity']}, genre: {n['matched_genre']})")
    
    return {
        "root": root,
        "neighbors": neighbors,
        "total_found": len(neighbors),
    }


# ─────────────────────────────────────────────
# 2. Discovered On — Playlist Reverse Engineering
# ─────────────────────────────────────────────

def get_discovered_on(artist_url: str, limit: int = 20) -> dict:
    """
    Finds playlists where an artist is featured.
    
    Since Spotify's API doesn't expose "Discovered On" directly,
    we search for playlists containing the artist and filter by:
    - Editorial playlists (owned by 'spotify')
    - High-follower algorithmic playlists
    - User-curated discovery playlists
    
    Returns:
        {
            "artist": { id, name },
            "playlists": [ { id, name, owner, description, followers, is_editorial } ],
            "entry_points": [ top playlists by authority ]
        }
    """
    artist_id = artist_url.split("/")[-1].split("?")[0]
    
    try:
        artist_info = sp.artist(artist_id)
        artist_name = artist_info["name"]
    except Exception as e:
        return {"error": f"Failed to fetch artist: {str(e)}"}
    
    print(f"🔍 Finding playlists for: {artist_name}")
    
    # Search for playlists containing this artist
    playlists = []
    seen_ids = set()
    
    # Multiple search queries to maximize coverage
    search_queries = [
        artist_name,
        f"{artist_name} mix",
        f"{artist_name} radio",
    ]
    
    for query in search_queries:
        try:
            results = sp.search(q=query, type="playlist", limit=limit)
            items = results.get("playlists", {}).get("items", [])
            
            for item in items:
                if not item or item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])
                
                owner = item.get("owner", {})
                owner_id = owner.get("id", "")
                
                playlists.append({
                    "id": item["id"],
                    "name": item["name"],
                    "owner": owner.get("display_name", owner_id),
                    "owner_id": owner_id,
                    "description": item.get("description", ""),
                    "is_editorial": owner_id == "spotify",
                    "url": item.get("external_urls", {}).get("spotify", ""),
                })
                
        except Exception as e:
            print(f"   ⚠️ Search error for '{query}': {e}")
            continue
    
    # Fetch follower counts for each playlist (requires individual calls)
    for pl in playlists:
        try:
            full_playlist = sp.playlist(pl["id"], fields="followers")
            pl["followers"] = full_playlist.get("followers", {}).get("total", 0)
        except Exception:
            pl["followers"] = 0
    
    # Sort by authority: editorial first, then by followers
    playlists.sort(key=lambda p: (p["is_editorial"], p["followers"]), reverse=True)
    
    # Top entry points = editorial + high follower playlists
    entry_points = [
        p for p in playlists
        if p["is_editorial"] or p["followers"] > 1000
    ][:10]
    
    print(f"✅ Found {len(playlists)} playlists, {len(entry_points)} entry points")
    
    return {
        "artist": {"id": artist_id, "name": artist_name},
        "playlists": playlists,
        "entry_points": entry_points,
        "total_found": len(playlists),
    }


# ─────────────────────────────────────────────
# 3. Sonic Twins — DNA-Based Artist Matching
# ─────────────────────────────────────────────

def find_sonic_twins(track_url: str, artist_url: str, top_n: int = 5) -> dict:
    """
    Finds artists whose sound is closest to a target track.
    
    Pipeline:
    1. Get target track's audio features
    2. Build neighborhood via genre search (since related_artists is deprecated)
    3. For each neighbor, fetch their top tracks' audio features
    4. Compute Euclidean distance in 9D DNA space
    5. Return the top_n closest "Sonic Twins"
    
    Returns:
        {
            "target": { name, artist, features },
            "twins": [ { artist, avg_distance, similarity_pct, closest_tracks } ]
        }
    """
    track_id = track_url.split("/")[-1].split("?")[0]
    artist_id = artist_url.split("/")[-1].split("?")[0]
    
    # 1. Get target track features
    print(f"🎯 Fetching target track features: {track_id}")
    
    try:
        track_info = sp.track(track_id)
        target_name = track_info["name"]
        target_artist = track_info["artists"][0]["name"]
    except Exception as e:
        return {"error": f"Failed to fetch track: {str(e)}"}
    
    # Get audio features from Reccobeats
    url = f"{RECCOBEATS_URL}?ids={track_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": f"Reccobeats returned {response.status_code} for target track"}
    
    content = response.json().get("content", [])
    if not content:
        return {"error": f"No audio features found for '{target_name}'"}
    
    target_features = content[0]
    target_vector = np.array([target_features.get(f, 0) for f in DNA_FEATURES])
    
    print(f"   🎵 Target: {target_name} - {target_artist}")
    
    # 2. Get neighborhood artists (using genre search since related_artists is deprecated)
    print(f"🔎 Building neighborhood for comparison...")
    
    neighborhood = build_artist_neighborhood(artist_url, max_artists=20)
    
    if "error" in neighborhood:
        return {"error": f"Failed to build neighborhood: {neighborhood['error']}"}
    
    neighbor_artists = neighborhood.get("neighbors", [])
    
    if not neighbor_artists:
        return {"error": "No neighboring artists found for comparison"}
    
    print(f"   📊 Comparing against {len(neighbor_artists)} neighborhood artists...")
    
    # 3. For each neighbor, get top tracks and compute distance
    artist_distances = []
    
    for artist in neighbor_artists:
        try:
            # Get top tracks
            top_tracks = sp.artist_top_tracks(artist["id"], country="US")
            tracks = top_tracks.get("tracks", [])[:5]  # Top 5 tracks
            
            if not tracks:
                continue
            
            track_ids = [t["id"] for t in tracks]
            
            # Fetch audio features via Reccobeats
            batch_url = f"{RECCOBEATS_URL}?ids={','.join(track_ids)}"
            batch_response = requests.get(batch_url)
            
            if batch_response.status_code != 200:
                continue
            
            batch_content = batch_response.json().get("content", [])
            
            if not batch_content:
                continue
            
            # Compute distances for each track
            distances = []
            track_names = []
            
            for feat in batch_content:
                feat_vector = np.array([feat.get(f, 0) for f in DNA_FEATURES])
                dist = np.linalg.norm(target_vector - feat_vector)
                distances.append(dist)
                
                # Find matching track name
                spotify_url = feat.get("href", "")
                feat_track_id = spotify_url.split("/")[-1]
                matching = [t for t in tracks if t["id"] == feat_track_id]
                track_name = matching[0]["name"] if matching else "Unknown"
                track_names.append({"name": track_name, "distance": round(dist, 4)})
            
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            
            # Convert distance to similarity percentage
            # Use exponential decay: closer = higher %
            # Typical raw distance range is 0-200 (tempo dominates)
            similarity_pct = round(np.exp(-avg_distance / 100) * 100, 2)
            
            artist_distances.append({
                "artist_id": artist["id"],
                "artist_name": artist["name"],
                "genres": artist.get("genres", []),
                "popularity": artist.get("popularity", 0),
                "image": artist.get("image"),
                "avg_distance": round(avg_distance, 4),
                "min_distance": round(min_distance, 4),
                "similarity_pct": similarity_pct,
                "tracks_analyzed": len(distances),
                "closest_tracks": sorted(track_names, key=lambda x: x["distance"])[:3],
            })
            
            print(f"   🎤 {artist['name']}: avg_dist={avg_distance:.2f}, sim={similarity_pct:.1f}%")
            
        except Exception as e:
            print(f"   ⚠️ Error processing {artist.get('name', '?')}: {e}")
            continue
    
    # 4. Sort by distance (ascending = most similar first)
    artist_distances.sort(key=lambda x: x["avg_distance"])
    twins = artist_distances[:top_n]
    
    print(f"\n🏆 Top {len(twins)} Sonic Twins:")
    for i, twin in enumerate(twins, 1):
        print(f"   {i}. {twin['artist_name']} — {twin['similarity_pct']}% match")
    
    return {
        "target": {
            "name": target_name,
            "artist": target_artist,
            "track_id": track_id,
        },
        "twins": twins,
        "total_compared": len(artist_distances),
    }

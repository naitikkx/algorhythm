import pandas as pd
import requests
from auth import get_spotify_client
from spotipy.exceptions import SpotifyException

RECCOBEATS_URL = "https://api.reccobeats.com/v1/audio-features"

sp = get_spotify_client()

def fetch_playlist_data(playlist_url: str):
    """
    Fetches all tracks from a playlist and returns a clean DataFrame.
    Includes Error Handling to catch bad URLs.
    """
    print(f"🔹 Analyzing URL: {playlist_url}")

    # 1. Extract Playlist ID
    try:
        playlist_id = playlist_url.split("/")[-1].split("?")[0]
    except IndexError:
        return {"error": "Invalid Playlist URL format"}

    print(f"📥 Fetching Playlist ID: {playlist_id}...")
    
    # 2. Fetch Tracks (Protected Block)
    try:
        playlist_info = sp.playlist(playlist_id, fields="name")
        playlist_name = playlist_info.get("name", "Unknown Playlist")
        
        results = sp.playlist_tracks(playlist_id)
        tracks = results['items']
        
        while results['next']:
            results = sp.next(results)
            tracks.extend(results['items'])
            
        print(f"✅ [{playlist_name}] Found {len(tracks)} raw tracks.")
        
    except SpotifyException as e:
        print(f"❌ Spotify API Error: {e}")
        return {"error": f"Spotify refused access. Check if playlist is Public. (Status: {e.http_status})"}
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return {"error": str(e)}

    # 3. Extract IDs
    track_ids = []
    track_meta = []
    
    for item in tracks:
        track = item.get('track')
        if not track or track['id'] is None:
            continue
        track_ids.append(track['id'])
        track_meta.append({
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "id": track['id'],
            "popularity": track['popularity']
        })

    # 4. Fetch Audio Features via Reccobeats API (batches of 20)
    audio_features = []
    print(f"🧬 Extracting DNA for {len(track_ids)} songs via Reccobeats...")
    
    try:
        for i in range(0, len(track_ids), 20):
            batch = track_ids[i:i + 20]
            url = f"{RECCOBEATS_URL}?ids={','.join(batch)}"
            response = requests.get(url)
            
            if response.status_code != 200:
                print(f"   ⚠️ Batch {i//20 + 1}: API returned {response.status_code}, skipping...")
                continue
                
            data = response.json().get("content", [])
            
            # Map Reccobeats 'href' back to Spotify track ID for merging
            for item in data:
                spotify_url = item.get("href", "")
                item["id"] = spotify_url.split("/")[-1]
            
            audio_features.extend(data)
            print(f"   📦 Batch {i//20 + 1}: got {len(data)}/{len(batch)} tracks")
            
    except Exception as e:
        print(f"❌ Crash during DNA Extraction: {e}")
        return {"error": "Failed to fetch audio features. See terminal for details."}

    # 5. Merge Data
    df_meta = pd.DataFrame(track_meta)
    df_features = pd.DataFrame(audio_features)
    
    full_df = pd.merge(df_meta, df_features, on='id', how='inner')
    
    key_columns = [
        'name', 'artist', 'id', 'danceability', 'energy', 'key', 
        'loudness', 'mode', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    clean_df = full_df[key_columns]
    
    print(f"✅ Successfully processed {len(clean_df)} tracks.")
    return {
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "data": clean_df
    }


def fetch_track_features(track_url: str) -> dict:
    """
    Fetches audio features for a single track.
    Used for scoring a song against a playlist DNA.
    """
    # 1. Extract Track ID
    try:
        track_id = track_url.split("/")[-1].split("?")[0]
    except IndexError:
        return {"error": "Invalid Track URL format"}
    
    print(f"🔍 Fetching features for track: {track_id}")
    
    # 2. Get track metadata from Spotify
    try:
        track_info = sp.track(track_id)
        track_name = track_info['name']
        track_artist = track_info['artists'][0]['name']
        print(f"   🎵 {track_name} - {track_artist}")
    except Exception as e:
        print(f"❌ Error fetching track metadata: {e}")
        return {"error": f"Could not fetch track: {str(e)}"}
    
    # 3. Get audio features from Reccobeats
    url = f"{RECCOBEATS_URL}?ids={track_id}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": f"Reccobeats returned {response.status_code} for track {track_id}"}
    
    content = response.json().get("content", [])
    
    if not content:
        return {"error": f"No audio features found for '{track_name}' in Reccobeats database"}
    
    features = content[0]
    features["name"] = track_name
    features["artist"] = track_artist
    features["id"] = track_id
    
    print(f"   ✅ Features retrieved successfully")
    return features
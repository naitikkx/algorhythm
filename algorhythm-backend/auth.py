import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_spotify_client():
    """
    Returns an authenticated Spotify client.
    Uses Client Credentials flow (best for server-side data fetching).
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("❌ Missing Spotify Credentials in .env file")

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    print("✅ Spotify Client Authenticated")
    return sp
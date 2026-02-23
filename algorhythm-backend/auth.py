import os
import ssl
import requests
from requests.adapters import HTTPAdapter

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─── SSL Fix for Sophos Firewall (MITM) ───
# This machine has Sophos doing SSL interception.
# Sophos replaces real certs with its own CA that Python doesn't trust.
# Disable SSL verification for dev environment.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class NoSSLVerifyAdapter(HTTPAdapter):
    """Requests adapter that disables SSL verification."""
    def send(self, *args, **kwargs):
        kwargs['verify'] = False
        return super().send(*args, **kwargs)


def get_spotify_client():
    """
    Returns an authenticated Spotify client.
    Uses Client Credentials flow (best for server-side data fetching).
    SSL verification disabled to work around Sophos firewall MITM.
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("❌ Missing Spotify Credentials in .env file")

    # Create a session that skips SSL verification
    session = requests.Session()
    session.mount("https://", NoSSLVerifyAdapter())

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    # Inject our no-verify session into the auth manager
    auth_manager._session = session

    sp = spotipy.Spotify(
        auth_manager=auth_manager,
        requests_session=session
    )
    print("✅ Spotify Client Authenticated (SSL verify=off for Sophos)")
    return sp
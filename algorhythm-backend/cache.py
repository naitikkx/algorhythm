"""
cache.py — SQLite Cache Layer for Algorhythm

Caches playlist data, DNA results, and semantic analysis
to avoid re-burning API credits on repeated analysis.

Tables:
  - playlist_cache: raw track data (Spotify + Reccobeats)
  - dna_cache: computed DNA response
  - semantic_cache: semantic DNA response (LRCLIB + Gemini)

Cache key: playlist_id
Expiry: 7 days
"""

import sqlite3
import json
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "algorhythm_cache.db")
CACHE_EXPIRY_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates cache tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS playlist_cache (
            playlist_id TEXT PRIMARY KEY,
            playlist_name TEXT,
            track_data TEXT,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS dna_cache (
            playlist_id TEXT PRIMARY KEY,
            track_hash TEXT,
            dna_response TEXT,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS semantic_cache (
            playlist_id TEXT PRIMARY KEY,
            track_hash TEXT,
            semantic_response TEXT,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS lyrics_cache (
            track_id TEXT PRIMARY KEY,
            track_name TEXT,
            artist_name TEXT,
            lyrics TEXT,
            created_at REAL
        );
    """)
    conn.commit()
    conn.close()
    print("📦 Cache DB initialized")


def _is_expired(created_at):
    return (time.time() - created_at) > CACHE_EXPIRY_SECONDS


def _track_hash(df):
    """Generate a hash from track IDs to detect if exclusions changed the subset."""
    ids = sorted(df["id"].tolist())
    return hash(tuple(ids))


# ─── Playlist Cache (Spotify + Reccobeats data) ───

def get_cached_playlist(playlist_id: str):
    """Returns cached (playlist_name, track_records_list) or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM playlist_cache WHERE playlist_id = ?", (playlist_id,)
    ).fetchone()
    conn.close()

    if row and not _is_expired(row["created_at"]):
        print(f"⚡ Cache HIT: playlist {playlist_id}")
        return {
            "playlist_name": row["playlist_name"],
            "track_data": json.loads(row["track_data"]),
        }
    return None


def save_playlist_cache(playlist_id: str, playlist_name: str, df):
    """Saves playlist track data to cache."""
    records = df.to_dict(orient="records")
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO playlist_cache 
           (playlist_id, playlist_name, track_data, created_at) 
           VALUES (?, ?, ?, ?)""",
        (playlist_id, playlist_name, json.dumps(records), time.time())
    )
    conn.commit()
    conn.close()
    print(f"💾 Cached playlist: {playlist_name} ({len(records)} tracks)")


# ─── DNA Cache ───

def get_cached_dna(playlist_id: str, df):
    """Returns cached DNA response dict or None. Checks track_hash to handle exclusions."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM dna_cache WHERE playlist_id = ?", (playlist_id,)
    ).fetchone()
    conn.close()

    if row and not _is_expired(row["created_at"]):
        if row["track_hash"] == str(_track_hash(df)):
            print(f"⚡ Cache HIT: DNA for {playlist_id}")
            return json.loads(row["dna_response"])
    return None


def save_dna_cache(playlist_id: str, df, dna_response: dict):
    """Saves DNA response to cache."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO dna_cache 
           (playlist_id, track_hash, dna_response, created_at) 
           VALUES (?, ?, ?, ?)""",
        (playlist_id, str(_track_hash(df)), json.dumps(dna_response), time.time())
    )
    conn.commit()
    conn.close()
    print(f"💾 Cached DNA for playlist {playlist_id}")


# ─── Semantic Cache ───

def get_cached_semantic(playlist_id: str, df):
    """Returns cached semantic DNA response dict or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM semantic_cache WHERE playlist_id = ?", (playlist_id,)
    ).fetchone()
    conn.close()

    if row and not _is_expired(row["created_at"]):
        if row["track_hash"] == str(_track_hash(df)):
            print(f"⚡ Cache HIT: Semantic DNA for {playlist_id}")
            return json.loads(row["semantic_response"])
    return None


def save_semantic_cache(playlist_id: str, df, semantic_response: dict):
    """Saves semantic DNA response to cache."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO semantic_cache 
           (playlist_id, track_hash, semantic_response, created_at) 
           VALUES (?, ?, ?, ?)""",
        (playlist_id, str(_track_hash(df)), json.dumps(semantic_response), time.time())
    )
    conn.commit()
    conn.close()
    print(f"💾 Cached Semantic DNA for playlist {playlist_id}")


# ─── Lyrics Cache (per-track, never expires) ───

def get_cached_lyrics(track_id: str):
    """Returns cached lyrics string or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT lyrics FROM lyrics_cache WHERE track_id = ?", (track_id,)
    ).fetchone()
    conn.close()
    if row and row["lyrics"]:
        return row["lyrics"]
    return None


def get_cached_lyrics_batch(track_ids: list) -> dict:
    """Returns {track_id: lyrics} for all cached tracks."""
    conn = _get_conn()
    placeholders = ",".join("?" for _ in track_ids)
    rows = conn.execute(
        f"SELECT track_id, lyrics FROM lyrics_cache WHERE track_id IN ({placeholders})",
        track_ids
    ).fetchall()
    conn.close()
    return {row["track_id"]: row["lyrics"] for row in rows if row["lyrics"]}


def save_lyrics_cache(track_id: str, track_name: str, artist_name: str, lyrics: str):
    """Saves lyrics for a single track."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO lyrics_cache 
           (track_id, track_name, artist_name, lyrics, created_at) 
           VALUES (?, ?, ?, ?, ?)""",
        (track_id, track_name, artist_name, lyrics, time.time())
    )
    conn.commit()
    conn.close()


# Initialize on import
init_db()

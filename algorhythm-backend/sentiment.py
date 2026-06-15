"""
sentiment.py — Batch Gemini Lyrics Sentiment Analyzer

Sends batches of 10 songs per Gemini API call to stay within
the free tier rate limit (5 req/min). For 70 tracks, that's
~7 calls over ~90 seconds instead of 70 calls over 14 minutes.

Extracts per-track:
  - Mood tags (energetic, melancholic, uplifting, etc.)
  - Themes (love, empowerment, loss, nature, etc.)
  - Emotional valence (-1 to +1)
  - Lyrical energy (0 to 1)
"""

import os
import json
import time
import numpy as np
from collections import Counter
from dotenv import load_dotenv

# ─── Gemini Setup ───

def _get_gemini_model():
    """Lazy-load Gemini model."""
    import google.generativeai as genai
    
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not set in .env")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model


# ─── Batch Prompt ───

BATCH_PROMPT = """Analyze the following {count} songs based on their lyrics. For EACH song, return a JSON object.

Return a JSON array of objects, one per song, in the SAME ORDER as listed below.
Each object must have these fields:
1. "mood" — array of 2-4 mood tags (e.g., "uplifting", "melancholic", "energetic", "romantic", "aggressive", "peaceful", "nostalgic", "defiant", "spiritual", "playful", "motivational")
2. "themes" — array of 2-4 theme tags (e.g., "love", "empowerment", "heartbreak", "party", "struggle", "nature", "faith", "ambition", "freedom", "friendship", "patriotism")
3. "emotional_valence" — float from -1.0 (very negative/dark) to +1.0 (very positive/bright). 0 = neutral.
4. "lyrical_energy" — float from 0.0 (calm/introspective) to 1.0 (intense/passionate)
5. "language" — detected language of lyrics (e.g., "english", "hindi", "spanish")
6. "summary" — one sentence describing the song's lyrical essence

Return ONLY valid JSON array, no markdown, no code blocks, just the raw JSON array.

{songs}
"""


def _build_batch_prompt(tracks_batch: list) -> str:
    """Builds a prompt with multiple songs."""
    songs_text = ""
    for i, (track_id, info, lyrics) in enumerate(tracks_batch, 1):
        # Truncate lyrics to ~1500 chars each to fit within token limits
        truncated = lyrics[:1500] if len(lyrics) > 1500 else lyrics
        songs_text += f"\n--- Song {i}: \"{info['name']}\" by {info['artist']} ---\n{truncated}\n"
    
    return BATCH_PROMPT.format(count=len(tracks_batch), songs=songs_text)


def _parse_batch_response(text: str, batch_size: int) -> list:
    """Parse Gemini's batch response into a list of sentiment dicts."""
    # Clean up response
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    
    results = json.loads(text)
    
    if not isinstance(results, list):
        results = [results]
    
    # Validate and clamp values
    cleaned = []
    for r in results:
        r["emotional_valence"] = max(-1, min(1, float(r.get("emotional_valence", 0))))
        r["lyrical_energy"] = max(0, min(1, float(r.get("lyrical_energy", 0.5))))
        r["mood"] = r.get("mood", [])[:4]
        r["themes"] = r.get("themes", [])[:4]
        r["language"] = r.get("language", "unknown")
        r["summary"] = r.get("summary", "")
        cleaned.append(r)
    
    return cleaned


# ─── Single Track Analysis (for /semantic-score) ───

SINGLE_PROMPT = """Analyze the following song lyrics and return a JSON object with these fields:

1. "mood" — array of 2-4 mood tags (e.g., "uplifting", "melancholic", "energetic", "romantic", "aggressive", "peaceful", "nostalgic", "defiant", "spiritual", "playful", "motivational")
2. "themes" — array of 2-4 theme tags (e.g., "love", "empowerment", "heartbreak", "party", "struggle", "nature", "faith", "ambition", "freedom", "friendship")
3. "emotional_valence" — float from -1.0 (very negative/dark) to +1.0 (very positive/bright). 0 = neutral.
4. "lyrical_energy" — float from 0.0 (calm/introspective) to 1.0 (intense/passionate)
5. "language" — detected language of lyrics (e.g., "english", "hindi", "spanish")
6. "summary" — one sentence describing the song's lyrical essence

Return ONLY valid JSON, no markdown, no code blocks.

Song: "{track_name}" by {artist_name}

Lyrics:
{lyrics}
"""


def analyze_track_sentiment(lyrics: str, track_name: str, artist_name: str) -> dict:
    """Analyzes a single track's lyrics via Gemini. Used for /semantic-score."""
    try:
        model = _get_gemini_model()
        truncated = lyrics[:3000] if len(lyrics) > 3000 else lyrics
        
        prompt = SINGLE_PROMPT.format(
            track_name=track_name,
            artist_name=artist_name,
            lyrics=truncated
        )
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        
        result = json.loads(text)
        result["emotional_valence"] = max(-1, min(1, float(result.get("emotional_valence", 0))))
        result["lyrical_energy"] = max(0, min(1, float(result.get("lyrical_energy", 0.5))))
        result["mood"] = result.get("mood", [])[:4]
        result["themes"] = result.get("themes", [])[:4]
        
        return result
        
    except Exception as e:
        print(f"   ⚠️ Gemini analysis failed for '{track_name}': {e}")
        return None


# ─── Playlist Semantic DNA (Batch) ───

BATCH_SIZE = 10  # songs per Gemini call
RATE_LIMIT_PAUSE = 15  # seconds to wait between batches (free tier: 5 req/min)


def build_semantic_dna(lyrics_dict: dict, track_info: dict, playlist_name: str, progress_callback=None) -> dict:
    """
    Builds semantic DNA using batch Gemini calls.
    Sends 10 songs per API call to stay within rate limits.
    """
    print(f"🧠 Building semantic DNA for [{playlist_name}] via batch Gemini...")
    
    # Prepare batches
    items = []
    for track_id, lyrics in lyrics_dict.items():
        info = track_info.get(track_id, {"name": "Unknown", "artist": "Unknown"})
        items.append((track_id, info, lyrics))
    
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    total_batches = len(batches)
    
    track_sentiments = []
    all_moods = []
    all_themes = []
    valences = []
    energies = []
    languages = {}
    
    model = _get_gemini_model()
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"   📦 Batch {batch_idx}/{total_batches} ({len(batch)} songs)...")
        
        if progress_callback:
            progress_callback(
                "gemini", 
                batch_idx, 
                total_batches,
                f"Gemini batch {batch_idx}/{total_batches}"
            )
        
        prompt = _build_batch_prompt(batch)
        
        try:
            response = model.generate_content(prompt)
            results = _parse_batch_response(response.text, len(batch))
            
            # Map results back to tracks
            for i, (track_id, info, _lyrics) in enumerate(batch):
                if i < len(results):
                    sentiment = results[i]
                    sentiment["track_id"] = track_id
                    sentiment["track_name"] = info["name"]
                    sentiment["artist_name"] = info["artist"]
                    track_sentiments.append(sentiment)
                    
                    all_moods.extend(sentiment.get("mood", []))
                    all_themes.extend(sentiment.get("themes", []))
                    valences.append(sentiment["emotional_valence"])
                    energies.append(sentiment["lyrical_energy"])
                    
                    lang = sentiment.get("language", "unknown").lower()
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    print(f"      ✅ {info['name']}: {sentiment['mood'][:2]}")
                    
        except json.JSONDecodeError as e:
            print(f"      ⚠️ Batch {batch_idx} returned invalid JSON: {e}")
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"      ⏳ Rate limited on batch {batch_idx}. Waiting 30s...")
                time.sleep(30)
                # Retry once
                try:
                    response = model.generate_content(prompt)
                    results = _parse_batch_response(response.text, len(batch))
                    for i, (track_id, info, _lyrics) in enumerate(batch):
                        if i < len(results):
                            sentiment = results[i]
                            sentiment["track_id"] = track_id
                            sentiment["track_name"] = info["name"]
                            sentiment["artist_name"] = info["artist"]
                            track_sentiments.append(sentiment)
                            all_moods.extend(sentiment.get("mood", []))
                            all_themes.extend(sentiment.get("themes", []))
                            valences.append(sentiment["emotional_valence"])
                            energies.append(sentiment["lyrical_energy"])
                            lang = sentiment.get("language", "unknown").lower()
                            languages[lang] = languages.get(lang, 0) + 1
                            print(f"      ✅ {info['name']}: {sentiment['mood'][:2]}")
                except Exception as retry_e:
                    print(f"      ❌ Batch {batch_idx} retry failed: {retry_e}")
            else:
                print(f"      ❌ Batch {batch_idx} failed: {e}")
        
        # Rate limit pause between batches (skip on last batch)
        if batch_idx < total_batches:
            print(f"      ⏳ Pausing {RATE_LIMIT_PAUSE}s for rate limit...")
            time.sleep(RATE_LIMIT_PAUSE)
    
    if not track_sentiments:
        return {"error": "No tracks could be analyzed by Gemini", "tracks_analyzed": 0}
    
    # ─── Aggregate ───
    mood_counts = Counter(m.lower() for m in all_moods)
    theme_counts = Counter(t.lower() for t in all_themes)
    n_tracks = len(track_sentiments)
    
    dominant_moods = sorted(
        [(mood, round(count / n_tracks, 3)) for mood, count in mood_counts.items()],
        key=lambda x: x[1], reverse=True
    )
    
    dominant_themes = sorted(
        [(theme, round(count / n_tracks, 3)) for theme, count in theme_counts.items()],
        key=lambda x: x[1], reverse=True
    )
    
    cohesion = compute_semantic_cohesion(
        mood_counts, theme_counts, n_tracks, valences, energies
    )
    
    aggregate = {
        "dominant_moods": dominant_moods[:8],
        "dominant_themes": dominant_themes[:8],
        "avg_valence": round(float(np.mean(valences)), 3),
        "valence_std": round(float(np.std(valences)), 3),
        "avg_energy": round(float(np.mean(energies)), 3),
        "energy_std": round(float(np.std(energies)), 3),
        "languages": languages,
    }
    
    print(f"✅ Semantic DNA built: {n_tracks} tracks analyzed")
    print(f"   🎭 Top moods: {[m[0] for m in dominant_moods[:3]]}")
    print(f"   📖 Top themes: {[t[0] for t in dominant_themes[:3]]}")
    print(f"   💡 Cohesion: {cohesion}")
    
    return {
        "playlist_name": playlist_name,
        "tracks_analyzed": n_tracks,
        "track_sentiments": track_sentiments,
        "aggregate": aggregate,
        "semantic_cohesion": cohesion,
    }


def compute_semantic_cohesion(
    mood_counts: dict, theme_counts: dict,
    n_tracks: int, valences: list, energies: list
) -> float:
    """Computes semantic cohesion (0-100)."""
    if n_tracks < 2:
        return 50.0
    
    theme_freqs = sorted(theme_counts.values(), reverse=True)
    top3_theme_coverage = sum(theme_freqs[:3]) / (n_tracks * 3)
    theme_concentration = min(top3_theme_coverage * 2, 1.0)
    
    mood_freqs = sorted(mood_counts.values(), reverse=True)
    top3_mood_coverage = sum(mood_freqs[:3]) / (n_tracks * 2)
    mood_concentration = min(top3_mood_coverage * 2, 1.0)
    
    valence_tightness = max(0, 1 - float(np.std(valences)) * 2)
    energy_tightness = max(0, 1 - float(np.std(energies)) * 2)
    emotional_consistency = (valence_tightness + energy_tightness) / 2
    
    cohesion = (
        theme_concentration * 40 +
        mood_concentration * 35 +
        emotional_consistency * 25
    )
    
    return round(min(100, max(0, cohesion)), 1)

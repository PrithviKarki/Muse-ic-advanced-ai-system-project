"""
YouTube Music Google Takeout Importer & Metadata Enricher for Muse-ic 2.0.

Reads Google Takeout playlist CSVs (`Video ID,Playlist Video Creation Timestamp`)
from `data/youtube_music/`, resolves track titles and artists via YouTube's
public oEmbed endpoint (with deterministic local fallback), infers audio telemetry
and RAG descriptions from the playlist taxonomy, and exports an enriched catalog.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

YTM_DIR = Path(__file__).resolve().parent.parent / "data" / "youtube_music"
CACHE_JSON = YTM_DIR / "resolved_ytm_tracks.json"

# Playlist-level musical profile priors based on user's actual YouTube Music playlists
PLAYLIST_PROFILES: Dict[str, Dict[str, Any]] = {
    "Japanese City Pop": {
        "genre": "city pop",
        "mood": "groovy",
        "base_energy": 0.76,
        "base_bpm": 116,
        "base_valence": 0.86,
        "base_danceability": 0.84,
        "base_acousticness": 0.24,
        "desc_template": "80s Japanese City Pop groove featuring slap bass, lush Rhodes synth-brass chords, and breezy nocturnal Tokyo highway nostalgia.",
        "themes": "japanese city pop, 80s retro, night drive, groovy, upbeat, funk, synthwave, pop, nostalgic, tokyo",
    },
    "Sophisti-pop": {
        "genre": "sophisti-pop",
        "mood": "relaxed",
        "base_energy": 0.58,
        "base_bpm": 102,
        "base_valence": 0.72,
        "base_danceability": 0.68,
        "base_acousticness": 0.48,
        "desc_template": "Polished sophisti-pop arrangement blending smooth jazz saxophone, velvet electric piano chords, and refined melodic hooks.",
        "themes": "sophisti-pop, smooth jazz, lounge, relaxed, chill, date night, romantic, evening, indie pop, r&b",
    },
    "Nepali favs": {
        "genre": "nepali indie",
        "mood": "nostalgic",
        "base_energy": 0.52,
        "base_bpm": 94,
        "base_valence": 0.66,
        "base_danceability": 0.56,
        "base_acousticness": 0.74,
        "desc_template": "Heartfelt Nepali folk-pop and indie acoustic balladry rich with warm guitar strumming, soulful vocals, and Himalayan nostalgic warmth.",
        "themes": "nepali favs, nepali indie, folk, acoustic, nostalgic, chill, peaceful, heartfelt, storytelling, travel",
    },
    "Nepali hype": {
        "genre": "nepali rock",
        "mood": "intense",
        "base_energy": 0.88,
        "base_bpm": 134,
        "base_valence": 0.80,
        "base_danceability": 0.79,
        "base_acousticness": 0.18,
        "desc_template": "High-energy Nepali hype anthem combining driving percussion, electric rock/hip-hop festival bounce, and crowd-chant momentum.",
        "themes": "nepali hype, workout, party, intense, happy, energetic, rock, hip hop, festive, celebration",
    },
    "Tasteful hindi": {
        "genre": "hindi indie",
        "mood": "romantic",
        "base_energy": 0.48,
        "base_bpm": 88,
        "base_valence": 0.68,
        "base_danceability": 0.58,
        "base_acousticness": 0.76,
        "desc_template": "Soulful Hindi indie and poetic Bollywood acoustic melody featuring intimate vocals, warm strings, and late-evening emotional depth.",
        "themes": "tasteful hindi, bollywood acoustic, romantic, soulful, relaxed, melancholic, peaceful, late night, love",
    },
}


def _clean_yt_title(raw_title: str, raw_author: str) -> Tuple[str, str]:
    """Clean YouTube video title and channel name into (Song Title, Artist)."""
    artist = re.sub(r"\s*-\s*Topic$", "", raw_author or "").strip()
    title = (raw_title or "").strip()

    # Strip common YouTube suffixes like (Official Video), [Official Audio], (Lyrics), etc.
    title = re.sub(
        r"\s*[\(\[](Official\s+[^\)\]]*|Lyric[^\)\]]*|Audio[^\)\]]*|HD|4K| HQ|Video|Full\s+Song[^\)\]]*)[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()

    # If title is formatted as "Artist - Song Title", split cleanly
    if " - " in title:
        parts = [p.strip() for p in title.split(" - ", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            artist = parts[0]
            title = parts[1]

    return title or "Untitled Track", artist or "Unknown Artist"


def fetch_oembed_metadata(video_id: str) -> Optional[Tuple[str, str]]:
    """Resolve (title, artist) for a YouTube Video ID via public oEmbed JSON."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed_url = f"https://www.youtube.com/oembed?url={urllib.parse.quote(watch_url, safe='')}&format=json"
    try:
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return _clean_yt_title(data.get("title", ""), data.get("author_name", ""))
    except Exception:
        return None


def _deterministic_jitter(video_id: str, base_val: float, spread: float = 0.08, low: float = 0.05, high: float = 0.98) -> float:
    """Add deterministic per-track variation to audio attributes based on video_id hash."""
    digest = int(hashlib.md5(video_id.encode("utf-8")).hexdigest()[:6], 16)
    offset = ((digest % 200) / 100.0 - 1.0) * spread
    return round(max(low, min(high, base_val + offset)), 2)


def build_ytm_catalog(resolve_online: bool = False, max_per_playlist: int = 60) -> List[Dict[str, Any]]:
    """
    Parse all `*-videos.csv` files in `data/youtube_music/`, resolve titles/artists,
    and return a list of Muse-ic song dictionaries.
    """
    cached_lookup: Dict[str, Dict[str, str]] = {}
    if CACHE_JSON.exists():
        try:
            with open(CACHE_JSON, "r", encoding="utf-8") as f:
                cached_lookup = json.load(f)
        except Exception:
            cached_lookup = {}

    tracks: List[Dict[str, Any]] = []
    seen_video_ids = set()
    next_id = 101

    if not YTM_DIR.exists():
        return tracks

    csv_files = sorted(YTM_DIR.glob("*-videos.csv"))
    updated_cache = False

    for csv_file in csv_files:
        playlist_name = csv_file.name.replace("-videos.csv", "").strip()
        profile = PLAYLIST_PROFILES.get(
            playlist_name,
            {
                "genre": "indie pop",
                "mood": "chill",
                "base_energy": 0.55,
                "base_bpm": 100,
                "base_valence": 0.65,
                "base_danceability": 0.65,
                "base_acousticness": 0.50,
                "desc_template": f"Curated track from YouTube Music playlist '{playlist_name}'.",
                "themes": f"{playlist_name.lower()}, youtube music, personal library",
            },
        )

        count_in_playlist = 0
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = (row.get("Video ID") or "").strip()
                if not vid or vid in seen_video_ids:
                    continue
                if count_in_playlist >= max_per_playlist:
                    break

                seen_video_ids.add(vid)
                count_in_playlist += 1

                if vid in cached_lookup:
                    title = cached_lookup[vid]["title"]
                    artist = cached_lookup[vid]["artist"]
                elif resolve_online:
                    resolved = fetch_oembed_metadata(vid)
                    if resolved:
                        title, artist = resolved
                        cached_lookup[vid] = {"title": title, "artist": artist, "playlist": playlist_name}
                        updated_cache = True
                    else:
                        title = f"{playlist_name} Track ({vid[:6]})"
                        artist = f"{playlist_name} Artist"
                else:
                    title = f"{playlist_name} Track ({vid[:6]})"
                    artist = f"{playlist_name} Artist"

                energy = _deterministic_jitter(vid + "e", profile["base_energy"], 0.09)
                valence = _deterministic_jitter(vid + "v", profile["base_valence"], 0.09)
                dance = _deterministic_jitter(vid + "d", profile["base_danceability"], 0.08)
                acoustic = _deterministic_jitter(vid + "a", profile["base_acousticness"], 0.10)
                bpm_jitter = int(_deterministic_jitter(vid + "b", 0.5, 0.4, 0.1, 0.9) * 24) - 12
                bpm = max(60, min(180, profile["base_bpm"] + bpm_jitter))

                tracks.append(
                    {
                        "id": next_id,
                        "title": title,
                        "artist": artist,
                        "genre": profile["genre"],
                        "mood": profile["mood"],
                        "energy": energy,
                        "tempo_bpm": float(bpm),
                        "valence": valence,
                        "danceability": dance,
                        "acousticness": acoustic,
                        "description": f"[{playlist_name} Playlist] {profile['desc_template']}",
                        "themes": f"{profile['themes']}, {artist.lower()}, {title.lower()}",
                        "youtube_url": f"https://music.youtube.com/watch?v={vid}",
                        "playlist": playlist_name,
                    }
                )
                next_id += 1

    if updated_cache:
        try:
            with open(CACHE_JSON, "w", encoding="utf-8") as f:
                json.dump(cached_lookup, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    return tracks


if __name__ == "__main__":
    resolved_tracks = build_ytm_catalog(resolve_online=True, max_per_playlist=60)
    print(f"Resolved {len(resolved_tracks)} YouTube Music tracks across your playlists.")
    for t in resolved_tracks[:20]:
        print(f"  • [{t['playlist']}] {t['title']} — {t['artist']} ({t['genre']} / {t['mood']}) -> {t['youtube_url']}")

"""
YouTube Music Google Takeout Importer & Metadata Enricher for Muse-ic 2.0.

Reads Google Takeout playlist CSVs (`Video ID,Playlist Video Creation Timestamp`)
from `data/youtube_music/`, resolves track titles and artists via YouTube's
public oEmbed endpoint (cached in `data/youtube_music/resolved_ytm_tracks.json`),
enriches audio telemetry and RAG descriptions from the playlist taxonomy, and
writes the project's primary `data/songs.csv` catalog exclusively from the user's
YouTube Music playlists.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YTM_DIR = DATA_DIR / "youtube_music"
CACHE_JSON = YTM_DIR / "resolved_ytm_tracks.json"
SONGS_CSV = DATA_DIR / "songs.csv"

# Playlist-level musical profile priors based on user's actual YouTube Music playlists
PLAYLIST_PROFILES: Dict[str, Dict[str, Any]] = {
    "Japanese City Pop": {
        "genre": "city pop",
        "moods": ["groovy", "happy", "nostalgic", "moody"],
        "base_energy": 0.78,
        "base_bpm": 118,
        "base_valence": 0.86,
        "base_danceability": 0.85,
        "base_acousticness": 0.22,
        "desc_template": "80s Japanese City Pop groove featuring slap bass, lush Rhodes synth-brass chords, and breezy nocturnal Tokyo highway nostalgia.",
        "themes": "japanese city pop, 80s retro, night drive, groovy, upbeat, funk, synthwave, pop, nostalgic, tokyo, cardio, party",
    },
    "Sophisti-pop": {
        "genre": "sophisti-pop",
        "moods": ["relaxed", "romantic", "chill", "focused"],
        "base_energy": 0.44,
        "base_bpm": 96,
        "base_valence": 0.72,
        "base_danceability": 0.68,
        "base_acousticness": 0.68,
        "desc_template": "Polished sophisti-pop arrangement blending smooth jazz saxophone, velvet electric piano chords, and refined late-night lounge warmth.",
        "themes": "sophisti-pop, smooth jazz, lounge, coffee, study, coding, relaxed, chill, date night, romantic, evening, piano, saxophone",
    },
    "Nepali favs": {
        "genre": "nepali indie",
        "moods": ["nostalgic", "chill", "peaceful", "melancholic", "focused"],
        "base_energy": 0.42,
        "base_bpm": 88,
        "base_valence": 0.58,
        "base_danceability": 0.54,
        "base_acousticness": 0.78,
        "desc_template": "Heartfelt Nepali folk-pop and indie acoustic balladry rich with warm guitar strumming, unplugged vocals, and Himalayan rainy-night nostalgia.",
        "themes": "nepali favs, nepali indie, folk, acoustic, guitar, piano, rain, rainy night, study, coding, pomodoro, debugging, chill, peaceful, nostalgic, melancholic, introspection",
    },
    "Nepali hype": {
        "genre": "nepali rock",
        "moods": ["intense", "festive", "confident", "happy"],
        "base_energy": 0.90,
        "base_bpm": 136,
        "base_valence": 0.82,
        "base_danceability": 0.82,
        "base_acousticness": 0.16,
        "desc_template": "High-voltage Nepali hype anthem combining driving percussion, electric rock and hip-hop bounce, and peak workout/party adrenaline.",
        "themes": "nepali hype, nepali rock, workout, gym, cardio, party, intense, aggressive, happy, energetic, rock, hip hop, festive, celebration",
    },
    "Tasteful hindi": {
        "genre": "hindi indie",
        "moods": ["romantic", "melancholic", "peaceful", "relaxed", "sorrowful"],
        "base_energy": 0.38,
        "base_bpm": 84,
        "base_valence": 0.52,
        "base_danceability": 0.52,
        "base_acousticness": 0.80,
        "desc_template": "Soulful Hindi indie and poetic Bollywood acoustic melody featuring intimate vocals, warm strings, and late-night emotional introspection.",
        "themes": "tasteful hindi, hindi indie, bollywood acoustic, romantic, soulful, relaxed, melancholic, sorrowful, existential, peaceful, late night, rain, sleep, meditation, guitar, piano",
    },
}

# Specific track overrides for standout songs in the user's YouTube Music library
TRACK_SPECIFIC_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "u6y5NSiRPOw": {  # Plastic Love — Mariya Takeuchi
        "mood": "groovy",
        "energy": 0.78,
        "tempo_bpm": 106.0,
        "acousticness": 0.24,
        "description": "[Japanese City Pop] Iconic 1984 Tokyo City Pop masterpiece by Mariya Takeuchi driven by infectious slap-bass funk, shimmering Rhodes chords, and neon night-drive euphoria.",
    },
    "_sOKkON_UnQ": {  # 4:00 A.M. — Taeko Ohnuki
        "mood": "moody",
        "energy": 0.68,
        "tempo_bpm": 102.0,
        "acousticness": 0.35,
        "description": "[Japanese City Pop] Sweeping nocturnal City Pop arrangement by Taeko Ohnuki capturing 4:00 A.M. rain-slicked Tokyo streets with lush strings and analog synth warmth.",
    },
    "ZS6p32gX_yk": {  # Stay With Me — Miki Matsubara
        "mood": "groovy",
        "energy": 0.82,
        "tempo_bpm": 114.0,
        "acousticness": 0.22,
        "description": "[Japanese City Pop] Soaring disco-funk City Pop classic by Miki Matsubara featuring brassy horn stabs, upbeat groove, and timeless 80s retro sparkle.",
    },
    "SLbvyz_Hk9E": {  # Paradise — Sade Adu
        "mood": "chill",
        "energy": 0.36,
        "tempo_bpm": 82.0,
        "acousticness": 0.76,
        "description": "[Sophisti-pop] Velvet sophisti-pop groove by Sade Adu pairing warm basslines, brushed percussion, and calm late-night coding & lounge focus.",
    },
    "mRNcPbCJNJ8": {  # Smooth Operator — Sade Adu
        "mood": "relaxed",
        "energy": 0.42,
        "tempo_bpm": 92.0,
        "acousticness": 0.72,
        "description": "[Sophisti-pop] Signature sophisti-pop classic by Sade Adu featuring silky tenor saxophone solos, Rhodes piano chords, and effortless coffee-lounge cool.",
    },
    "gelwwYuYGn0": {  # Just the Two of Us — Grover Washington, Jr.
        "mood": "romantic",
        "energy": 0.46,
        "tempo_bpm": 96.0,
        "acousticness": 0.74,
        "description": "[Sophisti-pop] Warm jazz-soul classic by Grover Washington, Jr. with Fender Rhodes electric piano, rain-on-windowpane lyricism, and smooth saxophone.",
    },
    "0lPdE7F31yI": {  # Deorali Darah — Bipul Chettri
        "mood": "peaceful",
        "energy": 0.30,
        "tempo_bpm": 76.0,
        "acousticness": 0.90,
        "description": "[Nepali favs] Intimate Himalayan acoustic folk storytelling by Bipul Chettri built on delicate fingerpicked guitar, mountain breeze calm, and peaceful study focus.",
    },
    "cwxiCWAI7wc": {  # Gauthali — Pahenlo Batti Muni
        "mood": "melancholic",
        "energy": 0.34,
        "tempo_bpm": 74.0,
        "acousticness": 0.86,
        "description": "[Nepali favs] Contemplative Nepali indie ballad by Pahenlo Batti Muni exploring existential introspection, somber late-night stillness, and warm acoustic guitar.",
    },
    "bTqGacqoIvU": {  # Malai Angali Deu — Sabin Rai
        "mood": "nostalgic",
        "energy": 0.44,
        "tempo_bpm": 84.0,
        "acousticness": 0.80,
        "description": "[Nepali favs] Soul-stirring Nepali acoustic rock ballad by Sabin Rai combining raspy emotive vocals with warm unplugged guitar melodies.",
    },
    "opwZ_PJ-F_E": {  # Choo Lo — The Local Train
        "mood": "melancholic",
        "energy": 0.48,
        "tempo_bpm": 86.0,
        "acousticness": 0.78,
        "description": "[Tasteful hindi] Poetic Hindi indie rock ballad by The Local Train blending raw acoustic strumming with late-night heartache and cathartic crescendo.",
    },
    "D395bK70m3w": {  # Masakali — Mohit Chauhan
        "mood": "happy",
        "energy": 0.64,
        "tempo_bpm": 108.0,
        "acousticness": 0.68,
        "description": "[Tasteful hindi] Uplifting A.R. Rahman & Mohit Chauhan Hindi melody with breezy accordion, acoustic rhythm guitar, and carefree morning warmth.",
    },
    "k6qo1PU2ipk": {  # BASANTI / HEARTBREAKER — VZN
        "mood": "intense",
        "energy": 0.92,
        "tempo_bpm": 140.0,
        "acousticness": 0.12,
        "description": "[Nepali hype] Explosive Nepali hype anthem by VZN engineered with heavy bass drops, rapid cadence, and peak gym/party adrenaline.",
    },
}


def _clean_yt_title(raw_title: str, raw_author: str) -> Tuple[str, str]:
    """Clean YouTube video title and channel name into (Song Title, Artist)."""
    artist = re.sub(r"\s*-\s*Topic$", "", raw_author or "").strip()
    title = (raw_title or "").strip()

    title = re.sub(
        r"\s*[\(\[](Official\s+[^\)\]]*|Lyric[^\)\]]*|Audio[^\)\]]*|HD|4K| HQ|Video|Full\s+Song[^\)\]]*)[\)\]]",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()

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
    next_id = 1

    if not YTM_DIR.exists():
        return tracks

    csv_files = sorted(YTM_DIR.glob("*-videos.csv"))
    updated_cache = False

    for csv_file in csv_files:
        playlist_name = csv_file.name.replace("-videos.csv", "").strip()
        profile = PLAYLIST_PROFILES.get(
            playlist_name,
            {
                "genre": "city pop",
                "moods": ["chill"],
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

                mood_list = profile["moods"]
                mood_idx = int(hashlib.md5(vid.encode("utf-8")).hexdigest()[:4], 16) % len(mood_list)
                mood = mood_list[mood_idx]

                energy = _deterministic_jitter(vid + "e", profile["base_energy"], 0.12)
                valence = _deterministic_jitter(vid + "v", profile["base_valence"], 0.12)
                dance = _deterministic_jitter(vid + "d", profile["base_danceability"], 0.08)
                acoustic = _deterministic_jitter(vid + "a", profile["base_acousticness"], 0.10)
                bpm_jitter = int(_deterministic_jitter(vid + "b", 0.5, 0.4, 0.1, 0.9) * 28) - 14
                bpm = float(max(62, min(176, profile["base_bpm"] + bpm_jitter)))
                desc = f"[{playlist_name}] {profile['desc_template']}"

                if vid in TRACK_SPECIFIC_OVERRIDES:
                    ov = TRACK_SPECIFIC_OVERRIDES[vid]
                    mood = ov.get("mood", mood)
                    energy = ov.get("energy", energy)
                    bpm = ov.get("tempo_bpm", bpm)
                    acoustic = ov.get("acousticness", acoustic)
                    desc = ov.get("description", desc)

                tracks.append(
                    {
                        "id": next_id,
                        "title": title,
                        "artist": artist,
                        "genre": profile["genre"],
                        "mood": mood,
                        "energy": energy,
                        "tempo_bpm": bpm,
                        "valence": valence,
                        "danceability": dance,
                        "acousticness": acoustic,
                        "description": desc,
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


def export_ytm_to_songs_csv(csv_path: Path = SONGS_CSV) -> List[Dict[str, Any]]:
    """Overwrite `data/songs.csv` exclusively with the user's resolved YouTube Music library."""
    tracks = build_ytm_catalog(resolve_online=False, max_per_playlist=60)
    fieldnames = [
        "id",
        "title",
        "artist",
        "genre",
        "mood",
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
        "description",
        "themes",
        "youtube_url",
        "playlist",
    ]
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in tracks:
            writer.writerow(t)
    return tracks


if __name__ == "__main__":
    exported = export_ytm_to_songs_csv()
    print(f"Wrote {len(exported)} YouTube Music tracks directly to {SONGS_CSV} (0 starter songs remaining).")

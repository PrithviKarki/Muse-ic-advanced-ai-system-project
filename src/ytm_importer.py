"""
YouTube Music Google Takeout Importer & Metadata Enricher for Muse-ic 2.0.

Reads Google Takeout playlist CSVs (`Video ID,Playlist Video Creation Timestamp`)
from `data/youtube_music/` (`Japanese City Pop` and `Sophisti-pop`), resolves track
titles and artists via YouTube's public oEmbed endpoint (cached in
`data/youtube_music/resolved_ytm_tracks.json`), enriches audio telemetry and RAG
descriptions, and exports `data/songs.csv`.
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

# Playlist-level musical profile priors for Japanese City Pop & Sophisti-pop
PLAYLIST_PROFILES: Dict[str, Dict[str, Any]] = {
    "Japanese City Pop": {
        "genre": "city pop",
        "moods": ["groovy", "happy", "nostalgic", "moody"],
        "base_energy": 0.78,
        "base_bpm": 116,
        "base_valence": 0.84,
        "base_danceability": 0.84,
        "base_acousticness": 0.24,
        "desc_template": "80s Japanese City Pop groove featuring slap bass, lush Rhodes synth-brass chords, and breezy nocturnal Tokyo highway nostalgia.",
        "themes": "japanese city pop, 80s retro, night drive, groovy, upbeat, funk, synthwave, pop, nostalgic, tokyo",
    },
    "Sophisti-pop": {
        "genre": "sophisti-pop",
        "moods": ["relaxed", "romantic", "chill", "focused"],
        "base_energy": 0.44,
        "base_bpm": 94,
        "base_valence": 0.72,
        "base_danceability": 0.68,
        "base_acousticness": 0.72,
        "desc_template": "Polished sophisti-pop arrangement blending smooth jazz saxophone, velvet electric piano chords, and refined late-night lounge warmth.",
        "themes": "sophisti-pop, smooth jazz, lounge, coffee, study, coding, relaxed, chill, date night, romantic, evening, piano, saxophone",
    },
}

# Rich per-track profiles for all 17 tracks across Japanese City Pop & Sophisti-pop
TRACK_SPECIFIC_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "u6y5NSiRPOw": {  # Plastic Love — Mariya Takeuchi
        "mood": "groovy",
        "energy": 0.80,
        "tempo_bpm": 106.0,
        "valence": 0.86,
        "acousticness": 0.24,
        "description": "[Japanese City Pop] Iconic 1984 Tokyo City Pop masterpiece by Mariya Takeuchi driven by infectious slap-bass funk, shimmering Rhodes chords, and neon night-drive euphoria.",
        "themes": "japanese city pop, night drive, groovy, funk, 80s retro, tokyo, upbeat, pop, synthwave",
    },
    "_sOKkON_UnQ": {  # 4:00 A.M. — Taeko Ohnuki
        "mood": "melancholic",
        "energy": 0.36,
        "tempo_bpm": 78.0,
        "valence": 0.34,
        "acousticness": 0.76,
        "description": "[Japanese City Pop] Sweeping nocturnal City Pop ballad by Taeko Ohnuki capturing 4:00 A.M. rain-slicked Tokyo streets, existential introspection, and warm acoustic piano & strings.",
        "themes": "japanese city pop, melancholic, sorrowful, existential dread, somber, late night, rain, introspection, acoustic, piano, strings",
    },
    "ZS6p32gX_yk": {  # 真夜中のドア〜stay with me — Miki Matsubara
        "mood": "happy",
        "energy": 0.84,
        "tempo_bpm": 114.0,
        "valence": 0.88,
        "acousticness": 0.20,
        "description": "[Japanese City Pop] Soaring disco-funk City Pop classic by Miki Matsubara featuring brassy horn stabs, upbeat groove, and timeless 80s retro sparkle.",
        "themes": "japanese city pop, happy, groovy, upbeat, disco, funk, party, night drive, 80s",
    },
    "hci81q8Q49Q": {  # Fly-day Chinatown — Yasuha
        "mood": "intense",
        "energy": 0.90,
        "tempo_bpm": 130.0,
        "valence": 0.88,
        "acousticness": 0.16,
        "description": "[Japanese City Pop] High-voltage 1981 City Pop anthem by Yasuha propelled by rapid slap-bass riffs, punchy brass hits, and adrenaline-charged workout & highway momentum.",
        "themes": "japanese city pop, intense, workout, gym, cardio, running, high energy, fast, night drive, upbeat",
    },
    "3agC6do-624": {  # Remember Summer Days — Anri
        "mood": "nostalgic",
        "energy": 0.76,
        "tempo_bpm": 118.0,
        "valence": 0.82,
        "acousticness": 0.26,
        "description": "[Japanese City Pop] Breezy coastal City Pop gem by Anri blending warm horns, nostalgic summer memories, and shimmering synth-funk rhythm.",
        "themes": "japanese city pop, nostalgic, summer, beach, groovy, happy, night drive, pop",
    },
    "7pPb5fmumNo": {  # Midnight Pretenders — Tomoko Aran
        "mood": "moody",
        "energy": 0.64,
        "tempo_bpm": 100.0,
        "valence": 0.48,
        "acousticness": 0.42,
        "description": "[Japanese City Pop] Atmospheric midnight synth-funk groove by Tomoko Aran with velvet analog pads, melancholic saxophone, and late-night Tokyo cruising.",
        "themes": "japanese city pop, moody, melancholic, midnight, night drive, synthwave, saxophone, retro",
    },
    "QLHMhVonF-s": {  # 黄昏のBAY CITY — Junko Yagami
        "mood": "intense",
        "energy": 0.86,
        "tempo_bpm": 124.0,
        "valence": 0.80,
        "acousticness": 0.18,
        "description": "[Japanese City Pop] Driving twilight Bay City synth-funk anthem by Junko Yagami with powerful vocals, electronic basslines, and high-energy pace.",
        "themes": "japanese city pop, intense, high energy, workout, cardio, night drive, synth, upbeat",
    },
    "CnMzjiYVXYo": {  # サンセット・ロード — Reiko Takahashi
        "mood": "peaceful",
        "energy": 0.34,
        "tempo_bpm": 76.0,
        "valence": 0.68,
        "acousticness": 0.80,
        "description": "[Japanese City Pop] Serene sunset coastal ballad by Reiko Takahashi featuring gentle Rhodes electric piano, warm acoustic guitar, and weightless evening calm.",
        "themes": "japanese city pop, peaceful, chill, calm, sleep, meditation, sunset, acoustic, piano, study",
    },
    "H9R1Hl6sJeE": {  # 悲しみがとまらない I CAN'T STOP THE LONELINESS — Anri
        "mood": "happy",
        "energy": 0.88,
        "tempo_bpm": 126.0,
        "valence": 0.85,
        "acousticness": 0.20,
        "description": "[Japanese City Pop] Exuberant horn-driven City Pop hit by Anri pairing dramatic melody with infectious high-tempo disco-pop energy.",
        "themes": "japanese city pop, happy, upbeat, workout, cardio, party, groovy, disco, pop",
    },
    "ytZd9n6ODGw": {  # Shibuya After Rain（渋谷アフター・レイン） — Japanese City Pop
        "mood": "chill",
        "energy": 0.32,
        "tempo_bpm": 74.0,
        "valence": 0.62,
        "acousticness": 0.84,
        "description": "[Japanese City Pop] Rainy-night Shibuya instrumental & warm Rhodes piano study groove crafted for late-night coding, pomodoro focus, and quiet rain foley.",
        "themes": "japanese city pop, chill, rain, rainy night, coding, study, pomodoro, debugging, piano, lofi, calm",
    },
    "SLbvyz_Hk9E": {  # Paradise — Sade Adu
        "mood": "chill",
        "energy": 0.35,
        "tempo_bpm": 82.0,
        "valence": 0.68,
        "acousticness": 0.78,
        "description": "[Sophisti-pop] Velvet sophisti-pop groove by Sade Adu pairing warm basslines, brushed percussion, and calm late-night coding & pomodoro study focus.",
        "themes": "sophisti-pop, chill, study, coding, pomodoro, debugging, exam, calm, piano, lounge, focus",
    },
    "mRNcPbCJNJ8": {  # Smooth Operator (Single Version) — Sade Adu
        "mood": "relaxed",
        "energy": 0.44,
        "tempo_bpm": 92.0,
        "valence": 0.70,
        "acousticness": 0.74,
        "description": "[Sophisti-pop] Signature sophisti-pop classic by Sade Adu featuring silky tenor saxophone solos, Rhodes piano chords, and effortless coffee-lounge cool.",
        "themes": "sophisti-pop, relaxed, jazz, saxophone, coffee, cafe, lounge, smooth, evening, piano",
    },
    "RcejfvTXusM": {  # The Sweetest Taboo — Sade Adu
        "mood": "peaceful",
        "energy": 0.30,
        "tempo_bpm": 76.0,
        "valence": 0.66,
        "acousticness": 0.86,
        "description": "[Sophisti-pop] Quiet storm sophisti-pop arrangement by Sade Adu opening with real rain foley, gentle acoustic guitar, and tranquil late-night warmth.",
        "themes": "sophisti-pop, peaceful, chill, rain, rainy night, acoustic, guitar, sleep, meditation, study, coding",
    },
    "z2t9RmOe4K0": {  # Hang On to Your Love (Long Version) — Sade Adu
        "mood": "focused",
        "energy": 0.48,
        "tempo_bpm": 96.0,
        "valence": 0.64,
        "acousticness": 0.68,
        "description": "[Sophisti-pop] Hypnotic mid-tempo sophisti-pop groove by Sade Adu built on steady bass pulse and warm electric piano ideal for deep work and concentration.",
        "themes": "sophisti-pop, focused, focus, coding, programming, deep work, study, piano, smooth",
    },
    "MpDM6ygsb20": {  # Part-Time Lover — Stevie Wonder
        "mood": "intense",
        "energy": 0.92,
        "tempo_bpm": 136.0,
        "valence": 0.90,
        "acousticness": 0.15,
        "description": "[Sophisti-pop] High-energy 1985 synth-pop and R&B powerhouse by Stevie Wonder with driving electronic percussion and peak cardio workout cadence.",
        "themes": "sophisti-pop, intense, happy, workout, gym, cardio, running, upbeat, party, synth, pop",
    },
    "VOiZC020nl0": {  # Caribbean Queen (No More Love On the Run) — Billy Ocean
        "mood": "groovy",
        "energy": 0.88,
        "tempo_bpm": 128.0,
        "valence": 0.86,
        "acousticness": 0.18,
        "description": "[Sophisti-pop] Electrifying 80s post-disco sophisti-pop hit by Billy Ocean featuring razor-sharp synth bass, saxophone solo, and high-energy dancefloor momentum.",
        "themes": "sophisti-pop, groovy, intense, workout, party, dance, 80s, synth, saxophone, upbeat",
    },
    "gelwwYuYGn0": {  # Just the Two of Us — Grover Washington, Jr.
        "mood": "romantic",
        "energy": 0.38,
        "tempo_bpm": 86.0,
        "valence": 0.76,
        "acousticness": 0.84,
        "description": "[Sophisti-pop] Warm jazz-soul classic by Grover Washington, Jr. and Bill Withers with Fender Rhodes electric piano, rain-on-windowpane lyricism, and smooth saxophone.",
        "themes": "sophisti-pop, romantic, chill, relaxed, jazz, piano, saxophone, rain, coffee, study, coding, pomodoro, acoustic",
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
    Parse `*-videos.csv` files in `data/youtube_music/`, resolve titles/artists,
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
        if playlist_name not in PLAYLIST_PROFILES:
            continue
        profile = PLAYLIST_PROFILES[playlist_name]

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
                themes = f"{profile['themes']}, {artist.lower()}, {title.lower()}"

                if vid in TRACK_SPECIFIC_OVERRIDES:
                    ov = TRACK_SPECIFIC_OVERRIDES[vid]
                    mood = ov.get("mood", mood)
                    energy = ov.get("energy", energy)
                    bpm = ov.get("tempo_bpm", bpm)
                    valence = ov.get("valence", valence)
                    acoustic = ov.get("acousticness", acoustic)
                    desc = ov.get("description", desc)
                    if "themes" in ov:
                        themes = f"{ov['themes']}, {artist.lower()}, {title.lower()}"

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
                        "themes": themes,
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
    """Overwrite `data/songs.csv` exclusively with the user's Japanese City Pop & Sophisti-pop library."""
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
    print(f"Wrote {len(exported)} YouTube Music tracks (Japanese City Pop & Sophisti-pop) to {SONGS_CSV}.")

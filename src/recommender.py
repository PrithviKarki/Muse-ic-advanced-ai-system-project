import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k recommended songs for a user profile sorted by compatibility and diversity."""
        ranked = recommend_songs(user, self.songs, k=k)
        return [song for song, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Generate a human-readable explanation string detailing why a song was recommended."""
        _, reasons = score_song(user, song)
        return ", ".join(reasons) if reasons else "General musical match"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries with typed attributes."""
    songs: List[Dict] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict = {
                "id": int(row["id"]) if "id" in row and row["id"].strip().isdigit() else row.get("id"),
                "title": row.get("title", "").strip(),
                "artist": row.get("artist", "").strip(),
                "genre": row.get("genre", "").strip(),
                "mood": row.get("mood", "").strip(),
                "energy": float(row["energy"]) if "energy" in row else 0.0,
                "tempo_bpm": float(row["tempo_bpm"]) if "tempo_bpm" in row else 0.0,
                "valence": float(row["valence"]) if "valence" in row else 0.0,
                "danceability": float(row["danceability"]) if "danceability" in row else 0.0,
                "acousticness": float(row["acousticness"]) if "acousticness" in row else 0.0,
            }
            songs.append(song)
    return songs

def _extract_attr(obj, key: str, alt_key: Optional[str] = None, default=None):
    """Safely extract an attribute or dictionary value using primary or alternate keys."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if alt_key and hasattr(obj, alt_key):
        return getattr(obj, alt_key)
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        if alt_key and alt_key in obj:
            return obj[alt_key]
    return default


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song against user preferences using weighted feature matching and return (score, reasons)."""
    weight_genre = 0.35
    weight_mood = 0.30
    weight_energy = 0.20
    weight_acoustic = 0.15

    reasons: List[str] = []

    # 1. Genre Score (Categorical / Substring match)
    user_genre = _extract_attr(user_prefs, "favorite_genre", "genre", "")
    song_genre = _extract_attr(song, "genre", default="")
    ug = str(user_genre).strip().lower()
    sg = str(song_genre).strip().lower()

    if ug and sg:
        if ug == sg:
            genre_score = 1.0
            reasons.append(f"genre match '{song_genre}' (+{weight_genre * genre_score:.2f})")
        elif ug in sg or sg in ug:
            genre_score = 0.5
            reasons.append(f"related genre '{song_genre}' (+{weight_genre * genre_score:.2f})")
        else:
            genre_score = 0.0
    else:
        genre_score = 0.0

    # 2. Mood Score (Categorical match)
    user_mood = _extract_attr(user_prefs, "favorite_mood", "mood", "")
    song_mood = _extract_attr(song, "mood", default="")
    um = str(user_mood).strip().lower()
    sm = str(song_mood).strip().lower()

    if um and sm and um == sm:
        mood_score = 1.0
        reasons.append(f"mood match '{song_mood}' (+{weight_mood * mood_score:.2f})")
    else:
        mood_score = 0.0

    # 3. Energy Proximity (Continuous similarity)
    user_energy = _extract_attr(user_prefs, "target_energy", "energy", default=None)
    song_energy = _extract_attr(song, "energy", default=0.0)

    if user_energy is not None:
        try:
            target_e = float(user_energy)
            song_e = float(song_energy)
            energy_diff = abs(song_e - target_e)
            energy_score = max(0.0, 1.0 - energy_diff)
            if energy_diff <= 0.15:
                reasons.append(f"energy match ({song_e:.2f} near target {target_e:.2f}) (+{weight_energy * energy_score:.2f})")
        except (ValueError, TypeError):
            energy_score = 0.0
    else:
        energy_score = 0.0

    # 4. Acoustic Alignment (Directional match)
    user_acoustic = _extract_attr(user_prefs, "likes_acoustic", default=None)
    song_acoustic = _extract_attr(song, "acousticness", default=0.0)

    if user_acoustic is not None:
        try:
            song_a = float(song_acoustic)
            if bool(user_acoustic):
                acoustic_score = max(0.0, min(1.0, song_a))
                if song_a >= 0.5:
                    reasons.append(f"acoustic sound match (+{weight_acoustic * acoustic_score:.2f})")
            else:
                acoustic_score = max(0.0, min(1.0, 1.0 - song_a))
                if song_a <= 0.5:
                    reasons.append(f"electronic/non-acoustic match (+{weight_acoustic * acoustic_score:.2f})")
        except (ValueError, TypeError):
            acoustic_score = 0.0
    else:
        acoustic_score = 0.5

    # Weighted Total Score
    final_score = (
        (weight_genre * genre_score)
        + (weight_mood * mood_score)
        + (weight_energy * energy_score)
        + (weight_acoustic * acoustic_score)
    )

    if not reasons:
        reasons.append("general catalog match")

    return round(final_score, 4), reasons

def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    artist_penalty: float = 0.15,
    genre_penalty: float = 0.10,
) -> List[Tuple[Dict, float, str]]:
    """Score all songs and iteratively select top k candidates applying artist and genre diversity penalties."""
    remaining_pool = []
    for song in songs:
        base_score, base_reasons = score_song(user_prefs, song)
        remaining_pool.append((song, base_score, list(base_reasons)))

    selected_recommendations: List[Tuple[Dict, float, str]] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}

    while remaining_pool and len(selected_recommendations) < k:
        best_idx = 0
        best_adjusted_score = -1.0
        best_adjusted_reasons: List[str] = []

        for idx, (song, base_score, base_reasons) in enumerate(remaining_pool):
            artist = str(_extract_attr(song, "artist", default="")).strip().lower()
            genre = str(_extract_attr(song, "genre", default="")).strip().lower()

            adjusted_score = base_score
            adjusted_reasons = list(base_reasons)

            # Apply artist diversity penalty if this artist is already in the top list
            prev_artist_count = artist_counts.get(artist, 0)
            if artist and prev_artist_count > 0:
                penalty = artist_penalty * prev_artist_count
                adjusted_score = max(0.0, adjusted_score - penalty)
                artist_name = _extract_attr(song, "artist", default="")
                adjusted_reasons.append(f"diversity penalty: repeat artist '{artist_name}' (-{penalty:.2f})")

            # Apply genre saturation penalty if this genre already appears 2+ times in the top list
            prev_genre_count = genre_counts.get(genre, 0)
            if genre and prev_genre_count >= 2:
                penalty = genre_penalty * (prev_genre_count - 1)
                adjusted_score = max(0.0, adjusted_score - penalty)
                genre_name = _extract_attr(song, "genre", default="")
                adjusted_reasons.append(f"diversity penalty: genre '{genre_name}' saturation (-{penalty:.2f})")

            adjusted_score = round(adjusted_score, 4)
            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_idx = idx
                best_adjusted_reasons = adjusted_reasons

        chosen_song, _, _ = remaining_pool.pop(best_idx)
        chosen_artist = str(_extract_attr(chosen_song, "artist", default="")).strip().lower()
        chosen_genre = str(_extract_attr(chosen_song, "genre", default="")).strip().lower()

        if chosen_artist:
            artist_counts[chosen_artist] = artist_counts.get(chosen_artist, 0) + 1
        if chosen_genre:
            genre_counts[chosen_genre] = genre_counts.get(chosen_genre, 0) + 1

        explanation = ", ".join(best_adjusted_reasons) if best_adjusted_reasons else "General musical match"
        selected_recommendations.append((chosen_song, best_adjusted_score, explanation))

    return selected_recommendations

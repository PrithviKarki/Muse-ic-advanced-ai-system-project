"""
Multi-Source Semantic Retrieval-Augmented Generation (RAG) & Specialized Local
Embedding Engine for Muse-ic.

Core capabilities:
1. Multi-Source Knowledge Indexing: Fuses three distinct data sources into the
   retrieval index:
   - Source 1 (CSV Telemetry & Track Metadata): data/songs.csv
   - Source 2 (JSON Musicology & Artist Lore Knowledge Base): data/musicology_guides.json
   - Source 3 (Markdown & User-Provided Custom Curation Notes): data/custom_listener_notes.md
2. Computes dense semantic embeddings using a 15-dimensional Specialized Musical
   Concept Vectorizer hybridized with IDF-weighted lexical embeddings (100% offline
   reproducible, zero external API dependency required).
3. Parses natural language listener prompts into structured UserProfile constraints
   using Google Gemini (gemini-2.5-flash) with automatic local semantic fallback.
4. Synthesizes grounded AI DJ explanations that actively cite retrieved song
   documents, psychoacoustic musicology guides, custom notes, and mathematical sub-scores.
"""

import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.logger import get_logger
except ModuleNotFoundError:
    from logger import get_logger

logger = get_logger("muse_ic.rag")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MUSICOLOGY_JSON = DATA_DIR / "musicology_guides.json"
DEFAULT_CUSTOM_NOTES_MD = DATA_DIR / "custom_listener_notes.md"


# ---------------------------------------------------------------------------
# Specialized Musical Taxonomy & Synonym Knowledge Graph
# ---------------------------------------------------------------------------
MOOD_SYNONYM_CLUSTERS: Dict[str, Dict[str, float]] = {
    "chill": {"chill": 1.0, "relaxed": 0.85, "peaceful": 0.80, "focused": 0.75, "melancholic": 0.45, "nostalgic": 0.55, "romantic": 0.50},
    "relaxed": {"relaxed": 1.0, "chill": 0.85, "peaceful": 0.85, "nostalgic": 0.65, "romantic": 0.60, "focused": 0.65},
    "peaceful": {"peaceful": 1.0, "relaxed": 0.85, "chill": 0.80, "nostalgic": 0.60, "focused": 0.60, "melancholic": 0.45},
    "focused": {"focused": 1.0, "chill": 0.75, "relaxed": 0.65, "peaceful": 0.60, "moody": 0.55, "confident": 0.50},
    "happy": {"happy": 1.0, "festive": 0.85, "groovy": 0.80, "euphoric": 0.75, "confident": 0.60, "peaceful": 0.45},
    "festive": {"festive": 1.0, "happy": 0.85, "groovy": 0.80, "euphoric": 0.70},
    "groovy": {"groovy": 1.0, "happy": 0.80, "festive": 0.80, "confident": 0.65, "romantic": 0.55},
    "intense": {"intense": 1.0, "aggressive": 0.85, "euphoric": 0.70, "confident": 0.65, "moody": 0.55},
    "aggressive": {"aggressive": 1.0, "intense": 0.85, "confident": 0.65, "moody": 0.50},
    "euphoric": {"euphoric": 1.0, "happy": 0.75, "intense": 0.70, "festive": 0.70, "groovy": 0.65},
    "melancholic": {"melancholic": 1.0, "sorrowful": 0.85, "nostalgic": 0.70, "moody": 0.65, "peaceful": 0.50, "chill": 0.45},
    "sorrowful": {"sorrowful": 1.0, "melancholic": 0.85, "nostalgic": 0.70, "moody": 0.60},
    "existential dread": {"melancholic": 0.85, "sorrowful": 0.80, "moody": 0.75, "intense": 0.60, "aggressive": 0.50},
    "moody": {"moody": 1.0, "melancholic": 0.65, "focused": 0.55, "intense": 0.55, "sorrowful": 0.60},
    "nostalgic": {"nostalgic": 1.0, "melancholic": 0.70, "relaxed": 0.65, "peaceful": 0.60, "sorrowful": 0.65},
    "romantic": {"romantic": 1.0, "relaxed": 0.60, "groovy": 0.55, "chill": 0.50, "peaceful": 0.50},
    "confident": {"confident": 1.0, "groovy": 0.65, "intense": 0.65, "happy": 0.60, "focused": 0.50},
}

GENRE_AFFINITY_CLUSTERS: Dict[str, Dict[str, float]] = {
    "lofi": {"lofi": 1.0, "ambient": 0.75, "jazz": 0.75, "hip hop": 0.65, "folk": 0.50, "classical": 0.50},
    "ambient": {"ambient": 1.0, "lofi": 0.75, "classical": 0.70, "jazz": 0.60, "folk": 0.55},
    "jazz": {"jazz": 1.0, "lofi": 0.75, "r&b": 0.70, "blues": 0.70, "ambient": 0.60, "classical": 0.55},
    "pop": {"pop": 1.0, "indie pop": 0.85, "edm": 0.70, "funk": 0.70, "synthwave": 0.65, "r&b": 0.60, "latin": 0.60},
    "indie pop": {"indie pop": 1.0, "pop": 0.85, "folk": 0.60, "synthwave": 0.60, "funk": 0.60},
    "rock": {"rock": 1.0, "metal": 0.80, "indie pop": 0.50, "blues": 0.55, "synthwave": 0.50},
    "metal": {"metal": 1.0, "rock": 0.80, "edm": 0.45, "synthwave": 0.50},
    "classical": {"classical": 1.0, "ambient": 0.70, "jazz": 0.55, "folk": 0.50, "lofi": 0.50},
    "edm": {"edm": 1.0, "synthwave": 0.80, "pop": 0.70, "funk": 0.60, "hip hop": 0.50},
    "synthwave": {"synthwave": 1.0, "edm": 0.80, "pop": 0.65, "indie pop": 0.60, "rock": 0.50},
    "folk": {"folk": 1.0, "blues": 0.70, "indie pop": 0.60, "reggae": 0.55, "ambient": 0.55, "jazz": 0.50},
    "r&b": {"r&b": 1.0, "jazz": 0.70, "hip hop": 0.70, "funk": 0.70, "pop": 0.60, "blues": 0.65},
    "hip hop": {"hip hop": 1.0, "r&b": 0.70, "lofi": 0.65, "funk": 0.65, "pop": 0.50},
    "blues": {"blues": 1.0, "jazz": 0.70, "folk": 0.70, "r&b": 0.65, "rock": 0.55},
    "funk": {"funk": 1.0, "pop": 0.70, "r&b": 0.70, "latin": 0.65, "hip hop": 0.65, "indie pop": 0.60},
    "latin": {"latin": 1.0, "reggae": 0.70, "funk": 0.65, "pop": 0.60, "jazz": 0.55},
    "reggae": {"reggae": 1.0, "latin": 0.70, "folk": 0.55, "r&b": 0.55, "lofi": 0.50},
    "city pop": {"city pop": 1.0, "funk": 0.85, "synthwave": 0.80, "sophisti-pop": 0.80, "pop": 0.75, "indie pop": 0.70},
    "sophisti-pop": {"sophisti-pop": 1.0, "city pop": 0.80, "jazz": 0.80, "indie pop": 0.75, "r&b": 0.75, "pop": 0.70},
    "nepali indie": {"nepali indie": 1.0, "folk": 0.80, "indie pop": 0.80, "nepali rock": 0.75, "hindi indie": 0.70},
    "nepali rock": {"nepali rock": 1.0, "rock": 0.85, "nepali indie": 0.75, "pop": 0.65, "hip hop": 0.60},
    "hindi indie": {"hindi indie": 1.0, "indie pop": 0.80, "folk": 0.75, "nepali indie": 0.70, "r&b": 0.65},
}

MUSICAL_CONCEPT_DIMENSIONS: List[Tuple[str, List[str]]] = [
    ("calm_study", ["chill", "study", "studying", "calm", "quiet", "library", "reading", "homework", "peaceful", "relaxed", "serene", "tranquil", "pomodoro", "exam"]),
    ("deep_focus", ["focus", "focused", "coding", "programming", "productive", "productivity", "concentration", "flow", "work", "instrumental", "debugging", "engineering"]),
    ("rain_night", ["rain", "rainy", "midnight", "night", "late", "windowpane", "afterhours", "moonlight", "nocturne", "stars", "stargazing", "foley"]),
    ("sleep_meditation", ["sleep", "meditation", "mindfulness", "anxiety", "rest", "weightless", "drift", "floating", "ambient", "drone", "beatless"]),
    ("coffee_lounge", ["coffee", "cafe", "espresso", "lounge", "cozy", "warm", "sunday", "conversation", "intimate", "velvet", "smooth"]),
    ("melancholy_sorrow", ["melancholic", "sorrowful", "sad", "existential", "dread", "heartache", "lonely", "somber", "weeping", "wistful", "introspection", "catharsis"]),
    ("happy_upbeat", ["happy", "upbeat", "joyful", "cheerful", "sunny", "sunrise", "summer", "positive", "feel-good", "bright", "shimmering", "breezy"]),
    ("party_dance", ["party", "dance", "dancing", "disco", "funk", "groovy", "festive", "celebration", "rooftop", "carnival", "social", "friends"]),
    ("workout_cardio", ["workout", "gym", "cardio", "running", "sprint", "hiit", "fitness", "training", "sweat", "motivation", "commute", "cadence"]),
    ("heavy_power", ["intense", "aggressive", "powerlifting", "weightlifting", "adrenaline", "heavy", "metal", "thrash", "blast", "rage", "extreme", "thunder"]),
    ("cyber_retro", ["cyberpunk", "synthwave", "retro", "80s", "neon", "highway", "drive", "gaming", "sci-fi", "arpeggio", "laser", "darksynth", "city pop", "japanese", "tokyo"]),
    ("acoustic_organic", ["acoustic", "piano", "guitar", "strings", "cello", "violin", "fiddle", "harmonica", "saxophone", "upright", "wood", "organic", "unplugged", "rhodes", "felt", "sophisti-pop"]),
    ("electronic_synth", ["electronic", "synth", "synthesizer", "edm", "rave", "festival", "sub-bass", "drop", "808", "electro", "digital"]),
    ("urban_groove", ["hip hop", "boom-bap", "cypher", "swagger", "confident", "hustle", "street", "basketball", "beats", "sample", "soul", "r&b"]),
    ("nature_folk", ["folk", "campfire", "rustic", "mountain", "forest", "pine", "hiking", "autumn", "trail", "coastal", "beach", "tropical", "reggae", "nepali", "hindi", "bollywood", "himalayan"]),
]


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words and stems."""
    words = re.findall(r"[a-z0-9&]+", text.lower())
    expanded = list(words)
    for w in words:
        if len(w) > 5 and w.endswith("ing"):
            expanded.append(w[:-3])
        elif len(w) > 4 and w.endswith("ed"):
            expanded.append(w[:-2])
        elif len(w) > 4 and w.endswith("s"):
            expanded.append(w[:-1])
    return expanded


def get_semantic_mood_similarity(user_mood: str, song_mood: str) -> float:
    """Return semantic similarity [0.0, 1.0] between two mood labels using taxonomy clusters."""
    um = str(user_mood or "").strip().lower()
    sm = str(song_mood or "").strip().lower()
    if not um or not sm:
        return 0.0
    if um == sm:
        return 1.0
    if um in MOOD_SYNONYM_CLUSTERS and sm in MOOD_SYNONYM_CLUSTERS[um]:
        return MOOD_SYNONYM_CLUSTERS[um][sm]
    for key, cluster in MOOD_SYNONYM_CLUSTERS.items():
        if (key in um or um in key) and sm in cluster:
            return cluster[sm]
    return 0.0


def get_semantic_genre_similarity(user_genre: str, song_genre: str) -> float:
    """Return semantic similarity [0.0, 1.0] between two genre labels using musical affinity."""
    ug = str(user_genre or "").strip().lower()
    sg = str(song_genre or "").strip().lower()
    if not ug or not sg:
        return 0.0
    if ug == sg:
        return 1.0
    if ug in sg or sg in ug:
        return 0.75
    if ug in GENRE_AFFINITY_CLUSTERS and sg in GENRE_AFFINITY_CLUSTERS[ug]:
        return GENRE_AFFINITY_CLUSTERS[ug][sg]
    return 0.0


def load_external_knowledge_sources(
    musicology_json_path: Optional[Path] = None,
    custom_notes_md_path: Optional[Path] = None,
    extra_custom_notes: str = "",
) -> Dict[str, Any]:
    """
    Load secondary RAG knowledge sources:
    - JSON Musicology & Artist Profiles (data/musicology_guides.json)
    - Markdown Custom Listener Notes (data/custom_listener_notes.md + optional runtime text)
    """
    json_path = musicology_json_path or DEFAULT_MUSICOLOGY_JSON
    md_path = custom_notes_md_path or DEFAULT_CUSTOM_NOTES_MD

    artist_profiles: Dict[str, Dict[str, str]] = {}
    activity_guides: List[Dict[str, Any]] = []
    custom_notes_lines: List[str] = []
    active_sources: List[str] = ["songs.csv (Track Telemetry & Descriptions)"]

    if json_path.exists():
        try:
            with open(json_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
                artist_profiles = data.get("artist_profiles", {})
                activity_guides = data.get("activity_guides", [])
                active_sources.append(f"{json_path.name} (Artist Lore & Psychoacoustic Guides)")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not load {json_path}: {exc}")

    if md_path.exists():
        try:
            with open(md_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("#"):
                        custom_notes_lines.append(cleaned.lstrip("-* ").strip())
                active_sources.append(f"{md_path.name} (Custom Listener Curation Notes)")
        except OSError as exc:
            logger.warning(f"Could not load {md_path}: {exc}")

    if extra_custom_notes and extra_custom_notes.strip():
        for line in extra_custom_notes.strip().splitlines():
            cleaned = line.strip().lstrip("-* ").strip()
            if cleaned:
                custom_notes_lines.append(cleaned)
        active_sources.append("User Runtime Custom Document (Interactive Notes)")

    return {
        "artist_profiles": artist_profiles,
        "activity_guides": activity_guides,
        "custom_notes": custom_notes_lines,
        "active_sources": active_sources,
    }


class SongKnowledgeIndex:
    """
    Multi-Source Specialized Local Embedding & Vector Retrieval Index.

    Fuses three distinct data sources per track:
    1. Song CSV telemetry & descriptions (data/songs.csv)
    2. Artist bios, signature sounds & psychoacoustic activity guides (data/musicology_guides.json)
    3. Custom Markdown listener notes & runtime user documents (data/custom_listener_notes.md)
    """

    def __init__(self, songs: List[Any], extra_custom_notes: str = ""):
        self.songs = songs
        self.extra_custom_notes = extra_custom_notes
        self.external_knowledge = load_external_knowledge_sources(extra_custom_notes=extra_custom_notes)
        self.documents: Dict[int, str] = {}
        self.concept_vectors: Dict[int, List[float]] = {}
        self.tfidf_vectors: Dict[int, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.backend_name: str = "Multi-Source Specialized Musical Concept + TF-IDF Hybrid Embedding Engine"
        self._build_index()

    def update_custom_notes(self, extra_custom_notes: str) -> None:
        """Rebuild the multi-source index when custom user documents/notes are added at runtime."""
        if extra_custom_notes != self.extra_custom_notes:
            self.extra_custom_notes = extra_custom_notes
            self.external_knowledge = load_external_knowledge_sources(extra_custom_notes=extra_custom_notes)
            self._build_index()
            logger.info("Re-indexed Multi-Source RAG Knowledge Base with updated custom user notes.")

    def _get_field(self, song: Any, key: str, default: Any = "") -> Any:
        if isinstance(song, dict):
            return song.get(key, default)
        return getattr(song, key, default)

    def build_song_document(self, song: Any) -> str:
        """Construct a multi-source RAG knowledge document fusing CSV, JSON lore, and Markdown notes."""
        title = str(self._get_field(song, "title", "Untitled"))
        artist = str(self._get_field(song, "artist", "Unknown"))
        genre = str(self._get_field(song, "genre", ""))
        mood = str(self._get_field(song, "mood", ""))
        energy = float(self._get_field(song, "energy", 0.5) or 0.5)
        acousticness = float(self._get_field(song, "acousticness", 0.5) or 0.5)
        valence = float(self._get_field(song, "valence", 0.5) or 0.5)
        tempo = float(self._get_field(song, "tempo_bpm", 100.0) or 100.0)
        description = str(self._get_field(song, "description", ""))
        themes = str(self._get_field(song, "themes", ""))

        energy_tag = "high energy intense upbeat" if energy >= 0.70 else ("low energy calm quiet slow" if energy <= 0.42 else "medium energy moderate")
        acoustic_tag = "acoustic organic unplugged piano guitar strings" if acousticness >= 0.55 else "electronic synthesized digital beats synth"
        valence_tag = "happy cheerful positive uplifting" if valence >= 0.68 else ("melancholic dark somber emotional" if valence <= 0.40 else "balanced mood")
        tempo_tag = "fast tempo" if tempo >= 125 else ("slow tempo" if tempo <= 82 else "mid-tempo")

        # Fuse Source 2: Artist Profile & Activity Guide from musicology_guides.json
        artist_info = self.external_knowledge["artist_profiles"].get(artist, {})
        artist_lore = ""
        if artist_info:
            artist_lore = (
                f" | Artist Lore (musicology_guides.json): {artist_info.get('bio', '')} "
                f"Signature Sound: {artist_info.get('signature_sound', '')}. "
                f"Ideal Moments: {artist_info.get('recommended_moments', '')}."
            )

        # Fuse Source 3: Matching Custom Listener Notes from custom_listener_notes.md
        matched_notes = [
            note
            for note in self.external_knowledge["custom_notes"]
            if title.lower() in note.lower() or artist.lower() in note.lower() or (genre and genre.lower() in note.lower())
        ]
        custom_note_str = (
            f" | Custom Curation Note (custom_listener_notes.md): {' '.join(matched_notes)}"
            if matched_notes
            else ""
        )

        return (
            f"Track: {title} by {artist} | Genre: {genre} | Mood: {mood} | "
            f"Audio Profile: {energy_tag}, {acoustic_tag}, {valence_tag}, {tempo_tag} ({tempo:.0f} BPM, energy {energy:.2f}, acousticness {acousticness:.2f}) | "
            f"Description: {description} | Themes & Activities: {themes}"
            f"{artist_lore}{custom_note_str}"
        )

    def _compute_concept_vector(self, text: str) -> List[float]:
        """Project text into the 15-dimensional Specialized Musical Concept space."""
        tokens = set(_tokenize(text))
        lower_text = text.lower()
        vec: List[float] = []

        for _, keywords in MUSICAL_CONCEPT_DIMENSIONS:
            hits = 0.0
            for kw in keywords:
                if kw in tokens or kw in lower_text:
                    hits += 1.0
            vec.append(min(1.0, hits / 2.5))

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-8:
            vec = [v / norm for v in vec]
        return vec

    def _build_index(self) -> None:
        """Index all songs across all connected knowledge sources."""
        doc_tokens: Dict[int, List[str]] = {}
        df_counts: Counter = Counter()
        n_docs = max(1, len(self.songs))

        for idx, song in enumerate(self.songs):
            song_id = int(self._get_field(song, "id", idx + 1) or (idx + 1))
            doc = self.build_song_document(song)
            self.documents[song_id] = doc
            self.concept_vectors[song_id] = self._compute_concept_vector(doc)

            tokens = _tokenize(doc)
            doc_tokens[song_id] = tokens
            for unique_tok in set(tokens):
                df_counts[unique_tok] += 1

        for tok, df in df_counts.items():
            self.idf[tok] = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0

        for song_id, tokens in doc_tokens.items():
            tf_counts = Counter(tokens)
            raw_vec = {tok: (count / max(1, len(tokens))) * self.idf.get(tok, 1.0) for tok, count in tf_counts.items()}
            norm = math.sqrt(sum(v * v for v in raw_vec.values()))
            if norm > 1e-8:
                raw_vec = {k: v / norm for k, v in raw_vec.items()}
            self.tfidf_vectors[song_id] = raw_vec

    def compute_query_similarity(self, query_text: str, song: Any) -> Tuple[float, List[str]]:
        """
        Compute hybrid semantic similarity [0.0, 1.0] between a user query/profile
        and a song's multi-source knowledge document, returning (score, matched_concepts).
        """
        if not query_text or not query_text.strip():
            return 0.0, []

        song_id = int(self._get_field(song, "id", -1) or -1)
        if song_id not in self.documents:
            doc = self.build_song_document(song)
            song_concept_vec = self._compute_concept_vector(doc)
            tokens = _tokenize(doc)
            tf_counts = Counter(tokens)
            raw_vec = {tok: count * self.idf.get(tok, 1.0) for tok, count in tf_counts.items()}
            norm = math.sqrt(sum(v * v for v in raw_vec.values()))
            song_tfidf_vec = {k: v / norm for k, v in raw_vec.items()} if norm > 1e-8 else {}
        else:
            doc = self.documents[song_id]
            song_concept_vec = self.concept_vectors[song_id]
            song_tfidf_vec = self.tfidf_vectors[song_id]

        # 1. Concept-space cosine similarity (65% weight)
        query_concept_vec = self._compute_concept_vector(query_text)
        concept_sim = sum(a * b for a, b in zip(query_concept_vec, song_concept_vec))

        # 2. Lexical TF-IDF cosine similarity (35% weight)
        q_tokens = _tokenize(query_text)
        q_counts = Counter(q_tokens)
        q_tfidf = {tok: count * self.idf.get(tok, 1.0) for tok, count in q_counts.items()}
        q_norm = math.sqrt(sum(v * v for v in q_tfidf.values()))
        if q_norm > 1e-8:
            q_tfidf = {k: v / q_norm for k, v in q_tfidf.items()}
        lexical_sim = sum(val * song_tfidf_vec.get(tok, 0.0) for tok, val in q_tfidf.items())

        doc_lower = doc.lower()
        matched_keywords: List[str] = []
        meaningful_stop = {"the", "and", "for", "with", "that", "this", "from", "music", "song", "songs", "track", "tracks", "want", "need", "like", "some"}
        for tok in q_counts:
            if len(tok) >= 3 and tok not in meaningful_stop:
                if tok in doc_lower:
                    matched_keywords.append(tok)

        combined_sim = min(1.0, (0.65 * concept_sim) + (0.35 * min(1.0, lexical_sim * 1.6)))
        return round(combined_sim, 4), matched_keywords[:4]

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Tuple[Any, float, List[str], str]]:
        """Retrieve top_k songs ranked purely by multi-source semantic vector similarity."""
        scored = []
        for song in self.songs:
            sim, matched_kw = self.compute_query_similarity(query_text, song)
            song_id = int(self._get_field(song, "id", 0) or 0)
            doc = self.documents.get(song_id, self.build_song_document(song))
            scored.append((song, sim, matched_kw, doc))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_multi_source_context(self, query_text: str) -> Dict[str, Any]:
        """
        Retrieve matching psychoacoustic activity guides (from musicology_guides.json)
        and custom listener notes (from custom_listener_notes.md) for a given query.
        """
        q_lower = (query_text or "").lower()
        q_tokens = set(_tokenize(q_lower))

        best_guide: Optional[Dict[str, Any]] = None
        best_guide_hits = 0
        for guide in self.external_knowledge.get("activity_guides", []):
            hits = sum(1 for kw in guide.get("keywords", []) if kw in q_lower or kw in q_tokens)
            if hits > best_guide_hits:
                best_guide_hits = hits
                best_guide = guide

        matched_custom_notes: List[str] = []
        for note in self.external_knowledge.get("custom_notes", []):
            note_tokens = set(_tokenize(note))
            overlap = len(q_tokens.intersection(note_tokens))
            if overlap >= 2:
                matched_custom_notes.append(note)

        return {
            "matched_guide": best_guide,
            "matched_custom_notes": matched_custom_notes[:2],
            "active_sources": self.external_knowledge.get("active_sources", []),
        }


# ---------------------------------------------------------------------------
# Few-Shot Exemplars & Constrained Tone/Style Specialization (Stretch Goal)
# ---------------------------------------------------------------------------
FEW_SHOT_INTENT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "user_request": "I need calm rainy night coding music with warm acoustic piano and zero vocals",
        "output": {
            "label": "Rainy Night Coding & Deep Focus",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.32,
            "likes_acoustic": True,
            "target_valence": 0.55,
        },
    },
    {
        "user_request": "High-voltage 80s neon cyberpunk highway night drive with pulsing analog synths",
        "output": {
            "label": "Neon Cyberpunk Highway Drive",
            "genre": "synthwave",
            "mood": "moody",
            "energy": 0.86,
            "likes_acoustic": False,
            "target_valence": 0.50,
        },
    },
    {
        "user_request": "Existential dread and midnight heartache with raw weeping acoustic instruments",
        "output": {
            "label": "Midnight Catharsis & Introspection",
            "genre": "blues",
            "mood": "sorrowful",
            "energy": 0.35,
            "likes_acoustic": True,
            "target_valence": 0.28,
        },
    },
]

SPECIALIZED_DJ_STYLES: Dict[str, Dict[str, str]] = {
    "musicologist": {
        "name": "Psychoacoustic Musicologist (Specialized Default)",
        "instruction": (
            "Write in a precise, analytical psychoacoustic musicologist tone. Cite BPM tempo ranges, "
            "harmonic timbre (felt piano, Rhodes, sub-bass), and cognitive/physiological entrainment."
        ),
        "few_shot_example": (
            "Exemplar: '[Psychoacoustic Analysis — 72 BPM Entrainment]: Library Rain (Paper Lanterns) pairs "
            "high acoustic warmth (0.86) with non-intrusive rain foley to prevent verbal working-memory interference.'"
        ),
    },
    "late_night_radio": {
        "name": "Late-Night FM Radio Host (Constrained Persona)",
        "instruction": (
            "Write in the warm, intimate, poetic voice of a midnight FM vinyl radio host. Include smooth "
            "[Crossfade Transition Cue] markers and evocative nocturnal imagery."
        ),
        "few_shot_example": (
            "Exemplar: '🎙️ [On-Air 02:00 AM — Crossfade 4s]: Let the rain outside your window melt into the "
            "dusty Rhodes chords of Paper Lanterns as we ease into Library Rain...'"
        ),
    },
    "hype_coach": {
        "name": "High-Intensity Athletic Coach (Constrained Persona)",
        "instruction": (
            "Write in the punchy, high-adrenaline cadence of an athletic performance coach. Emphasize stride "
            "cadence, BPM pacing, and peak power output with [Cadence Zone] markers."
        ),
        "few_shot_example": (
            "Exemplar: '🔥 [Cadence Zone: 132 BPM Peak Output]: Lock in your pace with Gym Hero (Max Pulse)—"
            "0.93 energy electro-synth drive built to push you through your final heavy set!'"
        ),
    },
    "baseline": {
        "name": "Unspecialized Baseline (Plain Checklist)",
        "instruction": "Output a plain, unstyled list of recommended songs and their raw numeric scores without domain persona.",
        "few_shot_example": "Exemplar: '1. Library Rain by Paper Lanterns (Score: 0.95) - genre match, mood match.'",
    },
}


# ---------------------------------------------------------------------------
# Natural Language Query Parser & Grounded Multi-Source RAG Synthesizer
# ---------------------------------------------------------------------------
def _resolve_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini API key from explicit parameter or environment variables."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for env_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for filename in (".env", ".env.example"):
        env_path = os.path.join(project_root, filename)
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(("GEMINI_API_KEY=", "GOOGLE_API_KEY=")):
                            key_val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if key_val and key_val != "your_key_here":
                                return key_val
            except OSError:
                pass
    return None


def _call_gemini_api(prompt: str, api_key: str, json_mode: bool = False) -> Optional[str]:
    """Call Google Gemini 2.5 Flash API with safe timeout and fallback handling."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    generation_config: Dict[str, Any] = {"temperature": 0.2, "maxOutputTokens": 650}
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
        logger.warning(f"Gemini API call fell back to local engine: {exc}")
    return None


def parse_natural_language_request(
    prompt: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse a free-form natural language music request into a structured UserProfile
    dictionary. Uses Gemini 2.5 Flash when an API key is available, or the local
    semantic intent analyzer offline.
    """
    cleaned_prompt = (prompt or "").strip()
    resolved_key = _resolve_gemini_api_key(api_key)

    if resolved_key and cleaned_prompt:
        few_shot_block = "\n".join(
            f'Example Input: "{ex["user_request"]}"\nExample JSON Output: {json.dumps(ex["output"])}'
            for ex in FEW_SHOT_INTENT_EXAMPLES
        )
        llm_prompt = (
            "You are a specialized music recommendation intent parser. Follow the few-shot examples below to extract "
            "a JSON object from the user's request with keys:\n"
            '- "label": short descriptive persona title (string)\n'
            '- "genre": primary genre from [pop, lofi, rock, ambient, jazz, synthwave, indie pop, reggae, classical, edm, folk, r&b, hip hop, metal, latin, blues, funk, city pop, sophisti-pop, nepali indie, nepali rock, hindi indie] (string)\n'
            '- "mood": primary mood from [happy, chill, intense, relaxed, moody, focused, peaceful, melancholic, euphoric, nostalgic, romantic, confident, aggressive, festive, sorrowful, groovy] (string)\n'
            '- "energy": target energy float between 0.0 and 1.0\n'
            '- "likes_acoustic": boolean (true if acoustic/organic/piano/guitar/strings preferred, false if electronic/synth/heavy amplification preferred)\n'
            '- "target_valence": float between 0.0 (sad/dark) and 1.0 (happy/uplifting)\n\n'
            f"Few-Shot Examples:\n{few_shot_block}\n\n"
            f'User Request: "{cleaned_prompt}"'
        )
        raw_json = _call_gemini_api(llm_prompt, resolved_key, json_mode=True)
        if raw_json:
            try:
                parsed = json.loads(raw_json)
                parsed["natural_query"] = cleaned_prompt
                parsed["parser_backend"] = "Google Gemini 2.5 Flash"
                logger.info(f"Parsed natural query via Gemini API: {parsed.get('label')} ({parsed.get('genre')}/{parsed.get('mood')})")
                return parsed
            except json.JSONDecodeError:
                pass

    text = cleaned_prompt.lower()

    # 1. Detect genre
    genre = "lofi"
    genre_keywords = [
        ("city pop", ["city pop", "japanese", "tokyo", "mariya", "anri", "matsubara", "80s japanese"]),
        ("sophisti-pop", ["sophisti-pop", "sophistipop", "smooth pop", "sade"]),
        ("nepali rock", ["nepali hype", "nepali rock", "nepali party", "nepali workout"]),
        ("nepali indie", ["nepali", "nepal", "sabin rai", "axix", "kta haru", "kathmandu"]),
        ("hindi indie", ["hindi", "bollywood", "tasteful hindi", "desi", "urdu"]),
        ("synthwave", ["synthwave", "cyberpunk", "retro 80s", "night drive", "darksynth", "neon"]),
        ("lofi", ["lofi", "lo-fi", "study", "coding", "programming", "homework", "library", "pomodoro", "debugging", "exam"]),
        ("ambient", ["ambient", "meditation", "sleep", "spacewalk", "stargazing", "weightless", "anxiety"]),
        ("jazz", ["jazz", "coffee", "espresso", "cafe", "saxophone", "bossa"]),
        ("classical", ["classical", "piano", "sonata", "nocturne", "orchestra", "cello", "chamber"]),
        ("metal", ["metal", "thrash", "powerlifting", "heavy metal", "headbanging", "blast"]),
        ("rock", ["rock", "guitar", "stadium", "adrenaline"]),
        ("edm", ["edm", "rave", "festival", "house", "club", "drop", "euphoric"]),
        ("funk", ["funk", "disco", "groovy", "dancefloor"]),
        ("hip hop", ["hip hop", "rap", "boom-bap", "cypher", "808", "basketball", "swagger"]),
        ("r&b", ["r&b", "soul", "romantic", "date night", "velvet"]),
        ("folk", ["folk", "campfire", "hiking", "mountain", "rustic", "autumn"]),
        ("blues", ["blues", "heartache", "sorrow", "crossroads"]),
        ("latin", ["latin", "samba", "copacabana", "festive"]),
        ("reggae", ["reggae", "beach", "island", "tropical", "coastal"]),
        ("indie pop", ["indie", "rooftop", "jangle"]),
        ("pop", ["pop", "upbeat", "morning", "cardio", "workout", "happy", "sunny"]),
    ]
    for g_name, kw_list in genre_keywords:
        if any(kw in text for kw in kw_list):
            genre = g_name
            break

    # 2. Detect mood
    mood = "chill"
    mood_keywords = [
        ("melancholic", ["melancholic", "sad", "sorrow", "existential", "dread", "lonely", "somber", "heartache"]),
        ("chill", ["chill", "rainy", "rain", "late night", "study", "lofi", "pomodoro"]),
        ("focused", ["focus", "focused", "coding", "programming", "productive", "deep work", "concentration", "debugging", "exam"]),
        ("peaceful", ["peaceful", "sleep", "meditation", "serene", "calm", "tranquil", "nature", "sunrise"]),
        ("relaxed", ["relaxed", "coffee", "cafe", "sunday", "lounge", "cozy"]),
        ("aggressive", ["aggressive", "rage", "powerlifting", "heavy", "metal", "extreme"]),
        ("intense", ["intense", "workout", "gym", "sprint", "hiit", "action", "adrenaline", "fast"]),
        ("euphoric", ["euphoric", "rave", "festival", "ecstatic"]),
        ("groovy", ["groovy", "disco", "funk", "dancefloor", "city pop", "japanese", "mariya", "anri"]),
        ("festive", ["festive", "party", "celebration", "carnival", "latin"]),
        ("romantic", ["romantic", "date", "love", "velvet", "smooth", "hindi", "bollywood", "tasteful hindi"]),
        ("confident", ["confident", "swagger", "hustle", "basketball"]),
        ("moody", ["moody", "night drive", "dark", "cyberpunk"]),
        ("nostalgic", ["nostalgic", "campfire", "autumn", "memories", "nepali fav", "nepali", "sabin rai", "axix"]),
        ("happy", ["happy", "upbeat", "joyful", "cheerful", "sunny", "summer", "morning"]),
    ]
    for m_name, kw_list in mood_keywords:
        if any(kw in text for kw in kw_list):
            mood = m_name
            break

    high_energy_terms = ["workout", "gym", "intense", "fast", "running", "cardio", "party", "dance", "heavy", "metal", "edm", "rave", "energetic", "high energy", "power", "hype", "city pop"]
    low_energy_terms = ["sleep", "calm", "quiet", "peaceful", "study", "coding", "library", "rain", "rainy", "meditation", "chill", "relaxed", "slow", "low energy", "late night", "pomodoro", "debugging", "hindi", "nepali fav"]

    if any(t in text for t in high_energy_terms):
        energy = 0.80 if "city pop" in text else 0.86
    elif any(t in text for t in low_energy_terms):
        energy = 0.48 if ("hindi" in text or "nepali" in text) else 0.32
    else:
        energy = 0.55

    electronic_terms = ["electronic", "synth", "synthwave", "edm", "cyberpunk", "80s", "rave", "disco", "electric", "metal", "rock", "808", "gym", "city pop", "hype"]
    acoustic_terms = ["acoustic", "piano", "guitar", "organic", "strings", "unplugged", "rain", "coffee", "folk", "classical", "jazz", "lofi", "campfire", "nepali", "hindi", "bollywood", "sophisti-pop"]

    if any(t in text for t in electronic_terms) and not any(t in text for t in ["acoustic", "piano"]):
        likes_acoustic = False
    elif any(t in text for t in acoustic_terms):
        likes_acoustic = True
    else:
        likes_acoustic = energy < 0.55

    sad_terms = ["sad", "melancholic", "sorrow", "existential", "dread", "dark", "rainy", "lonely"]
    happy_terms = ["happy", "upbeat", "joyful", "party", "festive", "groovy", "sunny", "celebration"]
    if any(t in text for t in sad_terms):
        valence = 0.32
    elif any(t in text for t in happy_terms):
        valence = 0.84
    else:
        valence = 0.58

    parsed_profile = {
        "label": f"AI DJ Intent: {cleaned_prompt[:48]}" if cleaned_prompt else "Custom Listener",
        "genre": genre,
        "favorite_genre": genre,
        "mood": mood,
        "favorite_mood": mood,
        "energy": energy,
        "target_energy": energy,
        "likes_acoustic": likes_acoustic,
        "target_valence": valence,
        "natural_query": cleaned_prompt,
        "parser_backend": "Specialized Local Semantic Parser",
    }
    logger.info(
        f"Parsed natural query via Local Semantic Parser: genre='{genre}', mood='{mood}', "
        f"energy={energy:.2f}, likes_acoustic={likes_acoustic}"
    )
    return parsed_profile


def generate_rag_explanation(
    user_prefs: Dict[str, Any],
    recommendations: List[Tuple[Dict[str, Any], float, str]],
    api_key: Optional[str] = None,
    rag_index: Optional[SongKnowledgeIndex] = None,
    dj_style: str = "musicologist",
) -> Tuple[str, str]:
    """
    Generate a grounded Multi-Source RAG explanation with Few-Shot Constrained Tone/Style:
    - 'musicologist': Psychoacoustic & harmonic analysis (Default Specialized Style)
    - 'late_night_radio': Poetic midnight FM vinyl host with [Crossfade Cue] transitions
    - 'hype_coach': High-adrenaline athletic cadence coach with [Cadence Zone] markers
    - 'baseline': Unstyled plain checklist for baseline comparison
    """
    style_key = dj_style if dj_style in SPECIALIZED_DJ_STYLES else "musicologist"
    style_cfg = SPECIALIZED_DJ_STYLES[style_key]

    if not recommendations:
        return "No matching tracks were retrieved from the catalog.", f"Local Grounded RAG ({style_cfg['name']})"

    resolved_key = _resolve_gemini_api_key(api_key)
    query_str = user_prefs.get("natural_query") or (
        f"{user_prefs.get('genre', '')} / {user_prefs.get('mood', '')} "
        f"(target energy {float(user_prefs.get('energy', 0.5)):.2f})"
    )

    # Unspecialized Baseline Mode (for measurable side-by-side comparison)
    if style_key == "baseline":
        base_lines = [f"Baseline Unspecialized Output for '{query_str}':"]
        for rank, (song, score, reasons) in enumerate(recommendations, start=1):
            title = song.get("title", "Untitled") if isinstance(song, dict) else getattr(song, "title", "Untitled")
            artist = song.get("artist", "Unknown") if isinstance(song, dict) else getattr(song, "artist", "Unknown")
            base_lines.append(f"{rank}. {title} by {artist} (Score: {score:.2f}) - {reasons}")
        return "\n".join(base_lines), "Baseline Unspecialized Output"

    multi_source_ctx = (
        rag_index.retrieve_multi_source_context(query_str)
        if rag_index is not None
        else load_external_knowledge_sources()
    )
    matched_guide = multi_source_ctx.get("matched_guide") if isinstance(multi_source_ctx, dict) else None
    matched_notes = multi_source_ctx.get("matched_custom_notes", []) if isinstance(multi_source_ctx, dict) else []

    retrieved_context_lines = []
    if matched_guide:
        retrieved_context_lines.append(
            f"[Source: musicology_guides.json — {matched_guide.get('title')}]: {matched_guide.get('insight')}"
        )
    for note in matched_notes:
        retrieved_context_lines.append(f"[Source: custom_listener_notes.md]: {note}")

    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        title = song.get("title", "Untitled") if isinstance(song, dict) else getattr(song, "title", "Untitled")
        artist = song.get("artist", "Unknown") if isinstance(song, dict) else getattr(song, "artist", "Unknown")
        genre = song.get("genre", "") if isinstance(song, dict) else getattr(song, "genre", "")
        mood = song.get("mood", "") if isinstance(song, dict) else getattr(song, "mood", "")
        desc = song.get("description", "") if isinstance(song, dict) else getattr(song, "description", "")
        themes = song.get("themes", "") if isinstance(song, dict) else getattr(song, "themes", "")
        retrieved_context_lines.append(
            f"[Source: songs.csv] Rank #{rank}: '{title}' by {artist} ({genre}/{mood}, Score: {score*100:.1f}%) | "
            f"Description: {desc} | Themes: {themes} | Scoring Breakdown: {reasons}"
        )

    retrieved_block = "\n".join(retrieved_context_lines)

    if resolved_key:
        rag_prompt = (
            f"You are Muse-ic's Explainable Multi-Source AI DJ operating in '{style_cfg['name']}' mode.\n"
            f"Constrained Style Instruction: {style_cfg['instruction']}\n"
            f"{style_cfg['few_shot_example']}\n\n"
            "Using ONLY the retrieved multi-source context below (songs.csv, musicology_guides.json, and custom_listener_notes.md), "
            "write a concise 1-paragraph setlist curator summary followed by 1 bullet point per track.\n"
            "STRICT RULE: Do NOT invent or mention any song not listed in Retrieved Context.\n\n"
            f"Listener Vibe / Request: {query_str}\n\n"
            f"Retrieved Multi-Source Context:\n{retrieved_block}"
        )
        llm_response = _call_gemini_api(rag_prompt, resolved_key, json_mode=False)
        if llm_response:
            logger.info(f"Synthesized grounded Multi-Source RAG explanation via Gemini 2.5 Flash ({style_key}).")
            return llm_response, f"Google Gemini 2.5 Flash ({style_cfg['name']})"

    top_song, top_score, _ = recommendations[0]
    top_title = top_song.get("title", "Untitled") if isinstance(top_song, dict) else getattr(top_song, "title", "Untitled")
    top_artist = top_song.get("artist", "Unknown") if isinstance(top_song, dict) else getattr(top_song, "artist", "Unknown")

    if style_key == "late_night_radio":
        header = (
            f"🎙️ **[On-Air Midnight FM Vinyl Broadcast — _{query_str}_]:** "
            f"Welcome back to the night shift. We pulled **{len(recommendations)} tracks** from our 3-source crate "
            f"(`songs.csv`, `musicology_guides.json`, `custom_listener_notes.md`), opening with the velvet resonance of "
            f"**\"{top_title}\"** by *{top_artist}* ({top_score * 100:.1f}% resonance):\n"
        )
    elif style_key == "hype_coach":
        header = (
            f"🔥 **[Performance Cadence Briefing — _{query_str}_]:** "
            f"Locked in **{len(recommendations)} high-impact tracks** across **3 telemetry sources** "
            f"(`songs.csv`, `musicology_guides.json`, `custom_listener_notes.md`). Leading the pace at #1 is "
            f"**\"{top_title}\"** by *{top_artist}* ({top_score * 100:.1f}% target lock):\n"
        )
    else:
        header = (
            f"**🎧 [Psychoacoustic Musicology Curation — _{query_str}_]:** "
            f"Retrieved and ranked **{len(recommendations)} catalog tracks** across **3 knowledge sources** "
            f"(`songs.csv`, `musicology_guides.json`, and `custom_listener_notes.md`). Anchored by **\"{top_title}\"** "
            f"by *{top_artist}* ({top_score * 100:.1f}% match):\n"
        )

    lines = [header]

    if matched_guide:
        lines.append(
            f"> 🧠 **Retrieved Musicology Insight (`musicology_guides.json` — {matched_guide.get('title')}):** "
            f"{matched_guide.get('insight').strip()}\n"
        )
    if matched_notes:
        lines.append(
            f"> 📝 **Retrieved Custom Curation Note (`custom_listener_notes.md`):** {matched_notes[0]}\n"
        )

    artist_profiles = (
        rag_index.external_knowledge.get("artist_profiles", {})
        if rag_index is not None
        else {}
    )

    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        title = song.get("title", "Untitled") if isinstance(song, dict) else getattr(song, "title", "Untitled")
        artist = song.get("artist", "Unknown") if isinstance(song, dict) else getattr(song, "artist", "Unknown")
        bpm = int(float(song.get("tempo_bpm", 100) if isinstance(song, dict) else getattr(song, "tempo_bpm", 100)))
        energy = float(song.get("energy", 0.5) if isinstance(song, dict) else getattr(song, "energy", 0.5))
        desc = (song.get("description", "") if isinstance(song, dict) else getattr(song, "description", "")) or "Matched via acoustic profile."
        artist_bio = artist_profiles.get(artist, {}).get("bio", "")
        lore_snippet = f" *(Artist Lore: {artist_bio})*" if artist_bio and rank <= 2 else ""

        if style_key == "late_night_radio":
            style_tag = f"🎙️ `[Crossfade Cue {bpm} BPM]`"
        elif style_key == "hype_coach":
            style_tag = f"⚡ `[Cadence Zone: {bpm} BPM | Energy {energy:.2f}]`"
        else:
            style_tag = f"🔬 `[Acoustic Telemetry: {bpm} BPM | Energy {energy:.2f}]`"

        lines.append(
            f"- **#{rank} \"{title}\" — {artist}** (`{score * 100:.1f}%`) {style_tag}: {desc}{lore_snippet} "
            f"*[Sub-scores: {reasons}]*"
        )

    logger.info(f"Synthesized grounded Multi-Source RAG explanation via Local Synthesizer (style={style_key}).")
    return "\n".join(lines), f"Local Grounded Multi-Source RAG ({style_cfg['name']})"

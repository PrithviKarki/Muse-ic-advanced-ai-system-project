"""
Input and output guardrails for the Muse-ic recommendation pipeline.

Enforces:
1. Automatic weight normalization (preventing score > 1.0 overflow).
2. User profile & natural language prompt sanitization (clamping out-of-bounds
   audio features, stripping control characters, blocking prompt-injection patterns).
3. Post-recommendation verification (checking score bounds, catalog grounding,
   and detecting catalog starvation / low-confidence matches).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from src.logger import get_logger
except ModuleNotFoundError:
    from logger import get_logger

logger = get_logger("muse_ic.guardrails")

DEFAULT_WEIGHTS: Dict[str, float] = {
    "genre": 0.25,
    "mood": 0.20,
    "energy": 0.20,
    "acoustic": 0.15,
    "semantic_rag": 0.20,
}

CLASSIC_WEIGHTS: Dict[str, float] = {
    "genre": 0.35,
    "mood": 0.30,
    "energy": 0.20,
    "acoustic": 0.15,
    "semantic_rag": 0.00,
}

MAX_QUERY_LENGTH = 500
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"forget\s+everything",
]


@dataclass
class GuardrailReport:
    """Structured report of all guardrail checks and interventions on a run."""

    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    normalized_weights: Dict[str, float] = field(default_factory=dict)
    low_confidence_detected: bool = False
    hallucination_blocked: bool = False


def sanitize_query_text(query: Optional[str]) -> Tuple[str, List[str]]:
    """Sanitize a free-text user prompt for length, control chars, and injection patterns."""
    warnings: List[str] = []
    if not query or not isinstance(query, str):
        return "", warnings

    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", query).strip()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            msg = "Guardrail blocked potential prompt-injection pattern in natural query."
            logger.warning(msg)
            warnings.append(msg)
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if len(cleaned) > MAX_QUERY_LENGTH:
        msg = f"Guardrail truncated natural query from {len(cleaned)} to {MAX_QUERY_LENGTH} chars."
        logger.warning(msg)
        warnings.append(msg)
        cleaned = cleaned[:MAX_QUERY_LENGTH].strip()

    return cleaned, warnings


def validate_and_normalize_weights(
    weights: Optional[Dict[str, float]] = None,
    use_semantic_rag: bool = True,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Ensure scoring weights are non-negative and sum strictly to 1.0.

    Directly prevents the Experiment 1 weight overflow bug (where weights summed
    to 1.025 and produced scores > 100%).
    """
    warnings: List[str] = []
    base_defaults = DEFAULT_WEIGHTS if use_semantic_rag else CLASSIC_WEIGHTS

    if not weights:
        return dict(base_defaults), warnings

    sanitized: Dict[str, float] = {}
    for key in base_defaults:
        raw_val = weights.get(key, base_defaults[key])
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            val = base_defaults[key]
            msg = f"Guardrail replaced non-numeric weight '{key}={raw_val}' with default {val:.2f}."
            logger.warning(msg)
            warnings.append(msg)

        if val < 0.0:
            msg = f"Guardrail clamped negative weight '{key}={val:.2f}' to 0.00."
            logger.warning(msg)
            warnings.append(msg)
            val = 0.0
        sanitized[key] = val

    total = sum(sanitized.values())
    if total <= 1e-6:
        msg = "Guardrail detected all-zero weights; restored default normalized weights."
        logger.warning(msg)
        warnings.append(msg)
        return dict(base_defaults), warnings

    if abs(total - 1.0) > 1e-4:
        msg = f"Guardrail normalized scoring weights (sum was {total:.4f} -> 1.0000)."
        logger.warning(msg)
        warnings.append(msg)
        sanitized = {k: round(v / total, 6) for k, v in sanitized.items()}

    return sanitized, warnings


def sanitize_user_profile(user_prefs: Any) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and clamp user preference values into safe numeric and string bounds.
    Accepts either a UserProfile dataclass instance or a dictionary.
    """
    warnings: List[str] = []

    def _get(key: str, alt_key: Optional[str] = None, default: Any = None) -> Any:
        if hasattr(user_prefs, key):
            return getattr(user_prefs, key)
        if alt_key and hasattr(user_prefs, alt_key):
            return getattr(user_prefs, alt_key)
        if isinstance(user_prefs, dict):
            if key in user_prefs:
                return user_prefs[key]
            if alt_key and alt_key in user_prefs:
                return user_prefs[alt_key]
        return default

    genre = str(_get("favorite_genre", "genre", "") or "").strip()
    mood = str(_get("favorite_mood", "mood", "") or "").strip()
    label = str(_get("label", default="Custom Listener") or "Custom Listener").strip()

    # Clamp target_energy to [0.0, 1.0]
    raw_energy = _get("target_energy", "energy", 0.5)
    try:
        energy = float(raw_energy) if raw_energy is not None else 0.5
        if energy < 0.0 or energy > 1.0:
            clamped = max(0.0, min(1.0, energy))
            msg = f"Guardrail clamped target_energy from {energy} to {clamped:.2f}."
            logger.warning(msg)
            warnings.append(msg)
            energy = clamped
    except (ValueError, TypeError):
        energy = 0.5
        msg = f"Guardrail reset invalid target_energy '{raw_energy}' to 0.50."
        logger.warning(msg)
        warnings.append(msg)

    # Optional target_valence in [0.0, 1.0]
    raw_valence = _get("target_valence", "valence", None)
    valence: Optional[float] = None
    if raw_valence is not None:
        try:
            v_val = float(raw_valence)
            if v_val < 0.0 or v_val > 1.0:
                v_val = max(0.0, min(1.0, v_val))
                warnings.append(f"Guardrail clamped target_valence to {v_val:.2f}.")
            valence = v_val
        except (ValueError, TypeError):
            valence = None

    # Optional target_tempo_bpm in [40.0, 220.0]
    raw_tempo = _get("target_tempo_bpm", "tempo_bpm", None)
    tempo_bpm: Optional[float] = None
    if raw_tempo is not None:
        try:
            t_val = float(raw_tempo)
            if t_val < 40.0 or t_val > 220.0:
                t_val = max(40.0, min(220.0, t_val))
                warnings.append(f"Guardrail clamped target_tempo_bpm to {t_val:.0f} BPM.")
            tempo_bpm = t_val
        except (ValueError, TypeError):
            tempo_bpm = None

    raw_acoustic = _get("likes_acoustic", default=None)
    likes_acoustic = bool(raw_acoustic) if raw_acoustic is not None else None

    raw_query = _get("natural_query", "query", "")
    cleaned_query, query_warnings = sanitize_query_text(raw_query)
    warnings.extend(query_warnings)

    raw_ignored = _get("ignored_artists", default=None)
    ignored_artists: List[str] = []
    if isinstance(raw_ignored, (list, set, tuple)):
        ignored_artists = [str(a).strip().lower() for a in raw_ignored if str(a).strip()]
    elif isinstance(raw_ignored, str) and raw_ignored.strip():
        ignored_artists = [a.strip().lower() for a in raw_ignored.split(",") if a.strip()]

    sanitized_profile = {
        "label": label,
        "favorite_genre": genre,
        "genre": genre,
        "favorite_mood": mood,
        "mood": mood,
        "target_energy": energy,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
        "target_valence": valence,
        "target_tempo_bpm": tempo_bpm,
        "natural_query": cleaned_query,
        "ignored_artists": ignored_artists,
    }
    return sanitized_profile, warnings


def verify_recommendations(
    recommendations: List[Tuple[Dict[str, Any], float, str]],
    catalog_titles: Optional[Set[str]] = None,
    min_top_score: float = 0.42,
) -> GuardrailReport:
    """
    Verify recommended tracks for catalog grounding, valid score bounds [0.0, 1.0],
    and low-confidence / catalog starvation conditions.
    """
    report = GuardrailReport(passed=True)

    if not recommendations:
        report.passed = False
        report.low_confidence_detected = True
        msg = "Guardrail alert: recommendation list is empty."
        logger.warning(msg)
        report.warnings.append(msg)
        return report

    top_score = recommendations[0][1]
    if top_score < min_top_score:
        report.low_confidence_detected = True
        msg = (
            f"Low-confidence match detected (top score {top_score:.2f} < {min_top_score:.2f}). "
            "Semantic RAG fallback broadened search across related vibes."
        )
        logger.info(msg)
        report.warnings.append(msg)

    for song, score, _ in recommendations:
        title = song.get("title", "") if isinstance(song, dict) else getattr(song, "title", "")
        if catalog_titles and title not in catalog_titles:
            report.passed = False
            report.hallucination_blocked = True
            msg = f"Guardrail blocked ungrounded track '{title}' not present in catalog."
            logger.error(msg)
            report.warnings.append(msg)

        if score < 0.0 or score > 1.0001:
            msg = f"Guardrail detected out-of-range score {score:.4f} for '{title}'."
            logger.warning(msg)
            report.warnings.append(msg)

    return report


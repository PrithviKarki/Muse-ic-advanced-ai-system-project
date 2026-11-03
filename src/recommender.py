import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from src.guardrails import (
        GuardrailReport,
        sanitize_user_profile,
        validate_and_normalize_weights,
        verify_recommendations,
    )
    from src.logger import append_agent_trace, get_logger
    from src.rag_engine import (
        SongKnowledgeIndex,
        generate_rag_explanation,
        get_semantic_genre_similarity,
        get_semantic_mood_similarity,
        parse_natural_language_request,
    )
except ModuleNotFoundError:
    from guardrails import (
        GuardrailReport,
        sanitize_user_profile,
        validate_and_normalize_weights,
        verify_recommendations,
    )
    from logger import append_agent_trace, get_logger
    from rag_engine import (
        SongKnowledgeIndex,
        generate_rag_explanation,
        get_semantic_genre_similarity,
        get_semantic_mood_similarity,
        parse_natural_language_request,
    )

logger = get_logger("muse_ic.recommender")


@dataclass
class Song:
    """
    Represents a song and its attributes, including enriched RAG knowledge fields.
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
    description: str = ""
    themes: str = ""


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences, supporting both explicit attributes
    and natural-language RAG queries.
    Required by tests/test_recommender.py
    """

    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    natural_query: str = ""
    target_valence: Optional[float] = None
    target_tempo_bpm: Optional[float] = None
    ignored_artists: Optional[List[str]] = None


class Recommender:
    """
    OOP implementation of the hybrid Semantic RAG + Content-Based recommendation logic.
    Required by tests/test_recommender.py
    """

    def __init__(self, songs: List[Any]):
        self.songs = songs
        self.rag_index = SongKnowledgeIndex(songs)

    def recommend(self, user: UserProfile, k: int = 5) -> List[Any]:
        """Return the top k recommended songs for a user profile sorted by compatibility and diversity."""
        ranked = recommend_songs(user, self.songs, k=k, rag_index=self.rag_index)
        return [song for song, _, _ in ranked]

    def explain_recommendation(self, user: UserProfile, song: Any) -> str:
        """Generate a human-readable explanation string detailing why a song was recommended."""
        _, reasons = score_song(user, song, rag_index=self.rag_index)
        return ", ".join(reasons) if reasons else "General musical match"

    def recommend_with_rag(
        self,
        prompt_or_profile: Any,
        k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        artist_penalty: float = 0.15,
        genre_penalty: float = 0.10,
        api_key: Optional[str] = None,
        custom_notes: str = "",
        dj_style: str = "musicologist",
    ) -> Dict[str, Any]:
        """
        Execute the Multi-Step Agentic RAG Workflow:
        - Step 1 [PLAN]: Parse intent, run input guardrails, and inspect catalog genre/energy feasibility.
        - Step 2 [TOOL CALL #1]: Retrieve multi-source RAG context and execute initial hybrid ranking.
        - Step 3 [CRITIQUE & DECIDE]: Evaluate Pass #1 confidence & catalog starvation; if out-of-distribution
          or contradictory constraints degrade scores, autonomously rebalance weights and invoke [TOOL CALL #2].
        - Step 4 [SYNTHESIZE]: Verify catalog grounding, synthesize AI DJ explanation, and persist trace.
        """
        agent_trace: List[Dict[str, Any]] = []

        if custom_notes is not None:
            self.rag_index.update_custom_notes(custom_notes)

        # ---------------------------------------------------------------------
        # STEP 1: [PLAN] Intent Parsing, Input Guardrails & Catalog Inspection
        # ---------------------------------------------------------------------
        if isinstance(prompt_or_profile, str):
            raw_profile = parse_natural_language_request(prompt_or_profile, api_key=api_key)
        else:
            raw_profile = prompt_or_profile

        sanitized_profile, input_warnings = sanitize_user_profile(raw_profile)
        norm_weights, weight_warnings = validate_and_normalize_weights(weights, use_semantic_rag=True)

        target_genre = str(sanitized_profile.get("genre", "")).strip().lower()
        target_mood = str(sanitized_profile.get("mood", "")).strip().lower()
        target_energy = float(sanitized_profile.get("energy", 0.5))

        genre_tracks = [
            s
            for s in self.songs
            if str(_extract_attr(s, "genre", default="")).strip().lower() == target_genre
        ]
        exact_genre_count = len(genre_tracks)
        avg_genre_energy = (
            sum(float(_extract_attr(s, "energy", default=0.5)) for s in genre_tracks) / exact_genre_count
            if exact_genre_count > 0
            else target_energy
        )
        energy_conflict_gap = abs(target_energy - avg_genre_energy) if exact_genre_count > 0 else 0.0

        plan_obs = (
            f"Parsed target genre='{target_genre}' ({exact_genre_count} exact tracks in catalog), "
            f"mood='{target_mood}', energy={target_energy:.2f} (genre avg energy={avg_genre_energy:.2f}, "
            f"gap={energy_conflict_gap:.2f})."
        )
        plan_decision = (
            "Flagged potential OOD/Contradiction risk for Critic inspection after Pass #1."
            if (exact_genre_count == 0 or energy_conflict_gap >= 0.45)
            else "Standard catalog coverage confirmed; proceed to Pass #1 retrieval."
        )
        agent_trace.append(
            {
                "step": 1,
                "phase": "PLAN & GUARDRAILS",
                "tool": "parse_natural_language_request + sanitize_user_profile",
                "thought": "Extract structured listener constraints and audit catalog coverage before scoring.",
                "observation": plan_obs,
                "decision": plan_decision,
            }
        )

        # ---------------------------------------------------------------------
        # STEP 2: [TOOL CALL #1] Multi-Source RAG Retrieval & Pass #1 Ranking
        # ---------------------------------------------------------------------
        query_str = sanitized_profile.get("natural_query") or f"{target_genre} {target_mood}"
        multi_ctx = self.rag_index.retrieve_multi_source_context(query_str)
        pass1_recs = recommend_songs(
            sanitized_profile,
            self.songs,
            k=k,
            artist_penalty=artist_penalty,
            genre_penalty=genre_penalty,
            weights=norm_weights,
            rag_index=self.rag_index,
        )
        pass1_top_score = pass1_recs[0][1] if pass1_recs else 0.0
        pass1_avg_score = (
            sum(score for _, score, _ in pass1_recs) / len(pass1_recs) if pass1_recs else 0.0
        )
        guide_title = (
            multi_ctx["matched_guide"]["title"]
            if multi_ctx.get("matched_guide")
            else "General Acoustic Profile"
        )

        agent_trace.append(
            {
                "step": 2,
                "phase": "TOOL CALL #1 (RETRIEVE & RANK PASS 1)",
                "tool": "SongKnowledgeIndex.retrieve_multi_source_context + recommend_songs",
                "thought": (
                    f"Retrieve across {len(multi_ctx.get('active_sources', []))} data sources and rank top-{k} "
                    f"using baseline weights (genre={norm_weights['genre']:.2f}, mood={norm_weights['mood']:.2f}, "
                    f"energy={norm_weights['energy']:.2f}, rag={norm_weights['semantic_rag']:.2f})."
                ),
                "observation": (
                    f"Matched Musicology Guide: '{guide_title}'. Pass #1 Top-1 Score = {pass1_top_score*100:.1f}%, "
                    f"Top-{len(pass1_recs)} Mean Score = {pass1_avg_score*100:.1f}%."
                ),
                "decision": "Send Pass #1 candidate pool to Critic Decision Node.",
            }
        )

        # ---------------------------------------------------------------------
        # STEP 3: [CRITIQUE & DECIDE] Autonomous Self-Correction Loop
        # ---------------------------------------------------------------------
        recommendations = pass1_recs
        agent_self_corrected = False
        active_weights = dict(norm_weights)

        if weights is None and (
            exact_genre_count == 0 or energy_conflict_gap >= 0.45 or pass1_top_score < 0.65
        ):
            if exact_genre_count == 0:
                rebalanced_raw = {
                    "genre": 0.05,
                    "mood": 0.25,
                    "energy": 0.20,
                    "acoustic": 0.15,
                    "semantic_rag": 0.35,
                }
                reason_str = (
                    f"Out-of-distribution genre '{target_genre}' has 0 exact catalog tracks, "
                    "penalizing all candidates by 25%."
                )
            else:
                rebalanced_raw = {
                    "genre": 0.20,
                    "mood": 0.20,
                    "energy": 0.30,
                    "acoustic": 0.10,
                    "semantic_rag": 0.20,
                }
                reason_str = (
                    f"Genre-Energy contradiction detected (target energy {target_energy:.2f} vs "
                    f"'{target_genre}' catalog average {avg_genre_energy:.2f})."
                )

            rebalanced_weights, _ = validate_and_normalize_weights(rebalanced_raw, use_semantic_rag=True)
            pass2_recs = recommend_songs(
                sanitized_profile,
                self.songs,
                k=k,
                artist_penalty=artist_penalty,
                genre_penalty=genre_penalty,
                weights=rebalanced_weights,
                rag_index=self.rag_index,
            )
            pass2_top_score = pass2_recs[0][1] if pass2_recs else 0.0
            pass2_avg_score = (
                sum(score for _, score, _ in pass2_recs) / len(pass2_recs) if pass2_recs else 0.0
            )

            recommendations = pass2_recs
            active_weights = rebalanced_weights
            agent_self_corrected = True

            agent_trace.append(
                {
                    "step": 3,
                    "phase": "CRITIQUE & SELF-CORRECTION (TOOL CALL #2)",
                    "tool": "adaptive_weight_rebalancer + recommend_songs (Pass 2)",
                    "thought": f"Critic flagged Pass #1: {reason_str} Rebalancing weights to prioritize Semantic RAG & viable audio features.",
                    "observation": (
                        f"Re-ran `recommend_songs` with adaptive weights (genre={rebalanced_weights['genre']:.2f}, "
                        f"mood={rebalanced_weights['mood']:.2f}, energy={rebalanced_weights['energy']:.2f}, "
                        f"rag={rebalanced_weights['semantic_rag']:.2f}). "
                        f"Top-1 Score shifted {pass1_top_score*100:.1f}% -> {pass2_top_score*100:.1f}% "
                        f"(Mean {pass1_avg_score*100:.1f}% -> {pass2_avg_score*100:.1f}%)."
                    ),
                    "decision": "SELF_CORRECTED: Adopted Pass #2 adaptive ranking.",
                }
            )
        else:
            agent_trace.append(
                {
                    "step": 3,
                    "phase": "CRITIQUE & DECISION NODE",
                    "tool": "verify_candidate_confidence",
                    "thought": "Verify whether Pass #1 top score and genre/energy alignment meet quality thresholds (>= 65.0%).",
                    "observation": (
                        f"Pass #1 Top-1 Score ({pass1_top_score*100:.1f}%) and Mean Score "
                        f"({pass1_avg_score*100:.1f}%) satisfy quality and diversity invariants."
                    ),
                    "decision": "APPROVED: Retain Pass #1 ranking without weight override.",
                }
            )

        # ---------------------------------------------------------------------
        # STEP 4: [SYNTHESIZE] Output Verification & Grounded RAG Commentary
        # ---------------------------------------------------------------------
        catalog_titles = {
            (s.get("title", "") if isinstance(s, dict) else getattr(s, "title", ""))
            for s in self.songs
        }
        guardrail_report = verify_recommendations(recommendations, catalog_titles=catalog_titles)
        guardrail_report.warnings = input_warnings + weight_warnings + guardrail_report.warnings
        guardrail_report.normalized_weights = active_weights

        rag_explanation, rag_backend = generate_rag_explanation(
            sanitized_profile,
            recommendations,
            api_key=api_key,
            rag_index=self.rag_index,
            dj_style=dj_style,
        )

        top_final_title = (
            _extract_attr(recommendations[0][0], "title", "N/A") if recommendations else "None"
        )
        agent_trace.append(
            {
                "step": 4,
                "phase": "VERIFY & SYNTHESIZE",
                "tool": "verify_recommendations + generate_rag_explanation",
                "thought": "Confirm zero hallucinated titles against catalog and synthesize multi-source AI DJ setlist commentary.",
                "observation": (
                    f"Guardrail passed={guardrail_report.passed}; anchored explanation on #1 '{top_final_title}' "
                    f"via {rag_backend}."
                ),
                "decision": "OUTPUT_READY: Persist reasoning trace to logs/agent_reasoning_traces.log.",
            }
        )

        append_agent_trace(
            profile_label=str(sanitized_profile.get("label", "Custom Listener")),
            query_summary=str(query_str),
            trace_steps=agent_trace,
        )

        return {
            "profile": sanitized_profile,
            "recommendations": recommendations,
            "rag_explanation": rag_explanation,
            "rag_backend": rag_backend,
            "embedding_backend": self.rag_index.backend_name,
            "active_sources": self.rag_index.external_knowledge.get("active_sources", []),
            "guardrail_report": guardrail_report,
            "agent_trace": agent_trace,
            "agent_self_corrected": agent_self_corrected,
        }


def load_songs(csv_path: str) -> List[Dict[str, Any]]:
    """Load songs from a CSV file and return a list of dictionaries with typed attributes and RAG fields."""
    songs: List[Dict[str, Any]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song: Dict[str, Any] = {
                "id": int(row["id"]) if "id" in row and str(row["id"]).strip().isdigit() else row.get("id"),
                "title": (row.get("title") or "").strip(),
                "artist": (row.get("artist") or "").strip(),
                "genre": (row.get("genre") or "").strip(),
                "mood": (row.get("mood") or "").strip(),
                "energy": float(row["energy"]) if row.get("energy") not in (None, "") else 0.0,
                "tempo_bpm": float(row["tempo_bpm"]) if row.get("tempo_bpm") not in (None, "") else 0.0,
                "valence": float(row["valence"]) if row.get("valence") not in (None, "") else 0.0,
                "danceability": float(row["danceability"]) if row.get("danceability") not in (None, "") else 0.0,
                "acousticness": float(row["acousticness"]) if row.get("acousticness") not in (None, "") else 0.0,
                "description": (row.get("description") or "").strip(),
                "themes": (row.get("themes") or "").strip(),
            }
            songs.append(song)
    logger.info(f"Loaded {len(songs)} songs from '{csv_path}'.")
    return songs


def _extract_attr(obj: Any, key: str, alt_key: Optional[str] = None, default: Any = None) -> Any:
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


def score_song(
    user_prefs: Any,
    song: Any,
    weights: Optional[Dict[str, float]] = None,
    rag_index: Optional[SongKnowledgeIndex] = None,
) -> Tuple[float, List[str]]:
    """
    Score a song against user preferences using Hybrid Semantic RAG + Multi-Attribute
    Audio Matching and return (score, reasons).

    Integrates:
    - Genre exact, substring, and semantic affinity matching
    - Mood exact and semantic synonym matching (resolving Synonym Blindness)
    - Energy proximity (plus optional Valence & Tempo BPM proximity)
    - Acoustic/Electronic alignment
    - Dense Semantic RAG document retrieval similarity over song descriptions & themes
    """
    sanitized_user, _ = sanitize_user_profile(user_prefs)

    song_desc = str(_extract_attr(song, "description", default="") or "").strip()
    song_themes = str(_extract_attr(song, "themes", default="") or "").strip()
    natural_query = str(sanitized_user.get("natural_query", "") or "").strip()
    has_rag_context = bool(song_desc or song_themes or natural_query or weights is not None)

    norm_weights, _ = validate_and_normalize_weights(weights, use_semantic_rag=has_rag_context)
    weight_genre = norm_weights["genre"]
    weight_mood = norm_weights["mood"]
    weight_energy = norm_weights["energy"]
    weight_acoustic = norm_weights["acoustic"]
    weight_rag = norm_weights.get("semantic_rag", 0.0)

    reasons: List[str] = []

    # 1. Genre Score (Exact, Substring, or Semantic Affinity)
    user_genre = sanitized_user.get("favorite_genre", "")
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
            affinity = get_semantic_genre_similarity(ug, sg)
            if affinity > 0.0:
                genre_score = round(affinity * 0.65, 4)
                reasons.append(
                    f"semantic genre affinity '{song_genre}' (~'{user_genre}') (+{weight_genre * genre_score:.2f})"
                )
            else:
                genre_score = 0.0
    else:
        genre_score = 0.0

    # 2. Mood Score (Exact or Semantic Synonym Cluster Match)
    user_mood = sanitized_user.get("favorite_mood", "")
    song_mood = _extract_attr(song, "mood", default="")
    um = str(user_mood).strip().lower()
    sm = str(song_mood).strip().lower()

    if um and sm:
        if um == sm:
            mood_score = 1.0
            reasons.append(f"mood match '{song_mood}' (+{weight_mood * mood_score:.2f})")
        else:
            mood_sim = get_semantic_mood_similarity(um, sm)
            if mood_sim > 0.0:
                mood_score = round(mood_sim * 0.80, 4)
                reasons.append(
                    f"semantic mood synonym '{song_mood}' (~'{user_mood}') (+{weight_mood * mood_score:.2f})"
                )
            else:
                mood_score = 0.0
    else:
        mood_score = 0.0

    # 3. Energy Proximity & Extended Audio Telemetry (Valence / Tempo)
    target_e = sanitized_user.get("target_energy", 0.5)
    song_energy = _extract_attr(song, "energy", default=0.0)
    try:
        song_e = max(0.0, min(1.0, float(song_energy)))
        energy_diff = abs(song_e - target_e)
        base_energy_score = max(0.0, 1.0 - energy_diff)

        # Blend optional target_valence and target_tempo_bpm when specified
        target_v = sanitized_user.get("target_valence")
        target_t = sanitized_user.get("target_tempo_bpm")
        sub_audio_scores = [base_energy_score]

        if target_v is not None:
            song_v = max(0.0, min(1.0, float(_extract_attr(song, "valence", default=0.5) or 0.5)))
            sub_audio_scores.append(max(0.0, 1.0 - abs(song_v - target_v)))
        if target_t is not None:
            song_t = float(_extract_attr(song, "tempo_bpm", default=100.0) or 100.0)
            tempo_diff_norm = min(1.0, abs(song_t - target_t) / 100.0)
            sub_audio_scores.append(max(0.0, 1.0 - tempo_diff_norm))

        energy_score = sum(sub_audio_scores) / len(sub_audio_scores)
        if energy_diff <= 0.15:
            reasons.append(
                f"energy match ({song_e:.2f} near target {target_e:.2f}) (+{weight_energy * energy_score:.2f})"
            )
    except (ValueError, TypeError):
        energy_score = 0.0

    # 4. Acoustic Alignment (Directional match)
    user_acoustic = sanitized_user.get("likes_acoustic")
    song_acoustic = _extract_attr(song, "acousticness", default=0.0)

    if user_acoustic is not None:
        try:
            song_a = max(0.0, min(1.0, float(song_acoustic)))
            if bool(user_acoustic):
                acoustic_score = song_a
                if song_a >= 0.5:
                    reasons.append(f"acoustic sound match (+{weight_acoustic * acoustic_score:.2f})")
            else:
                acoustic_score = 1.0 - song_a
                if song_a <= 0.5:
                    reasons.append(f"electronic/non-acoustic match (+{weight_acoustic * acoustic_score:.2f})")
        except (ValueError, TypeError):
            acoustic_score = 0.0
    else:
        acoustic_score = 0.5

    # 5. Semantic RAG Retrieval Score (Vector Similarity over Song Knowledge Doc)
    rag_score = 0.0
    if weight_rag > 0.0:
        active_index = rag_index if rag_index is not None else SongKnowledgeIndex([song])
        composite_query = " ".join(
            part for part in [natural_query, ug, um, "acoustic" if user_acoustic else "electronic"] if part
        )
        rag_score, matched_terms = active_index.compute_query_similarity(composite_query, song)
        if rag_score >= 0.20:
            term_tag = f" [{'/'.join(matched_terms[:3])}]" if matched_terms else ""
            reasons.append(f"RAG semantic context{term_tag} (+{weight_rag * rag_score:.2f})")

    # Weighted Total Score (strictly normalized in [0.0, 1.0])
    final_score = (
        (weight_genre * genre_score)
        + (weight_mood * mood_score)
        + (weight_energy * energy_score)
        + (weight_acoustic * acoustic_score)
        + (weight_rag * rag_score)
    )
    final_score = max(0.0, min(1.0, final_score))

    if not reasons:
        reasons.append("general catalog match")

    return round(final_score, 4), reasons


def recommend_songs(
    user_prefs: Any,
    songs: List[Any],
    k: int = 5,
    artist_penalty: float = 0.15,
    genre_penalty: float = 0.10,
    weights: Optional[Dict[str, float]] = None,
    rag_index: Optional[SongKnowledgeIndex] = None,
) -> List[Tuple[Any, float, str]]:
    """
    Score all songs using Hybrid Semantic RAG + Audio Telemetry and iteratively
    select the top k candidates while applying artist and genre diversity penalties
    and guardrail checks.
    """
    sanitized_user, guardrail_warnings = sanitize_user_profile(user_prefs)
    ignored_artists = set(sanitized_user.get("ignored_artists") or [])

    active_index = rag_index if rag_index is not None else SongKnowledgeIndex(songs)

    remaining_pool = []
    for song in songs:
        artist_clean = str(_extract_attr(song, "artist", default="")).strip().lower()
        if artist_clean and artist_clean in ignored_artists:
            logger.info(f"Filtered out track by excluded artist '{artist_clean}'.")
            continue

        base_score, base_reasons = score_song(
            sanitized_user, song, weights=weights, rag_index=active_index
        )
        remaining_pool.append((song, base_score, list(base_reasons)))

    selected_recommendations: List[Tuple[Any, float, str]] = []
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

    top_title = (
        _extract_attr(selected_recommendations[0][0], "title", "N/A")
        if selected_recommendations
        else "None"
    )
    top_score = selected_recommendations[0][1] if selected_recommendations else 0.0
    logger.info(
        f"Recommended {len(selected_recommendations)} tracks for profile '{sanitized_user.get('label')}' "
        f"(genre='{sanitized_user.get('genre')}', mood='{sanitized_user.get('mood')}') -> "
        f"#1 '{top_title}' ({top_score:.4f}) | warnings={len(guardrail_warnings)}"
    )

    return selected_recommendations

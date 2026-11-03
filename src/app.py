"""
Streamlit Web Application for Muse-ic: Hybrid Semantic RAG & Explainable Music Recommender.

Run with:
    streamlit run src/app.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st

try:
    from src.guardrails import validate_and_normalize_weights
    from src.logger import read_recent_logs
    from src.main import USER_PROFILES
    from src.rag_engine import _resolve_gemini_api_key
    from src.recommender import Recommender, load_songs
except ModuleNotFoundError:
    from guardrails import validate_and_normalize_weights
    from logger import read_recent_logs
    from main import USER_PROFILES
    from rag_engine import _resolve_gemini_api_key
    from recommender import Recommender, load_songs


st.set_page_config(
    page_title="Muse-ic | Hybrid Semantic RAG Music Recommender",
    page_icon="🎵",
    layout="wide",
)


@st.cache_resource
def get_recommender_engine() -> tuple[Recommender, list[dict]]:
    """Load the enriched song catalog and initialize the Specialized Vector RAG Index."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
    if not csv_path.exists():
        csv_path = Path("data/songs.csv")
    songs = load_songs(str(csv_path))
    return Recommender(songs), songs


def render_recommendations_section(result: dict) -> None:
    """Render profile telemetry, guardrail status, grounded RAG synthesis, and ranked song cards."""
    profile = result["profile"]
    recommendations = result["recommendations"]
    rag_explanation = result["rag_explanation"]
    rag_backend = result["rag_backend"]
    guardrail_report = result["guardrail_report"]

    # 1. Extracted Intent & Guardrail Status Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Target Genre", str(profile.get("genre", "any")).title())
    c2.metric("Target Mood", str(profile.get("mood", "any")).title())
    c3.metric("Target Energy", f"{float(profile.get('energy', 0.5)):.2f}")
    c4.metric(
        "Instrument Style",
        "Acoustic / Organic" if profile.get("likes_acoustic") else "Electronic / Amplified",
    )
    top_score = recommendations[0][1] * 100 if recommendations else 0.0
    c5.metric("Top Match Score", f"{top_score:.1f}%")

    # 2. Guardrail Banner
    if guardrail_report.warnings:
        with st.expander("⚠️ Active Guardrail & Normalization Interventions", expanded=True):
            for w in guardrail_report.warnings:
                st.warning(w)
    else:
        st.success(
            f"🛡️ **Guardrails Verified:** All weights normalized (Σw = 1.00), inputs in bounds, "
            f"and {len(recommendations)} tracks verified against catalog. | **RAG Backend:** `{rag_backend}`"
        )

    # 3. Grounded AI DJ RAG Commentary & Agentic Workflow Trace
    agent_trace = result.get("agent_trace", [])
    agent_self_corrected = result.get("agent_self_corrected", False)
    if agent_trace:
        with st.expander(
            f"🧠 Multi-Step Agentic Workflow Trace ({len(agent_trace)} Steps | "
            f"Self-Correction Triggered: {agent_self_corrected})",
            expanded=agent_self_corrected,
        ):
            for step in agent_trace:
                st.markdown(
                    f"**Step {step['step']} — `{step['phase']}`** | 🛠️ Tool: `{step['tool']}`\n"
                    f"- **Thought:** {step['thought']}\n"
                    f"- **Observation:** {step['observation']}\n"
                    f"- **Decision:** `{step['decision']}`"
                )

    st.subheader("🤖 Grounded AI DJ Setlist Commentary (RAG)")
    st.info(rag_explanation)

    # 4. Top-K Ranked Songs Table & Detailed RAG Cards
    st.subheader(f"🎧 Top {len(recommendations)} Recommended Tracks")
    table_data = []
    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        table_data.append(
            {
                "Rank": f"#{rank}",
                "Title": song.get("title"),
                "Artist": song.get("artist"),
                "Genre": song.get("genre"),
                "Mood": song.get("mood"),
                "Energy": f"{song.get('energy', 0.0):.2f}",
                "Acousticness": f"{song.get('acousticness', 0.0):.2f}",
                "Valence": f"{song.get('valence', 0.0):.2f}",
                "BPM": int(song.get("tempo_bpm", 0)),
                "Match Score": f"{score * 100:.1f}%",
                "Explainable Scoring Breakdown": reasons,
            }
        )
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("#### 🔍 Retrieved Song Knowledge Documents & Telemetry")
    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        with st.expander(
            f"#{rank} — {song.get('title')} by {song.get('artist')} "
            f"[{song.get('genre')} / {song.get('mood')}] — {score * 100:.1f}% Match",
            expanded=(rank == 1),
        ):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"**📖 Retrieved Musical Description:** {song.get('description', 'N/A')}")
                st.markdown(f"**🏷️ Activity & Vibe Themes:** `{song.get('themes', 'N/A')}`")
                st.markdown("**🧮 Pointwise & Diversity Score Breakdown:**")
                for clause in reasons.split(", "):
                    st.markdown(f"- `{clause}`")
            with col_b:
                st.progress(min(1.0, max(0.0, float(song.get("energy", 0.0)))), text=f"Energy: {song.get('energy', 0.0):.2f}")
                st.progress(min(1.0, max(0.0, float(song.get("acousticness", 0.0)))), text=f"Acousticness: {song.get('acousticness', 0.0):.2f}")
                st.progress(min(1.0, max(0.0, float(song.get("valence", 0.0)))), text=f"Valence (Positivity): {song.get('valence', 0.0):.2f}")
                st.progress(min(1.0, max(0.0, float(song.get("danceability", 0.0)))), text=f"Danceability: {song.get('danceability', 0.0):.2f}")


def main() -> None:
    engine, songs = get_recommender_engine()

    st.title("🎵 Muse-ic 2.0: Hybrid Semantic RAG Music Recommender")
    st.caption(
        "Explainable Content-Based Filtering + Specialized Local Vector Embeddings + "
        "Grounded Gemini RAG Synthesis + Runtime Guardrails"
    )

    # -------------------------------------------------------------------------
    # Sidebar: AI Configuration, Scoring Weight Lab & Diversity Guardrails
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ AI & Algorithm Controls")
        api_key = st.text_input(
            "🔑 Google Gemini API Key (Optional)",
            type="password",
            help="Paste a free GEMINI_API_KEY from Google AI Studio (or set it in .env) to enable Gemini 2.5 Flash intent parsing & DJ commentary.",
        )
        effective_key = _resolve_gemini_api_key(api_key)
        if effective_key:
            st.success("Gemini 2.5 Flash Key Configured (.env / Sidebar)")
        else:
            st.info("Offline-Ready Local Semantic RAG Active")

        dj_style_options = {
            "musicologist": "🔬 Psychoacoustic Musicologist (Default)",
            "late_night_radio": "🎙️ Late-Night FM Radio Host",
            "hype_coach": "🔥 High-Intensity Athletic Coach",
            "baseline": "📋 Unspecialized Baseline (Plain Checklist)",
        }
        dj_style = st.selectbox(
            "🎙️ Constrained AI DJ Tone / Style (Few-Shot Specialization)",
            options=list(dj_style_options.keys()),
            format_func=lambda k: dj_style_options[k],
        )

        top_k = st.slider("Number of Recommendations (k)", min_value=1, max_value=10, value=5)

        st.divider()
        st.subheader("⚖️ Hybrid Scoring Weights")
        st.caption("Guardrail automatically normalizes weights so Σw = 1.00.")
        w_genre = st.slider("Genre Weight", 0.0, 1.0, 0.25, 0.05)
        w_mood = st.slider("Mood Weight", 0.0, 1.0, 0.20, 0.05)
        w_energy = st.slider("Energy & Audio Telemetry Weight", 0.0, 1.0, 0.20, 0.05)
        w_acoustic = st.slider("Acoustic / Electronic Weight", 0.0, 1.0, 0.15, 0.05)
        w_rag = st.slider("Semantic RAG Embedding Weight", 0.0, 1.0, 0.20, 0.05)

        raw_weights = {
            "genre": w_genre,
            "mood": w_mood,
            "energy": w_energy,
            "acoustic": w_acoustic,
            "semantic_rag": w_rag,
        }
        norm_weights, _ = validate_and_normalize_weights(raw_weights, use_semantic_rag=True)
        raw_sum = sum(raw_weights.values())
        if abs(raw_sum - 1.0) > 1e-3:
            st.warning(f"Raw weight sum = `{raw_sum:.2f}` → Auto-normalized to `1.00`")

        st.divider()
        st.subheader("🎨 Diversity & Guest Filter")
        artist_penalty = st.slider("Repeat Artist Penalty", 0.0, 0.40, 0.15, 0.05)
        genre_penalty = st.slider("Genre Saturation Penalty", 0.0, 0.30, 0.10, 0.05)

        all_artists = sorted({s.get("artist", "") for s in songs if s.get("artist")})
        ignored_artists = st.multiselect(
            "🚫 Exclude Artists (Dislike / Guest Filter)",
            options=all_artists,
            default=[],
        )

    # -------------------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------------------
    tab_rag, tab_personas, tab_observability = st.tabs(
        [
            "🤖 AI DJ & Natural Language RAG",
            "🎛️ Persona & Explicit Profile Lab",
            "🛡️ Guardrails, Logs & Vector Catalog",
        ]
    )

    # --- TAB 1: Natural Language Semantic RAG ---
    with tab_rag:
        st.subheader("Ask Muse-ic in Plain English")
        st.markdown(
            "Describe a **moment, activity, atmosphere, or emotional state**. "
            "Muse-ic parses your intent, retrieves matching tracks from the vector knowledge base, "
            "and scores them using hybrid semantic + acoustic telemetry."
        )

        preset_prompts = [
            "Late-night rainy coding session with warm acoustic piano and calm focus",
            "High-energy 80s neon cyberpunk synthwave highway night drive",
            "Existential dread and somber midnight introspection with acoustic instruments",
            "Explosive gym workout cardio and heavy adrenaline motivation",
            "Cozy Sunday morning coffee shop reading with relaxed upright bass and jazz",
        ]
        selected_preset = st.selectbox("✨ Try a sample prompt (or type your own below):", ["Custom prompt..."] + preset_prompts)

        default_query = (
            selected_preset
            if selected_preset != "Custom prompt..."
            else "Late-night rainy coding session with warm acoustic piano and calm focus"
        )
        user_query = st.text_area("🎧 Your Vibe / Request:", value=default_query, height=80)

        with st.expander("📚 Multi-Source RAG: Add Custom Listener Document / Curation Notes (Stretch Goal)"):
            st.caption(
                "Extend retrieval beyond `songs.csv` and `musicology_guides.json` by adding custom markdown notes or uploading a `.txt`/`.md` file."
            )
            uploaded_doc = st.file_uploader("Upload Custom Notes (`.md` or `.txt`)", type=["md", "txt"])
            custom_doc_text = st.text_area(
                "Or Paste Custom Curation Notes:",
                placeholder="e.g., For my exam week pomodoro sessions, prioritize Paper Lanterns and LoRoom tracks with felt piano...",
                height=70,
            )
            combined_custom_notes = custom_doc_text
            if uploaded_doc is not None:
                combined_custom_notes = (
                    uploaded_doc.getvalue().decode("utf-8", errors="ignore") + "\n" + custom_doc_text
                ).strip()

        if st.button("🚀 Curate Grounded RAG Playlist", type="primary"):
            profile_dict = engine.recommend_with_rag(
                user_query,
                k=top_k,
                weights=raw_weights,
                artist_penalty=artist_penalty,
                genre_penalty=genre_penalty,
                api_key=api_key,
                custom_notes=combined_custom_notes,
                dj_style=dj_style,
            )
            if ignored_artists:
                profile_input = dict(profile_dict["profile"])
                profile_input["ignored_artists"] = ignored_artists
                profile_dict = engine.recommend_with_rag(
                    profile_input,
                    k=top_k,
                    weights=raw_weights,
                    artist_penalty=artist_penalty,
                    genre_penalty=genre_penalty,
                    api_key=api_key,
                    custom_notes=combined_custom_notes,
                    dj_style=dj_style,
                )
            render_recommendations_section(profile_dict)

    # --- TAB 2: Explicit Persona & Profile Lab ---
    with tab_personas:
        st.subheader("Test Standard & Adversarial Personas or Build a Custom Profile")
        col_left, col_right = st.columns([1, 2])

        with col_left:
            persona_options = {
                k: f"{v['label']} ({v['genre']} / {v['mood']})" for k, v in USER_PROFILES.items()
            }
            persona_options["custom"] = "🛠️ Custom Explicit Profile"
            chosen_key = st.radio(
                "Select Profile:",
                options=list(persona_options.keys()),
                format_func=lambda x: persona_options[x],
            )

            base_p = USER_PROFILES.get(chosen_key, USER_PROFILES["chill_lofi"])
            genres_list = sorted({s.get("genre", "") for s in songs}) + ["deathcore polka"]
            moods_list = sorted({s.get("mood", "") for s in songs}) + ["existential dread"]

            custom_genre = st.selectbox(
                "Favorite Genre",
                options=genres_list,
                index=genres_list.index(base_p["genre"]) if base_p["genre"] in genres_list else 0,
            )
            custom_mood = st.selectbox(
                "Favorite Mood",
                options=moods_list,
                index=moods_list.index(base_p["mood"]) if base_p["mood"] in moods_list else 0,
            )
            custom_energy = st.slider("Target Energy", 0.0, 1.0, float(base_p["energy"]), 0.01)
            custom_acoustic = st.checkbox("Prefers Acoustic Instruments", value=bool(base_p["likes_acoustic"]))
            custom_query = st.text_input("Optional Semantic Context Keywords", value=base_p.get("natural_query", ""))

        with col_right:
            active_profile = {
                "label": base_p["label"] if chosen_key != "custom" else "Custom Explicit Profile",
                "genre": custom_genre,
                "mood": custom_mood,
                "energy": custom_energy,
                "likes_acoustic": custom_acoustic,
                "natural_query": custom_query,
                "ignored_artists": ignored_artists,
            }
            result = engine.recommend_with_rag(
                active_profile,
                k=top_k,
                weights=raw_weights,
                artist_penalty=artist_penalty,
                genre_penalty=genre_penalty,
                api_key=api_key,
                dj_style=dj_style,
            )
            render_recommendations_section(result)

    # --- TAB 3: Guardrails, Logs & Enriched Vector Catalog ---
    with tab_observability:
        st.subheader("🛡️ System Observability & Enriched RAG Catalog")
        col_w, col_l = st.columns(2)

        with col_w:
            st.markdown("#### Normalized Active Scoring Weights (`Σw = 1.00`)")
            st.json(norm_weights)
            st.markdown(f"**Vector Index Backend:** `{engine.rag_index.backend_name}`")
            st.markdown(f"**Indexed Catalog Size:** `{len(songs)} enriched tracks`")

        with col_l:
            st.markdown("#### 📜 Live Structured Telemetry Log (`logs/muse_ic.log`)")
            recent_logs = read_recent_logs(40)
            if recent_logs:
                st.code("\n".join(recent_logs), language="text")
            else:
                st.caption("No log entries recorded yet.")

        st.divider()
        st.markdown("#### 🗂️ Enriched Song Knowledge Base (`data/songs.csv`)")
        st.dataframe(pd.DataFrame(songs), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()


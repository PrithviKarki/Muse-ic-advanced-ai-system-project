"""
Command line runner for the Muse-ic Hybrid Semantic RAG Music Recommender.

Supports:
1. Multi-profile simulation (Standard & Adversarial personas):
   python -m src.main
2. Single-profile lookup:
   python -m src.main lofi
3. Natural-language Semantic RAG query:
   python -m src.main --query "calm rainy night coding music with warm piano"
"""

import os
import sys
from pathlib import Path

try:
    from src.evaluate import run_evaluation_suite
    from src.recommender import Recommender, load_songs, recommend_songs
except ModuleNotFoundError:
    from evaluate import run_evaluation_suite
    from recommender import Recommender, load_songs, recommend_songs


# -------------------------------------------------------------------------
# User Taste Profiles (Standard & Adversarial Edge Cases)
# -------------------------------------------------------------------------
USER_PROFILES = {
    # --- Standard Personas ---
    "high_energy_pop": {
        "label": "High-Energy Pop Workout",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "likes_acoustic": False,
        "natural_query": "upbeat high energy pop workout cardio anthem",
    },
    "chill_lofi": {
        "label": "Late-Night Chill Lofi Study",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
        "natural_query": "late night chill lofi study coding beats with warm piano and rain",
    },
    "deep_intense_rock": {
        "label": "Deep Intense Rock / Metal",
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
        "natural_query": "intense heavy rock and metal adrenaline workout",
    },

    # --- Adversarial & Edge-Case Personas ---
    "contradiction": {
        "label": "Adversarial: The Contradiction (Max Energy Classical)",
        "genre": "classical",
        "mood": "melancholic",
        "energy": 0.98,
        "likes_acoustic": False,
        "natural_query": "dramatic melancholic classical piano and high intensity orchestral power",
    },
    "out_of_distribution": {
        "label": "Adversarial: Unknown Genre/Mood (Zero-Match Trap)",
        "genre": "deathcore polka",
        "mood": "existential dread",
        "energy": 0.50,
        "likes_acoustic": True,
        "natural_query": "existential dread somber late night acoustic introspection",
    },
    "zero_bound": {
        "label": "Adversarial: Zero-Bound Minimalist (Extreme Limits)",
        "genre": "ambient",
        "mood": "chill",
        "energy": 0.00,
        "likes_acoustic": True,
        "natural_query": "weightless ambient meditation sleep stargazing zero energy calm",
    },
}


def format_recommendations_table(recommendations: list) -> str:
    """Format top recommendations as a clean multi-line ASCII summary table including reasons."""
    headers = ["Rank", "Song (Artist)", "Genre / Mood", "Energy", "Score", "Reasons (Why Recommended)"]

    table_rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        title = song.get("title", "Untitled") if isinstance(song, dict) else getattr(song, "title", "Untitled")
        artist = song.get("artist", "Unknown") if isinstance(song, dict) else getattr(song, "artist", "Unknown")
        genre = song.get("genre", "N/A") if isinstance(song, dict) else getattr(song, "genre", "N/A")
        mood = song.get("mood", "N/A") if isinstance(song, dict) else getattr(song, "mood", "N/A")
        energy = song.get("energy", 0.0) if isinstance(song, dict) else getattr(song, "energy", 0.0)

        reasons = [f"• {r.strip()}" for r in explanation.split(", ")] if explanation else ["• General musical match"]

        table_rows.append([
            [f"#{rank}"],
            [f"{title} ({artist})"],
            [f"{genre} / {mood}"],
            [f"{energy:.2f}"],
            [f"{score:.2f} ({score * 100:.1f}%)"],
            reasons,
        ])

    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for col_idx, cell_lines in enumerate(row):
            for line in cell_lines:
                col_widths[col_idx] = max(col_widths[col_idx], len(line))

    def make_border(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"

    lines = [make_border("-")]
    header_line = "|" + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|"
    lines.append(header_line)
    lines.append(make_border("="))

    for row in table_rows:
        max_cell_lines = max(len(cell) for cell in row)
        for line_idx in range(max_cell_lines):
            rendered_cells = []
            for col_idx, cell in enumerate(row):
                text = cell[line_idx] if line_idx < len(cell) else ""
                rendered_cells.append(f" {text:<{col_widths[col_idx]}} ")
            lines.append("|" + "|".join(rendered_cells) + "|")
        lines.append(make_border("-"))

    return "\n".join(lines)


def display_recommendations(
    prompt_or_profile: object,
    songs: list,
    k: int = 5,
    engine: Recommender = None,
) -> None:
    """Run the hybrid Semantic RAG recommendation pipeline and display table + grounded synthesis."""
    rec_engine = engine if engine is not None else Recommender(songs)
    result = rec_engine.recommend_with_rag(prompt_or_profile, k=k)

    profile = result["profile"]
    recommendations = result["recommendations"]
    rag_explanation = result["rag_explanation"]
    rag_backend = result["rag_backend"]
    guardrail_report = result["guardrail_report"]
    agent_trace = result.get("agent_trace", [])
    agent_self_corrected = result.get("agent_self_corrected", False)

    print("=" * 118)
    print(f"👤 PROFILE: {profile.get('label', 'Custom User')}")
    print(
        f"   Preferences -> Genre: {profile.get('genre')} | Mood: {profile.get('mood')} | "
        f"Target Energy: {profile.get('energy'):.2f} | Likes Acoustic: {profile.get('likes_acoustic')}"
    )
    if profile.get("natural_query"):
        print(f"   Semantic RAG Query -> \"{profile.get('natural_query')}\"")
    print(
        f"   Guardrails -> Passed: {guardrail_report.passed} | "
        f"Agent Self-Corrected: {agent_self_corrected} | "
        f"RAG Synthesis: {rag_backend}"
    )
    if guardrail_report.warnings:
        for w in guardrail_report.warnings:
            print(f"   ⚠️  Guardrail Note: {w}")
    print("=" * 118)

    if agent_trace:
        print("\n🧠 Multi-Step Agentic Workflow Trace (Plan -> Tool Call -> Critique/Decide -> Synthesize):")
        for step in agent_trace:
            print(f"   [Step {step['step']} | {step['phase']}] Tool: {step['tool']}")
            print(f"     • Observation : {step['observation']}")
            print(f"     • Decision    : {step['decision']}")
    print(f"\n🎧 Top {len(recommendations)} Recommendations Summary Table:\n")
    print(format_recommendations_table(recommendations))
    print("\n🤖 Grounded RAG AI DJ Synthesis:")
    print("-" * 118)
    print(rag_explanation)
    print("-" * 118 + "\n")


def _resolve_songs_csv() -> str:
    """Locate data/songs.csv relative to either current working directory or project root."""
    default_path = Path("data/songs.csv")
    if default_path.exists():
        return str(default_path)
    project_root_csv = Path(__file__).resolve().parent.parent / "data" / "songs.csv"
    return str(project_root_csv)


def main() -> None:
    csv_path = _resolve_songs_csv()
    songs = load_songs(csv_path)
    engine = Recommender(songs)

    # Check if user specified a natural language query or profile via command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--eval", "-e"):
            run_evaluation_suite(print_report=True)
            return

        if sys.argv[1] in ("--query", "-q") and len(sys.argv) > 2:
            natural_prompt = " ".join(sys.argv[2:]).strip()
            print("\n" + "#" * 76)
            print("       🎵 MUSE-IC NATURAL LANGUAGE SEMANTIC RAG QUERY")
            print(f"       Prompt: \"{natural_prompt}\"")
            print(f"       Catalog loaded: {len(songs)} songs from {csv_path}")
            print("#" * 76 + "\n")
            display_recommendations(natural_prompt, songs, k=5, engine=engine)
            return

        query = " ".join(sys.argv[1:]).lower().strip()
        matched_key = None
        for key in USER_PROFILES:
            if query in key or query in USER_PROFILES[key]["label"].lower():
                matched_key = key
                break

        if matched_key:
            print("\n" + "#" * 64)
            print(f"       🎵 SINGLE PROFILE TEST: '{matched_key}'")
            print(f"       Catalog loaded: {len(songs)} songs from {csv_path}")
            print("#" * 64 + "\n")
            display_recommendations(USER_PROFILES[matched_key], songs, k=5, engine=engine)
            return
        else:
            # Treat free-form CLI argument as a natural language RAG prompt!
            raw_query = " ".join(sys.argv[1:]).strip()
            print("\n" + "#" * 76)
            print(f"       🎵 FREE-FORM SEMANTIC RAG PROMPT: '{raw_query}'")
            print(f"       Catalog loaded: {len(songs)} songs from {csv_path}")
            print("#" * 76 + "\n")
            display_recommendations(raw_query, songs, k=5, engine=engine)
            return

    # Default: Run top 5 recommendations for all profiles
    print("\n" + "#" * 76)
    print("       🎵 MUSE-IC HYBRID SEMANTIC RAG RECOMMENDER - MULTI-PROFILE RUNNER")
    print(f"       Catalog loaded: {len(songs)} songs from {csv_path}")
    print("#" * 76 + "\n")

    standard_keys = ["high_energy_pop", "chill_lofi", "deep_intense_rock"]
    for key in standard_keys:
        display_recommendations(USER_PROFILES[key], songs, k=5, engine=engine)

    print("\n" + "!" * 76)
    print("       🧪 RUNNING ADVERSARIAL & EDGE-CASE TEST PROFILES (TOP 5)")
    print("!" * 76 + "\n")
    adversarial_keys = ["contradiction", "out_of_distribution", "zero_bound"]
    for key in adversarial_keys:
        display_recommendations(USER_PROFILES[key], songs, k=5, engine=engine)


if __name__ == "__main__":
    main()

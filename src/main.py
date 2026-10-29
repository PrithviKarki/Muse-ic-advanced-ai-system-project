"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import sys

try:
    from src.recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs


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
    },
    "chill_lofi": {
        "label": "Late-Night Chill Lofi Study",
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "likes_acoustic": True,
    },
    "deep_intense_rock": {
        "label": "Deep Intense Rock / Metal",
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "likes_acoustic": False,
    },

    # --- Adversarial & Edge-Case Personas ---
    "contradiction": {
        "label": "Adversarial: The Contradiction (Max Energy Classical)",
        "genre": "classical",
        "mood": "melancholic",
        "energy": 0.98,
        "likes_acoustic": False,
    },
    "out_of_distribution": {
        "label": "Adversarial: Unknown Genre/Mood (Zero-Match Trap)",
        "genre": "deathcore polka",
        "mood": "existential dread",
        "energy": 0.50,
        "likes_acoustic": True,
    },
    "zero_bound": {
        "label": "Adversarial: Zero-Bound Minimalist (Extreme Limits)",
        "genre": "ambient",
        "mood": "chill",
        "energy": 0.00,
        "likes_acoustic": True,
    },
}


def format_recommendations_table(recommendations: list) -> str:
    """Format top recommendations as a clean multi-line ASCII summary table including reasons."""
    headers = ["Rank", "Song (Artist)", "Genre / Mood", "Energy", "Score", "Reasons (Why Recommended)"]

    # Build structured rows where each cell is a list of lines (supporting multi-line reasons)
    table_rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        title = song.get("title", "Untitled")
        artist = song.get("artist", "Unknown")
        genre = song.get("genre", "N/A")
        mood = song.get("mood", "N/A")
        energy = song.get("energy", 0.0)

        reasons = [f"• {r.strip()}" for r in explanation.split(", ")] if explanation else ["• General musical match"]

        table_rows.append([
            [f"#{rank}"],
            [f"{title} ({artist})"],
            [f"{genre} / {mood}"],
            [f"{energy:.2f}"],
            [f"{score:.2f} ({score * 100:.1f}%)"],
            reasons,
        ])

    # Calculate column widths dynamically
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


def display_recommendations(profile: dict, songs: list, k: int = 5) -> None:
    """Run recommendation pipeline for a profile and display a formatted visual summary table."""
    recommendations = recommend_songs(profile, songs, k=k)

    print("=" * 118)
    print(f"👤 PROFILE: {profile.get('label', 'Custom User')}")
    print(f"   Preferences -> Genre: {profile.get('genre')} | Mood: {profile.get('mood')} | "
          f"Target Energy: {profile.get('energy'):.2f} | Likes Acoustic: {profile.get('likes_acoustic')}")
    print("=" * 118)
    print(f"\n🎧 Top {len(recommendations)} Recommendations Summary Table:\n")
    print(format_recommendations_table(recommendations))
    print()


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Check if user specified a specific profile via command-line argument
    # e.g.: python -m src.main lofi  OR  python -m src.main pop
    if len(sys.argv) > 1:
        query = sys.argv[1].lower().strip()
        matched_key = None
        for key in USER_PROFILES:
            if query in key or query in USER_PROFILES[key]["label"].lower():
                matched_key = key
                break

        if matched_key:
            print("\n" + "#" * 64)
            print(f"       🎵 SINGLE PROFILE TEST: '{matched_key}'")
            print(f"       Catalog loaded: {len(songs)} songs from data/songs.csv")
            print("#" * 64 + "\n")
            display_recommendations(USER_PROFILES[matched_key], songs, k=5)
            return
        else:
            print(f"\nProfile '{query}' not found. Available profiles:")
            for k in USER_PROFILES:
                print(f"  • {k} ({USER_PROFILES[k]['label']})")
            print()
            return

    # Default: Run top 5 recommendations for all profiles
    print("\n" + "#" * 64)
    print("       🎵 MUSIC RECOMMENDER SIMULATION - MULTI-PROFILE RUNNER")
    print(f"       Catalog loaded: {len(songs)} songs from data/songs.csv")
    print("#" * 64 + "\n")

    # 1. Run the Three Distinct Standard Personas (Top 5 each)
    standard_keys = ["high_energy_pop", "chill_lofi", "deep_intense_rock"]
    for key in standard_keys:
        display_recommendations(USER_PROFILES[key], songs, k=5)

    # 2. Run Adversarial / Edge Cases (Top 5 each)
    print("\n" + "!" * 64)
    print("       🧪 RUNNING ADVERSARIAL & EDGE-CASE TEST PROFILES (TOP 5)")
    print("!" * 64 + "\n")
    adversarial_keys = ["contradiction", "out_of_distribution", "zero_bound"]
    for key in adversarial_keys:
        display_recommendations(USER_PROFILES[key], songs, k=5)


if __name__ == "__main__":
    main()

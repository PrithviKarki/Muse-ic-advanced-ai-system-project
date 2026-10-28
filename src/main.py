"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

try:
    from src.recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Active user taste profile
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.80,
        "likes_acoustic": False,
    }

    k = 5
    recommendations = recommend_songs(user_prefs, songs, k=k)

    print("=" * 64)
    print("🎵 MUSIC RECOMMENDER SIMULATION")
    print("=" * 64)
    print(f"Catalog: {len(songs)} songs loaded from 'data/songs.csv'")
    print("Active User Taste Profile:")
    print(f"  • Genre:          {user_prefs.get('genre')}")
    print(f"  • Mood:           {user_prefs.get('mood')}")
    print(f"  • Target Energy:  {user_prefs.get('energy'):.2f}")
    print(f"  • Likes Acoustic: {user_prefs.get('likes_acoustic')}")
    print("=" * 64)
    print(f"\n🎧 Top {len(recommendations)} Recommendations for You:\n")

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        artist = song.get("artist", "Unknown Artist")
        title = song.get("title", "Untitled")
        genre = song.get("genre", "N/A")
        mood = song.get("mood", "N/A")
        energy = song.get("energy", 0.0)
        acoustic = song.get("acousticness", 0.0)

        print(f"#{rank}. {title} — {artist}")
        print(f"    Match Score : {score:.2f} ({score * 100:.1f}%)")
        print(f"    Attributes  : Genre: {genre} | Mood: {mood} | Energy: {energy:.2f} | Acoustic: {acoustic:.2f}")
        print("    Why recommended:")
        reasons = [r.strip() for r in explanation.split(", ")] if explanation else ["General musical match"]
        for reason in reasons:
            print(f"      ✔ {reason}")
        print("-" * 64)


if __name__ == "__main__":
    main()

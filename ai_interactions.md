# AI Interactions Log

> **Stretch features & AI Collaboration Log.** Documents our experience using an AI coding agent to design, build, test, and refine **Muse-ic**.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot, Antigravity) to make multi-step changes autonomously.

**What task did you give the agent?**

We tasked the AI agent with helping us build, test, and refine **Muse-ic**, an explainable CLI-first content-based music recommender system, across multiple engineering phases:
1. Expanding `data/songs.csv` from 10 to 20 tracks across new genres and moods while ensuring clean CSV parsing.
2. Designing and implementing a normalized weighted scoring rule (`score_song`) and an iterative diversity-aware ranking pipeline (`recommend_songs` — **Challenge 3: Diversity and Fairness Logic**).
3. Building a multi-profile CLI runner with an ASCII **Visual Summary Table** (`format_recommendations_table` — **Challenge 4: Visual Summary Table**) and six test profiles (3 standard + 3 adversarial edge cases).
4. Running weight-tuning experiments and documenting findings in `README.md` and `model_card.md`.

**Prompts used:**

* *"Generate 5–10 additional songs in a valid CSV format that includes the existing headers. Ensure the new songs represent a diverse range of genres and moods not already present in the starter file."*
* *"Find and implement the `score_song(user_prefs, song)` function. Based on our 'Algorithm Recipe' from Phase 2, calculate a score for numerical features like energy. Ensure our scoring logic returns both a numeric score and a list of 'reasons' so the user understands the recommendation."*
* *"Update `recommend_songs()` in `src/recommender.py` to implement greedy re-ranking with a Diversity Penalty. Instead of sorting all songs statically by base score, select the top `k` songs iteratively one slot at a time while tracking how many times each `artist` and `genre` have already been added to the recommendations list. If a candidate song's artist is already present in the selected list, subtract a `-0.15` repeat-artist penalty per prior appearance; if its genre already appears 2 or more times, subtract a `-0.10` genre-saturation penalty. Append these penalties to the song's explanation reasons so the user can see why the score was adjusted."*
* *"Improve the readability of your terminal output by providing a formatted table or summary... Ensure your prompt specifies that the table must include the 'reasons' for each score."*
* *"Run a temporary experiment and double the importance of energy and halve the importance of genre. Verify that our math remains valid."*

**What did the agent generate or change?**

* **`data/songs.csv`**: Added 10 new tracks (IDs 11–20) covering `reggae`, `classical`, `edm`, `folk`, `r&b`, `hip hop`, `metal`, `latin`, `blues`, and `funk`.
* **`src/recommender.py`**:
  * Implemented `load_songs()` using `csv.DictReader` with automatic type casting (`int` and `float`).
  * Implemented `score_song()` using a 4-feature weighted sum (`0.35` genre, `0.30` mood, `0.20` energy proximity, `0.15` acoustic alignment) and point-by-point reason strings.
  * Implemented `recommend_songs()` with **Challenge 3 Diversity & Fairness Logic** (`-0.15` repeat-artist penalty and `-0.10` genre-saturation penalty) and connected the OOP `Recommender` class methods.
* **`src/main.py`**:
  * Added 6 user profiles (`high_energy_pop`, `chill_lofi`, `deep_intense_rock`, `contradiction`, `out_of_distribution`, and `zero_bound`).
  * Implemented `format_recommendations_table()` (**Challenge 4**) to render a multi-line ASCII summary table with aligned columns and bulleted reason cells.
* **`README.md` & `model_card.md`**: Documented the architecture Mermaid flowchart, pairwise profile comparisons, filter bubble critiques, and personal reflections.

**What did you verify or fix manually?**

1. **Module Import Paths (`src.main` and `pytest`):** Initially, running `python -m src.main` raised `ModuleNotFoundError: No module named 'recommender'` because `main.py` used a bare import. We verified and fixed this using a `try / except ModuleNotFoundError` import block so both `python -m src.main` and `python src/main.py` work seamlessly.
2. **Weight Normalization Math Check:** During our temporary experiment (halving genre weight from `0.35 -> 0.175` and doubling energy weight from `0.20 -> 0.40`), we verified the math and caught that the raw weights summed to **`1.025`** instead of `1.00`, causing top-scoring songs like *Library Rain* to exceed 100% (`1.0040`). This confirmed why linear scoring weights must be constrained to sum to `1.0` (or normalized by dividing by $\sum w_i$).
3. **Diversity Penalty Verification:** We manually checked the *Chill Lofi Study* output table to verify that `Focus Flow` (by `LoRoom`, genre `lofi`) received both the `-0.15` repeat-artist penalty and `-0.10` genre-saturation penalty, promoting `Spacewalk Thoughts` (`Orbit Bloom`, `ambient`) to #3.

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

**Two-Stage Pipeline / Strategy-Adapter Pattern** (combining a Pointwise Scoring Strategy in `score_song` with a Stateful Greedy Re-Ranking Pipeline in `recommend_songs`, plus an Adapter helper `_extract_attr` that allows the same scoring engine to process both functional `dict` records and OOP `Song`/`UserProfile` dataclasses).

**How did AI help you brainstorm or implement it?**

* We noticed early on that `src/main.py` passes plain Python dictionaries (`dict`), whereas `tests/test_recommender.py` passes `Song` and `UserProfile` dataclass instances. Rather than duplicating the scoring logic in two places, the AI helped us design `_extract_attr()` as a lightweight adapter so `score_song()` and `recommend_songs()` serve both interfaces with zero code duplication.
* When we added **Challenge 3 (Diversity Penalty)**, the AI helped us separate **pointwise scoring** (judging one song in isolation in `score_song`) from **listwise re-ranking** (iteratively selecting the top $k$ songs while penalizing artist and genre repetition in `recommend_songs`).

**How does the pattern appear in your final code?**

* **Adapter Helper:** `_extract_attr(obj, key, alt_key, default)` in `src/recommender.py` (lines 68–79) seamlessly reads attributes from either `dict` keys (`"genre"`, `"energy"`) or dataclass fields (`favorite_genre`, `target_energy`).
* **Pointwise Scoring Strategy:** `score_song(user_prefs, song)` in `src/recommender.py` (lines 82–173) computes the base `[0.0, 1.0]` compatibility score and feature reasons for a single track.
* **Listwise Diversity Re-Ranker:** `recommend_songs(user_prefs, songs, k=5, artist_penalty=0.15, genre_penalty=0.10)` in `src/recommender.py` (lines 175–237) consumes the pointwise scores and applies stateful artist/genre diversity penalties before `format_recommendations_table()` in `src/main.py` renders the final visual table.

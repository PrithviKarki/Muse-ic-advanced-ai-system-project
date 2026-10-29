# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**Muse-ic** — An Explainable Content-Based Music Recommender

---

## 2. Intended Use & Goal / Task  

### Goal / Task
**Muse-ic** suggests the top 5 songs from a music catalog that best match a listener's current taste profile. It also explains *why* each song was picked in plain English.

### Intended Use
* **Classroom & Learning Exploration:** Built to show how content-based recommendation systems score, rank, and explain song matches.
* **Activity-Based Playlist Building:** Great for picking songs for a specific moment, like a high-energy workout or a quiet late-night study session.
* **Assumptions:** Assumes the user already knows what genre, mood, energy level, and instrument style they want right now.

### Non-Intended Use
* **Not for Large Commercial Streaming:** This model is not built for catalogs with millions of songs or real-time user tracking.
* **No Crowd / Social Discovery:** Because it does not use collaborative filtering (other people's playlists or listening history), it should not be used to predict trending hits.

---

## 3. How the Model Works (Algorithm Summary)  

Imagine **Muse-ic** as a judge grading every song in the library using a **100-point checklist**:

1. **Genre Match (35 points):** If the song's genre matches the user's favorite genre (like `"pop"` or `"lofi"`), it earns all 35 points. If it is a close cousin (like `"indie pop"` when the user asked for `"pop"`), it gets half credit (17.5 points).
2. **Mood Match (30 points):** If the song's mood matches what the user wants (like `"happy"` or `"chill"`), it earns 30 points.
3. **Energy Closeness (20 points):** Energy is measured from `0.0` (super calm) to `1.0` (super intense). The closer the song's energy is to the user's target energy, the more points it gets out of 20.
4. **Instrument Style (15 points):** Users choose whether they prefer warm acoustic instruments (like piano and guitar) or electronic beats (like synths and drum machines). Songs earn up to 15 points based on how well they fit that style.

### Ranking, Diversity & Visual Explaining
After every song gets a base score from `0.00` to `1.00` (0% to 100%), the system picks the **Top 5** one slot at a time while applying a **Diversity Penalty** (Challenge 3):
* **Repeat Artist Penalty (-15 points):** If an artist is already in the Top 5 list, their next song loses 15 points so one artist doesn't take over the playlist.
* **Genre Saturation Penalty (-10 points):** If a genre already appears twice in the Top 5 list, extra songs in that same genre lose 10 points to make room for fresh styles.

Finally, the results are printed in a clean **Visual Summary Table** (Challenge 4) showing the rank, song, artist, genre/mood, energy, score, and a bulleted list of reasons (including any diversity penalties).

---

## 4. Data Used  

* **Dataset Size:** Our catalog (`data/songs.csv`) has **20 songs** total. We started with 10 tracks and added 10 new tracks to cover more musical styles.
* **Features Available:** Every song has 10 attributes: `id`, `title`, `artist`, `genre`, `mood`, `energy`, `tempo_bpm`, `valence` (happiness), `danceability`, and `acousticness`.
* **Features Used by the Model:** We use four core features: `genre`, `mood`, `energy`, and `acousticness`.
* **Dataset Limits:**
  * **Small Catalog:** While `lofi` and `pop` have 3 songs each, 14 other genres (like `rock`, `classical`, `metal`, `jazz`, and `reggae`) only have **1 song each**.
  * **Missing Context:** The dataset does not include lyrics, language, release year (era), song length, or user play/skip history.

---

## 5. Strengths  

* **Clear Separation of Opposite Vibes:** The model never confuses calm study music with loud workout music. A *Chill Lofi* user gets 98% matches on relaxing tracks and ~11% on heavy rock.
* **Built-In Artist & Genre Diversity:** Our Diversity Penalty stops a single artist or genre from hogging the Top 3, automatically making room for complementary tracks.
* **100% Explainable Visual Table:** Unlike "black-box" AI models, our Visual Summary Table shows the exact points earned (and any diversity penalties subtracted) for every recommended track.
* **No Cold-Start Problem for New Songs:** Because the system grades songs by their musical features instead of popularity, a brand-new song can be recommended right away even with zero listens.

---

## 6. Observed Behavior, Limitations, and Bias  

1. **"Synonym Blindness" Filter Bubble (Label Bias):**
   Because Genre (35%) and Mood (30%) make up 65% of the total score using exact word matching, the system does not know that `"chill"`, `"relaxed"`, `"focused"`, and `"peaceful"` mean almost the same thing. During testing with our *Chill Lofi* profile, great study songs like *Coffee Shop Stories* (`jazz`, `"relaxed"`) and *Focus Flow* (`lofi`, `"focused"`) lost 30% to 65% of their points just because their mood label did not literally say `"chill"`.

2. **Mid-Range Energy Compression ("Energy Gap" Math):**
   Energy is scored by how close a song is to the user's target ($1.0 - |\text{song energy} - \text{target energy}|$). If a user picks a middle target like `0.50`, no song in the catalog can ever be more than `0.50` away. That means every song automatically gets at least half of the energy points, making the energy rule less helpful for moderate listeners than for extreme listeners.

3. **Equal Penalty for Too Loud vs. Too Quiet:**
   The energy rule treats missing the target above and below the exact same way. For someone studying at `0.35` energy, a quieter `0.20` ambient song is fine, while a louder `0.50` track is distracting—yet both get the exact same energy score.

4. **Catalog Starvation for Single-Song Genres:**
   If a user picks a genre that only has 1 song in our CSV (like `rock` or `classical`), the #1 song is a great match, but recommendations #2 through #5 drop sharply in score and pull from other genres.

---

## 7. Evaluation Process  

To check whether our recommender worked as expected, we tested **six distinct user profiles** (three everyday listener personas and three "stress-test" edge cases) across our 20-song catalog, plus a weight-changing experiment.

### User Profiles Tested
1. **High-Energy Pop Workout:** Wants `pop` genre, `happy` mood, high energy (`0.85`), and electronic sound.
2. **Late-Night Chill Lofi Study:** Wants `lofi` genre, `chill` mood, low energy (`0.35`), and acoustic instruments.
3. **Deep Intense Rock / Metal:** Wants `rock` genre, `intense` mood, very high energy (`0.92`), and electronic/amplified production.
4. **The Contradiction (Stress Test):** Wants `classical` music with a `melancholic` mood, but asks for maximum energy (`0.98`) and electronic instruments.
5. **The Unknown Genre (Stress Test):** Asks for `"deathcore polka"` and `"existential dread"` (which do not exist in our catalog) with medium energy (`0.50`) and acoustic sound.
6. **The Zero-Energy Minimalist (Stress Test):** Wants `ambient` genre, `chill` mood, and `0.00` energy.

### Pairwise Profile Comparisons (What Changed & Why It Makes Sense)

* **Pair 1: *High-Energy Pop* vs. *Late-Night Chill Lofi***
  * **What changed:** Their top 5 lists had **zero overlap**. The Pop profile picked upbeat electronic tracks (*Sunrise City* at 96.7%, *Rooftop Lights* at 75.4%, *Gym Hero* at 67.7%), while the Lofi profile picked slow, relaxing study beats (*Library Rain* at 97.9%, *Midnight Coding* at 94.2%, *Focus Flow* at 65.7%).
  * **Why it makes sense:** The Pop profile rewards fast dance energy and synthetic beats, whereas the Lofi profile shifts points toward quiet, organic instrumentals.

* **Pair 2: *High-Energy Pop* vs. *Deep Intense Rock***
  * **What changed:** The #1 song flipped from *Sunrise City* (happy pop) to *Storm Runner* (intense rock at 98.3%), but both profiles shared high-energy workout tracks like *Gym Hero* and *Neon Cyberpunk* in their top 5.
  * **Why it makes sense:** Both listeners want loud, energetic, non-acoustic music (`0.85`–`0.92` energy), so high-energy tracks appeal to both, while the #1 spot shifts to match each person's specific genre and mood.

* **Pair 3: *Late-Night Chill Lofi* vs. *Deep Intense Rock***
  * **What changed:** Every single song and score flipped. *Storm Runner* scored 98.3% for the Rock fan but dropped to the bottom for the Lofi listener, while *Library Rain* did the exact opposite.
  * **Why it makes sense:** These two profiles want opposite genres, opposite moods, opposite energy levels, and opposite instrument styles.

* **Pair 4: *The Contradiction (Max-Energy Classical)* vs. *The Unknown Genre***
  * **What changed:** For the Contradiction profile, #1 was a quiet piano piece (*Moonlight Sonata Flow* at 70.1%) and #2 was heavy metal (*Iron Thunder* at 34.8%). For the Unknown Genre profile, scores topped out around 32% with mid-tempo acoustic folk and jazz (*Rustic Campfire* and *Coffee Shop Stories*).
  * **Why it makes sense:** When a user asks for an impossible combo ("high-energy classical"), our system prioritizes genre and mood labels first (giving #1 to classical) and uses energy second (giving #2 to metal). When a user asks for a genre that isn't in the library at all, the system safely falls back to matching their energy and acoustic style.

### Surprising Findings: Why Does "Gym Hero" Keep Showing Up for "Happy Pop" Listeners?
During testing, the song **"Gym Hero"** showed up at **#3 (67.7%)** for people who wanted **"Happy Pop"**—and also showed up at **#2 (64.0%)** for people who wanted **"Intense Rock"**!

Here is why that happens in plain language:
* Our catalog only has **two** songs that are both Pop and Happy (*Sunrise City* and *Rooftop Lights*), which easily take 1st and 2nd place.
* To fill 3rd place, the system looks for the next highest score on our 100-point checklist. Even though *Gym Hero* has an **"Intense"** mood instead of a **"Happy"** mood (losing 30 mood points), it is still a **Pop** song (+35 points), has **high energy** (+18 points), and uses **electronic instruments** (+14 points). That gives it **67.7 points**—beating every other song left in the catalog!
* Likewise, when someone asks for **"Intense Rock,"** *Gym Hero* loses the 35 points for not being Rock, but wins **30 points** for its "Intense" mood plus **34 points** for its high energy and electronic punch—landing it in 2nd place right behind the only true Rock song in the catalog.

### Weight-Shifting & Diversity Experiments
* **Weight-Shifting Experiment:** When we cut the **Genre weight in half** ($0.35 \to 0.175$) and **doubled the Energy weight** ($0.20 \to 0.40$), our *Chill Lofi* profile swapped its #3 and #4 songs: an ambient song (*Spacewalk Thoughts*) jumped ahead of a lofi song (*Focus Flow*) because matching the "chill" mood and calm energy mattered more than having "lofi" on the genre tag.
* **Diversity Penalty Experiment (Challenge 3):** Even with our original weights, adding our **Diversity Penalty** (-15 points for a repeat artist and -10 points when a genre appears 2+ times) promoted *Spacewalk Thoughts* (`Orbit Bloom`, `ambient`, 62.4%) to **#3** ahead of *Focus Flow* (`LoRoom`, `lofi`, adjusted from 65.7% down to 40.7%), preventing `LoRoom` from taking two spots in the Top 3!

---

## 8. Ideas for Improvement (Future Work)  

1. **Group Similar Moods and Genres Together (Fuzzy Matching):**
   Give partial points when moods or genres are closely related (for example, treating `"chill"`, `"relaxed"`, and `"focused"` as friendly neighbors, or connecting `lofi`, `ambient`, and `jazz`). This would break the filter bubble without losing accuracy.
2. **Use More Audio Features (`valence` and `tempo_bpm`):**
   Add musical positivity (`valence`) and normalized speed (`tempo_bpm`) into the score so the recommender can tell the difference between a fast happy song and a fast angry song.
3. **Add a "Guest Mode" & Explicit Dislike Filter:**
   Allow users to mark songs they don't like or turn on a "Guest Mode" when friends play music on their account so curiosity listens and other people's tastes don't pollute their recommendation profile.

---

## 9. Personal Reflection  

* **Biggest Learning Moment (Spotify vs. YouTube Music):**
  My biggest takeaway was finally understanding *why* I get the specific song recommendations I see on Spotify versus YouTube Music—and why **YouTube Music's subscription is the superior choice for me**. Spotify leans heavily on playlist co-occurrence and strict genre/audio clusters, which can trap you in a repetitive "filter bubble." YouTube Music blends explicit thumbs-up/down signals with broader cross-video listening context, giving me much better variety and control over what plays next.

* **Working with AI Tools (And When to Double-Check Them):**
  AI tools were great for brainstorming our weighted scoring formula, expanding our CSV dataset to 20 diverse songs, debugging Python module import paths (`python -m src.main` and `pytest`), and generating stress-test profiles. However, I learned I had to double-check the math myself during our weight-shifting experiment: when we cut the Genre weight in half ($0.35 \to 0.175$) and doubled the Energy weight ($0.20 \to 0.40$), the total weights added up to **$1.025$** instead of $1.00$, pushing top scores over $100\%$ ($1.004$). Catching that overflow proved why human verification is always necessary.

* **How Simple Rules Feel "Smart"—and How Habits Can Trick the Algorithm:**
  It surprised me how a simple 100-point checklist could immediately "feel" like a real recommendation engine. It also made me realize how easily real-world tracking signals get polluted by everyday habits. For example, I sometimes listen to a bad song I don't even like all the way to the end just to hear the whole thing—and since streaming apps count a 100% completion rate as a strong positive vote, that habit actually messes up my recommendations! In the same way, letting friends or family play songs they like on my account feeds conflicting signals into my profile, skewing my algorithm just like our "Contradiction" stress-test profile did.

* **What I Would Try Next:**
  If I extended **Muse-ic**, I would import my own real playlist data and connect the recommender to a larger music API to discover newer songs that fit my actual taste—while adding a "guest mode" or explicit dislike filter so curiosity listens and other people's songs don't ruin my recommendations.

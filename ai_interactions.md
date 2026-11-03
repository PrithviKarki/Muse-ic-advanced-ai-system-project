# AI Interactions & Agentic Workflow Reasoning Traces Log

> **Stretch Features & AI Collaboration Log.** Documents our experience using an AI coding agent (Antigravity) to design, build, test, and elevate **Muse-ic 2.0**, and records the **intermediate multi-step reasoning and tool-calling traces** produced by our autonomous Agentic DJ Workflow.

---

## 1. Stretch Goal: Agentic Workflow Enhancement (Intermediate Reasoning & Tool-Call Traces)

* **Committed Reasoning Trace Log File:** [**`logs/agent_reasoning_traces.log`**](logs/agent_reasoning_traces.log)
* **Implementation Location:** [`Recommender.recommend_with_rag`](src/recommender.py) in [`src/recommender.py`](src/recommender.py) and [`append_agent_trace`](src/logger.py) in [`src/logger.py`](src/logger.py).

### How the 4-Step Decision-Making & Tool-Calling Chain Works
Every recommendation request executes a 4-stage agentic loop (`PLAN → TOOL CALL #1 → CRITIQUE & DECIDE (with conditional TOOL CALL #2 Self-Correction) → VERIFY & SYNTHESIZE`):

1. **Step 1 (`PLAN & GUARDRAILS` — Tool: `parse_natural_language_request + sanitize_user_profile`):**
   * Extracts structured listener constraints (`genre`, `mood`, `target_energy`, `likes_acoustic`) and audits [`data/songs.csv`](data/songs.csv) to check exact genre availability and whether the user's target energy contradicts the genre's natural energy distribution.
2. **Step 2 (`TOOL CALL #1: RETRIEVE & RANK PASS 1` — Tool: `SongKnowledgeIndex.retrieve_multi_source_context + recommend_songs`):**
   * Queries all 3 RAG knowledge sources (`songs.csv`, `musicology_guides.json`, and `custom_listener_notes.md`) and ranks the top-$k$ candidates using baseline weights (`genre=0.25, mood=0.20, energy=0.20, acoustic=0.15, semantic_rag=0.20`).
3. **Step 3 (`CRITIQUE & DECISION NODE` — Tool: `verify_candidate_confidence` OR `adaptive_weight_rebalancer + recommend_songs (Pass 2)`):**
   * **Branch A (`APPROVED`):** If Pass #1 Top-1 score $\ge 65.0\%$ and no genre/energy contradiction exists, the Critic approves Pass #1.
   * **Branch B (`SELF_CORRECTED — TOOL CALL #2`):** If Pass #1 detects an **Out-of-Distribution Genre** (0 exact catalog tracks) or a **Genre-Energy Contradiction** ($\Delta \text{Energy} \ge 0.45$), the Critic autonomously re-allocates scoring weights (e.g., dropping dead `genre` weight from `0.25 → 0.05` and boosting `semantic_rag` to `0.35` and `mood` to `0.25`) and calls `recommend_songs` a second time, lifting top match confidence from **56.6% $\to$ 70.1%**.
4. **Step 4 (`VERIFY & SYNTHESIZE` — Tool: `verify_recommendations + generate_rag_explanation`):**
   * Verifies zero hallucinated song titles against the catalog, synthesizes the multi-source AI DJ commentary, and appends the complete trace to [`logs/agent_reasoning_traces.log`](logs/agent_reasoning_traces.log).

---

### Embedded Intermediate Reasoning Trace #1: Autonomous Self-Correction Branch (`TOOL CALL #2` Triggered)
When running `python -m src.main out_of_distribution` (`genre="deathcore polka"`, `mood="existential dread"`), the agent detects that `"deathcore polka"` has `0` tracks in the catalog, critiques the suppressed Pass #1 score (`56.6%`), and invokes `adaptive_weight_rebalancer + recommend_songs (Pass 2)` to boost the top match (*Midnight Delta Blues*) to **70.1%**:

```text
================================================================================================
AGENTIC WORKFLOW TRACE | 2026-11-03 16:49:48 | Profile: Adversarial: Unknown Genre/Mood (Zero-Match Trap)
Query / Intent: existential dread somber late night acoustic introspection
------------------------------------------------------------------------------------------------
  [Step 1 | PLAN & GUARDRAILS] Tool: `parse_natural_language_request + sanitize_user_profile`
    • Thought     : Extract structured listener constraints and audit catalog coverage before scoring.
    • Observation : Parsed target genre='deathcore polka' (0 exact tracks in catalog), mood='existential dread', energy=0.50 (genre avg energy=0.50, gap=0.00).
    • Decision    : Flagged potential OOD/Contradiction risk for Critic inspection after Pass #1.
  [Step 2 | TOOL CALL #1 (RETRIEVE & RANK PASS 1)] Tool: `SongKnowledgeIndex.retrieve_multi_source_context + recommend_songs`
    • Thought     : Retrieve across 3 data sources and rank top-5 using baseline weights (genre=0.25, mood=0.20, energy=0.20, rag=0.20).
    • Observation : Matched Musicology Guide: 'Late-Night Introspection & Catharsis Guide'. Pass #1 Top-1 Score = 56.6%, Top-5 Mean Score = 47.9%.
    • Decision    : Send Pass #1 candidate pool to Critic Decision Node.
  [Step 3 | CRITIQUE & SELF-CORRECTION (TOOL CALL #2)] Tool: `adaptive_weight_rebalancer + recommend_songs (Pass 2)`
    • Thought     : Critic flagged Pass #1: Out-of-distribution genre 'deathcore polka' has 0 exact catalog tracks, penalizing all candidates by 25%. Rebalancing weights to prioritize Semantic RAG & viable audio features.
    • Observation : Re-ran `recommend_songs` with adaptive weights (genre=0.05, mood=0.25, energy=0.20, rag=0.35). Top-1 Score shifted 56.6% -> 70.1% (Mean 47.9% -> 57.2%).
    • Decision    : SELF_CORRECTED: Adopted Pass #2 adaptive ranking.
  [Step 4 | VERIFY & SYNTHESIZE] Tool: `verify_recommendations + generate_rag_explanation`
    • Thought     : Confirm zero hallucinated titles against catalog and synthesize multi-source AI DJ setlist commentary.
    • Observation : Guardrail passed=True; anchored explanation on #1 'Midnight Delta Blues' via Local Grounded Multi-Source RAG Synthesizer (3 Data Sources).
    • Decision    : OUTPUT_READY: Persist reasoning trace to logs/agent_reasoning_traces.log.
================================================================================================
```

---

### Embedded Intermediate Reasoning Trace #2: Genre-Energy Contradiction Self-Correction
When running `python -m src.main contradiction` (`genre="classical"`, `target_energy=0.98`), Step 1 detects a `0.74` energy gap between the user's `0.98` target and the catalog's `0.23` classical average, triggering Step 3 self-correction to raise the top-5 mean score from `46.7%` to `49.4%`:

```text
================================================================================================
AGENTIC WORKFLOW TRACE | 2026-11-03 16:49:48 | Profile: Adversarial: The Contradiction (Max Energy Classical)
Query / Intent: dramatic melancholic classical piano and high intensity orchestral power
------------------------------------------------------------------------------------------------
  [Step 1 | PLAN & GUARDRAILS] Tool: `parse_natural_language_request + sanitize_user_profile`
    • Thought     : Extract structured listener constraints and audit catalog coverage before scoring.
    • Observation : Parsed target genre='classical' (2 exact tracks in catalog), mood='melancholic', energy=0.98 (genre avg energy=0.23, gap=0.74).
    • Decision    : Flagged potential OOD/Contradiction risk for Critic inspection after Pass #1.
  [Step 2 | TOOL CALL #1 (RETRIEVE & RANK PASS 1)] Tool: `SongKnowledgeIndex.retrieve_multi_source_context + recommend_songs`
    • Thought     : Retrieve across 3 data sources and rank top-5 using baseline weights (genre=0.25, mood=0.20, energy=0.20, rag=0.20).
    • Observation : Matched Musicology Guide: 'Late-Night Introspection & Catharsis Guide'. Pass #1 Top-1 Score = 58.0%, Top-5 Mean Score = 46.7%.
    • Decision    : Send Pass #1 candidate pool to Critic Decision Node.
  [Step 3 | CRITIQUE & SELF-CORRECTION (TOOL CALL #2)] Tool: `adaptive_weight_rebalancer + recommend_songs (Pass 2)`
    • Thought     : Critic flagged Pass #1: Genre-Energy contradiction detected (target energy 0.98 vs 'classical' catalog average 0.23). Rebalancing weights to prioritize Semantic RAG & viable audio features.
    • Observation : Re-ran `recommend_songs` with adaptive weights (genre=0.20, mood=0.20, energy=0.30, rag=0.20). Top-1 Score shifted 58.0% -> 55.3% (Mean 46.7% -> 49.4%).
    • Decision    : SELF_CORRECTED: Adopted Pass #2 adaptive ranking.
  [Step 4 | VERIFY & SYNTHESIZE] Tool: `verify_recommendations + generate_rag_explanation`
    • Thought     : Confirm zero hallucinated titles against catalog and synthesize multi-source AI DJ setlist commentary.
    • Observation : Guardrail passed=True; anchored explanation on #1 'Moonlight Sonata Flow' via Local Grounded Multi-Source RAG Synthesizer (3 Data Sources).
    • Decision    : OUTPUT_READY: Persist reasoning trace to logs/agent_reasoning_traces.log.
================================================================================================
```

---

### Embedded Intermediate Reasoning Trace #3: High-Confidence Pass #1 Approval Branch
When running `python -m src.main --query "pomodoro exam week debugging session with felt piano and rain foley"`, Pass #1 achieves a **95.3%** top match (*Library Rain*) and **80.3%** top-5 mean score, so the Step 3 Critic approves Pass #1 without overriding weights:

```text
================================================================================================
AGENTIC WORKFLOW TRACE | 2026-11-03 16:49:48 | Profile: AI DJ Intent: pomodoro exam week debugging session with felt p
Query / Intent: pomodoro exam week debugging session with felt piano and rain foley
------------------------------------------------------------------------------------------------
  [Step 1 | PLAN & GUARDRAILS] Tool: `parse_natural_language_request + sanitize_user_profile`
    • Thought     : Extract structured listener constraints and audit catalog coverage before scoring.
    • Observation : Parsed target genre='lofi' (4 exact tracks in catalog), mood='chill', energy=0.32 (genre avg energy=0.37, gap=0.05).
    • Decision    : Standard catalog coverage confirmed; proceed to Pass #1 retrieval.
  [Step 2 | TOOL CALL #1 (RETRIEVE & RANK PASS 1)] Tool: `SongKnowledgeIndex.retrieve_multi_source_context + recommend_songs`
    • Thought     : Retrieve across 3 data sources and rank top-5 using baseline weights (genre=0.25, mood=0.20, energy=0.20, rag=0.20).
    • Observation : Matched Musicology Guide: 'Cognitive Flow & Coding Acoustics Guide'. Pass #1 Top-1 Score = 95.3%, Top-5 Mean Score = 80.3%.
    • Decision    : Send Pass #1 candidate pool to Critic Decision Node.
  [Step 3 | CRITIQUE & DECISION NODE] Tool: `verify_candidate_confidence`
    • Thought     : Verify whether Pass #1 top score and genre/energy alignment meet quality thresholds (>= 65.0%).
    • Observation : Pass #1 Top-1 Score (95.3%) and Mean Score (80.3%) satisfy quality and diversity invariants.
    • Decision    : APPROVED: Retain Pass #1 ranking without weight override.
  [Step 4 | VERIFY & SYNTHESIZE] Tool: `verify_recommendations + generate_rag_explanation`
    • Thought     : Confirm zero hallucinated titles against catalog and synthesize multi-source AI DJ setlist commentary.
    • Observation : Guardrail passed=True; anchored explanation on #1 'Library Rain' via Local Grounded Multi-Source RAG Synthesizer (3 Data Sources).
    • Decision    : OUTPUT_READY: Persist reasoning trace to logs/agent_reasoning_traces.log.
================================================================================================
```

---

## 2. AI Coding Agent Collaboration Log (Module 3 & Module 4)

**What tasks did we give the AI coding agent?**

1. **Multi-Source Semantic RAG & Specialized Local Embeddings (`src/rag_engine.py`):** Expanding `data/songs.csv` to 30 tracks, adding `data/musicology_guides.json` and `data/custom_listener_notes.md`, and building `SongKnowledgeIndex` with `MOOD_SYNONYM_CLUSTERS` and `GENRE_AFFINITY_CLUSTERS`.
2. **Multi-Step Agentic Workflow (`src/recommender.py`):** Implementing the 4-step `Plan → Tool Call #1 → Critique & Self-Correction (Tool Call #2) → Verify & Synthesize` workflow and persisting intermediate traces to [`logs/agent_reasoning_traces.log`](logs/agent_reasoning_traces.log).
3. **Runtime Guardrails & Automated Testing (`src/guardrails.py`, `tests/test_recommender.py`):** Enforcing $\sum w_i = 1.0000$ weight normalization, input clamping, prompt-injection sanitization, and 11 automated `pytest` tests.

**What did we verify or revise manually?**

1. **ASCII Table Comma-Splitting Fix:** Changed the internal delimiter of matched RAG keywords in `score_song` from commas (`[rainy, night, acoustic]`) to slashes (`[pomodoro/exam/week]`) so `format_recommendations_table` in `src/main.py` renders each RAG reason on a single clean bullet line.
2. **Natural-Language Mood Priority Tuning:** Re-ordered mood keyword rules in `parse_natural_language_request` so `"rainy"` maps to `"chill"` rather than prematurely triggering `"melancholic"`.
3. **API Key Security (`.env` vs. `.env.example`):** Moved live API credentials into a git-ignored `.env` file while keeping `.env.example` clean for version control.

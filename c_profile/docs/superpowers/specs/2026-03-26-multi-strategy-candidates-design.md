# Multi-Strategy Candidate Generation via OpenRouter

**Date:** 2026-03-26
**Goal:** Beat Anik's 61% BIRD mini-dev accuracy by combining OpenRouter inference with multi-strategy prompting, feedback loop, and enhanced voting.
**Target:** 67-72% execution accuracy on 500-question BIRD mini-dev benchmark.

---

## Context

Current state:
- Local HuggingFace inference with gpt-oss-120b achieves 41% (no feedback) / 47% (with feedback)
- Anik's implementation using OpenRouter achieves 61% (no feedback, no voting)
- Gap is primarily due to local inference issues: truncation at 4096 tokens, broken chat templates, max_new_tokens=512, no marker extraction for reasoning model output

Reference implementation: `/Users/vora/Downloads/Anik/` (llm_backends_local.py, generate_candidates.py)

## Architecture

Phases 1-3 (profiles, indexes, schema linking) remain unchanged. Phase 4 (candidates.py) and LLM backend (llm.py) are enhanced.

```
Phase 4 (Enhanced):
  For each question:
    1. Build 3 structurally different prompts
       - Candidate A: Full schema + short profiles inline + few-shot
       - Candidate B: Pruned schema + full profiles block + few-shot
       - Candidate C: Pruned schema + long profiles + NO few-shot
    2. Generate SQL via OpenRouter (gpt-oss-120b)
       - With ###FINAL_BEGIN### / ###FINAL_END### marker extraction
       - max_new_tokens: 1900
    3. Feedback loop on each candidate
       - Execute -> if error or empty -> re-prompt with error/context
       - Up to 3 retries per candidate
       - Cross-candidate SQL hints when available
    4. Two-stage voting
       - Stage 1: Majority vote (2+ agree -> pick)
       - Stage 2: LLM tiebreaker (replaces random fallback)
```

---

## 1. OpenRouter Backend (llm.py)

### New class: OpenRouterBackend

- Uses OpenRouter API (OpenAI-compatible endpoint: https://openrouter.ai/api/v1)
- Default model: `openai/gpt-oss-120b`
- Requires `OPENROUTER_API_KEY` environment variable (reads via `os.environ`)
- Raises clear error if key not set
- max_new_tokens: 1900 (up from 512)
- Context: 131K tokens available (no truncation)

### Marker Injection

The gpt-oss-120b is a reasoning model that outputs chain-of-thought before answers. Markers ensure clean extraction.

Injected into BOTH system message and last user message:
```
IMPORTANT OUTPUT FORMAT (MANDATORY):
1) Put your FINAL answer between these exact markers:
###FINAL_BEGIN###
<final answer>
###FINAL_END###
2) Do NOT write anything inside the markers except the final answer.
3) You may write reasoning OUTSIDE the markers.
```

### Marker Extraction

1. Find `###FINAL_BEGIN###` and `###FINAL_END###` in raw output
2. Extract content between markers
3. If markers not found: fall back to existing regex extraction (fence match, then SELECT/WITH detection)
4. Return `{text: clean_sql, raw: full_output, thoughts: reasoning}`

### make_backend update

```python
make_backend("openrouter")                                    # -> gpt-oss-120b
make_backend("openrouter", model_id="openai/gpt-oss-20b")    # override
make_backend("huggingface")                                   # existing local (unchanged)
```

---

## 2. Three Candidate Strategies (candidates.py)

### Candidate A: "Wide Net" — Full schema + short profiles + few-shot

- **Schema:** All tables, all columns
- **Profiles:** Short profiles as inline comments on linked columns only
- **Linked marker:** `/* LINKED */` on schema-linked columns
- **Primary keys:** Labeled with `-- primary key`
- **Few-shot:** 8 masked examples from BIRD training set
- **Temperature:** 0 (deterministic)
- **Purpose:** Catches cases where schema linking missed a table

Prompt structure:
```
You are a SQLite expert. Given the database schema and column profiles below,
write a SQL query to answer the question.

### Examples
{8 few-shot examples}

### Database Schema
CREATE TABLE customers (
  CustomerID INTEGER,  -- primary key
  Segment TEXT,
  Currency TEXT  -- Customer segment type  /* LINKED */
);
...all tables...

### Hint
{evidence}

### Question
{question}

### SQL
Write only the SQL query with no explanation.

SQL:
```

### Candidate B: "Deep Focus" — Pruned schema + full profiles + few-shot

- **Schema:** Only linked tables and linked columns (pruned, bare name + type)
- **Profiles:** Full profiles as separate `### Column Profiles` block (stats + samples + dev docs)
- **Few-shot:** 8 masked examples
- **Temperature:** 0 (deterministic)
- **Purpose:** Maximum context about columns that matter

Prompt structure:
```
You are a SQLite expert. Given the database schema and column profiles below,
write a SQL query to answer the question.

### Examples
{8 few-shot examples}

### Database Schema
CREATE TABLE customers (
  CustomerID INTEGER,
  Currency TEXT
);

### Column Profiles
**customers.Currency**
[PROFILE] Type: TEXT, Distinct: 3, Nulls: 0. Top values: EUR (4500), CZK (3200), USD (1300).
[DEV DOC] ISO 4217 currency code for customer's primary account.

### Hint
{evidence}

### Question
{question}

### SQL
Write only the SQL query with no explanation.

SQL:
```

### Candidate C: "Independent Thinker" — Pruned schema + long profiles + NO few-shot

- **Schema:** Only linked tables and linked columns (pruned, bare name + type)
- **Profiles:** Long profiles (LLM-enhanced statistical descriptions)
- **Few-shot:** NONE
- **Temperature:** 0.7, seed: 42
- **Purpose:** Independent thinking without few-shot bias, stats-focused

Prompt structure:
```
You are a SQLite expert. Given the database schema and column profiles below,
write a SQL query to answer the question.

### Database Schema
CREATE TABLE customers (
  CustomerID INTEGER,
  Currency TEXT
);

### Column Descriptions
Field customers.Currency means: The currency column stores ISO currency codes.
3 distinct values observed. Most common: EUR (4500 rows), CZK (3200 rows).
Values are uppercase 3-letter codes.

### Hint
{evidence}

### Question
{question}

### SQL
Write only the SQL query with no explanation.

SQL:
```

---

## 3. Enhanced Feedback Loop (candidates.py)

### Change 1: Retry on empty generation

Current behavior: gives up immediately on empty SQL.
New behavior: treats empty generation as a retryable error.

Re-prompt for empty generation:
```
The previous attempt produced no valid SQL output.
Generate a valid SQLite SQL query for the following question.

### Database Schema
{schema}

### Column Profiles
{profiles}

### Hint
{evidence}

### Question
{question}

Write only the SQL query.
SQL:
```

Empty SQL is never accepted as a final result. If all 3 retries produce empty, candidate is marked failed.

### Change 2: Cross-candidate SQL hints

When correcting a failing candidate, if another candidate already succeeded, include it as reference:

```
You are a SQLite expert. The SQL query below has an error.
Fix the SQL query and return ONLY the corrected SQL with no explanation.

### Database Schema
{schema}

### Column Profiles
{profiles}

### Hint
{evidence}

### Question
{question}

### Working SQL from another approach
{successful_candidate_sql}

### Your SQL (with error)
{failed_sql}

### Error Message
{error_msg}

### Corrected SQL
Write only the corrected SQL query.

SQL:
```

Cross-candidate hints are only used when available (a prior candidate succeeded). Order of generation: A, B, C — so B can reference A's success, C can reference A or B.

---

## 4. Two-Stage Voting (candidates.py)

### Stage 1: Majority Vote (unchanged)

- Execute all 3 candidates against SQLite
- Group by result sets (frozenset equality)
- If 2+ agree -> pick that SQL. Done.

### Stage 2: LLM Tiebreaker (replaces random fallback)

Triggered only when no majority agreement (~17% of questions).

- Filter to candidates that executed successfully
- If only 1 valid -> pick it (no LLM call needed)
- If 0 valid -> pick first non-empty candidate
- If 2+ valid but disagree -> ask LLM:

```
You are a SQLite expert. Given a question and multiple SQL queries that produce
different results, select the most correct one.

### Question
{question}

### Hint
{evidence}

### Candidates
A) {sql_a}

B) {sql_b}

C) {sql_c}

Which candidate is most likely correct? Reply with ONLY the letter.

Answer:
```

Parse first letter from response. If parse fails -> pick first valid candidate (not random).

---

## 5. Files Modified

### llm.py
- Add `OpenRouterBackend` class
- Add `_inject_marker_instruction()` helper
- Add `_extract_final_and_thoughts()` helper
- Add `_fallback_extract_answer()` helper
- Update `make_backend()` to support `"openrouter"` kind

### candidates.py
- Add `render_full_schema_highlighted()` — full schema with linked columns marked
- Add `build_full_schema_prompt()` — Candidate A prompt builder
- Add `build_long_profile_prompt()` — Candidate C prompt builder
- Modify `process_question()` — 3 different strategies instead of 3 same-prompt
- Modify `generate_with_correction()` — retry on empty, cross-candidate hints
- Add `llm_tiebreaker()` — Stage 2 voting
- Modify `majority_vote()` — call tiebreaker instead of random fallback
- Update CLI: add `--backend openrouter` option

### Files NOT changed
- pipeline.py, profiles.py, indexes.py — unchanged
- Phase4.sh — CLI arg update only (`--backend openrouter`)

---

## 6. Expected Impact

| Change | Estimated Impact |
|--------|-----------------|
| OpenRouter (fixes local inference) | 41% -> ~61% (matches Anik baseline) |
| 3 diverse prompt strategies | +2-3% |
| Full profiles in Candidate B | +1-2% |
| Feedback loop with empty retry fix | +3-5% |
| Cross-candidate correction hints | +1-2% |
| LLM tiebreaker (replaces random) | +1-2% |
| **Total estimated** | **~67-72%** |

---

## 7. Cost Estimate

OpenRouter gpt-oss-120b: $0.04/1M input, $0.19/1M output.

For 500 questions with feedback:
- ~2000-2500 API calls
- ~6M input tokens, ~1.2M output tokens
- **Total: ~$0.47** (or $0.00 with free tier)

---

## 8. CLI Usage

```bash
# Basic run with OpenRouter
python candidates.py --input results/schema_links_all_dbs_*.json --backend openrouter

# With feedback loop
python candidates.py --input results/schema_links_all_dbs_*.json --backend openrouter --feedback

# With feedback + custom output
python candidates.py --input results/schema_links_all_dbs_*.json --backend openrouter --feedback --out results/candidates_openrouter.json

# Resume partial run
python candidates.py --input results/schema_links_all_dbs_*.json --backend openrouter --feedback --resume results/candidates_openrouter.json
```

Environment: `export OPENROUTER_API_KEY=your_key_here`

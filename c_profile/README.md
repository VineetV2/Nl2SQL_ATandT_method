# Schema Linking for Text-to-SQL

Implementation of **"Automatic Metadata Extraction for Text-to-SQL"** (arXiv: 2505.19988v2) by the AT&T CDO team, which achieved #1 on the BIRD leaderboard.

Given a natural language question and a relational database, this pipeline identifies which tables and columns are needed to answer it (**schema linking**), then generates and selects the best SQL candidate via majority voting.

---

## Pipeline Overview

```
                         ┌──────────────────────────────┐
                         │     SQLite Database           │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────────────────────────┐
                    │   Phase 1: Database Profiling          │
                    │   (profiles.py)                        │
                    │                                        │
                    │   1a. Raw SQLite stats per column       │
                    │   1b. LLM long descriptions (3-6 sent) │
                    │   1c. Merge with dev-doc CSVs           │
                    │   1d. LLM short descriptions (1 sent)   │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────────────────────────┐
                    │   Phase 2: Index Building              │
                    │   (indexes.py)                         │
                    │                                        │
                    │   2a. FAISS semantic index (embeddings) │
                    │   2b. LSH index (k-shingle MinHash)     │
                    └───────────────────┬───────────────────┘
                                        │
          ┌─────────────────────────────────────────────────────┐
          │   Phase 3: Schema Linking (pipeline.py)              │
          │                                                      │
          │   Per question:                                      │
          │   ├─ Extract literals from question                  │
          │   ├─ FAISS top-10 + LSH literal matches → focused   │
          │   ├─ Build 5 schema/profile combinations             │
          │   ├─ For each combo: 3 SQL candidates (seed+shuffle) │
          │   ├─ Correction loop (≤3 retries per candidate)      │
          │   ├─ SQLglot validation (4 fixes)                    │
          │   ├─ 8 few-shot examples from BIRD train set         │
          │   └─ Schema links = union of all referenced columns  │
          └─────────────────────┬───────────────────────────────┘
                                │
          ┌─────────────────────────────────────────────────────┐
          │   Phase 4: Candidate Generation (candidates.py)      │
          │                                                      │
          │   Per question (using schema links from Phase 3):    │
          │   ├─ 3 candidates: same prompt, different seed/temp  │
          │   ├─ 8 few-shot examples from BIRD train set         │
          │   ├─ Majority voting (execute + compare result sets) │
          │   └─ Fallback: random among valid candidates         │
          └─────────────────────┬───────────────────────────────┘
                                │
                                v
                    results/schema_links_*.json
                    results/candidates_*.json
```

---

## Phase 1: Database Profiling (`profiles.py`)

Generates column profiles in 4 sequential steps:

### Step 1a — Raw SQLite Statistics

For every column, computes directly from SQLite:

| Statistic | Description |
|-----------|-------------|
| `n_rows` | Total row count |
| `null_count` | Number of NULLs |
| `distinct_count` | Distinct non-null values |
| `min` / `max` | Min and max values |
| `avg` / `std` | Mean and standard deviation (numeric columns) |
| `top_values` | Top-5 most frequent values with counts |
| `samples` | First 5 non-null values |
| `shape` | `avg_len`, `min_len`, `max_len`, `pct_digits`, `pct_upper`, `pct_lower`, `pct_other`, `common_prefix` |

Output: `<db_id>.long_profiles.jsonl`

### Step 1b — LLM Long Descriptions (Paper Section 2.1)

Calls GPT-5.2 to generate a **3-6 sentence** natural language description of each column based on its raw stats and dev docs. These descriptions:
- Explain what the column stores
- Describe value format, range, and distribution
- Guide the SQL generator on proper literal usage

The LLM description is prepended to the raw stats in `profile_long_en`. This enriched text is what FAISS indexes in Phase 2.

### Step 1c — Full Profiles (merge with dev docs)

Merges long profiles with developer CSV documentation from BIRD's `database_description/` directory. Output has `[PROFILE]` + `[DEV DOC]` sections.

Output: `<db_id>.full_profiles.jsonl`

### Step 1d — LLM Short Descriptions

Calls GPT-5.2 to compress the full profile into **1 sentence (≤25 words)**. Used as lightweight inline SQL comments in schema prompts.

Output: `<db_id>.short_profiles.jsonl`

---

## Phase 2: Index Building (`indexes.py`)

### FAISS Semantic Index

Embeds each column's long profile text (with LLM description) via `text-embedding-3-small` (1536-dim). Normalized vectors stored in `faiss.IndexFlatIP` for cosine similarity search.

Output: `<db_id>.faiss` + `<db_id>.faiss_meta.json`

### LSH Index for Literal Matching (Paper Section 3)

For each column, fetches up to **N=10,000** distinct values from SQLite. Each value is normalized into variants (original, lower, upper, title, no-punct). Values are decomposed into **character 3-shingles** (paper footnote 6: `kshingle` library) before MinHash hashing (128 permutations, threshold=0.3).

At query time: exact match first, then approximate LSH fallback.

Output: `<db_id>.lsh.pkl` + `<db_id>.lsh_index.json`

---

## Phase 3: Schema Linking (`pipeline.py`)

### Per Question:

1. **Literal extraction** — Quoted strings, uppercase codes, title-case nouns, numbers (stopwords filtered)
2. **FAISS + LSH retrieval** — Top-10 semantic + literal matches → focused columns. PKs always included for JOINs.
3. **5 schema combinations** (paper Section 3):

| Combo | Tables | Columns | Profile |
|-------|--------|---------|---------|
| `focused_short` | Focused | Focused | Short (1-sentence) |
| `focused_long` | Focused | Focused | Long (LLM 3-6 sentences) |
| `full_short` | All | All | Short |
| `full_long` | All | All | Long |
| `focused_full` | Focused | All in focused tables | Short + Long + Dev docs |

4. **3 SQL candidates per combo** — `(temp=0, no shuffle)`, `(temp=0.7, seed=1, shuffle=1)`, `(temp=0.7, seed=2, shuffle=2)`
5. **8 few-shot examples** — Retrieved from BIRD training set (9428 questions) via masked-question FAISS similarity (paper Section 4)
6. **Question masking** — `"Fresno"` → `<value>`, `2013` → `<year>`, `CDS` → `<code>`, `42` → `<number>`
7. **Correction loop** (≤3 retries) — If SQL uses literals not found in any referenced column, augment schema + re-prompt LLM
8. **SQLglot 4 validation fixes** (paper Section 4):
   - NULLS LAST on ASC ORDER BY in LIMIT queries
   - Remove ORDER BY from scalar MIN()/MAX() without GROUP BY
   - Replace string concatenation (`||`) with separate SELECT columns
   - Rewrite `WHERE col = (SELECT MIN/MAX(...))` → `ORDER BY col LIMIT 1`
9. **Schema links** = union of all columns across 5 combos × 3 candidates

---

## Phase 4: Candidate Generation & Selection (`candidates.py`)

Takes schema links from Phase 3 and generates final SQL:

1. **Same-prompt approach** (paper Section 4) — All 3 candidates use identical pruned-schema prompt
2. **Diversity** via LLM seed + temperature + column order shuffling
3. **8 few-shot examples** from BIRD training set (masked-question retrieval)
4. **Majority voting** — Execute candidates, compare result sets as frozensets. 2+ agree → winner. No agreement → random among valid candidates.
5. **MCS-SQL alternative** — Optional LLM-based multiple-choice selection (`--mcs` flag)

---

## File Structure

```
c_profile/
├── profiles.py          Phase 1: column profiling (1a→1b→1c→1d)
├── indexes.py           Phase 2: FAISS + LSH index building & querying
├── pipeline.py          Phase 3: schema linking (SchemaLinker, FewShotRetriever,
│                        correction loop, SQLglot validation, pipeline runner)
├── candidates.py        Phase 4: SQL candidate generation & majority voting
├── llm.py               LLM backend (OpenAI GPT-5.2, seed support)
├── requirements.txt     Python dependencies
├── .env.example         API key template
├── .gitignore
├── MINIDEV/                         ← download separately (see below)
│   ├── mini_dev_sqlite.json         500 BIRD minidev questions
│   └── dev_databases/               11 SQLite databases with dev docs
│       └── <db_id>/
│           ├── <db_id>.sqlite
│           └── database_description/
└── results/                         ← generated at runtime
    ├── schema_links_*.json          Phase 3 output
    └── candidates_*.json            Phase 4 output
```

---

## Setup

### 1. Download the BIRD Benchmark Data

Download the BIRD minidev dataset from the official benchmark:

- **BIRD Benchmark**: https://bird-bench.github.io/
- **Minidev (500 questions + 11 SQLite databases)**: download and extract into `MINIDEV/` so the directory structure looks like:
  ```
  MINIDEV/
  ├── mini_dev_sqlite.json
  └── dev_databases/
      ├── california_schools/
      ├── card_games/
      ├── debit_card_specializing/
      ├── european_football_2/
      ├── financial/
      ├── formula_1/
      ├── student_club/
      ├── superhero/
      ├── thrombosis_prediction/
      ├── toxicology/
      └── works_cycles/
  ```
- **BIRD Training Set** (optional, for few-shot examples): download `train.json` (9428 questions) and place it at `../dataset/bird/train/train.json` relative to this directory.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

| Package | Role |
|---------|------|
| `openai` | GPT-5.2 (SQL generation, profiles) + embeddings (`text-embedding-3-small`) |
| `faiss-cpu` | Semantic similarity indexes (columns + few-shot questions) |
| `datasketch` | MinHash LSH for literal value matching |
| `sqlglot` | SQL AST parsing, column extraction, 4 validation fixes |
| `numpy` | Vector operations for FAISS |

---

## How to Run

### Full pipeline (single database)

```bash
# Phase 1: Generate profiles (run once per database)
python profiles.py --db debit_card_specializing

# Phase 2: Build indexes (run once per database)
python indexes.py --db debit_card_specializing

# Phase 3: Schema linking
python pipeline.py --db debit_card_specializing

# Phase 4: Candidate generation (uses Phase 3 output)
python candidates.py --input results/schema_links_debit_card_specializing_latest.json
```

### Full pipeline (all 11 databases)

```bash
python profiles.py --all
python indexes.py --all
python pipeline.py --all --questions_per_db 4
```

### Quick test (dry run, no LLM calls)

```bash
python pipeline.py --db debit_card_specializing --no_llm
```

### Common flags

| Flag | Available in | Description |
|------|-------------|-------------|
| `--db <name>` | profiles, indexes, pipeline | Process single database |
| `--all` | profiles, indexes, pipeline | Process all 11 databases |
| `--overwrite` | profiles, indexes | Regenerate existing files |
| `--no_llm` | pipeline | Dry run (schema linking without SQL generation) |
| `--questions_per_db N` | pipeline | Limit questions per database |
| `--top_k N` | pipeline | FAISS retrieval count (default: 10) |
| `--mcs` | candidates | Use MCS-SQL selection instead of majority voting |
| `--out <path>` | pipeline, candidates | Custom output path |

---

## Paper Reference

Papageorgiou, G., Krishnamurthy, S., Ahuja, K., Ko, W., and Katsogiannis-Meimarakis, G. (2025).
**Automatic Metadata Extraction for Text-to-SQL.**
arXiv preprint arXiv:2505.19988v2. AT&T Chief Data Office.
https://arxiv.org/abs/2505.19988

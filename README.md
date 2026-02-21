# Schema Linking for Text-to-SQL

An implementation of the AT&T CDO team's paper **"Automatic Metadata Extraction for Text-to-SQL"** (arXiv: 2505.19988v2), which achieved the **#1 rank on the BIRD leaderboard** at time of publication.

Given a natural language question and a relational database, this system identifies which tables and columns are required to answer the question — a task called **schema linking**. Precise schema links reduce the search space for SQL generation and are a critical prerequisite for accurate text-to-SQL systems.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Phase 1 — Column Profiles](#phase-1--column-profiles-profilespy)
3. [Phase 2 — Search Indexes](#phase-2--search-indexes-indexespy)
4. [Phase 3 — Schema Linking](#phase-3--schema-linking-pipelinepy)
5. [Phase 4a — Evaluate Schema Links](#phase-4a--evaluate-schema-links-evaluatepy)
6. [Phase 4b — Phrase-to-Column Alignment](#phase-4b--phrase-to-column-alignment-analysispy)
7. [Phase 4c — Explain Errors](#phase-4c--explain-errors-explain_errorspy)
8. [Why It Works](#why-it-works)
9. [Key Results](#key-results)
10. [Repository Structure](#repository-structure)
11. [Setup and Installation](#setup-and-installation)
12. [How to Run](#how-to-run)
13. [Output Format](#output-format)
14. [Paper Reference](#paper-reference)

---

## The Big Picture

The pipeline answers one core question:

> **Given a natural language question like "How many schools scored above 400 in Math?", which database columns are relevant?**

It is a 4-phase system. All phases must be completed in order before schema linking can be run.

```
SQLite Database
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 1: profiles.py — Column Profiles                  │
│                                                         │
│  1a. Long profiles   ← raw SQLite statistics per column │
│  1b. Full profiles   ← long + developer CSV docs        │
│  1c. Short profiles  ← GPT-5.2 one-sentence summaries   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 2: indexes.py — Search Indexes                    │
│                                                         │
│  2a. FAISS index  ← semantic similarity over profiles   │
│  2b. LSH index    ← literal value matching (N=10,000)   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 3: pipeline.py — Schema Linking (per question)    │
│                                                         │
│  3a. Extract literals from question                     │
│  3b. FAISS → top-10 semantically similar columns        │
│  3c. LSH   → columns containing literal values          │
│  3d. Union → focused columns + focused tables + PKs     │
│  3e. Build 5 schema+profile combinations                │
│  3f. Generate 3 SQL candidates per combo (15 total)     │
│  3g. Correction loop (≤3 retries per candidate)         │
│  3h. SQLglot validation: fix NULL ordering + MIN/MAX    │
│  3i. Schema links = union of all referenced columns     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Phase 4: analysis.py — Evaluation + Alignment           │
│                                                         │
│  4a. Evaluate schema link recall/precision/F1 vs gold   │
│  4b. Phrase-to-column alignment via LLM                 │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Column Profiles (`profiles.py`)

**Goal:** For every column in every database, generate a rich text description so the AI can understand what the column stores.

### Step 1a — Long Profiles (SQLite statistics)

For every `table.column` pair, the code queries the SQLite database and computes:

| Field | Description |
|-------|-------------|
| `n_rows` | Total row count for the table |
| `null_count` | Number of NULL values in this column |
| `distinct_count` | Number of distinct non-null values |
| `min` / `max` | Minimum and maximum observed values |
| `top_values` | Top-5 values by frequency with counts |
| `samples` | First 5 non-null values |
| `shape` | Character-level stats: `avg_len`, `pct_digits`, `pct_upper`, `pct_lower`, `common_prefix` |

**Example output:**
```
Column: schools.Virtual
Declared type: TEXT
Primary key: False
Stats: n_rows=17686, null_count=0, distinct_count=5, min=A, max=P
Shape: avg_len=1.0, min_len=1, max_len=1; upper=100%
Top values: 'F' (9123), 'N' (5891), 'P' (1444), 'A' (901), 'Y' (327)
Samples: 'F', 'N', 'F', 'P', 'N'
```

This tells the AI exactly what kind of data the column holds — numeric ranges, text lengths, top values — without it needing to query the database itself.

Output: `<db_id>.long_profiles.jsonl` — one JSON record per column.

---

### Step 1b — Full Profiles (merge with developer documentation)

Each BIRD database ships with `database_description/` CSV files written by human annotators. These describe what each column *means* in plain English. The code merges them with the long profile:

```
[PROFILE]
Column: schools.Virtual
Stats: n_rows=17686, distinct_count=5 ...
Top values: 'F' (9123), 'N' (5891) ...

[DEV DOC]
Description: Indicates if the school is virtual
Values: F=Exclusively Virtual, N=Not Virtual, P=Partial
```

Now each profile has both **statistical truth** (what the data actually looks like) and **semantic meaning** (what it represents). The code robustly handles different CSV column name conventions across databases.

Output: `<db_id>.full_profiles.jsonl` — one JSON record per column.

---

### Step 1c — Short Profiles (LLM one-sentence summaries)

The full profile is often hundreds of words — too long to include in every SQL prompt. So GPT-5.2 is given the full profile and asked to compress it into a single sentence of at most 25 words:

```
Input  →  Full profile for schools.Virtual (stats + dev doc)
GPT    →  "Indicates whether a school is exclusively virtual (F),
           not virtual (N), or partially virtual (P)."
```

The result is post-processed to enforce the 25-word limit and terminal punctuation. These one-sentence descriptions are later injected as inline SQL comments, keeping token usage low for large schemas.

Output: `<db_id>.short_profiles.jsonl` — one JSON record per column.

---

## Phase 2 — Search Indexes (`indexes.py`)

**Goal:** Build two indexes so that at query time, relevant columns can be retrieved instantly without scanning the entire schema.

### FAISS Index (semantic similarity)

- Each column's long profile text is converted to a 1536-dimensional vector using OpenAI's `text-embedding-3-small` model
- Vectors are L2-normalized and stored in a `faiss.IndexFlatIP` (inner product = cosine similarity after normalization)
- At query time: embed the question → find the **top-10 most semantically similar columns**

**Example:**
```
Question: "How many schools scored above 400 in Math?"
FAISS finds:
  schools.Virtual    (score: 0.41)
  satscores.AvgScrMath (score: 0.40)
  schools.School     (score: 0.35)
  ...
```

FAISS catches columns where the column *name* differs from the question word (e.g., "Math score" → `AvgScrMath`) because it matches on the *meaning* of the profile text, not just the column name.

Output files per database:
- `<db_id>.faiss` — binary FAISS index
- `<db_id>.faiss_meta.json` — column metadata aligned to FAISS row indices

---

### LSH Index (literal value matching)

- For each column, up to **N=10,000 distinct non-null values** are fetched directly from SQLite (per paper Section 3 — this is the key difference from using only the top-5 samples in profiles)
- Each value is normalized into multiple string variants (original, lowercased, uppercased, title-cased, punctuation-stripped)
- A `MinHash` object is built from character trigrams of all variants (128 permutations)
- All MinHash objects are inserted into a `MinHashLSH` index (Jaccard threshold = 0.3)

At query time:
1. **Exact match first** — check if the literal appears in any column's value list
2. **Approximate LSH match** — if no exact match, use MinHash similarity as fallback

**Example:**
```
Question: "...customers paid in CZK currency..."
Literal extracted: "CZK"
LSH exact match → transactions_1k.Currency contains 'CZK'
```

LSH catches filter values ("CZK", "F", "2013") and finds exactly which column stores them — something semantic search cannot do reliably.

Output files per database:
- `<db_id>.lsh.pkl` — serialized `MinHashLSH` object
- `<db_id>.lsh_index.json` — per-column value lists for exact lookup

---

## Phase 3 — Schema Linking (`pipeline.py`)

This is the core of the pipeline. For each natural language question, it runs a multi-step process to identify which columns are needed.

### Step 3a — Extract Literals from the Question

Regex patterns scan the question for candidate literal values:
- Quoted strings: `"CZK"`, `'France'`
- Uppercase abbreviations: `CZK`, `EUR`, `SAT`
- Title-case proper nouns: `France`, `Math`
- Numeric tokens: `400`, `2013`, `0.5`

Common English question words (`What`, `Which`, `How`, `List`, etc.) are filtered as stopwords.

---

### Step 3b+c — FAISS + LSH Retrieval → Focused Columns

```
Question: "How many schools with average Math score > 400 are exclusively virtual?"
Evidence: "Exclusively virtual refers to Virtual = 'F'"

FAISS hits:  schools.Virtual (0.41), satscores.AvgScrMath (0.40), ...
LSH hits:    satscores.AvgScrMath ← '400', frpm.Enrollment ← '400'

Focused columns (union + PKs for JOINs):
  satscores.AvgScrMath, satscores.cds, schools.CDSCode, schools.Virtual, ...

Focused tables: {satscores, schools, frpm}
```

Primary keys of all focused tables are always added unconditionally — they're needed for JOIN conditions even if the question doesn't mention them.

---

### Step 3d — Five Schema Combinations

The same question is sent to GPT with **5 different presentations of the schema**. The paper found this diversity significantly improves column recall:

| Combo | Tables | Columns | Profile comments |
|-------|--------|---------|-----------------|
| `focused_short` | Focused only | Focused only | 1-sentence |
| `focused_long` | Focused only | Focused only | Raw statistics |
| `full_short` | All tables | All columns | 1-sentence |
| `full_long` | All tables | All columns | Raw statistics |
| `focused_full` | Focused only | ALL their columns | Stats + dev doc combined |

Each combo produces `CREATE TABLE` blocks with profile text as inline SQL comments:

```sql
CREATE TABLE schools (
  CDSCode TEXT  -- Unique identifier linking to SAT scores and FRPM data.
  Virtual TEXT  -- Indicates if school is exclusively virtual (F), not virtual (N), or partial (P).
  School  TEXT  -- Official name of the school.
);
```

The intuition: `focused_short` is fast and targeted; `full_short` ensures no column is missed due to FAISS/LSH retrieval gaps; `focused_full` gives GPT the richest context for the most likely tables.

---

### Step 3e — Three SQL Candidates per Combo (15 total)

For each of the 5 combos, GPT-5.2 is called **3 times** with different settings:

| Candidate | Temperature | Column order |
|-----------|-------------|-------------|
| 0 | 0 (deterministic) | Natural DB order |
| 1 | 0.7 (creative) | Shuffled (seed=1) |
| 2 | 0.7 (creative) | Shuffled (seed=2) |

Column order shuffling (paper Section 4) reduces GPT's positional bias toward columns that appear early in the schema. Temperature variation generates more diverse SQL, covering more column combinations. Result: **5 × 3 = 15 SQL candidates per question**.

---

### Step 3f — Few-Shot Examples (Structure-Aware Retrieval)

Before generating each SQL, the code retrieves **8 similar questions** from the 500-question minidev pool to use as in-context examples.

The retrieval uses masked question embeddings:
```
Original:  "How many schools scored above 400 in Math?"
Masked:    "How many [MASK] scored above [NUMBER] in [MASK]?"
```

Entity tokens (quoted strings, numbers, years, short codes) are replaced by typed placeholders so the similarity is based on **question structure**, not specific entity words. The 8 most similar questions by masked-embedding cosine similarity are retrieved (leave-one-out — the query question is excluded from its own pool).

---

### Step 3g — Correction Loop (up to 3 retries)

After GPT generates SQL, the code checks: *does the SQL use literal values that don't appear in any of the columns it referenced?*

**Example:**
```sql
-- GPT generated:
SELECT COUNT(*) FROM transactions_1k WHERE Currency = 'CZK'

-- Columns GPT referenced in SQL: transactions_1k.Amount
-- LSH says 'CZK' is in:         transactions_1k.Currency
-- → Mismatch! 'CZK' used but Currency column not referenced.
```

The correction prompt tells GPT:
> *"You used the literal 'CZK' but did not reference any field containing it. The field `transactions_1k.Currency` contains this literal. Please revise the SQL."*

The schema is augmented with the missing column and GPT is re-asked. Up to 3 retries per candidate (paper Section 3, steps d–e).

---

### Step 3h — SQLglot Validation

Two systematic LLM SQL errors are corrected via SQLglot AST rewriting (paper Section 4):

1. **Missing `NULLS LAST`** — on ascending `ORDER BY` inside `LIMIT` queries (SQLite returns NULLs first by default, which causes wrong top-N results)
2. **Spurious `ORDER BY`** — on scalar `MIN()`/`MAX()` expressions without `GROUP BY` (ordering is meaningless and wastes tokens)

---

### Step 3i — Schema Link Aggregation

The final schema links for a question are the **union of all `(table, column)` pairs** referenced across all 15 SQL candidates (5 combos × 3 candidates). Column extraction uses SQLglot AST analysis with full alias resolution (T1/T2 → canonical table names), with a regex-based fallback if SQLglot fails to parse the SQL.

```json
"schema_links": [
  {"table": "satscores", "column": "AvgScrMath"},
  {"table": "satscores", "column": "cds"},
  {"table": "schools",   "column": "CDSCode"},
  {"table": "schools",   "column": "Virtual"}
]
```

---

## Phase 4a — Evaluate Schema Links (`evaluate.py`)

Compares the schema links found by Phase 3 against the columns actually used in the **gold SQL answer** from the BIRD benchmark.

**How it works:**

1. Parses the gold SQL using `sqlglot` into an AST
2. Resolves all table aliases (`T1` → `satscores`, `T2` → `schools`)
3. Extracts every `(table, column)` pair used in the gold SQL
4. Compares against the `schema_links` from Phase 3

```
Gold SQL uses: satscores.AvgScrMath, schools.Virtual, schools.CDSCode, satscores.cds
Phase 3 found: satscores.AvgScrMath, schools.Virtual, schools.CDSCode, satscores.cds

Recall    = 4/4 = 100%  ← all gold columns were found
Precision = 4/6 = 67%   ← 2 extra columns included (not harmful)
F1        = 80%
```

Reports are broken down by:
- Overall aggregate (Recall / Precision / F1 / Perfect recall count)
- Per database (11 databases, sorted by recall)
- Per difficulty level (simple / moderate / challenging)
- Per question with missed columns listed

**Run:**
```bash
# Evaluate all schema_links_*.json files in results/
python evaluate.py

# Evaluate a specific file only
python evaluate.py --input results/schema_links_all_dbs_4per_20260218_125609.json
```

---

## Phase 4b — Phrase-to-Column Alignment (`analysis.py`)

A new LLM task: for each question, identify every DB-related phrase and explicitly map it to the candidate column(s) it refers to.

**What is a "DB part"?**
Any word or phrase in the question that corresponds to:
- A column by name or semantics (e.g., "average score in Math" → `satscores.AvgScrMath`)
- A literal value stored in a column (e.g., "400" → threshold for `satscores.AvgScrMath`)
- A table concept (e.g., "SAT test" → `satscores` table columns)
- A filter condition from the evidence/hint

**Example output:**
```json
[
  {"phrase": "average score in Math", "columns": ["satscores.AvgScrMath"]},
  {"phrase": "400",                   "columns": ["satscores.AvgScrMath"]},
  {"phrase": "SAT test",              "columns": ["satscores.AvgScrMath", "satscores.cds"]},
  {"phrase": "exclusively virtual",   "columns": ["schools.Virtual"]}
]
```

**Prompt structure:**

Each LLM call is built from 4 parts:

1. **System message** — task definition, rules, and output format (JSON array only, no markdown)
2. **Few-shot examples** — up to 3 worked examples (controlled by `--num_examples`), each containing:
   - Full `CREATE TABLE` schema for that example's database (hardcoded)
   - The question + evidence
   - The candidate columns
   - Gold alignments
3. **Live database schema** — `CREATE TABLE` blocks loaded directly from the target question's SQLite file via `_render_db_schema()` (cached per database — loaded once, reused for all questions in the same DB)
4. **The actual question** — question, evidence, and the candidate columns from Phase 3

Schema is included in both the few-shot block and the real question so the LLM sees column types and relationships, not just column names.

**Controllable options:**
- `--num_examples` — number of few-shot examples to include (0, 1, 2, or 3). Use `0` for zero-shot (no examples, relies only on task description and schema).
- Results are saved incrementally every `--save_every` questions and support resuming interrupted runs.
- Only maps to columns from `focused_columns` — hallucinated columns are filtered out at validation time.

**Run:**
```bash
python analysis.py \
    --input       results/schema_links_all_dbs_4per_20260218_125609.json \
    --output      results/phrase_column_alignment_$(date +%Y%m%d_%H%M%S).json \
    --num_examples 3    # 0 = zero-shot, 1/2/3 = few-shot (default: 3)
```

---

## Phase 4c — Explain Errors (`explain_errors.py`)

Finds all questions where Phase 3 recall was less than 100% (i.e. at least one gold column was missed), then uses GPT to explain **why** each column was missed.

**What it provides for each wrong question:**
- Which columns were missed and which were correctly found
- The FAISS semantic hits (with similarity scores) that were returned
- The LSH literal hits that were returned
- The SQL candidates the system generated for each schema combo
- A GPT-generated explanation of *why* the specific columns were missed

**Example explanation:**
> *"The column `schools.school` was missed because the question uses the word 'schools' generically as a concept rather than referencing the `School` name column directly. FAISS matched on higher-scoring structural columns (`CDSCode`, `Virtual`). LSH found no literal values pointing to the `School` column. Since no SQL candidate referenced `school` by name, it was excluded from the final schema links."*

**Run:**
```bash
# Explain all wrong questions
python explain_errors.py \
    --input results/schema_links_all_dbs_4per_20260218_125609.json

# Explain only the first 5 wrong questions (for quick testing)
python explain_errors.py \
    --input results/schema_links_all_dbs_4per_20260218_125609.json \
    --n 5
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Phase 3 schema links JSON |
| `--output` | auto-timestamped | Where to save explanations |
| `--n` | all wrong | Limit to N wrong questions |
| `--model` | `gpt-5.2` | OpenAI model to use |
| `--save_every` | `3` | Save every N explanations (for resuming) |

---

## Why It Works

| Design Choice | Why It Helps |
|---------------|-------------|
| **FAISS semantic search** | Finds columns where the name differs from the question word ("Math score" → `AvgScrMath`) |
| **LSH on N=10,000 values** | Finds exactly which column stores a filter value ("CZK", "F", "2013") — semantic search can't do this reliably |
| **5 schema combinations** | Reduces the chance GPT misses a column due to information overload or context window bias |
| **3 candidates per combo** | Temperature variation + column shuffling generates SQL diversity, covering more column combinations |
| **Correction loop** | Catches SQL that uses a literal value without referencing the column that contains it |
| **Short profiles as comments** | GPT understands what `Virtual = 'F'` means without needing to guess from the column name alone |
| **Structure-aware few-shot retrieval** | Shows GPT the expected SQL pattern for structurally similar questions, not just topically similar ones |
| **SQLglot post-processing** | Silently fixes two systematic GPT SQL errors before they cause wrong evaluation results |

---

## Key Results

### debit_card_specializing — 30 questions (full run)

| Metric | Value |
|--------|-------|
| Mean Recall | 91.1% |
| Mean Precision | 91.7% |
| Mean F1 | 89.2% |
| Perfect recall | 22 / 30 (73%) |
| Zero recall | 0 / 30 |

### All 11 databases — 69 questions total

| Metric | Value |
|--------|-------|
| Mean Recall | **95.6%** |
| Mean Precision | 89.1% |
| Mean F1 | 90.5% |
| Perfect recall | 58 / 69 (84.1%) |

**Per-database highlights:**

| Database | Recall | Notes |
|----------|--------|-------|
| formula_1 | 100% | |
| superhero | 100% | |
| thrombosis_prediction | 100% | |
| toxicology | 71.1% | Subquery alias edge cases |

**Per-difficulty breakdown:**

| Difficulty | Recall |
|------------|--------|
| Simple | 93.9% |
| Moderate | 97.2% |
| Challenging | 96.2% |

---

## Repository Structure

The codebase is organized into **8 files**:

```
c_profile/
│
├── profiles.py          Phase 1 — Build column profiles
│                        Generates long (SQLite stats), full (stats + dev docs),
│                        and short (LLM one-sentence) profiles for every column.
│                        CLI: python profiles.py --all
│
├── indexes.py           Phase 2 — Build search indexes
│                        Creates FAISS (semantic) and LSH (literal value) indexes
│                        from the long profiles. Also provides query_faiss() and
│                        query_lsh() functions used by pipeline.py.
│                        CLI: python indexes.py --all
│
├── pipeline.py          Phase 3 — Schema linking + SQL generation
│                        Contains SchemaLinker class (FAISS+LSH retrieval, 5 schema
│                        combos, 3 SQL candidates, correction loop, SQLglot validation,
│                        column extraction) and FewShotRetriever class.
│                        CLI: python pipeline.py --all --questions_per_db 4
│
├── evaluate.py          Phase 4a — Evaluate schema links
│                        Compares schema_links from Phase 3 against gold SQL columns.
│                        Reports recall/precision/F1 per question, per database,
│                        and per difficulty level.
│                        CLI: python evaluate.py
│                             python evaluate.py --input results/schema_links_*.json
│
├── analysis.py          Phase 4b — Phrase-to-column alignment
│                        Maps every NL phrase in a question to the specific candidate
│                        columns it refers to, using GPT with few-shot examples and
│                        the full live database schema.
│                        CLI: python analysis.py --input results/schema_links_*.json
│                             python analysis.py --num_examples 0   # zero-shot
│
├── explain_errors.py    Phase 4c — Explain schema linking errors
│                        Finds all questions where recall < 100%, then calls GPT to
│                        explain why each column was missed (FAISS/LSH failure mode,
│                        alias issues, indirect references, etc.).
│                        CLI: python explain_errors.py --input results/schema_links_*.json
│                             python explain_errors.py --n 5   # first 5 wrong only
│
├── llm.py               Utility — LLM backend abstraction
│                        OpenAIBackend (GPT-5.2 via API, primary),
│                        HFTransformersBackend (Qwen local models),
│                        OSSHFPBackend (GPT-OSS via HF pipeline).
│                        Factory: make_backend("openai", model_id="gpt-5.2")
│                        Backends are cached to avoid reloading model weights.
│
├── MINIDEV/
│   ├── mini_dev_sqlite.json             500 BIRD minidev questions with gold SQL
│   └── dev_databases/
│       └── <db_id>/
│           ├── <db_id>.sqlite
│           ├── database_description/         Developer CSV docs per table (input)
│           ├── <db_id>.long_profiles.jsonl   ← Phase 1a output
│           ├── <db_id>.full_profiles.jsonl   ← Phase 1b output
│           ├── <db_id>.short_profiles.jsonl  ← Phase 1c output
│           ├── <db_id>.faiss                 ← Phase 2 output
│           ├── <db_id>.faiss_meta.json       ← Phase 2 output
│           ├── <db_id>.lsh.pkl               ← Phase 2 output
│           └── <db_id>.lsh_index.json        ← Phase 2 output
│
└── results/
    ├── schema_links_<tag>_<timestamp>.json          Phase 3 output
    ├── phrase_column_alignment_<timestamp>.json     Phase 4b output
    └── error_explanations_<tag>_<timestamp>.json    Phase 4c output
```

---

## Setup and Installation

```bash
# 1. Navigate to the project directory
cd /path/to/DAIL-SQL/c_profile

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key (required for all LLM calls and embeddings)
export OPENAI_API_KEY=your_api_key_here
```

**Key dependencies:**

| Package | Role |
|---------|------|
| `openai` | GPT-5.2 for SQL generation and short profile generation; `text-embedding-3-small` for FAISS vectors |
| `faiss-cpu` | Semantic column retrieval index and few-shot question retrieval |
| `datasketch` | MinHash LSH index for literal value matching |
| `sqlglot` | AST-based SQL column extraction and SQL validation/fixing |
| `numpy` | Vector arithmetic for FAISS |

---

## How to Run

All four phases must be run in order for each database before schema linking can be used.

### Step 1: Generate profiles

```bash
# All 11 databases
python profiles.py --all

# Single database
python profiles.py --db debit_card_specializing

# Regenerate (overwrite existing files)
python profiles.py --all --overwrite
```

Generates three JSONL files next to each `.sqlite` file:
- `<db_id>.long_profiles.jsonl`
- `<db_id>.full_profiles.jsonl`
- `<db_id>.short_profiles.jsonl`

> **Note:** Only needs to be run once per database. Existing files are skipped automatically.
> Estimated API cost for all 11 databases: ~$2–5 (short profile LLM calls).

---

### Step 2: Build indexes

```bash
# All 11 databases
python indexes.py --all

# Single database
python indexes.py --db debit_card_specializing

# Rebuild (overwrite existing indexes)
python indexes.py --all --overwrite
```

Generates four index files next to each `.sqlite` file:
- `<db_id>.faiss` — binary FAISS index (cosine similarity, 1536-dim)
- `<db_id>.faiss_meta.json` — column metadata aligned to FAISS row indices
- `<db_id>.lsh.pkl` — serialized MinHashLSH object (threshold=0.3, 128 permutations)
- `<db_id>.lsh_index.json` — per-column value lists for exact lookup

> FAISS index construction requires one OpenAI embedding API call per column.

---

### Step 3: Run schema linking

```bash
# Full run: all 30 questions in debit_card_specializing
python pipeline.py --db debit_card_specializing

# Quick test: first 4 questions across all 11 databases
python pipeline.py --all --questions_per_db 4

# Dry run (no LLM calls — shows focused schema only, skips SQL generation)
python pipeline.py --db debit_card_specializing --no_llm

# Custom output path
python pipeline.py --db debit_card_specializing --out /tmp/my_results.json
```

Results are saved incrementally every 5 questions to `results/schema_links_<tag>_<timestamp>.json`.

---

### Step 4a: Evaluate schema links

```bash
# Evaluate all schema_links_*.json files in results/ (combines duplicates)
python evaluate.py

# Evaluate a single specific file
python evaluate.py --input results/schema_links_all_dbs_4per_20260218_125609.json
```

Prints a full report: overall recall/precision/F1, per-database breakdown, per-difficulty breakdown, and a per-question table showing missed columns.

---

### Step 4b: Phrase-to-column alignment

```bash
# 3-shot (default — best accuracy)
python analysis.py \
    --input        results/schema_links_all_dbs_4per_20260218_125609.json \
    --output       results/phrase_column_alignment_$(date +%Y%m%d_%H%M%S).json \
    --model        gpt-5.2 \
    --num_examples 3

# Zero-shot (no examples — faster, lower cost)
python analysis.py \
    --input        results/schema_links_all_dbs_4per_20260218_125609.json \
    --num_examples 0

# 1-shot or 2-shot
python analysis.py \
    --input        results/schema_links_all_dbs_4per_20260218_125609.json \
    --num_examples 1
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to Phase 3 schema links JSON |
| `--output` | auto-timestamped | Output path for alignment results |
| `--model` | `gpt-5.2` | OpenAI model to use |
| `--num_examples` | `3` | Few-shot examples in prompt (0–3) |
| `--save_every` | `5` | Save results every N questions (for resuming) |

---

### Step 4c: Explain errors

```bash
# Explain all wrong questions (recall < 100%)
python explain_errors.py \
    --input results/schema_links_all_dbs_4per_20260218_125609.json

# Test on first 5 wrong questions only
python explain_errors.py \
    --input results/schema_links_all_dbs_4per_20260218_125609.json \
    --n 5

# Save to a specific file
python explain_errors.py \
    --input  results/schema_links_all_dbs_4per_20260218_125609.json \
    --output results/error_explanations_$(date +%Y%m%d_%H%M%S).json
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Phase 3 schema links JSON |
| `--output` | auto-timestamped | Where to save explanations |
| `--n` | all wrong | Limit to N wrong questions |
| `--model` | `gpt-5.2` | OpenAI model to use |
| `--save_every` | `3` | Save every N explanations (for resuming) |

---

## Output Format

### Phase 3 output — `schema_links_<tag>_<timestamp>.json`

```json
{
  "results": [
    {
      "question_id": 1471,
      "db_id": "debit_card_specializing",
      "question": "What is the ratio of customers who pay in EUR against customers who pay in CZK?",
      "evidence": "ratio = count(Currency='EUR') / count(Currency='CZK')",
      "literals": ["EUR", "CZK"],
      "focused_tables": ["customers", "transactions_1k"],
      "focused_columns": ["customers.Currency", "customers.CustomerID", "..."],
      "faiss_hits": [
        {"table": "customers", "column": "Currency", "score": 0.5894}
      ],
      "lsh_hits": [
        {"table": "customers", "column": "Currency", "literal": "EUR", "match_type": "exact"},
        {"table": "customers", "column": "Currency", "literal": "CZK", "match_type": "exact"}
      ],
      "combo_sqls": {
        "focused_short": "SELECT ...",
        "focused_long":  "SELECT ...",
        "full_short":    "SELECT ...",
        "full_long":     "SELECT ...",
        "focused_full":  "SELECT ..."
      },
      "combo_columns": {
        "focused_short": [{"table": "customers", "column": "Currency"}]
      },
      "schema_links": [
        {"table": "customers", "column": "Currency"}
      ],
      "gold_sql": "SELECT CAST(SUM(CASE WHEN Currency='EUR' THEN 1 ELSE 0 END) AS REAL) / ...",
      "difficulty": "simple"
    }
  ],
  "errors": []
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `schema_links` | **Final answer**: union of all columns referenced across 5 combos × 3 candidates |
| `focused_tables` | Tables selected by FAISS + LSH before SQL generation |
| `focused_columns` | Columns selected by FAISS + LSH (including PKs for JOINs) |
| `faiss_hits` | Top-10 semantic matches with cosine similarity scores |
| `lsh_hits` | Literal-matched columns with match type (exact or approximate) |
| `combo_sqls` | Deterministic (temp=0) SQL candidate for each of the 5 combos |
| `combo_columns` | Columns referenced across all 3 candidates for each combo |
| `gold_sql` | Reference SQL from the BIRD benchmark (not used during linking) |
| `difficulty` | BIRD difficulty label: simple / moderate / challenging |

---

### Phase 4b output — `phrase_column_alignment_<timestamp>.json`

```json
{
  "results": [
    {
      "question_id": 5,
      "db_id": "california_schools",
      "question": "How many schools with average Math score > 400 are exclusively virtual?",
      "evidence": "Exclusively virtual refers to Virtual = 'F'",
      "db_schema": "CREATE TABLE frpm (\n  CDSCode TEXT, ...\n);\n\nCREATE TABLE schools (\n  ...\n);",
      "candidate_columns": ["satscores.AvgScrMath", "schools.Virtual", "..."],
      "alignments": [
        {"phrase": "average score in Math", "columns": ["satscores.AvgScrMath"]},
        {"phrase": "400",                   "columns": ["satscores.AvgScrMath"]},
        {"phrase": "exclusively virtual",   "columns": ["schools.Virtual"]}
      ],
      "raw_llm_response": "[{\"phrase\": \"average score in Math\", ...}]"
    }
  ]
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `alignments` | **Final answer**: validated phrase → column mappings (only from candidate set) |
| `candidate_columns` | The focused columns from Phase 3 that were given to the LLM |
| `db_schema` | The full `CREATE TABLE` schema loaded from SQLite (included for reference/debugging) |
| `raw_llm_response` | Raw text returned by the LLM before parsing (useful for debugging parse failures) |

---

### Phase 4c output — `error_explanations_<tag>_<timestamp>.json`

```json
{
  "results": [
    {
      "question_id": 5,
      "db_id": "california_schools",
      "difficulty": "simple",
      "question": "How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?",
      "evidence": "Exclusively virtual refers to Virtual = 'F'",
      "gold_sql": "SELECT COUNT(*) FROM satscores T1 JOIN schools T2 ...",
      "recall": 0.8,
      "missed_cols": ["schools.school"],
      "found_cols":  ["satscores.avgscrmath", "schools.virtual", "schools.cdscode", "satscores.cds"],
      "gold_cols":   ["satscores.avgscrmath", "schools.virtual", "schools.cdscode", "satscores.cds", "schools.school"],
      "explanation": "The column schools.school was missed because the question never uses the word 'school name' — it only says 'schools' generically. FAISS matched on CDSCode and Virtual which had higher semantic similarity. LSH found no literal value pointing to the School column. Since no SQL candidate referenced the school name column, it was excluded from the final schema links."
    }
  ]
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `recall` | Recall score for this question (< 1.0 means some columns were missed) |
| `missed_cols` | Columns the system failed to include |
| `found_cols` | Columns the system correctly returned |
| `gold_cols` | All columns required by the gold SQL |
| `explanation` | GPT's analysis of why the specific columns were missed |

---

## Paper Reference

Papageorgiou, G., Krishnamurthy, S., Ahuja, K., Ko, W., and Katsogiannis-Meimarakis, G. (2025).
**Automatic Metadata Extraction for Text-to-SQL.**
arXiv preprint arXiv:2505.19988v2. AT&T Chief Data Office.
https://arxiv.org/abs/2505.19988

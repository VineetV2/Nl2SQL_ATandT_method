Is # Schema Linking for Text-to-SQL

An implementation of the AT&T CDO team's paper **"Automatic Metadata Extraction for Text-to-SQL"** (arXiv: 2505.19988v2), which achieved the #1 rank on the BIRD leaderboard at time of publication.

Given a natural language question and a relational database, this system identifies which tables and columns are required to answer the question — a task called **schema linking**. Precise schema links reduce the search space for SQL generation and are a critical prerequisite for accurate text-to-SQL systems.

---

## Table of Contents

1. [Architecture and Pipeline](#architecture-and-pipeline)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Setup and Installation](#setup-and-installation)
5. [How to Run](#how-to-run)
6. [Output Format](#output-format)
7. [Paper Reference](#paper-reference)

---

## Architecture and Pipeline

The pipeline has three sequential phases. All phases must be run in order for a given database before schema linking can be performed.

```
SQLite database
      |
      v
[ Phase 1: Database Profiling ]
  1a. Long profiles    -- raw SQLite statistics per column
  1b. Full profiles    -- long profile + developer CSV documentation
  1c. Short profiles   -- LLM-generated 1-sentence summaries (GPT-5.2)
      |
      v
[ Phase 2: Index Building ]
  2a. FAISS index      -- semantic similarity over long profile embeddings
  2b. LSH index        -- literal value matching over up to N=10,000 distinct values
      |
      v
[ Phase 3: Schema Linking ]  (per question)
  3a. Extract literals from question (quoted, numeric, uppercase tokens)
  3b. FAISS: top-10 semantically similar columns
  3c. LSH:   columns containing literal values (exact then approximate)
  3d. Union -> focused columns + focused tables + PKs (for JOINs)
  3e. Build 5 schema+profile combinations
  3f. For each combo: generate 3 SQL candidates (temp=0 + 2x temp=0.7 with shuffled col order)
  3g. Correction loop (<=3 retries): re-ask LLM when SQL literals are unmapped
  3h. SQLglot validation: fix NULL ordering and wrong MIN/MAX patterns
  3i. Schema links = union of all columns referenced across 5 combos x 3 candidates
      |
      v
results/schema_links_<db>_<timestamp>.json
```

### Phase 1: Database Profiling

**Step 1a — Long profiles** (`profiles_sqlite_local.py`)

For every `(table, column)` pair in the SQLite database, a structured text profile is computed from raw SQL statistics. Each profile includes:

| Field | Description |
|-------|-------------|
| `n_rows` | Total row count for the table |
| `null_count` | Number of NULL values in this column |
| `distinct_count` | Number of distinct non-null values |
| `min` / `max` | Minimum and maximum observed values |
| `top_values` | Top-5 values by frequency with counts |
| `samples` | First 5 non-null values (insertion order) |
| `shape` | Character-level statistics: `avg_len`, `pct_digits`, `pct_upper`, `pct_lower`, `common_prefix` |

The shape statistics, including the longest common prefix shared by at least 80% of values, follow the approach described in Section 2 of the paper.

Output: `<db_id>.long_profiles.jsonl` — one JSON record per column.

**Step 1b — Full profiles** (`profiles_full_local.py`)

The long profile is merged with human-authored developer documentation from the BIRD benchmark's `database_description/` CSV files. The CSVs supply column descriptions, data format notes, and value explanations.

The merged document is structured with a `[PROFILE]` section (raw stats) followed by a `[DEV DOC]` section (developer text). This full profile is later used directly in the `focused_full` schema combination and as context for the LLM in step 1c.

Output: `<db_id>.full_profiles.jsonl` — one JSON record per column.

**Step 1c — Short profiles** (`short_profiles_local.py`)

A GPT-5.2 call is made for each column to compress the full profile into a single sentence of at most 25 words. The system prompt instructs the model to describe the column's purpose and obvious value format with no markdown or commentary. The result is post-processed to enforce the 25-word limit and terminal punctuation.

Short profiles serve as lightweight inline comments in schema prompts, keeping token usage low for large schemas.

Output: `<db_id>.short_profiles.jsonl` — one JSON record per column.

---

### Phase 2: Index Building (`build_indexes.py`)

**FAISS semantic index**

Each column's long profile text is embedded via the OpenAI `text-embedding-3-small` model (1536-dimensional vectors). Vectors are L2-normalized and stored in a `faiss.IndexFlatIP` (inner product, equivalent to cosine similarity after normalization). At query time, the question text is embedded with the same model and the top-10 most similar columns are retrieved.

Output files per database:
- `<db_id>.faiss` — binary FAISS index
- `<db_id>.faiss_meta.json` — column metadata aligned to FAISS row indices

**LSH index for literal value matching**

For each column, up to N=10,000 distinct non-null values are fetched directly from SQLite (per paper Section 3). Each value is normalized into multiple string variants (original, lowercased, uppercased, title-cased, punctuation-stripped). A `MinHash` object is built from character trigrams of all variants, using 128 permutations. All MinHash objects are inserted into a `MinHashLSH` index with a Jaccard similarity threshold of 0.3.

At query time, literals extracted from the question are first matched exactly against the stored value lists. If no exact match is found, approximate MinHash LSH matching is used as a fallback.

Output files per database:
- `<db_id>.lsh.pkl` — serialized `MinHashLSH` object
- `<db_id>.lsh_index.json` — per-column value lists for exact matching

---

### Phase 3: Schema Linking (`schema_linking.py`, `run_pipeline.py`)

**Literal extraction**

The question string is parsed by regex to identify candidate literal values: quoted strings, uppercase abbreviations (e.g., `CZK`, `EUR`), title-case proper nouns, and numeric tokens. Common English question words (`What`, `Which`, `How`, etc.) are filtered as stopwords.

**FAISS + LSH retrieval**

FAISS retrieves the top-10 semantically relevant columns. LSH retrieves columns whose stored values contain any of the extracted literals. The union of both result sets forms the **focused column set**. Primary keys of all focused tables are added unconditionally to enable JOIN construction.

**Five schema combinations**

Five distinct schema+profile prompt blocks are constructed to cover complementary signal:

| Combo | Tables | Columns | Profile |
|-------|--------|---------|---------|
| `focused_short` | Focused only | Focused only | 1-sentence |
| `focused_long` | Focused only | Focused only | Raw stats |
| `full_short` | All tables | All columns | 1-sentence |
| `full_long` | All tables | All columns | Raw stats |
| `focused_full` | Focused only | All columns in those tables | Stats + dev doc |

Each combo produces a `CREATE TABLE` block per table with profile text as inline SQL comments. Columns not in the filter set are omitted from the schema text.

**Three SQL candidates per combo**

For each of the 5 combos, 3 SQL candidates are generated:
- Candidate 0: temperature=0, natural column order
- Candidate 1: temperature=0.7, column order shuffled with seed 1
- Candidate 2: temperature=0.7, column order shuffled with seed 2

Column order shuffling (following paper Section 4) reduces bias toward columns that appear early in the schema text.

**Few-shot retrieval**

A FAISS index is built over all 500 BIRD minidev questions, with entity tokens (quoted strings, numbers, years, short codes) replaced by typed placeholders (`<value>`, `<number>`, `<year>`, `<code>`). For each query question, the 8 most similar questions by masked-embedding cosine similarity are retrieved and included as in-context examples, with the query question excluded from its own pool (leave-one-out). This follows the structure-aware few-shot strategy in paper Section 4.

**Correction loop**

After generating each SQL, string literals appearing in the SQL are extracted and looked up in the LSH index. If a literal is used in the SQL but the column containing it is not referenced in the parsed SQL, the system augments the schema with the missing column and re-asks the LLM with an explicit correction prompt explaining the discrepancy. Up to 3 correction attempts are made per candidate (paper Section 3, steps d–e).

**SQLglot validation**

Two systematic LLM SQL errors are corrected via SQLglot AST rewriting (paper Section 4):
1. Missing `NULLS LAST` on ascending `ORDER BY` inside `LIMIT` queries
2. Spurious `ORDER BY` on scalar `MIN()` / `MAX()` expressions without `GROUP BY`

**Schema link aggregation**

The final schema links for a question are the union of all `(table, column)` pairs referenced across all 5 combos and all 3 candidates. Column extraction uses SQLglot AST analysis with alias resolution (T1/T2 aliases to canonical table names), falling back to regex-based extraction if SQLglot is unavailable or the SQL fails to parse.

---

## Key Results

### debit_card_specializing — 30 questions (full run)

| Metric | Value |
|--------|-------|
| Mean Recall | 91.1% |
| Mean Precision | 91.7% |
| Mean F1 | 89.2% |
| Perfect recall (all gold columns found) | 22 / 30 (73%) |
| Zero recall | 0 / 30 |

### All 11 databases — 69 questions total

| Metric | Value |
|--------|-------|
| Mean Recall | 95.6% |
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

```
c_profile/
├── run_profiles.py              Phase 1 runner: generates long, full, and short profiles
│                                for one or all databases
├── build_indexes.py             Phase 2: builds FAISS semantic index and MinHash LSH
│                                index from profiles; also provides query functions
│                                used by Phase 3
├── run_pipeline.py              Phase 3 runner: loads questions from minidev,
│                                orchestrates SchemaLinker per question, saves results
├── schema_linking.py            Core Phase 3 logic: SchemaLinker class, literal
│                                extraction, 5-combo schema rendering, 3-candidate
│                                SQL generation, correction loop, SQLglot validation,
│                                column extraction, FewShotRetriever class
├── profiles_sqlite_local.py     Long profile generation: raw SQLite statistics
│                                (n_rows, null_count, distinct_count, min, max,
│                                top-5 values, samples, shape stats) per column
├── profiles_full_local.py       Full profile generation: merges long profiles with
│                                developer CSV documentation from database_description/
├── short_profiles_local.py      Short profile generation: builds prompts from full
│                                profiles and calls LLM backend to produce one-sentence
│                                column summaries (<=25 words)
├── llm_backends_local.py        LLM backend abstraction: OpenAIBackend (GPT-5.2 via
│                                API), HFTransformersBackend (Qwen), OSSHFPBackend
│                                (GPT-OSS via HF pipeline); backend instance caching
├── MINIDEV/
│   ├── mini_dev_sqlite.json     500 BIRD minidev questions with gold SQL and difficulty
│   └── dev_databases/           One subdirectory per database, each containing:
│       └── <db_id>/
│           ├── <db_id>.sqlite
│           ├── database_description/   Developer CSV documentation per table
│           ├── <db_id>.long_profiles.jsonl    (generated by Phase 1a)
│           ├── <db_id>.full_profiles.jsonl    (generated by Phase 1b)
│           ├── <db_id>.short_profiles.jsonl   (generated by Phase 1c)
│           ├── <db_id>.faiss                  (generated by Phase 2)
│           ├── <db_id>.faiss_meta.json        (generated by Phase 2)
│           ├── <db_id>.lsh.pkl               (generated by Phase 2)
│           └── <db_id>.lsh_index.json        (generated by Phase 2)
└── results/
    └── schema_links_<db>_<timestamp>.json     Pipeline output
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
pip install openai faiss-cpu datasketch sqlglot numpy

# 4. Set your OpenAI API key (required for all LLM calls and embeddings)
export OPENAI_API_KEY=your_api_key_here
```

The key dependencies and their roles:

| Package | Role |
|---------|------|
| `openai` | GPT-5.2 for SQL generation, short profile generation, and embeddings (`text-embedding-3-small`) |
| `faiss-cpu` | Semantic column retrieval index and few-shot question retrieval index |
| `datasketch` | MinHash LSH index for literal value matching |
| `sqlglot` | AST-based SQL column extraction and SQL validation/fixing |
| `numpy` | Vector arithmetic for FAISS |

### Quick Start

Once setup is complete, run the full pipeline with these three commands:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key_here
python run_profiles.py --all
python build_indexes.py --all
python run_pipeline.py --all --questions_per_db 4
```

> **Note:** `run_profiles.py --all` and `build_indexes.py --all` only need to be run once per database. Re-running is safe — existing files are skipped automatically unless `--overwrite` is passed.

---

## How to Run

All three phases must be completed for each database before schema linking can be run.

### Step 1: Generate profiles

```bash
# All 11 databases
python run_profiles.py --all

# Single database
python run_profiles.py --db debit_card_specializing

# Regenerate (overwrite existing files)
python run_profiles.py --all --overwrite
```

This generates three JSONL files next to each `.sqlite` file:
- `<db_id>.long_profiles.jsonl`
- `<db_id>.full_profiles.jsonl`
- `<db_id>.short_profiles.jsonl`

Estimated API cost for all 11 databases: approximately $2–5 in OpenAI calls (short profile generation for all columns).

### Step 2: Build FAISS and LSH indexes

```bash
# All 11 databases
python build_indexes.py --all

# Single database
python build_indexes.py --db debit_card_specializing

# Rebuild (overwrite existing indexes)
python build_indexes.py --all --overwrite
```

This generates four index files next to each `.sqlite` file:
- `<db_id>.faiss` — binary FAISS index (cosine similarity, 1536-dim)
- `<db_id>.faiss_meta.json` — column metadata aligned to FAISS row indices
- `<db_id>.lsh.pkl` — serialized MinHashLSH object (threshold=0.3, num_perm=128)
- `<db_id>.lsh_index.json` — per-column distinct value lists for exact lookup

FAISS index construction requires one OpenAI embedding API call per column.

### Step 3: Run schema linking

```bash
# Full run: all 30 questions in debit_card_specializing
python run_pipeline.py --db debit_card_specializing

# Full run with explicit question count
python run_pipeline.py --db debit_card_specializing --questions_per_db 30

# Quick test: first 4 questions across all 11 databases
python run_pipeline.py --all --questions_per_db 4

# Dry run (no LLM calls — shows focused schema only, no SQL generation)
python run_pipeline.py --db debit_card_specializing --no_llm

# Custom output path
python run_pipeline.py --db debit_card_specializing --out /tmp/my_results.json

# Single question via the SchemaLinker CLI
python schema_linking.py --db debit_card_specializing \
    --question "How many customers paid in CZK currency?"
```

Results are saved incrementally every 5 questions to `results/schema_links_<tag>_<timestamp>.json`.

---

## Output Format

Each result file is a JSON object with a `results` list and an `errors` list:

```json
{
  "results": [
    {
      "question_id": 1471,
      "db_id": "debit_card_specializing",
      "question": "What is the ratio of customers who pay in EUR against customers who pay in CZK?",
      "evidence": "ratio = count(Currency = 'EUR') / count(Currency = 'CZK')",
      "literals": ["EUR", "CZK"],
      "focused_tables": ["customers", "gasstations", "products", "transactions_1k", "yearmonth"],
      "focused_columns": ["customers.Currency", "customers.CustomerID", "..."],
      "faiss_hits": [
        {"table": "customers", "column": "Currency", "score": 0.5894},
        {"table": "products",  "column": "Description", "score": 0.41}
      ],
      "lsh_hits": [
        {"table": "customers", "column": "Currency", "literal": "EUR", "match_type": "exact"},
        {"table": "customers", "column": "Currency", "literal": "CZK", "match_type": "exact"}
      ],
      "combo_sqls": {
        "focused_short": "SELECT ... FROM customers;",
        "focused_long":  "SELECT ... FROM customers;",
        "full_short":    "SELECT ... FROM customers;",
        "full_long":     "SELECT ... FROM customers;",
        "focused_full":  "SELECT ... FROM customers;"
      },
      "combo_columns": {
        "focused_short": [{"table": "customers", "column": "Currency"}],
        "..."
      },
      "schema_links": [
        {"table": "customers", "column": "Currency"}
      ],
      "gold_sql": "SELECT CAST(SUM(CASE WHEN Currency = 'EUR' THEN 1 ELSE 0 END) AS REAL) / ...",
      "difficulty": "simple"
    }
  ],
  "errors": []
}
```

Key fields:

| Field | Description |
|-------|-------------|
| `schema_links` | Final answer: union of all columns referenced across 5 combos x 3 candidates |
| `focused_tables` | Tables selected by FAISS + LSH before SQL generation |
| `focused_columns` | Columns selected by FAISS + LSH (including PKs) |
| `faiss_hits` | Top-10 semantic matches with cosine similarity scores |
| `lsh_hits` | Literal-matched columns with match type (exact or approximate) |
| `combo_sqls` | Deterministic (temp=0) SQL candidate for each of the 5 combos |
| `combo_columns` | Columns referenced across all 3 candidates for each combo |
| `gold_sql` | Reference SQL from the BIRD benchmark (not used during linking) |
| `difficulty` | BIRD difficulty label: simple, moderate, or challenging |

---

## Paper Reference

Papageorgiou, G., Krishnamurthy, S., Ahuja, K., Ko, W., and Katsogiannis-Meimarakis, G. (2025).
**Automatic Metadata Extraction for Text-to-SQL.**
arXiv preprint arXiv:2505.19988v2. AT&T Chief Data Office.
https://arxiv.org/abs/2505.19988

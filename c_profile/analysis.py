"""
analysis.py
-----------
Phase 4b: Phrase-to-Column Alignment.

Maps natural language phrases in a question to the specific database
columns they refer to, using the candidate columns from Phase 3.

Usage:
  python analysis.py align \
      --input  results/schema_links_all_dbs_4per_20260218_125609.json \
      --output results/phrase_column_alignment_<timestamp>.json \
      --num_examples 3    # 0 = zero-shot, 1/2/3 = few-shot (default: 3)

For schema link evaluation, use evaluate.py instead:
  python evaluate.py
  python evaluate.py --input results/schema_links_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE        = Path(__file__).parent
MINIDEV_JSON = _HERE / "MINIDEV" / "mini_dev_sqlite.json"
DB_ROOT      = _HERE / "MINIDEV" / "dev_databases"


# ══════════════════════════════════════════════════════════════
# Phrase-to-Column Alignment
# ══════════════════════════════════════════════════════════════

# ── Schema renderer ───────────────────────────────────────────

def _render_db_schema(db_id: str) -> str:
    """
    Load the SQLite schema for db_id and render CREATE TABLE blocks.
    Returns an empty string if the database file is not found.
    """
    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        blocks = []
        for tbl in tables:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            cols = cur.fetchall()
            col_lines = [f"  {c[1]} {c[2] or 'TEXT'}" for c in cols]
            blocks.append(f"CREATE TABLE {tbl} (\n" + ",\n".join(col_lines) + "\n);")
        conn.close()
        return "\n\n".join(blocks)
    except Exception:
        return ""


# ── Few-shot examples (include schema so LLM sees the full picture) ──

_FEW_SHOT_EXAMPLES = [
    {
        "question": "Among the transactions made in the gas stations in the Czech Republic, how many of them take place after 2012/1/1?",
        "evidence": "Czech Republic is a country; after 2012/1/1 refers to a date filter on the transaction date",
        "schema": (
            "CREATE TABLE customers (\n"
            "  CustomerID INTEGER,\n"
            "  Segment TEXT,\n"
            "  Currency TEXT\n"
            ");\n\n"
            "CREATE TABLE gasstations (\n"
            "  GasStationID INTEGER,\n"
            "  ChainID INTEGER,\n"
            "  Country TEXT,\n"
            "  Segment TEXT\n"
            ");\n\n"
            "CREATE TABLE transactions_1k (\n"
            "  TransactionID INTEGER,\n"
            "  Date DATE,\n"
            "  CustomerID INTEGER,\n"
            "  CardID INTEGER,\n"
            "  GasStationID INTEGER,\n"
            "  ProductID INTEGER,\n"
            "  Amount INTEGER,\n"
            "  Price REAL\n"
            ");\n\n"
            "CREATE TABLE yearmonth (\n"
            "  CustomerID INTEGER,\n"
            "  Date TEXT,\n"
            "  Consumption REAL\n"
            ");"
        ),
        "candidate_columns": [
            "gasstations.Country",
            "gasstations.GasStationID",
            "transactions_1k.Date",
            "transactions_1k.GasStationID",
            "transactions_1k.TransactionID",
        ],
        "alignments": [
            {"phrase": "transactions",      "columns": ["transactions_1k.TransactionID"]},
            {"phrase": "gas stations",      "columns": ["gasstations.GasStationID", "transactions_1k.GasStationID"]},
            {"phrase": "Czech Republic",    "columns": ["gasstations.Country"]},
            {"phrase": "after 2012/1/1",    "columns": ["transactions_1k.Date"]},
        ],
    },
    {
        "question": "How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?",
        "evidence": "Exclusively virtual refers to Virtual = 'F'",
        "schema": (
            "CREATE TABLE frpm (\n"
            "  CDSCode TEXT,\n"
            "  School Name TEXT,\n"
            "  Enrollment (K-12) REAL,\n"
            "  Free Meal Count (K-12) REAL\n"
            ");\n\n"
            "CREATE TABLE satscores (\n"
            "  cds TEXT,\n"
            "  sname TEXT,\n"
            "  NumTstTakr INTEGER,\n"
            "  AvgScrRead INTEGER,\n"
            "  AvgScrMath INTEGER,\n"
            "  AvgScrWrite INTEGER\n"
            ");\n\n"
            "CREATE TABLE schools (\n"
            "  CDSCode TEXT,\n"
            "  School TEXT,\n"
            "  Virtual TEXT,\n"
            "  Magnet TEXT,\n"
            "  Latitude REAL,\n"
            "  Longitude REAL\n"
            ");"
        ),
        "candidate_columns": [
            "frpm.CDSCode",
            "satscores.AvgScrMath",
            "satscores.cds",
            "schools.CDSCode",
            "schools.Virtual",
        ],
        "alignments": [
            {"phrase": "average score in Math", "columns": ["satscores.AvgScrMath"]},
            {"phrase": "400",                   "columns": ["satscores.AvgScrMath"]},
            {"phrase": "SAT test",              "columns": ["satscores.AvgScrMath", "satscores.cds"]},
            {"phrase": "exclusively virtual",   "columns": ["schools.Virtual"]},
        ],
    },
    {
        "question": "List the home team names where the match resulted in a draw, along with the season year.",
        "evidence": "A draw means home team goals equal away team goals",
        "schema": (
            "CREATE TABLE match (\n"
            "  id INTEGER,\n"
            "  season TEXT,\n"
            "  date TEXT,\n"
            "  home_team_api_id INTEGER,\n"
            "  away_team_api_id INTEGER,\n"
            "  home_team_goal INTEGER,\n"
            "  away_team_goal INTEGER\n"
            ");\n\n"
            "CREATE TABLE team (\n"
            "  id INTEGER,\n"
            "  team_api_id INTEGER,\n"
            "  team_long_name TEXT,\n"
            "  team_short_name TEXT\n"
            ");"
        ),
        "candidate_columns": [
            "match.home_team_goal",
            "match.away_team_goal",
            "match.season",
            "team.team_long_name",
            "team.team_short_name",
        ],
        "alignments": [
            {"phrase": "home team names", "columns": ["team.team_long_name", "team.team_short_name"]},
            {"phrase": "draw",            "columns": ["match.home_team_goal", "match.away_team_goal"]},
            {"phrase": "season year",     "columns": ["match.season"]},
        ],
    },
]


def _format_few_shot(ex: Dict[str, Any]) -> str:
    """Render one few-shot example as a prompt block (schema + question + alignments)."""
    evid_str = f"\nEvidence: {ex['evidence']}" if ex.get("evidence") else ""
    cols_str = "\n".join(f"  - {c}" for c in ex["candidate_columns"])
    return (
        f"### Database Schema\n{ex['schema']}\n\n"
        f"### Question\n{ex['question']}{evid_str}\n\n"
        f"### Candidate Columns\n{cols_str}\n\n"
        f"### DB Parts (answer)\n{json.dumps(ex['alignments'], indent=2)}"
    )


def _build_alignment_prompt(
    question: str,
    evidence: str,
    candidate_columns: List[str],
    db_schema: str,
    num_examples: int = 3,
) -> Tuple[str, str]:
    """
    Build the system + user messages for phrase-to-column alignment.

    The prompt includes:
      1. Task definition + what counts as a DB part
      2. num_examples worked few-shot examples (0-3), each with:
           - Full CREATE TABLE schema
           - The question + evidence
           - Pre-filtered candidate columns
           - Gold alignments
      3. The actual question with its schema + candidate columns
    """
    examples = _FEW_SHOT_EXAMPLES[:max(0, num_examples)]
    few_shot_block = "\n\n---\n\n".join(_format_few_shot(ex) for ex in examples)

    cols_str = "\n".join(f"  - {c}" for c in candidate_columns)
    evid_str = f"\nEvidence: {evidence}" if evidence else ""

    system_msg = (
        "You are a database expert specializing in Text-to-SQL systems.\n\n"
        "## Task: Phrase-to-Column Alignment\n\n"
        "Given a natural language question and a relational database schema, identify every "
        "DB-related phrase in the question and map it to the specific column(s) it refers to.\n\n"
        "## What counts as a DB part?\n\n"
        "A DB part is any word or phrase in the question that corresponds to:\n"
        "  1. A TABLE concept   — e.g., 'transactions' → maps to the transactions table columns\n"
        "  2. A COLUMN concept  — e.g., 'average score in Math' → satscores.AvgScrMath\n"
        "  3. A LITERAL VALUE   — e.g., 'Czech Republic' → a value in gasstations.Country\n"
        "  4. A DATE / NUMBER   — e.g., '2012/1/1' → a date filter on transactions_1k.Date\n"
        "  5. A HINT condition  — phrases from the Evidence that map to a specific column\n\n"
        "## Rules\n\n"
        "1. Only map phrases to columns that appear in the Candidate Columns list.\n"
        "2. One phrase can map to multiple columns (e.g., a join key appears in two tables).\n"
        "3. Use exact format: table.column\n"
        "4. Ignore generic question words (How many, List, What is, etc.).\n"
        "5. Return ONLY a valid JSON array — no markdown fences, no explanation.\n\n"
        "## Output format\n\n"
        '[{"phrase": "<exact phrase from question or evidence>", "columns": ["table.column", ...]}, ...]'
    )

    if few_shot_block:
        examples_section = f"## Worked examples\n\n{few_shot_block}\n\n---\n\n"
    else:
        examples_section = ""

    user_msg = (
        f"{examples_section}"
        "## Now align the following question\n\n"
        f"### Database Schema\n{db_schema}\n\n"
        f"### Question\n{question}{evid_str}\n\n"
        f"### Candidate Columns\n{cols_str}\n\n"
        "### DB Parts (answer as JSON array only):"
    )

    return system_msg, user_msg


_RE_JSON_ARRAY = re.compile(r'\[\s*\{[\s\S]*\}\s*\]', re.DOTALL)


def _parse_alignments(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if not raw:
        return []
    fence = re.search(r'```(?:json)?\s*([\s\S]+?)```', raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    m = _RE_JSON_ARRAY.search(raw)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    print(f"    [PARSE WARN] Could not parse: {raw[:200]}")
    return []


def _validate_alignments(alignments: List[Dict], candidate_set: Set[str]) -> List[Dict]:
    return [{"phrase": a.get("phrase","").strip(),
             "columns": [c for c in a.get("columns",[]) if c in candidate_set]}
            for a in alignments if a.get("phrase","").strip()
            and [c for c in a.get("columns",[]) if c in candidate_set]]


def _save_alignment(output_path: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2)


def run_alignment(input_path: str, output_path: str,
                  model: str = "gpt-5.2", save_every: int = 5,
                  num_examples: int = 3) -> None:
    from llm import make_backend

    print(f"Loading: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    entries = data.get("results", data) if isinstance(data, dict) else data
    print(f"  {len(entries)} questions.")

    backend = make_backend("openai", model_id=model, cache=True)

    results: List[Dict] = []
    done_ids: Set[Tuple] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results  = existing.get("results", [])
        done_ids = {(r["question_id"], r["db_id"]) for r in results}
        print(f"  Resuming — {len(done_ids)} already done.")

    # Cache rendered schemas per db_id (avoid re-loading SQLite for every question)
    schema_cache: Dict[str, str] = {}

    for i, entry in enumerate(entries):
        qid   = entry.get("question_id")
        db_id = entry.get("db_id", "")
        if (qid, db_id) in done_ids:
            continue

        question        = entry.get("question", "")
        evidence        = entry.get("evidence", "")
        focused_columns = entry.get("focused_columns", [])
        if not question or not focused_columns:
            continue

        print(f"\n  [{i+1}/{len(entries)}] Q{qid} | {db_id}")
        print(f"  Q: {question[:80]}...")

        # Load full DB schema once per database
        if db_id not in schema_cache:
            schema_cache[db_id] = _render_db_schema(db_id)
            if schema_cache[db_id]:
                print(f"  Schema loaded for: {db_id}")
            else:
                print(f"  [WARN] Schema not found for: {db_id} — prompt will omit schema")

        db_schema = schema_cache[db_id]

        system_msg, user_msg = _build_alignment_prompt(
            question, evidence, focused_columns, db_schema,
            num_examples=num_examples,
        )
        messages = [{"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}]
        try:
            raw = backend.generate(messages, max_new_tokens=1024, temperature=0)
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            raw = ""

        candidate_set = set(focused_columns)
        alignments = _validate_alignments(_parse_alignments(raw), candidate_set)
        print(f"  Alignments: {len(alignments)}")
        for a in alignments:
            print(f"    '{a['phrase']}' → {a['columns']}")

        results.append({
            "question_id":      qid,
            "db_id":            db_id,
            "question":         question,
            "evidence":         evidence,
            "db_schema":        db_schema,
            "candidate_columns": focused_columns,
            "alignments":       alignments,
            "raw_llm_response": raw,
        })
        done_ids.add((qid, db_id))

        if len(results) % save_every == 0:
            _save_alignment(output_path, results)
            print(f"  [Saved] {len(results)} results.")

    _save_alignment(output_path, results)
    print(f"\nDone. {len(results)} results saved to {output_path}")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 4b: Phrase-to-column alignment")
    parser.add_argument("--input",  type=str,
                        default="results/schema_links_all_dbs_4per_20260218_125609.json",
                        help="Phase 3 schema links JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: auto-timestamped in results/)")
    parser.add_argument("--model",  type=str, default="gpt-5.2",
                        help="OpenAI model to use")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save progress every N questions (for resuming)")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of few-shot examples in the prompt (0-3, default: 3)")

    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/phrase_column_alignment_{ts}.json"

    run_alignment(args.input, args.output, model=args.model,
                  save_every=args.save_every, num_examples=args.num_examples)


if __name__ == "__main__":
    main()

"""
explain_errors.py
-----------------
Finds schema-linking errors (questions where recall < 100%) from Phase 3
results, then uses GPT to explain *why* each column was missed.

Usage:
  python explain_errors.py \
      --input   results/schema_links_all_dbs_4per_20260218_125609.json \
      --output  results/error_explanations_<timestamp>.json \
      --n       5          # how many wrong questions to explain (default: all)
      --model   gpt-5.2
      --save_every 3

Output JSON:
  {
    "results": [
      {
        "question_id": 5,
        "db_id": "california_schools",
        "question": "...",
        "recall": 0.6,
        "missed_columns": ["satscores.avgscrmath", "schools.virtual"],
        "found_columns":  ["satscores.cds", "schools.cdscode"],
        "gold_sql": "SELECT ...",
        "explanation": "The system missed ... because ..."
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_HERE        = Path(__file__).parent
MINIDEV_JSON = _HERE / "MINIDEV" / "mini_dev_sqlite.json"
DB_ROOT      = _HERE / "MINIDEV" / "dev_databases"


# ── Reuse evaluation logic from analysis.py ───────────────────────────────────

def _load_db_schema_map(db_id: str) -> Dict[str, Set[str]]:
    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    schema: Dict[str, Set[str]] = {}
    for tbl in tables:
        cur.execute(f'PRAGMA table_info("{tbl}")')
        schema[tbl.lower()] = {row[1].lower() for row in cur.fetchall()}
    conn.close()
    return schema


_schema_map_cache: Dict[str, Dict[str, Set[str]]] = {}

def _get_schema_map(db_id: str) -> Dict[str, Set[str]]:
    if db_id not in _schema_map_cache:
        _schema_map_cache[db_id] = _load_db_schema_map(db_id)
    return _schema_map_cache[db_id]


def _extract_gold_columns(gold_sql: str, db_id: str) -> Set[Tuple[str, str]]:
    import sqlglot
    import sqlglot.expressions as exp

    schema = _get_schema_map(db_id)
    result: Set[Tuple[str, str]] = set()
    try:
        ast = sqlglot.parse_one(gold_sql, dialect="sqlite")
    except Exception:
        try:
            ast = sqlglot.parse_one(gold_sql)
        except Exception:
            return result

    alias_map: Dict[str, str] = {}
    for node in ast.walk():
        if isinstance(node, exp.Table):
            tbl   = node.name.lower() if node.name else None
            alias = node.alias.lower() if node.alias else None
            if tbl and alias:
                alias_map[alias] = tbl
            elif tbl:
                alias_map[tbl] = tbl

    for col_node in ast.find_all(exp.Column):
        col_name = col_node.name.lower() if col_node.name else None
        if not col_name or col_name == "*":
            continue
        tbl_ref = col_node.table.lower() if col_node.table else None
        if tbl_ref:
            resolved = alias_map.get(tbl_ref, tbl_ref)
            result.add((resolved, col_name))
        else:
            owners = [t for t, cols in schema.items() if col_name in cols]
            if len(owners) == 1:
                result.add((owners[0], col_name))
            elif len(owners) > 1:
                from_tables = set(alias_map.values())
                candidates  = [t for t in owners if t in from_tables]
                for t in (candidates or owners):
                    result.add((t, col_name))
    return result


def _evaluate(entry: Dict, minidev: Dict) -> Dict:
    qid   = entry["question_id"]
    db_id = entry["db_id"]

    gold_info  = minidev.get(qid, {})
    gold_sql   = gold_info.get("SQL", entry.get("gold_sql", ""))
    difficulty = gold_info.get("difficulty", entry.get("difficulty", "unknown"))
    evidence   = gold_info.get("evidence", entry.get("evidence", ""))
    gold_cols  = _extract_gold_columns(gold_sql, db_id)

    found_cols: Set[Tuple[str, str]] = set()
    for sl in entry.get("schema_links", []):
        tbl, col = sl.get("table","").lower(), sl.get("column","").lower()
        if tbl and col:
            found_cols.add((tbl, col))

    if not gold_cols:
        return None  # can't evaluate

    intersection = gold_cols & found_cols
    missed       = gold_cols - found_cols
    recall       = len(intersection) / len(gold_cols)

    return {
        "question_id": qid,
        "db_id":       db_id,
        "difficulty":  difficulty,
        "question":    entry.get("question", ""),
        "evidence":    evidence,
        "gold_sql":    gold_sql,
        "recall":      recall,
        "gold_cols":   sorted(f"{t}.{c}" for t, c in gold_cols),
        "found_cols":  sorted(f"{t}.{c}" for t, c in found_cols),
        "missed_cols": sorted(f"{t}.{c}" for t, c in missed),
        "focused_columns": entry.get("focused_columns", []),
        "faiss_hits":  entry.get("faiss_hits", []),
        "lsh_hits":    entry.get("lsh_hits", []),
        "combo_sqls":  entry.get("combo_sqls", {}),
    }


# ── DB schema renderer (for prompt context) ──────────────────────────────────

def _render_schema(db_id: str) -> str:
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


# ── LLM prompt builder ────────────────────────────────────────────────────────

_SYSTEM_MSG = """\
You are a Text-to-SQL expert reviewing a schema linking system.

Schema linking is the task of identifying which database columns are needed to
answer a natural language question. The system uses two retrieval methods:
  1. FAISS semantic search — finds columns whose profile text is semantically
     similar to the question.
  2. LSH literal matching — finds columns that contain literal values
     mentioned in the question (e.g. 'CZK', 'F', '2013').

Your job: explain clearly and concisely WHY the system missed specific columns.

Focus on:
  - Was the column name/meaning too indirect for semantic search to find?
  - Was a literal value in the question not matched to the right column?
  - Did an alias or paraphrase in the question obscure the column's purpose?
  - Was the column a join key (FK/PK) that the question never directly mentions?
  - Was the column used only in a subquery or nested context?

Be specific. Reference the missed column names, the question phrasing, and the
gold SQL. Keep the explanation under 200 words.
"""


def _build_prompt(ev: Dict, db_schema: str) -> str:
    missed_str   = "\n".join(f"  - {c}" for c in ev["missed_cols"])
    found_str    = "\n".join(f"  - {c}" for c in ev["found_cols"])
    faiss_str    = "\n".join(
        f"  - {h.get('table','')}.{h.get('column','')}  (score: {h.get('faiss_score', h.get('score','?')):.3f})"
        for h in ev["faiss_hits"]
    ) or "  (none)"
    lsh_str      = "\n".join(
        f"  - {h.get('table','')}.{h.get('column','')}  literal='{h.get('literal', h.get('matched_values','?'))}'"
        f"  ({h.get('match_type','?')})"
        for h in ev["lsh_hits"]
    ) or "  (none)"
    combo_sqls   = ev.get("combo_sqls", {})
    sqls_str     = "\n".join(
        f"  [{combo}]  {sql[:120]}{'...' if len(sql)>120 else ''}"
        for combo, sql in combo_sqls.items()
        if sql
    ) or "  (none)"

    return f"""## Database Schema
{db_schema}

## Question
{ev['question']}
Evidence: {ev['evidence'] or '(none)'}

## Gold SQL (correct answer)
{ev['gold_sql']}

## Schema Linking Result
Recall: {ev['recall']*100:.0f}%

Columns the system FOUND (correct):
{found_str}

Columns the system MISSED (these should have been found):
{missed_str}

## What the system retrieved
FAISS semantic hits:
{faiss_str}

LSH literal hits:
{lsh_str}

SQL candidates the system generated (one per schema combo):
{sqls_str}

## Your Task
Explain in detail why the system missed the columns listed above.
Be specific about which retrieval method failed and why.
"""


# ── Main runner ───────────────────────────────────────────────────────────────

def _save(output_path: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"results": results}, f, indent=2)


def run_explain(
    input_path: str,
    output_path: str,
    n: Optional[int] = None,
    model: str = "gpt-5.2",
    save_every: int = 3,
) -> None:
    from llm import make_backend

    # Load Phase 3 results
    print(f"Loading Phase 3 results: {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    entries = data.get("results", data) if isinstance(data, dict) else data
    print(f"  {len(entries)} total questions.")

    # Load gold data
    print("Loading minidev gold data...")
    minidev = {e["question_id"]: e for e in json.loads(MINIDEV_JSON.read_text())}

    # Evaluate all entries, keep only wrong ones (recall < 1.0)
    wrong = []
    for entry in entries:
        ev = _evaluate(entry, minidev)
        if ev is not None and ev["recall"] < 1.0:
            wrong.append(ev)

    wrong.sort(key=lambda e: (e["db_id"], e["question_id"]))
    print(f"  {len(wrong)} questions with recall < 100%.")

    if n is not None:
        wrong = wrong[:n]
        print(f"  Limiting to {n} questions.")

    # Resume support
    results: List[Dict] = []
    done_ids: Set[Tuple] = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        results  = existing.get("results", [])
        done_ids = {(r["question_id"], r["db_id"]) for r in results}
        print(f"  Resuming — {len(done_ids)} already explained.")

    backend = make_backend("openai", model_id=model, cache=True)
    schema_cache: Dict[str, str] = {}

    for i, ev in enumerate(wrong):
        qid, db_id = ev["question_id"], ev["db_id"]
        if (qid, db_id) in done_ids:
            continue

        print(f"\n  [{i+1}/{len(wrong)}] Q{qid} | {db_id} | recall={ev['recall']*100:.0f}%")
        print(f"  Q: {ev['question'][:90]}...")
        print(f"  Missed: {ev['missed_cols']}")

        if db_id not in schema_cache:
            schema_cache[db_id] = _render_schema(db_id)

        user_msg = _build_prompt(ev, schema_cache[db_id])
        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user",   "content": user_msg},
        ]

        try:
            explanation = backend.generate(messages, max_new_tokens=512, temperature=0)
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            explanation = f"[ERROR] {e}"

        print(f"  Explanation preview: {explanation[:120]}...")

        results.append({
            "question_id":   qid,
            "db_id":         db_id,
            "difficulty":    ev["difficulty"],
            "question":      ev["question"],
            "evidence":      ev["evidence"],
            "gold_sql":      ev["gold_sql"],
            "recall":        round(ev["recall"], 4),
            "missed_cols":   ev["missed_cols"],
            "found_cols":    ev["found_cols"],
            "gold_cols":     ev["gold_cols"],
            "explanation":   explanation,
        })
        done_ids.add((qid, db_id))

        if len(results) % save_every == 0:
            _save(output_path, results)
            print(f"  [Saved {len(results)} results]")

    _save(output_path, results)
    print(f"\nDone. {len(results)} explanations saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ERRORS EXPLAINED")
    print("="*60)
    for r in results:
        print(f"  Q{r['question_id']:>4} [{r['db_id']:<28}] recall={r['recall']*100:.0f}%")
        print(f"         missed: {r['missed_cols']}")
        print(f"         reason: {r['explanation'][:120]}...")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explain schema linking errors using GPT"
    )
    parser.add_argument(
        "--input", type=str,
        default="results/schema_links_all_dbs_4per_20260218_125609.json",
        help="Phase 3 schema links JSON file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: auto-timestamped in results/)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of wrong questions to explain (default: all wrong ones)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5.2",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--save_every", type=int, default=3,
        help="Save progress every N explanations (for resuming)",
    )
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_tag = f"_n{args.n}" if args.n else ""
        args.output = f"results/error_explanations{n_tag}_{ts}.json"

    run_explain(
        input_path=args.input,
        output_path=args.output,
        n=args.n,
        model=args.model,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()

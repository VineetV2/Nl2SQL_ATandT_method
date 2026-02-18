"""
run_pipeline.py
---------------
Main runner for the Schema Linking pipeline.

Runs the full Phase 3 pipeline for:
  - All 30 questions in debit_card_specializing  (default)
  - Or 3-4 questions per database for all 11 databases  (--all)

For each question:
  1. Schema Linking: FAISS + LSH → focused schema
  2. Build 5 schema + profile combinations
  3. Call GPT-5.2 for each → 5 SQL queries
  4. Extract referenced columns from each SQL
  5. Union of columns = schema links

Output:
  results/schema_links_<db_id>_<timestamp>.json

Usage:
  # Run all 30 debit_card questions
  python run_pipeline.py --db debit_card_specializing

  # Run 3 questions per database for all 11 databases
  python run_pipeline.py --all --questions_per_db 3

  # Dry run (no LLM call, just show schemas)
  python run_pipeline.py --db debit_card_specializing --no_llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV" / "dev_databases"
MINIDEV_JSON = Path(__file__).parent / "MINIDEV" / "mini_dev_sqlite.json"
RESULTS_DIR  = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────
# Load Questions
# ─────────────────────────────────────────────────────────────

def load_questions(db_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """
    Load minidev questions from mini_dev_sqlite.json.
    If db_id is given, filter to that database only.
    If limit is given, take only the first N questions per database.
    """
    with open(MINIDEV_JSON) as f:
        all_qs = json.load(f)

    if db_id:
        qs = [q for q in all_qs if q["db_id"] == db_id]
    else:
        qs = all_qs

    if limit:
        # If filtering by db, just take first N
        # If all dbs, take first N per db
        if db_id:
            qs = qs[:limit]
        else:
            by_db: Dict[str, List] = {}
            for q in qs:
                by_db.setdefault(q["db_id"], []).append(q)
            qs = []
            for db, db_qs in sorted(by_db.items()):
                qs.extend(db_qs[:limit])

    return qs


# ─────────────────────────────────────────────────────────────
# Check Indexes Exist
# ─────────────────────────────────────────────────────────────

def check_indexes(db_id: str) -> bool:
    """Return True if FAISS + LSH indexes exist for this database."""
    db_dir = MINIDEV_ROOT / db_id
    faiss_ok = (db_dir / f"{db_id}.faiss").exists()
    lsh_ok   = (db_dir / f"{db_id}.lsh.pkl").exists()
    if not faiss_ok or not lsh_ok:
        print(f"  [WARN] Indexes missing for {db_id} — run build_indexes.py first")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Main Pipeline Runner
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    questions: List[Dict],
    backend=None,
    faiss_top_k: int = 10,
    output_path: Optional[Path] = None,
    few_shot_retriever=None,
) -> List[Dict]:
    """
    Run the schema linking pipeline for a list of questions.
    few_shot_retriever: FewShotRetriever instance (built once in main, shared across all questions).
    Returns list of result dicts.
    """
    from schema_linking import SchemaLinker

    # Cache one SchemaLinker per db_id
    linkers: Dict[str, SchemaLinker] = {}
    results: List[Dict] = []
    errors:  List[Dict] = []

    total = len(questions)
    print(f"\nProcessing {total} questions...")

    for i, q in enumerate(questions):
        db_id    = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "")
        qid      = q.get("question_id", i)

        print(f"\n[{i+1}/{total}] Q#{qid} | db={db_id}")

        # Check indexes
        if not check_indexes(db_id):
            errors.append({"question_id": qid, "db_id": db_id, "error": "missing indexes"})
            continue

        # Get or create linker
        if db_id not in linkers:
            try:
                linkers[db_id] = SchemaLinker(db_id)
                print(f"  Loaded SchemaLinker for: {db_id}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {db_id}: {e}")
                errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})
                continue

        linker = linkers[db_id]

        try:
            result = linker.run(
                question=question,
                evidence=evidence,
                backend=backend,
                faiss_top_k=faiss_top_k,
                question_id=qid,
                few_shot_retriever=few_shot_retriever,
            )
            # Attach gold SQL for reference/evaluation later
            result["gold_sql"] = q.get("SQL", "")
            result["difficulty"] = q.get("difficulty", "")
            results.append(result)

        except Exception as e:
            print(f"  [ERROR] Q#{qid}: {e}")
            errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})

        # Save incrementally every 5 questions
        if output_path and (i + 1) % 5 == 0:
            _save(results, errors, output_path)
            print(f"  [Saved] {i+1}/{total} done → {output_path.name}")

    # Final save
    if output_path:
        _save(results, errors, output_path)

    print(f"\n{'='*60}")
    print(f"Done: {len(results)} succeeded, {len(errors)} errors")
    return results


def _save(results: List[Dict], errors: List[Dict], path: Path):
    """Save results + errors to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Summary Printer
# ─────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    """Print a human-readable summary of schema linking results."""
    print(f"\n{'='*70}")
    print("SCHEMA LINKING SUMMARY")
    print(f"{'='*70}")

    for r in results:
        qid = r.get("question_id", "?")
        db  = r.get("db_id", "?")
        q   = r.get("question", "")[:60]
        links = r.get("schema_links", [])
        n_links = len(links)
        link_str = ", ".join(f"{l['table']}.{l['column']}" for l in links[:5])
        if len(links) > 5:
            link_str += f" ... (+{len(links)-5} more)"

        print(f"\nQ#{qid} [{db}]")
        print(f"  {q}...")
        print(f"  Literals: {r.get('literals', [])}")
        print(f"  Schema links ({n_links}): {link_str or '(none)'}")

    print(f"\nTotal questions: {len(results)}")
    avg_links = sum(len(r.get("schema_links", [])) for r in results) / max(len(results), 1)
    print(f"Avg schema links per question: {avg_links:.1f}")


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Schema Linking Pipeline")
    parser.add_argument("--db",  type=str, default="debit_card_specializing",
                        help="Single database to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all 11 databases (3-4 questions each)")
    parser.add_argument("--questions_per_db", type=int, default=4,
                        help="Questions per DB when using --all (default: 4)")
    parser.add_argument("--top_k",  type=int, default=10,
                        help="FAISS top-k columns to retrieve (default: 10)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM calls (dry run: just schema linking)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output JSON path (default: results/schema_links_<db>_<ts>.json)")
    args = parser.parse_args()

    # Load questions
    if args.all:
        questions = load_questions(db_id=None, limit=args.questions_per_db)
        tag = f"all_dbs_{args.questions_per_db}per"
    else:
        questions = load_questions(db_id=args.db)
        tag = args.db

    print(f"Loaded {len(questions)} questions")

    # Output path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (RESULTS_DIR / f"schema_links_{tag}_{ts}.json")
    print(f"Output → {out_path}")

    # Backend
    backend = None
    if not args.no_llm:
        from llm_backends_local import make_backend
        print("Initializing GPT-5.2 backend...")
        backend = make_backend("openai", model_id="gpt-5.2", cache=True)
    else:
        print("Dry run mode — no LLM calls")

    # Build few-shot retriever from the full minidev question pool (leave-one-out at query time)
    # Paper Section 4: use the 8 most similar questions as in-context examples per prompt.
    few_shot_retriever = None
    if backend is not None:
        from schema_linking import FewShotRetriever
        print("Building few-shot retriever from minidev questions...")
        all_minidev = load_questions()  # all 500 minidev questions as the pool
        few_shot_retriever = FewShotRetriever(all_minidev)
        few_shot_retriever.build()

    # Run pipeline
    t0 = time.time()
    results = run_pipeline(
        questions,
        backend=backend,
        faiss_top_k=args.top_k,
        output_path=out_path,
        few_shot_retriever=few_shot_retriever,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()

"""
candidates.py
--------------
Section 4: SQL Candidate Generation & Selection.

Takes schema linking results (from pipeline.py) and generates 3 SQL
candidates per question using the same pruned-schema prompt with diversity
from LLM seed + temperature + column order shuffling (paper Section 4).
Selects the best via majority voting or MCS-SQL multiple-choice selection.

Usage:
  python candidates.py --input results/schema_links_financial_*.json
  python candidates.py --input results/schema_links_financial_*.json --mcs
  python candidates.py --input results/schema_links_financial_*.json --backend openai --model gpt-5.2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Reuse from pipeline.py
from pipeline import (
    load_sqlite_schema,
    load_full_profiles,
    load_short_profiles,
    validate_and_fix_sql,
    FewShotRetriever,
    load_train_questions,
    MINIDEV_ROOT,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────
# Schema Rendering — Pruned Schema (paper Section 4)
# ─────────────────────────────────────────────────────────────

def render_pruned_schema(
    schema: Dict[str, List[Dict[str, Any]]],
    linked_keys: Set[str],
    col_order_seed: Optional[int] = None,
) -> str:
    """
    Pruned-schema strategy: only linked tables and linked columns.
    Bare rendering (name + type only, no inline profiles).
    Optionally shuffle column order for diversity.
    """
    # Determine which tables have linked columns
    linked_tables: Dict[str, List[Dict]] = {}
    for key in linked_keys:
        table, column = key.split(".", 1)
        if table not in linked_tables:
            linked_tables[table] = []

    # Collect linked columns per table (preserve schema order)
    for table in linked_tables:
        cols = []
        for col in schema.get(table, []):
            key = f"{table}.{col['column']}"
            if key in linked_keys:
                cols.append(col)
        if col_order_seed is not None:
            rng = random.Random(col_order_seed + abs(hash(table)) % 10000)
            rng.shuffle(cols)
        linked_tables[table] = cols

    blocks = []
    for table in schema:  # preserve original table order
        if table not in linked_tables or not linked_tables[table]:
            continue
        lines = [f"  {col['column']} {col['type']}" for col in linked_tables[table]]
        blocks.append(f"CREATE TABLE {table} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(blocks)


def render_column_profiles(
    linked_keys: Set[str],
    full_profiles: Dict[str, str],
) -> str:
    """
    Render detailed column profiles section for pruned-schema prompts.
    Each linked column gets its full [PROFILE] + [DEV DOC] text.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = full_profiles.get(key, "")
        if profile_text:
            sections.append(f"**{key}**\n{profile_text}")
        else:
            sections.append(f"**{key}**\n(no profile available)")
    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────
# Prompt Building
# ─────────────────────────────────────────────────────────────

def _render_few_shots(examples: List[Dict[str, Any]]) -> str:
    """
    Render few-shot examples block for the prompt (paper Section 4).

    Each example shows: Question → SQL (with optional hint).
    """
    if not examples:
        return ""
    blocks = []
    for i, ex in enumerate(examples, 1):
        q = ex.get("question", "")
        sql = ex.get("SQL", "")
        ev = ex.get("evidence", "")
        block = f"Example {i}:\nQuestion: {q}"
        if ev:
            block += f"\nHint: {ev}"
        block += f"\nSQL: {sql}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_pruned_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    profiles_text: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for pruned-schema candidate (with column profiles and few-shot examples)."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles "
        "below, write a SQL query to answer the question.\n\n"
        f"### Database Schema\n{schema_text}\n"
    ]
    if profiles_text:
        parts.append(f"\n### Column Profiles\n{profiles_text}")
    if few_shots:
        parts.append(f"\n### Examples\n{_render_few_shots(few_shots)}")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# SQL Generation
# ─────────────────────────────────────────────────────────────

def generate_sql(prompt: str, backend, temperature: float = 0,
                  seed: Optional[int] = None) -> str:
    """Call LLM, strip markdown fences, apply SQLglot fixes."""
    try:
        messages = [{"role": "user", "content": prompt}]
        gen_kwargs = {"temperature": temperature}
        if seed is not None:
            gen_kwargs["seed"] = seed
        response = backend.generate(messages, max_new_tokens=512, **gen_kwargs)
    except Exception as e:
        print(f"    [LLM ERROR] {e}")
        return ""

    sql = response.strip()
    fence_match = re.search(r'```(?:sql)?\s*([\s\S]+?)```', sql, re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    if sql:
        sql = validate_and_fix_sql(sql)
    return sql


# ─────────────────────────────────────────────────────────────
# SQL Execution
# ─────────────────────────────────────────────────────────────

def execute_sql(sql: str, db_path: str, timeout: float = 30.0) -> Optional[FrozenSet[Tuple]]:
    """
    Execute SQL against SQLite and return result as frozenset of tuples.
    Returns None on error or timeout.
    """
    if not sql or not sql.strip():
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(sql)
        rows = frozenset(cursor.fetchall())
        conn.close()
        return rows
    except Exception as e:
        print(f"    [EXEC ERROR] {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Selection Method A: Majority Voting (Paper Section 4)
# ─────────────────────────────────────────────────────────────

def majority_vote(
    candidate_sqls: List[str],
    candidate_names: List[str],
    db_path: str,
) -> Dict[str, Any]:
    """
    Execute each candidate SQL, compare result sets.
    If 2+ agree → pick that SQL. Otherwise fallback to candidate 0.
    """
    results = []
    for sql in candidate_sqls:
        results.append(execute_sql(sql, db_path))

    # Find agreement
    for i in range(len(results)):
        if results[i] is None:
            continue
        agreeing = [j for j in range(len(results)) if results[j] == results[i]]
        if len(agreeing) >= 2:
            winner_idx = agreeing[0]
            return {
                "winner_idx": winner_idx,
                "winner": candidate_names[winner_idx],
                "method": "majority",
                "agreement": len(agreeing),
                "sql": candidate_sqls[winner_idx],
            }

    # No agreement — fallback: pick randomly among valid candidates (paper Section 4)
    valid_indices = [i for i, sql in enumerate(candidate_sqls)
                     if sql and results[i] is not None]
    if valid_indices:
        i = random.choice(valid_indices)
        return {
            "winner_idx": i,
            "winner": candidate_names[i],
            "method": "fallback_random",
            "agreement": 1,
            "sql": candidate_sqls[i],
        }

    # All failed — pick randomly from all candidates
    i = random.randrange(len(candidate_sqls)) if candidate_sqls else 0
    return {
        "winner_idx": i,
        "winner": candidate_names[i] if candidate_names else "",
        "method": "fallback_random",
        "agreement": 0,
        "sql": candidate_sqls[i] if candidate_sqls else "",
    }


# ─────────────────────────────────────────────────────────────
# Selection Method B: MCS-SQL Multiple-Choice Selection [LPKP24]
# ─────────────────────────────────────────────────────────────

def mcs_select(
    candidate_sqls: List[str],
    candidate_names: List[str],
    db_path: str,
    question: str,
    evidence: str,
    backend,
) -> Dict[str, Any]:
    """
    MCS-SQL style selection:
    1. Execute all candidates, remove errors
    2. Group by result sets, compute confidence
    3. Present as multiple-choice to LLM for final selection
    """
    # Execute and group
    results = []
    valid_indices = []
    for i, sql in enumerate(candidate_sqls):
        result = execute_sql(sql, db_path)
        results.append(result)
        if result is not None:
            valid_indices.append(i)

    if not valid_indices:
        return {
            "winner_idx": 0,
            "winner": candidate_names[0],
            "method": "mcs_fallback",
            "agreement": 0,
            "confidence": 0.0,
            "sql": candidate_sqls[0] if candidate_sqls else "",
        }

    # If only one valid candidate, return it
    if len(valid_indices) == 1:
        idx = valid_indices[0]
        return {
            "winner_idx": idx,
            "winner": candidate_names[idx],
            "method": "mcs_single",
            "agreement": 1,
            "confidence": 1.0,
            "sql": candidate_sqls[idx],
        }

    # Group by result sets and compute confidence
    result_groups: Dict[int, List[int]] = {}  # group_id → list of candidate indices
    group_results: Dict[int, FrozenSet] = {}
    group_id = 0
    for i in valid_indices:
        matched = False
        for gid, gresult in group_results.items():
            if results[i] == gresult:
                result_groups[gid].append(i)
                matched = True
                break
        if not matched:
            result_groups[group_id] = [i]
            group_results[group_id] = results[i]
            group_id += 1

    total_valid = len(valid_indices)

    # Build multiple-choice prompt
    options = []
    option_map = {}  # letter → candidate index
    letters = "ABCDEFGHIJ"

    # Sort groups by confidence (descending) to offset position bias
    sorted_groups = sorted(
        result_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    letter_idx = 0
    for gid, indices in sorted_groups:
        representative_idx = indices[0]
        confidence = len(indices) / total_valid
        letter = letters[letter_idx]
        option_map[letter] = representative_idx
        options.append(
            f"{letter}) [Confidence: {confidence:.0%}]\n{candidate_sqls[representative_idx]}"
        )
        letter_idx += 1

    options_text = "\n\n".join(options)

    mcs_prompt = (
        "You are a SQLite expert. Given a question and multiple SQL candidate queries, "
        "select the most correct one.\n\n"
        f"### Question\n{question}\n"
    )
    if evidence:
        mcs_prompt += f"\n### Hint\n{evidence}\n"
    mcs_prompt += (
        f"\n### SQL Candidates\n{options_text}\n\n"
        "### Selection\n"
        "Which candidate is most likely correct? Reply with ONLY the letter "
        "(e.g., A) and a brief reason.\n\n"
        "Answer:"
    )

    # Ask LLM to choose
    try:
        messages = [{"role": "user", "content": mcs_prompt}]
        response = backend.generate(messages, max_new_tokens=200, temperature=0)
        # Parse letter from response
        chosen_letter = None
        for letter in option_map:
            if letter in response[:10]:  # look in first few chars
                chosen_letter = letter
                break
        if chosen_letter and chosen_letter in option_map:
            winner_idx = option_map[chosen_letter]
        else:
            # Fallback to highest confidence group
            winner_idx = sorted_groups[0][1][0]
    except Exception as e:
        print(f"    [MCS ERROR] {e}")
        winner_idx = sorted_groups[0][1][0]

    # Find confidence of winner
    winner_confidence = 0.0
    for gid, indices in result_groups.items():
        if winner_idx in indices:
            winner_confidence = len(indices) / total_valid
            break

    return {
        "winner_idx": winner_idx,
        "winner": candidate_names[winner_idx],
        "method": "mcs",
        "agreement": sum(1 for gid, indices in result_groups.items() if winner_idx in indices),
        "confidence": round(winner_confidence, 3),
        "sql": candidate_sqls[winner_idx],
    }


# ─────────────────────────────────────────────────────────────
# Database Loader (caches schema + profiles per db_id)
# ─────────────────────────────────────────────────────────────

class DBLoader:
    """Load and cache schema + profiles for a database."""

    def __init__(self, db_id: str):
        self.db_id = db_id
        self.db_dir = MINIDEV_ROOT / db_id
        self.db_path = str(self.db_dir / f"{db_id}.sqlite")

        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"SQLite not found: {self.db_path}")

        self.schema = load_sqlite_schema(Path(self.db_path))
        self.full_profiles = load_full_profiles(self.db_dir, db_id)


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

# Paper Section 4: 3 candidates from same pruned-schema prompt.
# Diversity via: (temperature, seed, col_order_seed)
_CANDIDATES = [
    {"name": "candidate_0", "temperature": 0,   "seed": None, "col_order_seed": None},
    {"name": "candidate_1", "temperature": 0.7, "seed": 1,    "col_order_seed": 1},
    {"name": "candidate_2", "temperature": 0.7, "seed": 2,    "col_order_seed": 2},
]


def process_question(
    q: Dict[str, Any],
    db: DBLoader,
    backend,
    use_mcs: bool = False,
    few_shot_retriever: Optional[FewShotRetriever] = None,
) -> Dict[str, Any]:
    """
    Generate 3 SQL candidates for a question and select the best one.

    Paper Section 4: same pruned-schema prompt for all candidates.
    Diversity from LLM seed + temperature + column order shuffling.
    Few-shot: 8 most similar masked questions from BIRD training set.
    """
    question = q["question"]
    evidence = q.get("evidence", "")
    schema_links = q.get("schema_links", [])

    # Build set of linked column keys
    linked_keys: Set[str] = set()
    for sl in schema_links:
        linked_keys.add(f"{sl['table']}.{sl['column']}")

    print(f"  Linked columns: {len(linked_keys)}")

    # Retrieve 8 few-shot examples (paper Section 4)
    few_shots: Optional[List[Dict[str, Any]]] = None
    if few_shot_retriever is not None:
        few_shots = few_shot_retriever.retrieve(question, k=8, exclude_question=question)
        print(f"  Few-shot examples: {len(few_shots)}")

    # Generate candidates — same prompt structure, different seeds/order
    candidate_sqls: List[str] = []
    candidate_names: List[str] = []
    candidates_detail: List[Dict[str, Any]] = []

    for i, cfg in enumerate(_CANDIDATES):
        name = cfg["name"]
        temp = cfg["temperature"]
        seed = cfg["seed"]
        col_seed = cfg["col_order_seed"]

        print(f"    [{i+1}/{len(_CANDIDATES)}] {name} (temp={temp}, seed={seed}, shuffle={col_seed})...")

        schema_text = render_pruned_schema(db.schema, linked_keys, col_order_seed=col_seed)
        profiles_text = render_column_profiles(linked_keys, db.full_profiles)
        prompt = build_pruned_prompt(question, evidence, schema_text, profiles_text,
                                     few_shots=few_shots)
        sql = generate_sql(prompt, backend, temperature=temp, seed=seed)
        print(f"    → {sql[:80]}..." if len(sql) > 80 else f"    → {sql}")

        candidate_sqls.append(sql)
        candidate_names.append(name)
        candidates_detail.append({
            "sql": sql,
            "temperature": temp,
            "seed": seed,
            "col_order_seed": col_seed,
        })

    # ── Selection ──
    if use_mcs:
        print("    Selecting via MCS-SQL multiple-choice...")
        vote = mcs_select(
            candidate_sqls, candidate_names, db.db_path,
            question, evidence, backend,
        )
    else:
        print("    Selecting via majority vote...")
        vote = majority_vote(candidate_sqls, candidate_names, db.db_path)

    print(f"    Winner: {vote['winner']} ({vote['method']}, agreement={vote['agreement']})")

    return {
        "question_id": q.get("question_id"),
        "db_id": q.get("db_id", db.db_id),
        "question": question,
        "evidence": evidence,
        "gold_sql": q.get("gold_sql", ""),
        "difficulty": q.get("difficulty", ""),
        "schema_links": schema_links,
        "candidates": candidates_detail,
        "voted_sql": vote["sql"],
        "vote_details": {
            "method": vote["method"],
            "winner": vote["winner"],
            "agreement": vote["agreement"],
        },
    }


def run_candidates(
    input_path: Path,
    backend,
    use_mcs: bool = False,
    output_path: Optional[Path] = None,
    few_shot_retriever: Optional[FewShotRetriever] = None,
) -> List[Dict[str, Any]]:
    """Run candidate generation + selection on all questions from schema links file."""

    with open(input_path) as f:
        data = json.load(f)

    questions = data.get("results", [])
    print(f"Loaded {len(questions)} questions from {input_path.name}")

    # Cache one DBLoader per db_id
    dbs: Dict[str, DBLoader] = {}
    results: List[Dict] = []
    errors: List[Dict] = []

    for i, q in enumerate(questions):
        db_id = q.get("db_id", "")
        qid = q.get("question_id", i)
        print(f"\n[{i+1}/{len(questions)}] Q#{qid} | db={db_id}")

        # Load DB
        if db_id not in dbs:
            try:
                dbs[db_id] = DBLoader(db_id)
                print(f"  Loaded DB: {db_id}")
            except Exception as e:
                print(f"  [ERROR] Failed to load {db_id}: {e}")
                errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})
                continue

        try:
            result = process_question(q, dbs[db_id], backend, use_mcs=use_mcs,
                                        few_shot_retriever=few_shot_retriever)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Q#{qid}: {e}")
            errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})

        # Incremental save every 5 questions
        if output_path and (i + 1) % 5 == 0:
            _save(results, errors, input_path, output_path, use_mcs)
            print(f"  [Saved] {i+1}/{len(questions)} → {output_path.name}")

    # Final save
    if output_path:
        _save(results, errors, input_path, output_path, use_mcs)

    print(f"\n{'='*60}")
    print(f"Done: {len(results)} succeeded, {len(errors)} errors")
    return results


def _save(
    results: List[Dict],
    errors: List[Dict],
    input_path: Path,
    output_path: Path,
    use_mcs: bool,
):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "input_file": str(input_path),
            "selection_method": "mcs" if use_mcs else "majority_vote",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
        "results": results,
        "errors": errors,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────

def print_summary(results: List[Dict]):
    """Print summary of candidate generation results."""
    print(f"\n{'='*70}")
    print("CANDIDATE GENERATION SUMMARY")
    print(f"{'='*70}")

    method_counts: Dict[str, int] = {}
    winner_counts: Dict[str, int] = {}

    for r in results:
        vote = r.get("vote_details", {})
        method = vote.get("method", "unknown")
        winner = vote.get("winner", "unknown")
        method_counts[method] = method_counts.get(method, 0) + 1
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    print(f"\nTotal questions: {len(results)}")
    print(f"\nSelection methods:")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")
    print(f"\nWinner distribution:")
    for winner, count in sorted(winner_counts.items()):
        print(f"  {winner}: {count}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Section 4: SQL Candidate Generation & Selection"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to schema_links JSON file (from pipeline.py)",
    )
    parser.add_argument(
        "--backend", type=str, default="openai",
        help="LLM backend: openai, qwen, gptoss (default: openai)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID (default: backend default, e.g. gpt-5.2 for openai)",
    )
    parser.add_argument(
        "--mcs", action="store_true",
        help="Use MCS-SQL multiple-choice selection instead of majority voting",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON path (default: results/candidates_<db>_<ts>.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    # Derive output path
    if args.out:
        output_path = Path(args.out)
    else:
        # Extract db tag from input filename
        stem = input_path.stem  # e.g., schema_links_financial_20260224_033607
        tag = stem.replace("schema_links_", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"candidates_{tag}_{ts}.json"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Selection: {'MCS-SQL' if args.mcs else 'Majority Vote'}")

    # Initialize backend
    from llm import make_backend
    backend_kwargs = {"kind": args.backend}
    if args.model:
        backend_kwargs["model_id"] = args.model
    backend = make_backend(**backend_kwargs, cache=True)
    print(f"Backend: {args.backend} ({args.model or 'default'})")

    # Build few-shot retriever from BIRD training set (paper Section 4)
    print("Building few-shot retriever from BIRD training set...")
    train_pool = load_train_questions()
    print(f"  Loaded {len(train_pool)} training questions for few-shot pool")
    few_shot_retriever = FewShotRetriever(train_pool)
    few_shot_retriever.build()

    # Run
    t0 = time.time()
    results = run_candidates(
        input_path=input_path,
        backend=backend,
        use_mcs=args.mcs,
        output_path=output_path,
        few_shot_retriever=few_shot_retriever,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved → {output_path}")


if __name__ == "__main__":
    main()

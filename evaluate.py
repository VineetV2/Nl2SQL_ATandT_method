"""
evaluate.py
-----------
Evaluate schema linking recall/precision/F1 against gold SQL answers.

Reads all schema_links_*.json files from the results/ directory,
compares them against the BIRD minidev gold SQL, and prints a report.

Usage:
  python evaluate.py
  python evaluate.py --input results/schema_links_all_dbs_4per_20260218_125609.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_HERE        = Path(__file__).parent
RESULTS_DIR  = _HERE / "results"
MINIDEV_JSON = _HERE / "MINIDEV" / "mini_dev_sqlite.json"
DB_ROOT      = _HERE / "MINIDEV" / "dev_databases"


# ── SQLite schema loader ──────────────────────────────────────────────────────

def _load_db_schema(db_id: str) -> Dict[str, Set[str]]:
    """Return {table_lower: {col_lower, ...}} for every table in the SQLite DB."""
    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]
    schema = {}
    for tbl in tables:
        cur.execute(f'PRAGMA table_info("{tbl}")')
        schema[tbl.lower()] = {row[1].lower() for row in cur.fetchall()}
    conn.close()
    return schema


_schema_cache: Dict[str, Dict[str, Set[str]]] = {}

def _get_schema(db_id: str) -> Dict[str, Set[str]]:
    if db_id not in _schema_cache:
        _schema_cache[db_id] = _load_db_schema(db_id)
    return _schema_cache[db_id]


# ── Gold column extractor ─────────────────────────────────────────────────────

def extract_gold_columns(gold_sql: str, db_id: str) -> Set[Tuple[str, str]]:
    """Parse gold SQL → set of (table_lower, column_lower) pairs."""
    import sqlglot
    import sqlglot.expressions as exp

    schema = _get_schema(db_id)
    result: Set[Tuple[str, str]] = set()
    try:
        ast = sqlglot.parse_one(gold_sql, dialect="sqlite")
    except Exception:
        try:
            ast = sqlglot.parse_one(gold_sql)
        except Exception:
            return result

    # Build alias map: T1 → satscores, T2 → schools, etc.
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
            # No table prefix — look up which table owns this column
            owners = [t for t, cols in schema.items() if col_name in cols]
            if len(owners) == 1:
                result.add((owners[0], col_name))
            elif len(owners) > 1:
                from_tables = set(alias_map.values())
                candidates  = [t for t in owners if t in from_tables]
                for t in (candidates or owners):
                    result.add((t, col_name))
    return result


# ── Per-question evaluator ────────────────────────────────────────────────────

def evaluate_entry(entry: Dict, minidev: Dict) -> Dict:
    qid          = entry["question_id"]
    db_id        = entry["db_id"]
    schema_links = entry.get("schema_links", [])

    gold_info  = minidev.get(qid, {})
    gold_sql   = gold_info.get("SQL", entry.get("gold_sql", ""))
    difficulty = gold_info.get("difficulty", entry.get("difficulty", "unknown"))
    gold_cols  = extract_gold_columns(gold_sql, db_id)

    found_cols: Set[Tuple[str, str]] = set()
    for sl in schema_links:
        tbl, col = sl.get("table", "").lower(), sl.get("column", "").lower()
        if tbl and col:
            found_cols.add((tbl, col))

    if not gold_cols:
        return {
            "question_id": qid, "db_id": db_id, "difficulty": difficulty,
            "question": entry.get("question", ""), "gold_sql": gold_sql,
            "gold_cols": gold_cols, "found_cols": found_cols,
            "intersection": set(), "missed": set(), "extra": found_cols,
            "recall": None, "precision": 0.0, "f1": 0.0, "perfect": False,
        }

    intersection = gold_cols & found_cols
    missed       = gold_cols - found_cols
    extra        = found_cols - gold_cols
    recall       = len(intersection) / len(gold_cols)
    precision    = len(intersection) / len(found_cols) if found_cols else 0.0
    f1           = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "question_id": qid, "db_id": db_id, "difficulty": difficulty,
        "question": entry.get("question", ""), "gold_sql": gold_sql,
        "gold_cols": gold_cols, "found_cols": found_cols,
        "intersection": intersection, "missed": missed, "extra": extra,
        "recall": recall, "precision": precision, "f1": f1,
        "perfect": recall == 1.0,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def _fmt_col(tc: Tuple[str, str]) -> str:
    return f"{tc[0]}.{tc[1]}"


def print_report(evaluations: List[Dict]) -> None:
    by_db: Dict[str, List[Dict]] = defaultdict(list)
    for ev in evaluations:
        by_db[ev["db_id"]].append(ev)

    db_stats: Dict[str, Dict] = {}
    for db_id, evs in sorted(by_db.items()):
        valid   = [e for e in evs if e["recall"] is not None]
        recalls = [e["recall"] for e in valid]
        precs   = [e["precision"] for e in valid]
        f1s     = [e["f1"] for e in valid]
        db_stats[db_id] = {
            "n": len(evs), "valid": len(valid),
            "mean_recall":    sum(recalls) / len(recalls) if recalls else 0.0,
            "mean_precision": sum(precs)   / len(precs)   if precs   else 0.0,
            "mean_f1":        sum(f1s)     / len(f1s)     if f1s     else 0.0,
            "perfect": sum(1 for e in valid if e["perfect"]),
            "partial": sum(1 for e in valid if e["recall"] is not None and 0 < e["recall"] < 1.0),
            "zero":    sum(1 for e in valid if e["recall"] == 0.0),
            "worst":   min(valid, key=lambda e: (e["recall"], e["question_id"])) if valid else None,
        }

    all_valid  = [e for e in evaluations if e["recall"] is not None]
    total_n    = len(evaluations)
    total_v    = len(all_valid)
    ovr_recall = sum(e["recall"]    for e in all_valid) / total_v if total_v else 0
    ovr_prec   = sum(e["precision"] for e in all_valid) / total_v if total_v else 0
    ovr_f1     = sum(e["f1"]        for e in all_valid) / total_v if total_v else 0

    by_diff: Dict[str, List[float]] = defaultdict(list)
    for e in all_valid:
        by_diff[e["difficulty"]].append(e["recall"])
    diff_stats = {d: (sum(v) / len(v), len(v)) for d, v in sorted(by_diff.items())}

    SEP, sep = "=" * 72, "-" * 72
    print(f"\n{SEP}\n  SCHEMA LINKING ACCURACY REPORT\n{SEP}")
    print(f"\nOVERALL SUMMARY\n{sep}")
    print(f"  Total questions  : {total_n}")
    print(f"  Mean Recall      : {ovr_recall*100:.1f}%")
    print(f"  Mean Precision   : {ovr_prec*100:.1f}%")
    print(f"  Mean F1          : {ovr_f1*100:.1f}%")
    if total_v:
        perfect = sum(1 for e in all_valid if e["perfect"])
        print(f"  Perfect recall   : {perfect}/{total_v}  ({perfect/total_v*100:.1f}%)")
    best_db  = max(db_stats, key=lambda d: db_stats[d]["mean_recall"])
    worst_db = min(db_stats, key=lambda d: db_stats[d]["mean_recall"])
    print(f"  Best DB          : {best_db}  ({db_stats[best_db]['mean_recall']*100:.1f}%)")
    print(f"  Worst DB         : {worst_db}  ({db_stats[worst_db]['mean_recall']*100:.1f}%)")

    print(f"\nPER-DIFFICULTY\n{sep}")
    for diff, (mean_r, cnt) in diff_stats.items():
        print(f"  {diff:<16}  N={cnt:>3}  Recall={mean_r*100:.1f}%")

    print(f"\nPER-DATABASE  (sorted by recall desc)\n{sep}")
    print(f"  {'Database':<28}  {'N':>3}  {'Recall':>7}  {'Prec':>7}  {'F1':>7}  {'Perfect':>8}")
    for db_id, s in sorted(db_stats.items(), key=lambda x: -x[1]["mean_recall"]):
        print(f"  {db_id:<28}  {s['n']:>3}  "
              f"{s['mean_recall']*100:>6.1f}%  {s['mean_precision']*100:>6.1f}%  "
              f"{s['mean_f1']*100:>6.1f}%  {s['perfect']:>4}/{s['n']:<3}")

    print(f"\nDETAILED PER-QUESTION\n{sep}")
    for db_id, evs in sorted(by_db.items()):
        print(f"\n  === {db_id} ===")
        for e in sorted(evs, key=lambda x: x["question_id"]):
            rec_str  = f"{e['recall']*100:.0f}%" if e["recall"] is not None else "N/A"
            missed_s = ", ".join(_fmt_col(c) for c in sorted(e["missed"])) if e["missed"] else "-"
            print(f"  Q#{e['question_id']:>4}  {e['difficulty']:<12}  "
                  f"recall={rec_str:>4}  prec={e['precision']*100:.0f}%  "
                  f"gold={len(e['gold_cols'])}  found={len(e['found_cols'])}  missed={missed_s}")

    print(f"\n{SEP}\n  END OF REPORT\n{SEP}\n")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_evaluate(input_path: Optional[str] = None) -> None:
    print("Loading minidev gold data...")
    minidev = {e["question_id"]: e for e in json.loads(MINIDEV_JSON.read_text())}
    print(f"  {len(minidev)} questions loaded.")

    if input_path:
        files = [Path(input_path)]
        print(f"Evaluating: {input_path}")
    else:
        files = sorted(RESULTS_DIR.glob("schema_links_*.json"))
        print(f"Found {len(files)} result file(s).")

    combined: Dict[Tuple[str, int], Dict] = {}
    for fpath in files:
        try:
            raw     = json.loads(fpath.read_text())
            entries = raw.get("results", raw) if isinstance(raw, dict) else raw
            for entry in (entries if isinstance(entries, list) else []):
                key = (entry.get("db_id", ""), entry.get("question_id", -1))
                combined[key] = entry
        except Exception as e:
            print(f"  [WARN] {fpath.name}: {e}")
    print(f"  {len(combined)} unique (db_id, question_id) entries.")

    evaluations = [evaluate_entry(entry, minidev) for entry in combined.values()]
    print_report(evaluations)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate schema linking recall/precision/F1 vs gold SQL"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to a specific schema_links JSON file (default: all schema_links_*.json in results/)",
    )
    args = parser.parse_args()
    run_evaluate(input_path=args.input)


if __name__ == "__main__":
    main()

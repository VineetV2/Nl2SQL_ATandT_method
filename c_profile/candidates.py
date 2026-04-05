"""
candidates.py
--------------
Section 4: SQL Candidate Generation & Selection.

Takes schema linking results (from pipeline.py) and generates 3 SQL
candidates per question using structurally different prompts:
  - Candidate A (wide_net):            Full schema + short profiles + few-shot
  - Candidate B (deep_focus):          Pruned schema + full profiles + few-shot
  - Candidate C (independent_thinker): Pruned schema + long profiles + NO few-shot

Selection: majority vote with LLM tiebreaker when no agreement.

Usage:
  python candidates.py --input results/schema_links_*.json --backend openrouter
  python candidates.py --input results/schema_links_*.json --backend openrouter --feedback
  python candidates.py --input results/schema_links_*.json --backend openrouter --feedback --max_retry 3
"""

from __future__ import annotations

import argparse
import json
import os
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
    load_long_profiles,
    load_short_profiles,
    validate_and_fix_sql,
    FewShotRetriever,
    load_train_questions,
    MINIDEV_ROOT,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ─────────────────────────────────────────────────────────────
# Schema Rendering
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


def render_full_schema_highlighted(
    schema: Dict[str, List[Dict[str, Any]]],
    linked_keys: Set[str],
    short_profiles: Dict[str, str],
) -> str:
    """
    Full-schema rendering: all tables, all columns.
    Linked columns get short profile inline comment + /* LINKED */ marker.
    Primary keys get -- primary key label.
    Non-linked columns: bare name + type.
    """
    blocks = []
    for table in schema:
        rendered = []
        for col in schema.get(table, []):
            key = f"{table}.{col['column']}"
            line = f"  {col['column']} {col['type']}"
            if col.get("pk"):
                line += "  -- primary key"
            if key in linked_keys:
                profile = short_profiles.get(key, "")
                if profile:
                    line += f"  -- {profile}"
                line += "  /* LINKED */"
            rendered.append(line)
        if rendered:
            blocks.append(f"CREATE TABLE {table} (\n" + ",\n".join(rendered) + "\n);")
    return "\n\n".join(blocks)


def render_column_profiles(
    linked_keys: Set[str],
    profiles: Dict[str, str],
) -> str:
    """
    Render column profile descriptions for the prompt (paper Section 3.1).
    Uses short one-line descriptions in 'Field X means: ...' format.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = profiles.get(key, "")
        if profile_text:
            sections.append(f"Field {key} means: {profile_text}")
    return "\n".join(sections)


def render_full_profiles_block(
    linked_keys: Set[str],
    full_profiles: Dict[str, str],
    long_profiles: Dict[str, str],
) -> str:
    """
    Render detailed profile block for Candidate B (pruned + full profiles).
    Uses full_profiles (stats + dev doc), falls back to long_profiles.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = full_profiles.get(key, "")
        if not profile_text:
            profile_text = long_profiles.get(key, "")
        if profile_text:
            sections.append(f"**{key}**\n{profile_text}")
    return "\n\n".join(sections)


def render_long_profiles_block(
    linked_keys: Set[str],
    long_profiles: Dict[str, str],
) -> str:
    """
    Render long profile descriptions for Candidate C (pruned + long profiles).
    Uses 'Field X means:' format with LLM-enhanced statistical descriptions.
    """
    sections = []
    for key in sorted(linked_keys):
        profile_text = long_profiles.get(key, "")
        if profile_text:
            sections.append(f"Field {key} means: {profile_text}")
    return "\n".join(sections)


# ─────────────────────────────────────────────────────────────
# Prompt Building
# ─────────────────────────────────────────────────────────────

def _render_few_shots(examples: List[Dict[str, Any]]) -> str:
    """
    Render few-shot examples block for the prompt (paper Section 4).

    Each example shows: Question -> SQL (with optional hint).
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
    parts = [f"### Database Schema\n{schema_text}\n"]
    if profiles_text:
        parts.append(f"\n### Column Descriptions\n{profiles_text}")
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


def build_full_schema_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for Candidate A: full schema + short profiles inline + few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    if few_shots:
        parts.append(f"\n### Examples\n{_render_few_shots(few_shots)}")
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


def build_deep_focus_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    profiles_block: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for Candidate B: pruned schema + full profiles block + few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    if few_shots:
        parts.append(f"\n### Examples\n{_render_few_shots(few_shots)}")
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if profiles_block:
        parts.append(f"\n### Column Profiles\n{profiles_block}\n")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


def build_independent_prompt(
    question: str,
    evidence: str,
    schema_text: str,
    profiles_text: str,
) -> str:
    """Build prompt for Candidate C: pruned schema + long profiles + NO few-shot."""
    parts = [
        "You are a SQLite expert. Given the database schema and column profiles below, "
        "write a SQL query to answer the question.\n"
    ]
    parts.append(f"\n### Database Schema\n{schema_text}\n")
    if profiles_text:
        parts.append(f"\n### Column Descriptions\n{profiles_text}")
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

SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant for SQLite.\n"
    "Return ONLY a single valid SQLite SQL query.\n"
    "No explanations."
)


def generate_sql(prompt: str, backend, temperature: float = 0,
                  seed: Optional[int] = None) -> str:
    """Call LLM, strip markdown fences, extract SQL, apply SQLglot fixes."""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        gen_kwargs = {"temperature": temperature}
        if seed is not None:
            gen_kwargs["seed"] = seed
        response = backend.generate(messages, max_new_tokens=1900, **gen_kwargs)
    except Exception as e:
        print(f"    [LLM ERROR] {e}")
        return ""

    sql = response.strip()

    # Strip markdown fences
    fence_match = re.search(r'```(?:sql)?\s*([\s\S]+?)```', sql, re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    else:
        # Reasoning model may output chain-of-thought before SQL.
        # Find the last SELECT/WITH that starts at the beginning of a line.
        last_match = None
        for m in re.finditer(r'(?m)^(SELECT|WITH)\b', sql, re.IGNORECASE):
            last_match = m
        if last_match:
            sql = sql[last_match.start():]
        else:
            # Fallback: find SELECT anywhere first (unambiguous),
            # then WITH only if followed by SQL syntax (not prose like "with Currency column")
            m = re.search(r'(?i)\bSELECT\b', sql)
            if not m:
                m = re.search(r'(?i)\bWITH\b\s+\w+\s+(AS|RECURSIVE)\b', sql)
            if m:
                sql = sql[m.start():]
            else:
                return ""  # pure reasoning text, no SQL found

    # Truncate at first blank line or semicolon
    blank = re.search(r'\n\s*\n', sql)
    if blank:
        sql = sql[:blank.start()].strip()
    semi = sql.find(';')
    if semi != -1:
        sql = sql[:semi + 1]

    if sql:
        sql = validate_and_fix_sql(sql)
    return sql


# ─────────────────────────────────────────────────────────────
# Feedback Loop: Generate -> Execute -> Correct (Paper Section 4)
# ─────────────────────────────────────────────────────────────

def generate_with_correction(
    prompt: str,
    backend,
    db_path: str,
    schema_text: str,
    question: str,
    evidence: str = "",
    profiles_text: str = "",
    temperature: float = 0,
    seed: Optional[int] = None,
    max_retry: int = 3,
    hint_sql: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate SQL, execute it, and if it fails or is empty, re-prompt the LLM
    with the error message for up to max_retry attempts (paper Section 4).

    hint_sql: optional working SQL from another candidate to reference in corrections.

    Returns:
        (final_sql, correction_log)  where correction_log tracks each attempt.
    """
    correction_log: List[Dict[str, Any]] = []

    # Initial generation
    sql = generate_sql(prompt, backend, temperature=temperature, seed=seed)

    # Handle empty generation — retry instead of giving up
    if not sql or not sql.strip():
        correction_log.append({"attempt": 0, "sql": "", "error": "empty_generation", "action": "will_retry"})

        for attempt in range(1, max_retry + 1):
            print(f"      [retry {attempt}/{max_retry}] empty generation, retrying...")
            retry_prompt = (
                "The previous attempt produced no valid SQL output.\n"
                "Generate a valid SQLite SQL query for the following question.\n\n"
                f"### Database Schema\n{schema_text}\n\n"
            )
            if profiles_text:
                retry_prompt += f"### Column Descriptions\n{profiles_text}\n\n"
            if evidence:
                retry_prompt += f"### Hint\n{evidence}\n\n"
            if hint_sql:
                retry_prompt += f"### Reference SQL from another approach\n{hint_sql}\n\n"
            retry_prompt += (
                f"### Question\n{question}\n\n"
                "Write only the SQL query.\n\nSQL:"
            )
            sql = generate_sql(retry_prompt, backend, temperature=temperature, seed=seed)
            if sql and sql.strip():
                correction_log.append({"attempt": attempt, "sql": sql, "error": None, "action": "recovered"})
                break
            correction_log.append({"attempt": attempt, "sql": "", "error": "empty_generation",
                                   "action": "will_retry" if attempt < max_retry else "exhausted"})

        if not sql or not sql.strip():
            return "", correction_log

    # Try executing
    exec_result = _try_execute(sql, db_path)
    if exec_result["success"]:
        correction_log.append({"attempt": 0, "sql": sql, "error": None, "action": "success"})
        return sql, correction_log

    # Initial execution failed — start correction loop
    correction_log.append({"attempt": 0, "sql": sql, "error": exec_result["error"], "action": "will_retry"})

    for attempt in range(1, max_retry + 1):
        error_msg = exec_result["error"]
        print(f"      [correction {attempt}/{max_retry}] error: {error_msg}")

        # Build correction prompt with the error message
        correction_prompt = (
            "You are a SQLite expert. The SQL query below has an error. "
            "Fix the SQL query and return ONLY the corrected SQL with no explanation.\n\n"
            f"### Database Schema\n{schema_text}\n\n"
        )
        if profiles_text:
            correction_prompt += f"### Column Descriptions\n{profiles_text}\n\n"
        if evidence:
            correction_prompt += f"### Hint\n{evidence}\n\n"
        if hint_sql:
            correction_prompt += (
                f"### Working SQL from another approach\n{hint_sql}\n\n"
            )
        correction_prompt += (
            f"### Question\n{question}\n\n"
            f"### Previous SQL (with error)\n{sql}\n\n"
            f"### Error Message\n{error_msg}\n\n"
            "### Corrected SQL\nWrite only the corrected SQL query.\n\n"
            "SQL:"
        )

        new_sql = generate_sql(correction_prompt, backend, temperature=temperature, seed=seed)

        if not new_sql or not new_sql.strip():
            correction_log.append({"attempt": attempt, "sql": "", "error": "empty_generation", "action": "stop"})
            break

        if new_sql.strip() == sql.strip():
            # LLM returned the same SQL — no point retrying
            correction_log.append({"attempt": attempt, "sql": new_sql, "error": "same_as_previous", "action": "stop"})
            break

        sql = new_sql
        exec_result = _try_execute(sql, db_path)

        if exec_result["success"]:
            correction_log.append({"attempt": attempt, "sql": sql, "error": None, "action": "fixed"})
            print(f"      [correction {attempt}/{max_retry}] fixed!")
            return sql, correction_log

        correction_log.append({"attempt": attempt, "sql": sql, "error": exec_result["error"],
                               "action": "will_retry" if attempt < max_retry else "exhausted"})

    # Exhausted retries — return the last SQL (may still be broken)
    return sql, correction_log


def _try_execute(sql: str, db_path: str) -> Dict[str, Any]:
    """
    Try executing SQL. Returns {success: bool, error: str|None, rows: frozenset|None}.
    """
    if not sql or not sql.strip():
        return {"success": False, "error": "empty SQL", "rows": None}
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        cursor = conn.execute(sql)
        rows = frozenset(cursor.fetchall())
        conn.close()
        return {"success": True, "error": None, "rows": rows}
    except Exception as e:
        return {"success": False, "error": str(e), "rows": None}


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
) -> Optional[Dict[str, Any]]:
    """
    Execute each candidate SQL, compare result sets.
    If 2+ agree -> pick that SQL. Otherwise return None (caller uses tiebreaker).
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

    # No agreement — return None to signal tiebreaker needed
    return None


# ─────────────────────────────────────────────────────────────
# LLM Tiebreaker (replaces random fallback)
# ─────────────────────────────────────────────────────────────

def llm_tiebreaker(
    candidate_sqls: List[str],
    candidate_names: List[str],
    db_path: str,
    question: str,
    evidence: str,
    backend,
) -> Dict[str, Any]:
    """
    LLM-based tiebreaker for when majority vote has no agreement.
    Asks the LLM to pick the best SQL from valid candidates.
    """
    # Filter to valid (executable) candidates
    valid = []
    for i, sql in enumerate(candidate_sqls):
        if sql and sql.strip():
            result = execute_sql(sql, db_path)
            if result is not None:
                valid.append((i, sql))

    # If only 1 valid, pick it
    if len(valid) == 1:
        idx, sql = valid[0]
        return {
            "winner_idx": idx,
            "winner": candidate_names[idx],
            "method": "tiebreaker_single",
            "agreement": 1,
            "sql": sql,
        }

    # If 0 valid, pick first non-empty
    if len(valid) == 0:
        for i, sql in enumerate(candidate_sqls):
            if sql and sql.strip():
                return {
                    "winner_idx": i,
                    "winner": candidate_names[i],
                    "method": "tiebreaker_fallback",
                    "agreement": 0,
                    "sql": sql,
                }
        return {
            "winner_idx": 0,
            "winner": candidate_names[0] if candidate_names else "",
            "method": "tiebreaker_fallback",
            "agreement": 0,
            "sql": candidate_sqls[0] if candidate_sqls else "",
        }

    # 2+ valid but disagree — ask LLM
    letters = "ABCDEFGHIJ"
    options = []
    option_map = {}  # letter -> candidate index
    for li, (idx, sql) in enumerate(valid):
        letter = letters[li]
        option_map[letter] = idx
        options.append(f"{letter}) {sql}")

    options_text = "\n\n".join(options)

    tiebreaker_prompt = (
        "You are a SQLite expert. Given a question and multiple SQL queries that produce "
        "different results, select the most correct one.\n\n"
        f"### Question\n{question}\n"
    )
    if evidence:
        tiebreaker_prompt += f"\n### Hint\n{evidence}\n"
    tiebreaker_prompt += (
        f"\n### Candidates\n{options_text}\n\n"
        "Which candidate is most likely correct? Reply with ONLY the letter.\n\n"
        "Answer:"
    )

    try:
        messages = [{"role": "user", "content": tiebreaker_prompt}]
        response = backend.generate(messages, max_new_tokens=50, temperature=0)
        # Parse letter from response
        chosen_letter = None
        for letter in option_map:
            if letter in response[:10]:
                chosen_letter = letter
                break
        if chosen_letter and chosen_letter in option_map:
            winner_idx = option_map[chosen_letter]
        else:
            # Fallback to first valid
            winner_idx = valid[0][0]
    except Exception as e:
        print(f"    [TIEBREAKER ERROR] {e}")
        winner_idx = valid[0][0]

    return {
        "winner_idx": winner_idx,
        "winner": candidate_names[winner_idx],
        "method": "tiebreaker_llm",
        "agreement": 1,
        "sql": candidate_sqls[winner_idx],
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
    result_groups: Dict[int, List[int]] = {}  # group_id -> list of candidate indices
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
    option_map = {}  # letter -> candidate index
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
        self.short_profiles = load_short_profiles(self.db_dir, db_id)
        self.long_profiles = load_long_profiles(self.db_dir, db_id)
        self.full_profiles = load_full_profiles(self.db_dir, db_id)


# ─────────────────────────────────────────────────────────────
# Multi-Strategy Candidate Configuration
# ─────────────────────────────────────────────────────────────

_CANDIDATE_STRATEGIES = [
    {"name": "wide_net",           "strategy": "full_schema",   "temperature": 0,   "seed": None, "use_fewshot": True},
    {"name": "deep_focus",         "strategy": "pruned_full",   "temperature": 0,   "seed": None, "use_fewshot": True},
    {"name": "independent_thinker","strategy": "pruned_long",   "temperature": 0.7, "seed": 42,   "use_fewshot": False},
]


def process_question(
    q: Dict[str, Any],
    db: DBLoader,
    backend,
    use_mcs: bool = False,
    few_shot_retriever: Optional[FewShotRetriever] = None,
    feedback: bool = False,
    max_retry: int = 3,
) -> Dict[str, Any]:
    """
    Generate 3 SQL candidates using structurally different prompts and select the best.

    Candidate A (wide_net):           Full schema + short profiles inline + few-shot, temp=0
    Candidate B (deep_focus):         Pruned schema + full profiles block + few-shot, temp=0
    Candidate C (independent_thinker): Pruned schema + long profiles + NO few-shot, temp=0.7

    Selection: majority vote, with LLM tiebreaker when no agreement.
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

    # Pre-render schema variants
    schema_pruned = render_pruned_schema(db.schema, linked_keys)
    schema_full = render_full_schema_highlighted(db.schema, linked_keys, db.short_profiles)
    profiles_full_block = render_full_profiles_block(linked_keys, db.full_profiles, db.long_profiles)
    profiles_long_text = render_long_profiles_block(linked_keys, db.long_profiles)

    # Generate candidates — 3 structurally different prompts
    candidate_sqls: List[str] = []
    candidate_names: List[str] = []
    candidates_detail: List[Dict[str, Any]] = []

    for i, cfg in enumerate(_CANDIDATE_STRATEGIES):
        name = cfg["name"]
        strategy = cfg["strategy"]
        temp = cfg["temperature"]
        seed = cfg["seed"]
        use_fewshot = cfg["use_fewshot"]

        print(f"    [{i+1}/{len(_CANDIDATE_STRATEGIES)}] {name} (strategy={strategy}, temp={temp})...")

        # Build strategy-specific prompt
        if strategy == "full_schema":
            prompt = build_full_schema_prompt(
                question, evidence, schema_full,
                few_shots=few_shots if use_fewshot else None,
            )
            schema_for_correction = schema_full
            profiles_for_correction = ""
        elif strategy == "pruned_full":
            prompt = build_deep_focus_prompt(
                question, evidence, schema_pruned, profiles_full_block,
                few_shots=few_shots if use_fewshot else None,
            )
            schema_for_correction = schema_pruned
            profiles_for_correction = profiles_full_block
        elif strategy == "pruned_long":
            prompt = build_independent_prompt(
                question, evidence, schema_pruned, profiles_long_text,
            )
            schema_for_correction = schema_pruned
            profiles_for_correction = profiles_long_text
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Cross-candidate hint: use first successful candidate's SQL
        hint_sql = ""
        if feedback and candidate_sqls:
            for prev_sql in candidate_sqls:
                if prev_sql and prev_sql.strip():
                    prev_result = _try_execute(prev_sql, db.db_path)
                    if prev_result["success"]:
                        hint_sql = prev_sql
                        break

        if feedback:
            sql, corr_log = generate_with_correction(
                prompt, backend, db.db_path, schema_for_correction, question, evidence,
                profiles_text=profiles_for_correction,
                temperature=temp, seed=seed, max_retry=max_retry,
                hint_sql=hint_sql,
            )
            n_corrections = len([c for c in corr_log if c.get("action") in ("fixed", "recovered")])
            if n_corrections > 0:
                print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")
                print(f"    fixed after {len(corr_log)-1} correction(s)")
            else:
                print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")
        else:
            sql = generate_sql(prompt, backend, temperature=temp, seed=seed)
            corr_log = []
            print(f"    -> {sql[:80]}..." if len(sql) > 80 else f"    -> {sql}")

        candidate_sqls.append(sql)
        candidate_names.append(name)
        detail = {
            "sql": sql,
            "strategy": strategy,
            "temperature": temp,
            "seed": seed,
            "use_fewshot": use_fewshot,
        }
        if corr_log:
            detail["correction_log"] = corr_log
        candidates_detail.append(detail)

    # ── Selection: Majority Vote + LLM Tiebreaker ──
    if use_mcs:
        print("    Selecting via MCS-SQL multiple-choice...")
        vote = mcs_select(
            candidate_sqls, candidate_names, db.db_path,
            question, evidence, backend,
        )
    else:
        print("    Selecting via majority vote...")
        vote = majority_vote(candidate_sqls, candidate_names, db.db_path)

        if vote is None:
            # No majority agreement — use LLM tiebreaker
            print("    No majority — using LLM tiebreaker...")
            vote = llm_tiebreaker(
                candidate_sqls, candidate_names, db.db_path,
                question, evidence, backend,
            )

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
    feedback: bool = False,
    max_retry: int = 3,
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

    # Resume: load already-done question IDs from output file
    done_ids: set = set()
    if output_path and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for r in existing.get("results", []):
                done_ids.add(r["question_id"])
            for r in existing.get("errors", []):
                done_ids.add(r["question_id"])
            results = existing.get("results", [])
            errors = existing.get("errors", [])
            print(f"[Resume] Skipping {len(done_ids)} already-done questions, {len(questions) - len(done_ids)} remaining")
        except Exception:
            pass

    for i, q in enumerate(questions):
        if q.get("question_id", i) in done_ids:
            continue
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
                                        few_shot_retriever=few_shot_retriever,
                                        feedback=feedback, max_retry=max_retry)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Q#{qid}: {e}")
            errors.append({"question_id": qid, "db_id": db_id, "error": str(e)})

        # Incremental save every 5 questions
        if output_path and (i + 1) % 5 == 0:
            _save(results, errors, input_path, output_path, use_mcs)
            print(f"  [Saved] {i+1}/{len(questions)} -> {output_path.name}")

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
            "selection_method": "mcs" if use_mcs else "majority_vote_with_tiebreaker",
            "feedback_loop": getattr(_save, '_feedback', False),
            "max_retry": getattr(_save, '_max_retry', 0),
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

    # ── Before vs After Voting Accuracy ──
    # not_voted (before): question is correct if ANY candidate matches gold
    # Voted (after):      question is correct if the VOTED SQL matches gold
    total_with_gold = 0
    not_voted_correct = 0
    voted_correct = 0

    for r in results:
        gold_sql = r.get("gold_sql", "")
        if not gold_sql:
            continue

        db_id = r.get("db_id", "")
        db_path = str(MINIDEV_ROOT / db_id / f"{db_id}.sqlite")

        gold_result = execute_sql(gold_sql, db_path)
        if gold_result is None:
            continue

        total_with_gold += 1

        # Voted accuracy (after voting)
        voted_sql = r.get("voted_sql", "")
        voted_result = execute_sql(voted_sql, db_path)
        if voted_result is not None and voted_result == gold_result:
            voted_correct += 1

        # Oracle accuracy (before voting — any candidate correct?)
        any_correct = False
        for cand in r.get("candidates", []):
            cand_sql = cand.get("sql", "")
            cand_result = execute_sql(cand_sql, db_path)
            if cand_result is not None and cand_result == gold_result:
                any_correct = True
                break
        if any_correct:
            not_voted_correct += 1

    if total_with_gold > 0:
        not_voted_pct = round(100 * not_voted_correct / total_with_gold, 2)
        voted_pct = round(100 * voted_correct / total_with_gold, 2)
        gap = round(not_voted_pct - voted_pct, 2)
        print(f"\n{'~'*70}")
        print(f"ACCURACY: Before vs After Voting")
        print(f"{'~'*70}")
        print(f"  Not Voted (any candidate correct):   {not_voted_correct}/{total_with_gold} = {not_voted_pct}%")
        print(f"  Voted     (final selection correct): {voted_correct}/{total_with_gold} = {voted_pct}%")
        print(f"  Gap       (not_voted - voted):       {gap}%")
        print(f"{'~'*70}")


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
        "--backend", type=str, default="openrouter",
        choices=["openrouter", "huggingface"],
        help="LLM backend (default: openrouter). Requires OPENROUTER_API_KEY env var.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model ID override (default: openai/gpt-oss-120b for openrouter)",
    )
    parser.add_argument(
        "--feedback", action="store_true",
        help="Enable feedback loop: re-prompt LLM on SQL execution errors (paper Section 4)",
    )
    parser.add_argument(
        "--max_retry", type=int, default=3,
        help="Max correction retries per candidate when --feedback is enabled (default: 3)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON path (default: results/candidates_<db>_<ts>.json)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing candidates JSON to resume from",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    # Derive output path
    if args.resume:
        output_path = Path(args.resume)
        if not output_path.exists():
            print(f"Error: resume file not found: {output_path}")
            return
        print(f"[Resume] Continuing from: {output_path}")
    elif args.out:
        output_path = Path(args.out)
    else:
        # Extract db tag from input filename
        stem = input_path.stem  # e.g., schema_links_financial_20260224_033607
        tag = stem.replace("schema_links_", "")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"candidates_{tag}_{ts}.json"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Selection: Majority Vote + LLM Tiebreaker")
    if args.feedback:
        print(f"Feedback loop: ENABLED (max {args.max_retry} retries per candidate)")

    # Store feedback settings for _save metadata
    _save._feedback = args.feedback
    _save._max_retry = args.max_retry if args.feedback else 0

    # Load .env file if present
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

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
        use_mcs=False,
        output_path=output_path,
        few_shot_retriever=few_shot_retriever,
        feedback=args.feedback,
        max_retry=args.max_retry,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved -> {output_path}")


if __name__ == "__main__":
    main()

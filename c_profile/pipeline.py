"""
pipeline.py
-----------
Phase 3: Schema Linking + Pipeline Runner.

Consolidates:
  - schema_linking.py  (core SchemaLinker class, FewShotRetriever, SQL generation)
  - run_pipeline.py    (CLI runner, load_questions, print_summary)

Usage:
  python pipeline.py --db debit_card_specializing
  python pipeline.py --all --questions_per_db 4
  python pipeline.py --db debit_card_specializing --no_llm


For each question: extract literals, query FAISS + LSH for focused columns,
build 5 schema/profile combos, generate 3 SQL candidates per combo,
apply correction loop, then union all referenced columns.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV " / "dev_databases"
MINIDEV_JSON = Path(__file__).parent / "MINIDEV " / "mini_dev_sqlite.json"
TRAIN_JSON   = Path(__file__).resolve().parent.parent / "dataset" / "bird" / "train.json"

sys.path.insert(0, str(Path(__file__).parent))


# -- Helpers ---------------------------------------------------

def _dedup_ordered(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving insertion order."""
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# -- Literal extraction (regex) --------------------------------

_RE_QUOTED = re.compile(r'["\']([^"\']+)["\']')
_RE_NUMBER = re.compile(r'\b\d{4}\b|\b\d+(?:\.\d+)?\b')
_RE_UPPER  = re.compile(r'\b[A-Z][A-Z0-9]{1,}\b')
_RE_TITLE  = re.compile(r'\b[A-Z][a-z]{2,}\b')

_STOPWORDS = {
    "What", "Which", "How", "Who", "When", "Where", "The", "Are", "Was",
    "Did", "Does", "Has", "Have", "List", "Find", "Show", "Give", "Tell",
    "Many", "Much", "Most", "All", "Any", "Each", "Every", "Some",
    "And", "For", "Not", "But", "With", "Than", "That", "This",
}


def extract_literals(question: str) -> List[str]:
    """Extract candidate literal values from a question for LSH lookup."""
    candidates: List[str] = []

    for m in _RE_QUOTED.finditer(question):
        candidates.append(m.group(1).strip())

    for pattern in (_RE_UPPER, _RE_TITLE):
        for m in pattern.finditer(question):
            if m.group(0) not in _STOPWORDS:
                candidates.append(m.group(0))

    for m in _RE_NUMBER.finditer(question):
        candidates.append(m.group(0))

    return _dedup_ordered(candidates)


_RE_SQL_STRINGS = re.compile(r"'([^']*)'")


def extract_string_literals_from_sql(sql: str) -> List[str]:
    """Extract quoted string literals from a generated SQL query."""
    return _dedup_ordered(
        m.group(1).strip() for m in _RE_SQL_STRINGS.finditer(sql)
    )


# -- Data Loaders ----------------------------------------------

def _load_profile_jsonl(path: Path, value_key: str) -> Dict[str, str]:
    """Load a profile JSONL into {table.column: value} dict."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[f"{r['table']}.{r['column']}"] = r.get(value_key, "")
    return out


def load_sqlite_schema(db_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load full schema: {table: [{column, type, pk, fk}, ...]}"""
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]

    fk_cols: Dict[str, Set[str]] = {t: set() for t in tables}
    for t in tables:
        try:
            cur.execute(f'PRAGMA foreign_key_list("{t}")')
            for row in cur.fetchall():
                fk_cols[t].add(row[3])
        except Exception:
            pass

    schema: Dict[str, List[Dict[str, Any]]] = {}
    for t in tables:
        cur.execute(f'PRAGMA table_info("{t}")')
        schema[t] = [
            {"column": r[1], "type": r[2] or "TEXT", "pk": bool(r[5]),
             "fk": r[1] in fk_cols.get(t, set())}
            for r in cur.fetchall()
        ]
    conn.close()
    return schema


def load_short_profiles(db_dir: Path, db_id: str) -> Dict[str, str]:
    """Load {table.column: one-sentence description}."""
    path = db_dir / f"{db_id}.short_profiles.jsonl"
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[f"{r['table']}.{r['column']}"] = (
                    r.get("extracted_final") or r.get("short_profile_en") or ""
                )
    return out


def load_long_profiles(db_dir: Path, db_id: str) -> Dict[str, str]:
    """Load {table.column: long profile text (raw stats)}."""
    return _load_profile_jsonl(db_dir / f"{db_id}.long_profiles.jsonl", "profile_long_en")


def load_full_profiles(db_dir: Path, db_id: str) -> Dict[str, str]:
    """Load {table.column: full profile text (dev_doc + long stats)}."""
    return _load_profile_jsonl(db_dir / f"{db_id}.full_profiles.jsonl", "profile_full_en")


# -- Index Queries ---------------------------------------------

def query_faiss(question: str, db_id: str, db_dir: Path, top_k: int = 10) -> List[Dict]:
    from indexes import query_faiss as _qf
    return _qf(question, db_id, db_dir, top_k=top_k)


def query_lsh(literal: str, db_id: str, db_dir: Path) -> List[Dict]:
    from indexes import query_lsh as _ql
    return _ql(literal, db_id, db_dir)


# -- Column Extraction from SQL --------------------------------

_SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "GROUP", "BY", "ORDER", "HAVING",
    "AND", "OR", "NOT", "IN", "AS", "DISTINCT", "COUNT", "SUM", "AVG", "MIN",
    "MAX", "LIMIT", "OFFSET", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS",
    "UNION", "INTERSECT", "EXCEPT", "NULL", "IS", "BETWEEN", "LIKE", "CASE",
    "WHEN", "THEN", "ELSE", "END", "ASC", "DESC", "NULLS", "LAST", "FIRST",
    "ALL", "ANY", "EXISTS", "WITH", "RECURSIVE", "INSERT", "UPDATE", "DELETE",
    "CREATE", "DROP", "CAST", "IIF", "COALESCE", "IFNULL", "LENGTH", "TRIM",
    "UPPER", "LOWER", "SUBSTR", "REPLACE", "ROUND", "ABS", "STRFTIME",
    "TRUE", "FALSE", "OVER", "PARTITION", "ROWS", "RANGE",
}


def _build_schema_lookups(
    schema: Dict[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[str, str]]]]:
    """Build lowercased table name map and column-to-tables reverse index."""
    table_lower = {t.lower(): t for t in schema}
    col_to_tables: Dict[str, List[Tuple[str, str]]] = {}
    for tname, cols in schema.items():
        for c in cols:
            col_to_tables.setdefault(c["column"].lower(), []).append((tname, c["column"]))
    return table_lower, col_to_tables


def _resolve_ambiguous_column(
    matches: List[Tuple[str, str]], used_tables: Set[str]
) -> Set[Tuple[str, str]]:
    """Disambiguate a column name appearing in multiple tables using FROM/JOIN context."""
    return {(t, c) for t, c in matches if t in used_tables}


def _extract_columns_regex(
    sql: str, schema: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """Regex-based fallback: extract {table, column} pairs from SQL."""
    found: Set[Tuple[str, str]] = set()
    table_lower, col_to_tables = _build_schema_lookups(schema)

    sql_clean = re.sub(r"'[^']*'", "''", sql)
    sql_clean = re.sub(r'"[^"]*"', '""', sql_clean)

    # Pass 1: explicit table.column references
    for m in re.finditer(r'\b(\w+)\.(\w+)\b', sql_clean, re.IGNORECASE):
        traw, craw = m.group(1), m.group(2)
        treal = table_lower.get(traw.lower())
        if treal is None:
            continue
        for c in schema[treal]:
            if c["column"].lower() == craw.lower():
                found.add((treal, c["column"]))
                break

    # Pass 2: bare identifiers matched against schema columns
    used_tables: Set[str] = set()
    for tm in re.finditer(r'\b(?:FROM|JOIN)\s+([A-Za-z_]\w*)', sql_clean, re.IGNORECASE):
        t = table_lower.get(tm.group(1).lower())
        if t:
            used_tables.add(t)

    for m in re.finditer(r'\b([A-Za-z_]\w*)\b', sql_clean):
        ident = m.group(1)
        if ident.upper() in _SQL_KEYWORDS or ident.lower() in table_lower:
            continue
        matches = col_to_tables.get(ident.lower(), [])
        if len(matches) == 1:
            found.add(matches[0])
        elif len(matches) > 1:
            found |= _resolve_ambiguous_column(matches, used_tables)

    return [{"table": t, "column": c} for t, c in sorted(found)]


def _extract_columns_sqlglot(
    sql: str, schema: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """SQLglot AST-based column extraction. Falls back to regex on parse failure."""
    import sqlglot
    import sqlglot.expressions as exp

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return _extract_columns_regex(sql, schema)

    table_lower, col_to_tables = _build_schema_lookups(schema)

    # Build alias_map: resolve T1/T2/AS aliases to canonical table names
    alias_map: Dict[str, str] = {}
    for table_expr in tree.find_all(exp.Table):
        tname = table_expr.name or ""
        alias = table_expr.alias or ""
        real  = table_lower.get(tname.lower())
        if real:
            alias_map[tname.lower()] = real
            if alias:
                alias_map[alias.lower()] = real

    found: Set[Tuple[str, str]] = set()
    for col_expr in tree.find_all(exp.Column):
        col_name  = col_expr.name or ""
        table_ref = col_expr.table or ""

        if not col_name or col_name == "*":
            continue

        if table_ref:
            real_table = alias_map.get(table_ref.lower())
            if real_table:
                for c in schema[real_table]:
                    if c["column"].lower() == col_name.lower():
                        found.add((real_table, c["column"]))
                        break
        else:
            matches = col_to_tables.get(col_name.lower(), [])
            if len(matches) == 1:
                found.add(matches[0])
            elif len(matches) > 1:
                found |= _resolve_ambiguous_column(matches, set(alias_map.values()))

    return [{"table": t, "column": c} for t, c in sorted(found)]


def extract_columns_from_sql(
    sql: str, schema: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, str]]:
    """Parse SQL and return all {table, column} pairs. Uses SQLglot if available, else regex."""
    try:
        import sqlglot  # noqa: F401
        return _extract_columns_sqlglot(sql, schema)
    except ImportError:
        return _extract_columns_regex(sql, schema)


# -- Schema Renderer -------------------------------------------

def _get_column_comment(
    key: str,
    profile_mode: str,
    short_profiles: Dict[str, str],
    long_profiles: Dict[str, str],
    full_profiles: Optional[Dict[str, str]],
) -> str:
    """Return the inline comment for a column based on the profile mode."""
    if profile_mode == "short":
        desc = short_profiles.get(key, "")
        return desc

    if profile_mode == "long":
        desc = long_profiles.get(key, "")
        if desc:
            return desc.replace("\n", " | ")[:200]
        return ""

    if profile_mode == "full":
        ldesc = long_profiles.get(key, "").replace("\n", " | ")[:150]
        dev_text = ""
        if full_profiles:
            fp = full_profiles.get(key, "")
            if "[DEV DOC]" in fp:
                dev_text = fp.split("[DEV DOC]", 1)[1].strip().replace("\n", " | ")[:150]
        parts = []
        if dev_text:
            parts.append(dev_text)
        elif short_profiles.get(key):
            parts.append(short_profiles[key])
        if ldesc:
            parts.append(ldesc)
        return " || ".join(parts) if parts else ""

    return ""


def _render_schema(
    schema: Dict[str, List[Dict[str, Any]]],
    tables: List[str],
    col_filter: Optional[Set[str]],
    short_profiles: Dict[str, str],
    long_profiles: Dict[str, str],
    profile_mode: str,
    col_order_seed: Optional[int] = None,
    full_profiles: Optional[Dict[str, str]] = None,
) -> str:
    """Render CREATE TABLE blocks with optional profile comments per column."""
    blocks = []
    for table in tables:
        cols = list(schema.get(table, []))
        if col_order_seed is not None:
            rng = random.Random(col_order_seed + abs(hash(table)) % 10000)
            rng.shuffle(cols)

        rendered = []
        for col in cols:
            key = f"{table}.{col['column']}"
            if col_filter is not None and key not in col_filter:
                continue

            line = f"  {col['column']} {col['type']}"
            comment = _get_column_comment(
                key, profile_mode, short_profiles, long_profiles, full_profiles
            )
            if comment:
                line += f"  -- {comment}"
            rendered.append(line)

        if rendered:
            blocks.append(f"CREATE TABLE {table} (\n" + ",\n".join(rendered) + "\n);")

    return "\n\n".join(blocks)


# -- Prompt Builder --------------------------------------------

def build_prompt(
    question: str,
    schema_text: str,
    evidence: str = "",
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the LLM prompt for SQL generation."""
    parts = [
        "You are a SQLite expert. Given the database schema below, write a SQL query "
        "to answer the question.\n\n"
        "### Database Schema\n"
        f"{schema_text}\n"
    ]
    if few_shots:
        examples = "\n".join(f"Question: {ex['question']}\nSQL: {ex['SQL']}\n" for ex in few_shots)
        parts.append(f"\n### Examples\n{examples}")
    if evidence:
        parts.append(f"\n### Hint\n{evidence}")
    parts.append(
        f"\n### Question\n{question}\n\n"
        "### SQL\nWrite only the SQL query with no explanation.\n\n"
        "SQL:"
    )
    return "\n".join(parts)


# -- Few-Shot Question Masking ---------------------------------

_RE_MASK_QUOTED = re.compile(r'["\'][^"\']{1,100}["\']')
_RE_MASK_YEAR   = re.compile(r'\b(19|20)\d{2}\b')
_RE_MASK_NUMBER = re.compile(r'\b\d+(?:\.\d+)?\b')
_RE_MASK_CODE   = re.compile(r'\b[A-Z]{2,6}\b')

# SQL/common English words that happen to be ALL-CAPS — do not mask these
_MASK_SKIP = {"SQL", "ID", "DB", "NULL", "AND", "OR", "NOT", "IN", "IS", "BY", "ON"}


def mask_question(question: str) -> str:
    """Replace entity tokens with placeholders for structure-aware few-shot retrieval (paper Section 4)."""
    masked = _RE_MASK_QUOTED.sub("<value>", question)
    masked = _RE_MASK_YEAR.sub("<year>", masked)
    masked = _RE_MASK_NUMBER.sub("<number>", masked)

    def _replace_code(m: re.Match) -> str:
        return m.group(0) if m.group(0) in _MASK_SKIP else "<code>"

    masked = _RE_MASK_CODE.sub(_replace_code, masked)
    return masked


# -- Few-Shot Retriever ----------------------------------------

class FewShotRetriever:
    """FAISS-based few-shot retriever using masked question embeddings (paper Section 4)."""

    def __init__(self, questions: List[Dict[str, Any]]):
        self.pool = [q for q in questions if q.get("question") and q.get("SQL")]
        self._index = None

    def build(self, cache_path: Optional[str] = None) -> None:
        """Embed all pool questions (masked) and build FAISS index.
        If cache_path is given, saves/loads embeddings to avoid re-embedding."""
        from indexes import get_embeddings
        import faiss
        import numpy as np

        _cache = Path(cache_path) if cache_path else Path(__file__).parent / "fewshot_index.npy"

        texts = [mask_question(q["question"]) for q in self.pool]

        if _cache.exists():
            print(f"  [FewShot] Loading cached embeddings from {_cache.name}...")
            vecs = np.load(str(_cache))
        else:
            print(f"  [FewShot] Embedding {len(texts)} masked questions for few-shot pool...")
            embeddings = get_embeddings(texts)
            vecs = np.array(embeddings, dtype="float32")
            np.save(str(_cache), vecs)
            print(f"  [FewShot] Saved embeddings cache → {_cache.name}")

        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        print(f"  [FewShot] Index built ({dim}-dim, {len(texts)} vectors).")

    def retrieve(
        self,
        question: str,
        k: int = 8,
        exclude_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar questions; optionally exclude an exact match."""
        from indexes import get_embeddings
        import faiss
        import numpy as np

        if self._index is None:
            self.build()

        q_emb = get_embeddings([mask_question(question)])[0]
        q_vec = np.array([q_emb], dtype="float32")
        faiss.normalize_L2(q_vec)

        search_k = min(k + 10, self._index.ntotal)  # overfetch to allow exclusion
        scores, indices = self._index.search(q_vec, search_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            candidate = self.pool[idx]
            if exclude_question and candidate["question"].strip() == exclude_question.strip():
                continue
            results.append(candidate)
            if len(results) >= k:
                break

        return results


# -- SQLglot Validation (paper Section 4) ---------------------

def validate_and_fix_sql(sql: str) -> str:
    """
    Fix common LLM SQL errors via SQLglot AST (paper Section 4):
      1. NULL ordering: add NULLS LAST to ASC ORDER BY in LIMIT queries.
      2. Wrong min/max pattern: strip ORDER BY from scalar MIN()/MAX() without GROUP BY.
      3. String concatenation: replace col1 || ' ' || col2 with separate SELECT columns.
      4. Nested subquery min/max: replace WHERE col = (SELECT MIN/MAX(col) FROM t)
         with ORDER BY col ASC/DESC LIMIT 1.
    Returns original SQL if sqlglot is unavailable or parsing fails.
    """
    try:
        import sqlglot
        import sqlglot.expressions as exp
    except ImportError:
        return sql

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return sql  # unparseable SQL — return as-is, correction loop will handle it

    changed = False

    # Fix 1: NULLS LAST on ASC ORDER BY inside LIMIT queries
    if tree.find(exp.Limit):
        for ordered in tree.find_all(exp.Ordered):
            if not ordered.args.get("desc", False) and ordered.args.get("nullsfirst") is None:
                ordered.args["nullsfirst"] = False
                changed = True

    # Fix 2: remove meaningless ORDER BY on scalar MIN()/MAX()
    # Only check the SELECT's own expressions — not nested subqueries
    for select in tree.find_all(exp.Select):
        has_min_max = any(
            isinstance(expr, (exp.Min, exp.Max)) or
            (hasattr(expr, 'this') and isinstance(getattr(expr, 'this', None), (exp.Min, exp.Max)))
            for expr in select.expressions
        )
        has_group_by = bool(select.args.get("group"))
        has_order_by = bool(select.args.get("order"))
        if has_min_max and not has_group_by and has_order_by:
            select.set("order", None)
            changed = True

    # Fix 3: replace string concatenation (||) in SELECT with separate columns
    for select in tree.find_all(exp.Select):
        new_expressions = []
        select_changed = False
        for expr in select.expressions:
            # Check if expression is or contains DPipe (||)
            dpipe = expr.find(exp.DPipe) if not isinstance(expr, exp.DPipe) else expr
            if dpipe is None:
                new_expressions.append(expr)
                continue
            # Extract column references from the concatenation operands
            cols = list(dpipe.find_all(exp.Column))
            if len(cols) >= 2:
                # Replace concatenation with individual columns
                for col in cols:
                    new_expressions.append(col.copy())
                select_changed = True
            else:
                new_expressions.append(expr)
        if select_changed:
            select.set("expressions", new_expressions)
            changed = True

    # Fix 4: nested subquery min/max → ORDER BY + LIMIT 1
    # Pattern: WHERE col = (SELECT MIN(col) FROM t) → remove WHERE, add ORDER BY col ASC LIMIT 1
    # Pattern: WHERE col = (SELECT MAX(col) FROM t) → remove WHERE, add ORDER BY col DESC LIMIT 1
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        if where is None:
            continue
        # Look for EQ predicate: col = (SELECT MIN/MAX(...) ...)
        eq_node = where.find(exp.EQ)
        if eq_node is None:
            continue
        left, right = eq_node.left, eq_node.right
        # The subquery should be on the right side
        subquery = None
        outer_col = None
        if isinstance(right, exp.Subquery):
            subquery = right
            outer_col = left
        elif isinstance(left, exp.Subquery):
            subquery = left
            outer_col = right
        if subquery is None or not isinstance(outer_col, exp.Column):
            continue
        # Get the inner SELECT from the subquery
        inner_select = subquery.this
        if not isinstance(inner_select, exp.Select):
            continue
        # Check: inner SELECT has exactly one expression which is MIN or MAX
        inner_exprs = inner_select.expressions
        if len(inner_exprs) != 1:
            continue
        inner_expr = inner_exprs[0]
        is_min = isinstance(inner_expr, exp.Min)
        is_max = isinstance(inner_expr, exp.Max)
        if not is_min and not is_max:
            continue
        # Check: inner aggregate references the same column as outer
        agg_col = inner_expr.this
        if not isinstance(agg_col, exp.Column):
            continue
        # Check: inner query has no GROUP BY (scalar aggregate)
        if inner_select.args.get("group"):
            continue
        # Check: outer query doesn't already have ORDER BY or LIMIT
        if select.args.get("order") or select.args.get("limit"):
            continue
        # Check: the WHERE clause is ONLY this eq_node (no AND/OR with other conditions)
        # If the where has other conditions, skip — too complex to safely rewrite
        where_expr = where.this
        if where_expr is not eq_node:
            continue
        # Safe to rewrite: remove WHERE, add ORDER BY + LIMIT 1
        select.set("where", None)
        order_col = outer_col.copy()
        ordered = exp.Ordered(this=order_col, desc=is_max)
        select.set("order", exp.Order(expressions=[ordered]))
        select.set("limit", exp.Limit(expression=exp.Literal.number(1)))
        changed = True

    if changed:
        return tree.sql(dialect="sqlite")
    return sql


# -- SchemaLinker ----------------------------------------------

class SchemaLinker:
    """
    Full Phase 3 pipeline for one database:
      link() → FAISS + LSH → focused schema
      run()  → 5 combos → GPT-5.2 → extract columns → schema links
    """

    def __init__(self, db_id: str, db_dir: Optional[Path] = None):
        self.db_id   = db_id
        self.db_dir  = db_dir or (MINIDEV_ROOT / db_id)
        self.db_path = self.db_dir / f"{db_id}.sqlite"

        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite not found: {self.db_path}")

        self.schema         = load_sqlite_schema(self.db_path)
        self.short_profiles = load_short_profiles(self.db_dir, db_id)
        self.long_profiles  = load_long_profiles(self.db_dir, db_id)
        self.full_profiles  = load_full_profiles(self.db_dir, db_id)

    # Schema Linking

    def link(self, question: str, faiss_top_k: int = 10) -> Dict[str, Any]:
        """
        FAISS + LSH → focused columns + tables.
        PKs of focused tables are always included for JOIN capability.
        """
        # Semantic match
        faiss_hits = query_faiss(question, self.db_id, self.db_dir, top_k=faiss_top_k)

        # Literal value match
        literals   = extract_literals(question)
        lsh_hits: List[Dict] = []
        for lit in literals:
            try:
                hits = query_lsh(lit, self.db_id, self.db_dir)
                for h in hits:
                    lsh_hits.append({**h, "literal": lit})
            except Exception:
                pass

        # Build focused set
        focused: Set[str] = set()
        for r in faiss_hits:
            focused.add(f"{r['table']}.{r['column']}")
        for r in lsh_hits:
            if r.get("table") and r.get("column"):
                focused.add(f"{r['table']}.{r['column']}")

        # Focused tables
        focused_tables: Set[str] = {key.split(".")[0] for key in focused}

        # Always include PKs of focused tables (needed for JOINs)
        for table in list(focused_tables):
            for col in self.schema.get(table, []):
                if col["pk"]:
                    focused.add(f"{table}.{col['column']}")

        return {
            "faiss_hits":      faiss_hits,
            "lsh_hits":        lsh_hits,
            "focused_columns": focused,
            "focused_tables":  focused_tables,
            "literals":        literals,
        }

    # 5 Schema Combinations

    def build_five_schemas(
        self,
        link_result: Dict[str, Any],
        col_order_seed: Optional[int] = None,
    ) -> Dict[str, str]:
        """
        Build the 5 schema+profile prompt blocks.

        Combo key → (tables, col_filter, profile_mode)
        col_order_seed: when set, shuffles column order for diversity (paper Section 4).
        """
        focused_cols   = link_result["focused_columns"]    # Set[str]
        focused_tables = link_result["focused_tables"]     # Set[str]
        all_tables     = list(self.schema.keys())
        focused_list   = [t for t in all_tables if t in focused_tables]  # preserve DB order

        combos = {
            # 1. Focused tables, focused cols only, short profiles
            "focused_short": (focused_list, focused_cols, "short"),
            # 2. Focused tables, focused cols only, long profiles
            "focused_long":  (focused_list, focused_cols, "long"),
            # 3. All tables, all cols, short profiles
            "full_short":    (all_tables, None, "short"),
            # 4. All tables, all cols, long profiles
            "full_long":     (all_tables, None, "long"),
            # 5. Focused tables, ALL their cols, short+long combined
            "focused_full":  (focused_list, None, "full"),
        }

        schemas: Dict[str, str] = {}
        for name, (tables, col_filter, profile_mode) in combos.items():
            schemas[name] = _render_schema(
                self.schema, tables, col_filter,
                self.short_profiles, self.long_profiles, profile_mode,
                col_order_seed=col_order_seed,
                full_profiles=self.full_profiles,
            )
        return schemas

    # SQL Generation

    def generate_sql(self, prompt: str, backend, temperature: float = 0) -> str:
        """Call GPT-5.2 and extract the SQL from the response."""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = backend.generate(messages, max_new_tokens=512, temperature=temperature)
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

    # ── Correction Loop (Paper Section 3, steps d–e) ──────────

    def _run_correction_loop(
        self,
        question: str,
        evidence: str,
        schema_text: str,
        initial_sql: str,
        initial_cols: List[Dict[str, str]],
        backend,
        max_retry: int = 3,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Re-ask LLM when SQL uses literals not found in any referenced column (paper Section 3, steps d–e)."""
        sql = initial_sql
        cols = initial_cols

        for attempt in range(max_retry):
            sql_literals = extract_string_literals_from_sql(sql)
            if not sql_literals:
                break

            fields_q: Set[str] = {f"{c['table']}.{c['column']}" for c in cols}
            lit_fields_q: Set[str] = set()
            missing_lits: List[str] = []

            for lit in sql_literals:
                try:
                    lsh_hits = query_lsh(lit, self.db_id, self.db_dir)
                except Exception:
                    lsh_hits = []

                fields_containing_lit = {
                    f"{h['table']}.{h['column']}"
                    for h in lsh_hits
                    if h.get("table") and h.get("column")
                }

                if fields_containing_lit and not (fields_containing_lit & fields_q):
                    lit_fields_q |= fields_containing_lit
                    missing_lits.append(lit)

            if not lit_fields_q:
                break

            # Augment schema with columns that contain missing literals
            new_cols_by_table: Dict[str, List[str]] = {}
            for key in lit_fields_q:
                if key in fields_q:
                    continue
                tbl, col_name = key.split(".", 1)
                new_cols_by_table.setdefault(tbl, []).append(col_name)

            augmented_schema = schema_text
            if new_cols_by_table:
                extra_blocks = []
                for tbl, col_names in sorted(new_cols_by_table.items()):
                    lines = []
                    for c in self.schema.get(tbl, []):
                        if c["column"] in col_names:
                            key = f"{tbl}.{c['column']}"
                            line = f"  {c['column']} {c['type']}"
                            desc = self.short_profiles.get(key, "")
                            if desc:
                                line += f"  -- {desc}"
                            lines.append(line)
                    if lines:
                        extra_blocks.append(
                            f"-- Additional fields (contain literal values):\n"
                            f"CREATE TABLE {tbl} (\n" + ",\n".join(lines) + "\n);"
                        )
                if extra_blocks:
                    augmented_schema = schema_text + "\n\n" + "\n\n".join(extra_blocks)

            missing_str = ", ".join(f"'{l}'" for l in missing_lits)
            suggestion_fields = ", ".join(sorted(lit_fields_q))
            correction_prompt = (
                "You are a SQLite expert. The SQL query below uses the literal value(s) "
                f"{missing_str}, but no field containing those values was referenced. "
                f"Please revise the SQL to use one of these fields which contain the literal: "
                f"{suggestion_fields}.\n\n"
                "### Database Schema\n"
                f"{augmented_schema}\n\n"
            )
            if evidence:
                correction_prompt += f"### Hint\n{evidence}\n\n"
            correction_prompt += (
                f"### Question\n{question}\n\n"
                f"### Previous SQL\n{sql}\n\n"
                "### Revised SQL\nWrite only the revised SQL query with no explanation.\n\n"
                "SQL:"
            )

            new_sql = self.generate_sql(correction_prompt, backend)
            if not new_sql or new_sql == sql:
                break

            new_cols = extract_columns_from_sql(new_sql, self.schema)
            print(f"    [correction {attempt+1}/{max_retry}] missing literals={missing_lits} "
                  f"→ {len(new_cols)} col(s) after revision")
            sql = new_sql
            cols = new_cols

        return sql, cols

    # Full Pipeline

    # 3 candidates per combo: (temperature, col_order_seed) — paper Section 4
    _CANDIDATES = [(0, None), (0.7, 1), (0.7, 2)]

    @staticmethod
    def _combo_vote(candidate_sqls: List[str], db_path: str) -> Tuple[str, str]:
        """
        Vote among candidates within a single combo (paper Section 3).
        Execute each SQL, compare result sets. 2+ agree → pick that SQL.
        Returns (winning_sql, method).
        """
        results = []
        for sql in candidate_sqls:
            if not sql or not sql.strip():
                results.append(None)
                continue
            try:
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA busy_timeout = 5000")
                cursor = conn.execute(sql)
                rows = frozenset(cursor.fetchall())
                conn.close()
                results.append(rows)
            except Exception:
                results.append(None)

        # Check for majority agreement
        for i in range(len(results)):
            if results[i] is None:
                continue
            agreeing = [j for j in range(len(results)) if results[j] == results[i]]
            if len(agreeing) >= 2:
                return candidate_sqls[agreeing[0]], "majority"

        # No agreement — pick random valid candidate
        valid = [(i, sql) for i, sql in enumerate(candidate_sqls)
                 if sql and results[i] is not None]
        if valid:
            idx, sql = random.choice(valid)
            return sql, "random"

        # All failed — return first non-empty
        for sql in candidate_sqls:
            if sql and sql.strip():
                return sql, "fallback"
        return "", "none"

    def run(
        self,
        question: str,
        evidence: str = "",
        backend=None,
        faiss_top_k: int = 10,
        question_id: Any = None,
        few_shot_retriever: Optional["FewShotRetriever"] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: FAISS+LSH → 5 combos × 3 candidates → correction loop → vote → union of schema links.
        combo_sqls holds the majority-voted winner per combo.
        """
        print(f"\n  Q: {question[:70]}...")
        link = self.link(question, faiss_top_k=faiss_top_k)
        print(f"  Focused tables: {sorted(link['focused_tables'])}")
        print(f"  Focused columns ({len(link['focused_columns'])}): "
              f"{sorted(link['focused_columns'])[:6]}{'...' if len(link['focused_columns']) > 6 else ''}")
        print(f"  Literals: {link['literals']}")

        all_tables   = list(self.schema.keys())
        focused_cols = link["focused_columns"]
        focused_list = [t for t in all_tables if t in link["focused_tables"]]

        combos_params: Dict[str, Tuple] = {
            "focused_short": (focused_list, focused_cols, "short"),
            "focused_long":  (focused_list, focused_cols, "long"),
            "full_short":    (all_tables,   None,         "short"),
            "full_long":     (all_tables,   None,         "long"),
            "focused_full":  (focused_list, None,         "full"),
        }

        combo_sqls: Dict[str, str]           = {}
        combo_columns: Dict[str, List[Dict]] = {}
        all_schema_links: Set[Tuple[str, str]] = set()

        if backend is not None:
            few_shots: Optional[List[Dict]] = None
            if few_shot_retriever is not None:
                few_shots = few_shot_retriever.retrieve(
                    question, k=8, exclude_question=question
                )
                print(f"  Few-shot examples: {len(few_shots)} retrieved")

            for combo_name, (tables, col_filter, profile_mode) in combos_params.items():
                combo_links: Set[Tuple[str, str]] = set()
                cand_sqls: List[str] = []

                for cand_idx, (temp, col_seed) in enumerate(self._CANDIDATES):
                    schema_cand = _render_schema(
                        self.schema, tables, col_filter,
                        self.short_profiles, self.long_profiles, profile_mode,
                        col_order_seed=col_seed,
                        full_profiles=self.full_profiles,
                    )

                    prompt = build_prompt(question, schema_cand, evidence, few_shots)
                    sql    = self.generate_sql(prompt, backend, temperature=temp)
                    cols   = extract_columns_from_sql(sql, self.schema) if sql else []

                    if sql:  # correction loop (paper Section 3, steps d–e)
                        sql, cols = self._run_correction_loop(
                            question, evidence, schema_cand, sql, cols, backend
                        )

                    for c in cols:
                        combo_links.add((c["table"], c["column"]))

                    cand_sqls.append(sql)

                # Vote among the 3 candidates for this combo (paper Section 3)
                db_path = str(self.db_dir / f"{self.db_id}.sqlite")
                winner_sql, vote_method = self._combo_vote(cand_sqls, db_path)

                combo_sqls[combo_name]    = winner_sql
                combo_columns[combo_name] = [
                    {"table": t, "column": c} for t, c in sorted(combo_links)
                ]
                all_schema_links |= combo_links
                print(f"  [{combo_name}] {len(combo_links)} col(s) "
                      f"from {len(self._CANDIDATES)} candidates (vote: {vote_method})")
        else:
            # No backend: skip SQL generation
            for combo_name in combos_params:
                combo_sqls[combo_name]    = ""
                combo_columns[combo_name] = []

        schema_links = [
            {"table": t, "column": c}
            for t, c in sorted(all_schema_links)
        ]

        return {
            "question_id":     question_id,
            "db_id":           self.db_id,
            "question":        question,
            "evidence":        evidence,
            "literals":        link["literals"],
            "focused_tables":  sorted(link["focused_tables"]),
            "focused_columns": sorted(link["focused_columns"]),
            "faiss_hits": [
                {"table": r["table"], "column": r["column"],
                 "score": round(r.get("faiss_score", 0), 4)}
                for r in link["faiss_hits"]
            ],
            "lsh_hits": [
                {"table": r.get("table",""), "column": r.get("column",""),
                 "literal": r.get("literal",""), "match_type": r.get("match_type","")}
                for r in link["lsh_hits"]
            ],
            "combo_sqls":     combo_sqls,
            "combo_columns":  combo_columns,
            "schema_links":   schema_links,
        }


# -- CLI Demo --------------------------------------------------

def demo_single_question():
    """CLI demo: run schema linking on a single question."""
    parser = argparse.ArgumentParser(description="Phase 3: Schema Linking (single question)")
    parser.add_argument("--db",       type=str, default="debit_card_specializing")
    parser.add_argument("--question", type=str,
                        default="How many customers paid in CZK currency?")
    parser.add_argument("--evidence", type=str, default="")
    parser.add_argument("--top_k",   type=int, default=10)
    parser.add_argument("--no_llm",  action="store_true",
                        help="Skip LLM call (just show schemas)")
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    backend = None
    if not args.no_llm:
        from llm import make_backend
        backend = make_backend("huggingface", model_id="openai/gpt-oss-120b", cache=True)

    linker = SchemaLinker(args.db)
    result = linker.run(
        args.question,
        evidence=args.evidence,
        backend=backend,
        faiss_top_k=args.top_k,
    )

    print("\n" + "="*70)
    print("SCHEMA LINKS (columns used by LLM):")
    for sl in result["schema_links"]:
        print(f"  {sl['table']}.{sl['column']}")

    if args.no_llm:
        print("\n--- 5 SCHEMA COMBINATIONS (no LLM call) ---")
        linker2 = SchemaLinker(args.db)
        link = linker2.link(args.question, faiss_top_k=args.top_k)
        schemas = linker2.build_five_schemas(link)
        for name, s in schemas.items():
            print(f"\n[{name}]\n{s[:500]}...")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved → {args.json_out}")


# ══════════════════════════════════════════════════════════════
# Pipeline Runner (was run_pipeline.py)
# ══════════════════════════════════════════════════════════════


RESULTS_DIR = Path(__file__).parent / "results"


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


def load_train_questions() -> List[Dict]:
    """
    Load BIRD training set questions for few-shot retrieval pool.

    Paper Section 4: "few-shot examples taken from the train query set"
    (9428 questions with 'question', 'SQL', 'evidence', 'db_id').
    """
    if not TRAIN_JSON.exists():
        print(f"[WARN] BIRD train set not found at {TRAIN_JSON}, falling back to minidev pool")
        return load_questions()
    with open(TRAIN_JSON) as f:
        return json.load(f)


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

def _load_checkpoint(path: Path):
    """Load existing results/errors from a checkpoint file. Returns (results, errors, done_ids)."""
    if path and path.exists():
        with open(path) as f:
            data = json.load(f)
        results = data.get("results", [])
        errors  = data.get("errors", [])
        done_ids = {r["question_id"] for r in results} | {e["question_id"] for e in errors}
        print(f"[Checkpoint] Resuming: {len(results)} done, {len(errors)} errors, {len(done_ids)} total skipped")
        return results, errors, done_ids
    return [], [], set()


def run_pipeline(
    questions: List[Dict],
    backend=None,
    faiss_top_k: int = 10,
    output_path: Optional[Path] = None,
    few_shot_retriever=None,
    resume: bool = False,
) -> List[Dict]:
    """
    Run the schema linking pipeline for a list of questions.
    few_shot_retriever: FewShotRetriever instance (built once in main, shared across all questions).
    resume: if True, load existing output_path and skip already-completed questions.
    Returns list of result dicts.
    """
    # Cache one SchemaLinker per db_id
    linkers: Dict[str, SchemaLinker] = {}

    # Load checkpoint if resuming
    if resume and output_path:
        results, errors, done_ids = _load_checkpoint(output_path)
    else:
        results, errors, done_ids = [], [], set()

    total = len(questions)
    remaining = [q for q in questions if q.get("question_id", questions.index(q)) not in done_ids]
    print(f"\nProcessing {len(remaining)}/{total} questions (skipping {len(done_ids)} already done)...")

    for i, q in enumerate(remaining):
        db_id    = q["db_id"]
        question = q["question"]
        evidence = q.get("evidence", "")
        qid      = q.get("question_id", i)

        print(f"\n[{i+1}/{len(remaining)}] Q#{qid} | db={db_id}")

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
                        help="Process all 11 databases (all questions)")
    parser.add_argument("--questions_per_db", type=int, default=None,
                        help="Limit questions per DB when using --all (default: all questions)")
    parser.add_argument("--top_k",  type=int, default=10,
                        help="FAISS top-k columns to retrieve (default: 10)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM calls (dry run: just schema linking)")
    parser.add_argument("--out",    type=str, default=None,
                        help="Output JSON path (default: results/schema_links_<db>_<ts>.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing --out checkpoint, skipping already-done questions")
    args = parser.parse_args()

    # Load questions
    if args.all:
        questions = load_questions(db_id=None, limit=args.questions_per_db)
        tag = f"all_dbs_{args.questions_per_db}per"
    else:
        questions = load_questions(db_id=args.db)
        tag = args.db

    print(f"Loaded {len(questions)} questions")

    # Output path — keep a fixed name when resuming so checkpoint loads correctly
    if args.out:
        out_path = Path(args.out)
    elif args.resume:
        # Find the most recent matching results file to resume from
        import glob as _glob
        pattern = str(RESULTS_DIR / f"schema_links_{tag}_*.json")
        matches = sorted(_glob.glob(pattern))
        if matches:
            out_path = Path(matches[-1])
            print(f"[Resume] Found checkpoint: {out_path.name}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"schema_links_{tag}_{ts}.json"
            print(f"[Resume] No checkpoint found, starting fresh → {out_path.name}")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"schema_links_{tag}_{ts}.json"
    print(f"Output → {out_path}")

    # Backend
    backend = None
    if not args.no_llm:
        from llm import make_backend
        print("Initializing HuggingFace gpt-oss-120b backend...")
        backend = make_backend("huggingface", model_id="openai/gpt-oss-120b", cache=True)
    else:
        print("Dry run mode — no LLM calls")

    # Build few-shot retriever from the BIRD training set (paper Section 4)
    # "few-shot examples taken from the train query set" — 9428 questions
    few_shot_retriever = None
    if backend is not None:
        print("Building few-shot retriever from BIRD training set...")
        train_pool = load_train_questions()
        print(f"  Loaded {len(train_pool)} training questions for few-shot pool")
        few_shot_retriever = FewShotRetriever(train_pool)
        few_shot_retriever.build()

    # Run pipeline
    t0 = time.time()
    results = run_pipeline(
        questions,
        backend=backend,
        faiss_top_k=args.top_k,
        output_path=out_path,
        few_shot_retriever=few_shot_retriever,
        resume=args.resume,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()

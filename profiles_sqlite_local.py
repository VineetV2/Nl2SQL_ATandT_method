"""
profiles_sqlite_local.py
------------------------
Build LONG profiles for every (table, column) in an SQLite database. - 1. starting file , next 2. profiles_full_local.py

Writes JSONL next to the sqlite file by default.
"""


from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sqlite3
import json


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fetchall(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = _fetchall(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
    )
    return [r[0] for r in rows]


def table_info(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    rows = _fetchall(conn, f"PRAGMA table_info({_quote_ident(table)})")
    out = []
    for cid, name, typ, notnull, dflt, pk in rows:
        out.append(
            {
                "cid": cid,
                "name": name,
                "type": typ or "",
                "notnull": bool(notnull),
                "default": dflt,
                "pk": int(pk),
            }
        )
    return out


# Kept for future AR-DB work, but not used in LONG profiles (by request).
def foreign_keys(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    rows = _fetchall(conn, f"PRAGMA foreign_key_list({_quote_ident(table)})")
    out = []
    for _id, seq, to_table, from_col, to_col, on_update, on_delete, match in rows:
        out.append(
            {
                "to_table": to_table,
                "from_col": from_col,
                "to_col": to_col,
                "on_update": on_update,
                "on_delete": on_delete,
                "match": match,
            }
        )
    return out


def _safe_count(conn: sqlite3.Connection, table: str) -> Optional[int]:
    try:
        r = _fetchall(conn, f"SELECT COUNT(*) FROM {_quote_ident(table)}")
        return int(r[0][0])
    except Exception:
        return None


def _col_stats(conn: sqlite3.Connection, table: str, col: str, limit_for_distinct: Optional[int] = None) -> Dict[str, Any]:
    qt = _quote_ident(table)
    qc = _quote_ident(col)
    stats: Dict[str, Any] = {}

    nrows = _safe_count(conn, table)
    if nrows is not None:
        stats["n_rows"] = nrows

    try:
        r = _fetchall(conn, f"SELECT COUNT(*) FROM {qt} WHERE {qc} IS NULL")
        stats["null_count"] = int(r[0][0])
    except Exception:
        pass

    try:
        if limit_for_distinct is None:
            r = _fetchall(conn, f"SELECT COUNT(DISTINCT {qc}) FROM {qt}")
        else:
            r = _fetchall(
                conn,
                f"SELECT COUNT(DISTINCT {qc}) FROM (SELECT {qc} FROM {qt} LIMIT ?)",
                (limit_for_distinct,),
            )
        stats["distinct_count"] = int(r[0][0])
    except Exception:
        pass

    try:
        r = _fetchall(conn, f"SELECT MIN({qc}), MAX({qc}) FROM {qt} WHERE {qc} IS NOT NULL")
        stats["min"] = r[0][0]
        stats["max"] = r[0][1]
    except Exception:
        pass

    return stats


def _samples(conn: sqlite3.Connection, table: str, col: str, n: int = 5) -> List[Any]:
    qt = _quote_ident(table)
    qc = _quote_ident(col)
    try:
        rows = _fetchall(conn, f"SELECT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?", (n,))
        return [r[0] for r in rows]
    except Exception:
        return []


def _top_values(conn: sqlite3.Connection, table: str, col: str, k: int = 5) -> List[Dict[str, Any]]:
    qt = _quote_ident(table)
    qc = _quote_ident(col)
    try:
        rows = _fetchall(
            conn,
            f"SELECT {qc}, COUNT(*) AS c FROM {qt} WHERE {qc} IS NOT NULL GROUP BY {qc} ORDER BY c DESC LIMIT ?",
            (k,),
        )
        return [{"value": v, "count": int(c)} for v, c in rows]
    except Exception:
        return []


def _common_prefix(values: List[str], min_coverage: float = 0.8) -> str:
    """Longest prefix shared by at least min_coverage fraction of values (paper Section 2)."""
    if not values:
        return ""
    candidate = min(values, key=len)
    for length in range(len(candidate), 0, -1):
        prefix = candidate[:length]
        if sum(1 for v in values if v.startswith(prefix)) / len(values) >= min_coverage:
            return prefix
    return ""


def _shape_stats(
    conn: sqlite3.Connection, table: str, col: str, n_samples: int = 100
) -> Dict[str, Any]:
    """Character-level shape statistics: lengths, char-class percentages, common prefix (paper Section 2)."""
    qt = _quote_ident(table)
    qc = _quote_ident(col)
    try:
        rows = _fetchall(
            conn,
            f"SELECT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?",
            (n_samples,),
        )
        values = [str(r[0]) for r in rows if r[0] is not None]
    except Exception:
        return {}

    if not values:
        return {}

    lengths = [len(v) for v in values]
    total_chars = sum(lengths)
    digit_chars = upper_chars = lower_chars = other_chars = 0

    for v in values:
        for ch in v:
            if ch.isdigit():
                digit_chars += 1
            elif ch.isupper():
                upper_chars += 1
            elif ch.islower():
                lower_chars += 1
            else:
                other_chars += 1

    shape: Dict[str, Any] = {
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }
    if total_chars > 0:
        shape["pct_digits"] = round(100 * digit_chars / total_chars, 1)
        shape["pct_upper"]  = round(100 * upper_chars  / total_chars, 1)
        shape["pct_lower"]  = round(100 * lower_chars  / total_chars, 1)
        shape["pct_other"]  = round(100 * other_chars  / total_chars, 1)

    prefix = _common_prefix(values)
    if prefix and len(prefix) >= 2:
        shape["common_prefix"] = prefix

    return shape


def _render_long_profile(row: Dict[str, Any]) -> str:
    table = row["table"]
    col = row["column"]
    decl = row.get("decl_type", "")
    pk = row.get("is_pk", False)
    stats = row.get("stats", {})
    samples = row.get("samples", [])
    topv = row.get("top_values", [])

    lines = []
    lines.append(f"Column: {table}.{col}")
    if decl:
        lines.append(f"Declared type: {decl}")
    lines.append(f"Primary key: {pk}")

    if stats:
        parts = []
        for k in ["n_rows", "null_count", "distinct_count", "min", "max"]:
            if k in stats:
                parts.append(f"{k}={stats[k]}")
        if parts:
            lines.append("Stats: " + ", ".join(parts))

    shape = row.get("shape", {})
    if shape:
        shape_parts = []
        if "avg_len" in shape:
            shape_parts.append(
                f"avg_len={shape['avg_len']}, "
                f"min_len={shape['min_len']}, "
                f"max_len={shape['max_len']}"
            )
        char_dist = []
        for k, label in [
            ("pct_digits", "digits"), ("pct_upper", "upper"),
            ("pct_lower", "lower"),   ("pct_other", "other"),
        ]:
            if k in shape:
                char_dist.append(f"{label}={shape[k]}%")
        if char_dist:
            shape_parts.append(", ".join(char_dist))
        if "common_prefix" in shape:
            shape_parts.append(f"common_prefix={repr(shape['common_prefix'])}")
        if shape_parts:
            lines.append("Shape: " + "; ".join(shape_parts))

    if topv:
        tv = ", ".join([f"{repr(x['value'])} ({x['count']})" for x in topv])
        lines.append(f"Top values: {tv}")

    if samples:
        sm = ", ".join([repr(x) for x in samples])
        lines.append(f"Samples: {sm}")

    return "\n".join(lines)


def build_long_profiles(
    sqlite_path: Path,
    db_id: Optional[str] = None,
    sample_n: int = 5,
    topk: int = 5,
    distinct_limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sqlite_path = Path(sqlite_path)
    if db_id is None:
        db_id = sqlite_path.stem

    conn = sqlite3.connect(str(sqlite_path))
    try:
        tables = list_tables(conn)
        out_rows: List[Dict[str, Any]] = []

        for t in tables:
            info = table_info(conn, t)

            for c in info:
                col = c["name"]
                is_pk = c["pk"] > 0

                stats = _col_stats(conn, t, col, limit_for_distinct=distinct_limit)
                samples = _samples(conn, t, col, n=sample_n)
                top_values = _top_values(conn, t, col, k=topk)
                shape = _shape_stats(conn, t, col)

                row = {
                    "db_id": db_id,
                    "table": t,
                    "column": col,
                    "decl_type": c.get("type", ""),
                    "notnull": bool(c.get("notnull", False)),
                    "default": c.get("default", None),
                    "is_pk": is_pk,
                    "stats": stats,
                    "samples": samples,
                    "top_values": top_values,
                    "shape": shape,
                }
                row["profile_long_en"] = _render_long_profile(row)
                out_rows.append(row)

        return out_rows
    finally:
        conn.close()


def build_and_write_long_profiles(
    sqlite_path: Path,
    output_jsonl: Optional[Path] = None,
    **kwargs,
) -> Path:
    sqlite_path = Path(sqlite_path)
    if output_jsonl is None:
        output_jsonl = sqlite_path.with_suffix(".long_profiles.jsonl")

    rows = build_long_profiles(sqlite_path, db_id=kwargs.pop("db_id", None), **kwargs)
    write_jsonl(output_jsonl, rows)
    return Path(output_jsonl)

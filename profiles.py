"""
profiles.py
-----------
Phase 1: Build column profiles for all MINIDEV databases.

Consolidates:
  - profiles_sqlite_local.py  (Phase 1a: long profiles from SQLite stats)
  - profiles_full_local.py    (Phase 1b: merge long profiles with dev-doc CSVs)
  - short_profiles_local.py   (Phase 1c: LLM one-sentence short profiles)
  - run_profiles.py           (CLI orchestrator)

Pipeline per database:
  1. Long profiles  → <db_id>.long_profiles.jsonl
  2. Full profiles  → <db_id>.full_profiles.jsonl
  3. Short profiles → <db_id>.short_profiles.jsonl  (GPT-5.2)

Usage:
  python profiles.py --db debit_card_specializing
  python profiles.py --all
  python profiles.py --all --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV" / "dev_databases"


# ══════════════════════════════════════════════════════════════
# Shared JSONL helpers
# ══════════════════════════════════════════════════════════════

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ══════════════════════════════════════════════════════════════
# Phase 1a — Long Profiles (SQLite stats)
# ══════════════════════════════════════════════════════════════

def _fetchall(conn: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = _fetchall(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;",
    )
    return [r[0] for r in rows]


def _table_info(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    rows = _fetchall(conn, f"PRAGMA table_info({_quote_ident(table)})")
    out = []
    for cid, name, typ, notnull, dflt, pk in rows:
        out.append({"cid": cid, "name": name, "type": typ or "",
                    "notnull": bool(notnull), "default": dflt, "pk": int(pk)})
    return out


def _col_stats(conn: sqlite3.Connection, table: str, col: str,
               limit_for_distinct: Optional[int] = None) -> Dict[str, Any]:
    qt, qc = _quote_ident(table), _quote_ident(col)
    stats: Dict[str, Any] = {}
    try:
        stats["n_rows"] = int(_fetchall(conn, f"SELECT COUNT(*) FROM {qt}")[0][0])
    except Exception:
        pass
    try:
        stats["null_count"] = int(_fetchall(conn, f"SELECT COUNT(*) FROM {qt} WHERE {qc} IS NULL")[0][0])
    except Exception:
        pass
    try:
        if limit_for_distinct is None:
            r = _fetchall(conn, f"SELECT COUNT(DISTINCT {qc}) FROM {qt}")
        else:
            r = _fetchall(conn, f"SELECT COUNT(DISTINCT {qc}) FROM (SELECT {qc} FROM {qt} LIMIT ?)", (limit_for_distinct,))
        stats["distinct_count"] = int(r[0][0])
    except Exception:
        pass
    try:
        r = _fetchall(conn, f"SELECT MIN({qc}), MAX({qc}) FROM {qt} WHERE {qc} IS NOT NULL")
        stats["min"], stats["max"] = r[0][0], r[0][1]
    except Exception:
        pass
    return stats


def _samples(conn: sqlite3.Connection, table: str, col: str, n: int = 5) -> List[Any]:
    qt, qc = _quote_ident(table), _quote_ident(col)
    try:
        rows = _fetchall(conn, f"SELECT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?", (n,))
        return [r[0] for r in rows]
    except Exception:
        return []


def _top_values(conn: sqlite3.Connection, table: str, col: str, k: int = 5) -> List[Dict[str, Any]]:
    qt, qc = _quote_ident(table), _quote_ident(col)
    try:
        rows = _fetchall(
            conn,
            f"SELECT {qc}, COUNT(*) AS c FROM {qt} WHERE {qc} IS NOT NULL "
            f"GROUP BY {qc} ORDER BY c DESC LIMIT ?", (k,))
        return [{"value": v, "count": int(c)} for v, c in rows]
    except Exception:
        return []


def _common_prefix(values: List[str], min_coverage: float = 0.8) -> str:
    if not values:
        return ""
    candidate = min(values, key=len)
    for length in range(len(candidate), 0, -1):
        prefix = candidate[:length]
        if sum(1 for v in values if v.startswith(prefix)) / len(values) >= min_coverage:
            return prefix
    return ""


def _shape_stats(conn: sqlite3.Connection, table: str, col: str, n_samples: int = 100) -> Dict[str, Any]:
    qt, qc = _quote_ident(table), _quote_ident(col)
    try:
        rows = _fetchall(conn, f"SELECT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?", (n_samples,))
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
            if ch.isdigit():       digit_chars += 1
            elif ch.isupper():     upper_chars += 1
            elif ch.islower():     lower_chars += 1
            else:                  other_chars += 1
    shape: Dict[str, Any] = {
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "min_len": min(lengths), "max_len": max(lengths),
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
    lines = [f"Column: {row['table']}.{row['column']}"]
    if row.get("decl_type"):
        lines.append(f"Declared type: {row['decl_type']}")
    lines.append(f"Primary key: {row.get('is_pk', False)}")
    stats = row.get("stats", {})
    if stats:
        parts = [f"{k}={stats[k]}" for k in ["n_rows", "null_count", "distinct_count", "min", "max"] if k in stats]
        if parts:
            lines.append("Stats: " + ", ".join(parts))
    shape = row.get("shape", {})
    if shape:
        shape_parts = []
        if "avg_len" in shape:
            shape_parts.append(f"avg_len={shape['avg_len']}, min_len={shape['min_len']}, max_len={shape['max_len']}")
        char_dist = [f"{lbl}={shape[k]}%" for k, lbl in [
            ("pct_digits","digits"),("pct_upper","upper"),("pct_lower","lower"),("pct_other","other")
        ] if k in shape]
        if char_dist:
            shape_parts.append(", ".join(char_dist))
        if "common_prefix" in shape:
            shape_parts.append(f"common_prefix={repr(shape['common_prefix'])}")
        if shape_parts:
            lines.append("Shape: " + "; ".join(shape_parts))
    topv = row.get("top_values", [])
    if topv:
        lines.append("Top values: " + ", ".join(f"{repr(x['value'])} ({x['count']})" for x in topv))
    samples = row.get("samples", [])
    if samples:
        lines.append("Samples: " + ", ".join(repr(x) for x in samples))
    return "\n".join(lines)


def build_long_profiles(sqlite_path: Path, db_id: Optional[str] = None,
                        sample_n: int = 5, topk: int = 5,
                        distinct_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sqlite_path = Path(sqlite_path)
    if db_id is None:
        db_id = sqlite_path.stem
    conn = sqlite3.connect(str(sqlite_path))
    try:
        out_rows: List[Dict[str, Any]] = []
        for t in _list_tables(conn):
            for c in _table_info(conn, t):
                col = c["name"]
                row = {
                    "db_id": db_id, "table": t, "column": col,
                    "decl_type": c.get("type", ""), "notnull": bool(c.get("notnull", False)),
                    "default": c.get("default"), "is_pk": c["pk"] > 0,
                    "stats": _col_stats(conn, t, col, limit_for_distinct=distinct_limit),
                    "samples": _samples(conn, t, col, n=sample_n),
                    "top_values": _top_values(conn, t, col, k=topk),
                    "shape": _shape_stats(conn, t, col),
                }
                row["profile_long_en"] = _render_long_profile(row)
                out_rows.append(row)
        return out_rows
    finally:
        conn.close()


def build_and_write_long_profiles(sqlite_path: Path,
                                  output_jsonl: Optional[Path] = None,
                                  **kwargs) -> Path:
    sqlite_path = Path(sqlite_path)
    if output_jsonl is None:
        output_jsonl = sqlite_path.with_suffix(".long_profiles.jsonl")
    rows = build_long_profiles(sqlite_path, db_id=kwargs.pop("db_id", None), **kwargs)
    _write_jsonl(output_jsonl, rows)
    return Path(output_jsonl)


# ══════════════════════════════════════════════════════════════
# Phase 1b — Full Profiles (merge long + dev-doc CSVs)
# ══════════════════════════════════════════════════════════════

def _norm_header(h: str) -> str:
    return "".join(ch.lower() for ch in (h or "").strip() if ch.isalnum() or ch == "_")


def load_dev_docs_map(desc_dir: Path, debug: bool = False) -> Dict[Tuple[str, str], Dict[str, Any]]:
    desc_dir = Path(desc_dir)
    dev_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not desc_dir.exists():
        if debug:
            print(f"[WARN] database_description dir not found: {desc_dir}")
        return dev_map

    for csv_path in sorted(desc_dir.glob("*.csv")):
        table = csv_path.stem
        try:
            fh = csv_path.open("r", encoding="utf-8", newline="")
            fh.read(1024); fh.seek(0)
        except UnicodeDecodeError:
            fh = csv_path.open("r", encoding="latin-1", newline="")
        dict_reader = csv.DictReader(fh)
        field_norm_map = {_norm_header(fn): fn for fn in (dict_reader.fieldnames or [])}

        col_field_norm = next((c for c in [
            "original_column_name","originalcolumnname","column_name",
            "columnname","field_name","fieldname","column"
        ] if c in field_norm_map), None)
        if col_field_norm is None:
            if debug:
                print(f"[WARN] {csv_path.name}: could not find column name header.")
            continue
        col_key = field_norm_map[col_field_norm]

        desc_key = field_norm_map.get(next((c for c in ["column_description","columndescription","description","column_desc","columndesc"] if c in field_norm_map), ""))
        fmt_key  = field_norm_map.get(next((c for c in ["data_format","dataformat","format","datatype"] if c in field_norm_map), ""))
        val_key  = field_norm_map.get(next((c for c in ["value_description","valuedescription","values","value_desc","valuedesc"] if c in field_norm_map), ""))

        for raw in dict_reader:
            colname = (raw.get(col_key, "") or "").strip()
            if not colname:
                continue
            dev_map[(table, colname)] = {
                "table": table, "column": colname,
                "column_description": (raw.get(desc_key, "") or "").strip() if desc_key else "",
                "data_format":        (raw.get(fmt_key,  "") or "").strip() if fmt_key  else "",
                "value_description":  (raw.get(val_key,  "") or "").strip() if val_key  else "",
            }
    return dev_map


def _render_full_profile(long_row: Dict[str, Any], dev: Optional[Dict[str, Any]]) -> str:
    parts = ["[PROFILE]\n" + (long_row.get("profile_long_en", "") or "").strip()]
    if dev:
        dev_lines = []
        if dev.get("column_description"):
            dev_lines.append(f"Description: {dev['column_description']}")
        if dev.get("data_format"):
            dev_lines.append(f"Data format: {dev['data_format']}")
        if dev.get("value_description"):
            dev_lines.append(f"Values: {dev['value_description']}")
        if dev_lines:
            parts.append("[DEV DOC]\n" + "\n".join(dev_lines))
    return "\n\n".join(parts).strip()


def build_full_profiles(long_profiles_jsonl: Path, desc_dir: Path,
                        debug: bool = False) -> List[Dict[str, Any]]:
    long_rows = _read_jsonl(long_profiles_jsonl)
    dev_map = load_dev_docs_map(desc_dir, debug=debug)
    out = []
    matched = 0
    for r in long_rows:
        dev = dev_map.get((r.get("table", ""), r.get("column", "")))
        if dev:
            matched += 1
        rr = dict(r)
        rr["dev_doc"] = dev
        rr["profile_full_en"] = _render_full_profile(r, dev)
        out.append(rr)
    if debug:
        print(f"[INFO] Full profile merge: {matched}/{len(long_rows)} columns matched dev docs.")
    return out


def build_and_write_full_profiles(long_profiles_jsonl: Path, desc_dir: Path,
                                  output_jsonl: Optional[Path] = None,
                                  debug: bool = False) -> Path:
    long_profiles_jsonl = Path(long_profiles_jsonl)
    if output_jsonl is None:
        output_jsonl = long_profiles_jsonl.with_name(
            long_profiles_jsonl.name.replace(".long_profiles.jsonl", ".full_profiles.jsonl")
        )
    rows = build_full_profiles(long_profiles_jsonl, desc_dir, debug=debug)
    _write_jsonl(output_jsonl, rows)
    return Path(output_jsonl)


# ══════════════════════════════════════════════════════════════
# Phase 1c — Short Profiles (LLM one-sentence summaries)
# ══════════════════════════════════════════════════════════════

_SYS_SHORT = (
    "You are documenting database columns for a text-to-SQL system.\n"
    "Return exactly ONE sentence (<=25 words) describing the column and obvious value format.\n"
    "Do NOT include markdown fences, code blocks, or extra commentary."
)
_WS = re.compile(r"\s+")


def _postprocess_one_sentence(text: str) -> str:
    text = (text or "").strip().strip("`").replace("```", " ").replace("\n", " ")
    text = _WS.sub(" ", text).strip()
    words = text.split()
    if len(words) > 25:
        text = " ".join(words[:25]).rstrip(",.") + "."
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _build_short_prompt_row(full_row: Dict[str, Any], max_profile_chars: int = 3500) -> Dict[str, Any]:
    db_id  = full_row.get("db_id", "")
    table  = full_row.get("table", "")
    column = full_row.get("column", "")
    full_txt = (full_row.get("profile_full_en") or "").strip()
    if len(full_txt) > max_profile_chars:
        full_txt = full_txt[:max_profile_chars].rstrip() + "\n[TRUNCATED]"

    stats  = full_row.get("stats", {}) or {}
    topv   = full_row.get("top_values", []) or []
    samples = full_row.get("samples", []) or []
    dev    = full_row.get("dev_doc") or {}

    header_lines = [f"DB: {db_id}", f"Column: {table}.{column}"]
    if full_row.get("decl_type"):
        header_lines.append(f"Declared type: {full_row['decl_type']}")
    if stats:
        keep = [f"{k}={stats[k]}" for k in ("n_rows","null_count","distinct_count","min","max") if k in stats]
        if keep:
            header_lines.append("Stats: " + ", ".join(keep))
    if topv:
        header_lines.append("Top values: " + ", ".join(f"{repr(x.get('value'))} ({x.get('count')})" for x in topv[:5]))
    if samples:
        header_lines.append("Samples: " + ", ".join(repr(x) for x in samples[:5]))
    if isinstance(dev, dict):
        if dev.get("column_description"):
            header_lines.append("Dev description: " + dev["column_description"])
        if dev.get("data_format"):
            header_lines.append("Dev format: " + dev["data_format"])
        if dev.get("value_description"):
            header_lines.append("Dev values: " + dev["value_description"])

    user = (
        "Create a short column profile.\n\n"
        + "\n".join(header_lines)
        + "\n\nFULL PROFILE CONTEXT:\n" + full_txt
        + "\n\nReturn ONLY the one-sentence short profile."
    )
    return {"db_id": db_id, "table": table, "column": column, "system": _SYS_SHORT, "user": user}


def build_short_profile_prompts(full_profiles_jsonl: Path,
                                output_jsonl: Optional[Path] = None,
                                max_profile_chars: int = 3500) -> Path:
    full_profiles_jsonl = Path(full_profiles_jsonl)
    rows = _read_jsonl(full_profiles_jsonl)
    if output_jsonl is None:
        db_id = rows[0].get("db_id", full_profiles_jsonl.stem.split(".")[0]) if rows else full_profiles_jsonl.stem
        output_jsonl = full_profiles_jsonl.with_name(f"{db_id}.short_profile_prompts.jsonl")
    out_rows = [_build_short_prompt_row(r, max_profile_chars=max_profile_chars) for r in rows]
    _write_jsonl(output_jsonl, out_rows)
    return Path(output_jsonl)


def generate_short_profiles(prompts_jsonl: Path, backend,
                            output_jsonl: Optional[Path] = None,
                            max_new_tokens: int = 64,
                            overwrite: bool = False,
                            gen_kwargs: Optional[Dict[str, Any]] = None) -> Path:
    prompts_jsonl = Path(prompts_jsonl)
    rows = _read_jsonl(prompts_jsonl)
    if output_jsonl is None:
        db_id = rows[0].get("db_id", prompts_jsonl.stem.split(".")[0]) if rows else prompts_jsonl.stem
        output_jsonl = prompts_jsonl.with_name(f"{db_id}.short_profiles.jsonl")

    existing: Dict[Tuple[str, str, str], Any] = {}
    if not overwrite and Path(output_jsonl).exists():
        for r in _read_jsonl(output_jsonl):
            existing[(r.get("db_id",""), r.get("table",""), r.get("column",""))] = r

    out_rows: List[Dict[str, Any]] = []
    gen_kwargs = gen_kwargs or {}
    for r in rows:
        key = (r.get("db_id",""), r.get("table",""), r.get("column",""))
        if not overwrite and key in existing:
            continue
        messages = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        meta = backend.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        short = _postprocess_one_sentence((meta.get("text") or "").strip())
        out_rows.append({
            "db_id": r.get("db_id",""), "table": r.get("table",""), "column": r.get("column",""),
            "short_profile_en": short,
            "extracted_final": (meta.get("text") or "").strip(),
            "raw_model_output": (meta.get("raw") or "").strip(),
            "thoughts": (meta.get("thoughts") or "").strip(),
            "backend": backend.__class__.__name__, "max_new_tokens": max_new_tokens,
        })
        if len(out_rows) >= 50:
            _append_jsonl(output_jsonl, out_rows)
            out_rows = []
    if out_rows:
        _append_jsonl(output_jsonl, out_rows)
    return Path(output_jsonl)


# ══════════════════════════════════════════════════════════════
# CLI Orchestrator (was run_profiles.py)
# ══════════════════════════════════════════════════════════════

def run_for_db(db_id: str, backend, overwrite: bool = False) -> None:
    db_dir = MINIDEV_ROOT / db_id
    sqlite_path = db_dir / f"{db_id}.sqlite"
    desc_dir    = db_dir / "database_description"

    if not sqlite_path.exists():
        print(f"[SKIP] SQLite not found: {sqlite_path}")
        return

    print(f"\n{'='*60}\nProcessing: {db_id}\n{'='*60}")

    # Step 1: Long profiles
    long_path = db_dir / f"{db_id}.long_profiles.jsonl"
    if not long_path.exists() or overwrite:
        long_path = build_and_write_long_profiles(sqlite_path, db_id=db_id, sample_n=5, topk=5)
        print(f"  [1/3] Long profiles written: {long_path.name}")
    else:
        print(f"  [1/3] Long profiles already exist: {long_path.name}")

    # Step 2: Full profiles (merge with dev docs)
    full_path = db_dir / f"{db_id}.full_profiles.jsonl"
    if not full_path.exists() or overwrite:
        full_path = build_and_write_full_profiles(long_path, desc_dir, debug=True)
        print(f"  [2/3] Full profiles written: {full_path.name}")
    else:
        print(f"  [2/3] Full profiles already exist: {full_path.name}")

    # Step 3: Short profiles via LLM
    short_path = db_dir / f"{db_id}.short_profiles.jsonl"
    if not short_path.exists() or overwrite:
        prompts_path = build_short_profile_prompts(full_path, max_profile_chars=3500)
        print(f"  [3/3] Generating short profiles via GPT-5.2...")
        short_path = generate_short_profiles(
            prompts_path, backend=backend,
            max_new_tokens=64, overwrite=overwrite, gen_kwargs={"temperature": 0},
        )
        print(f"  [3/3] Short profiles written: {short_path.name}")
    else:
        print(f"  [3/3] Short profiles already exist: {short_path.name}")

    print(f"  Done: {db_id}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Build column profiles")
    parser.add_argument("--db",       type=str, default="debit_card_specializing")
    parser.add_argument("--all",      action="store_true", help="Process all 11 databases")
    parser.add_argument("--overwrite",action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    from llm import make_backend
    print("Initializing OpenAI GPT-5.2 backend...")
    backend = make_backend("openai", model_id="gpt-5.2", cache=True)

    if args.all:
        databases = sorted(d.name for d in MINIDEV_ROOT.iterdir() if d.is_dir())
        print(f"Processing all {len(databases)} databases: {databases}")
        for db_id in databases:
            run_for_db(db_id, backend, overwrite=args.overwrite)
    else:
        run_for_db(args.db, backend, overwrite=args.overwrite)

    print("\nAll profiles generated!")


if __name__ == "__main__":
    main()

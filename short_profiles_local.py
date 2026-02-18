"""
short_profiles_local.py - 3. comes after full profile, next 4, dep tree
-----------------------
Create SHORT profile prompts from FULL profiles and call an LLM backend to generate
one-sentence short profiles aligned to each (db_id, table, column).

Artifacts written next to the input .full_profiles.jsonl by default:
  - <db_id>.short_profile_prompts.jsonl
  - <db_id>.short_profiles.jsonl
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import re

from llm_backends_local import LLMBackend

SYS_SHORT = (
    "You are documenting database columns for a text-to-SQL system.\n"
    "Return exactly ONE sentence (<=25 words) describing the column and obvious value format.\n"
    "Do NOT include markdown fences, code blocks, or extra commentary."
)

_WS = re.compile(r"\s+")


def _postprocess_one_sentence(text: str) -> str:
    """Normalize down to one crisp sentence (<=25 words)."""
    text = (text or "").strip().strip("`").replace("```", " ").replace("\n", " ")
    text = _WS.sub(" ", text).strip()
    words = text.split()
    if len(words) > 25:
        text = " ".join(words[:25]).rstrip(",.") + "."
    if text and text[-1] not in ".!?":
        text += "."
    return text


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_short_profile_prompt_row(full_row: Dict[str, Any],
                                   max_profile_chars: int = 3500) -> Dict[str, Any]:
    """
    Build a prompt row from a FULL profile row.
    """
    db_id = full_row.get("db_id", "")
    table = full_row.get("table", "")
    column = full_row.get("column", "")

    full_txt = (full_row.get("profile_full_en") or "").strip()
    if max_profile_chars and len(full_txt) > max_profile_chars:
        full_txt = full_txt[:max_profile_chars].rstrip() + "\n[TRUNCATED]"

    decl = full_row.get("decl_type", "")
    stats = full_row.get("stats", {}) or {}
    samples = full_row.get("samples", []) or []
    topv = full_row.get("top_values", []) or []
    dev = full_row.get("dev_doc") or {}

    header_lines = [
        f"DB: {db_id}",
        f"Column: {table}.{column}",
    ]
    if decl:
        header_lines.append(f"Declared type: {decl}")
    if stats:
        keep = []
        for k in ("n_rows", "null_count", "distinct_count", "min", "max"):
            if k in stats:
                keep.append(f"{k}={stats[k]}")
        if keep:
            header_lines.append("Stats: " + ", ".join(keep))
    if topv:
        tv = ", ".join([f"{repr(x.get('value'))} ({x.get('count')})" for x in topv[:5]])
        header_lines.append("Top values: " + tv)
    if samples:
        header_lines.append("Samples: " + ", ".join([repr(x) for x in samples[:5]]))
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
        + "\n\nFULL PROFILE CONTEXT:\n"
        + full_txt
        + "\n\nReturn ONLY the one-sentence short profile."
    )

    return {
        "db_id": db_id,
        "table": table,
        "column": column,
        "system": SYS_SHORT,
        "user": user,
        "source_full_profile_chars": len(full_row.get("profile_full_en") or ""),
        "prompt_profile_chars": len(full_txt),
    }


def build_short_profile_prompts(full_profiles_jsonl: Path,
                                output_jsonl: Optional[Path] = None,
                                max_profile_chars: int = 3500) -> Path:
    """
    Read FULL profiles JSONL, write prompt rows JSONL.
    """
    full_profiles_jsonl = Path(full_profiles_jsonl)
    rows = read_jsonl(full_profiles_jsonl)

    if output_jsonl is None:
        db_id = rows[0].get("db_id", full_profiles_jsonl.stem.split(".")[0]) if rows else full_profiles_jsonl.stem
        output_jsonl = full_profiles_jsonl.with_name(f"{db_id}.short_profile_prompts.jsonl")

    out_rows = [build_short_profile_prompt_row(r, max_profile_chars=max_profile_chars) for r in rows]
    write_jsonl(output_jsonl, out_rows)
    return Path(output_jsonl)


def _load_existing_short_map(short_profiles_jsonl: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    m: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not Path(short_profiles_jsonl).exists():
        return m
    for r in read_jsonl(short_profiles_jsonl):
        key = (r.get("db_id", ""), r.get("table", ""), r.get("column", ""))
        m[key] = r
    return m


def generate_short_profiles(prompts_jsonl: Path,
                            backend: LLMBackend,
                            output_jsonl: Optional[Path] = None,
                            max_new_tokens: int = 64,
                            overwrite: bool = False,
                            gen_kwargs: Optional[Dict[str, Any]] = None) -> Path:
    """
    Call backend.generate_with_meta() for each prompt row and write short profiles JSONL.
    Notebook-friendly (no CLI).
    """
    prompts_jsonl = Path(prompts_jsonl)
    rows = read_jsonl(prompts_jsonl)

    if output_jsonl is None:
        db_id = rows[0].get("db_id", prompts_jsonl.stem.split(".")[0]) if rows else prompts_jsonl.stem
        output_jsonl = prompts_jsonl.with_name(f"{db_id}.short_profiles.jsonl")

    existing = {} if overwrite else _load_existing_short_map(output_jsonl)

    out_rows: List[Dict[str, Any]] = []
    gen_kwargs = gen_kwargs or {}

    for r in rows:
        key = (r.get("db_id", ""), r.get("table", ""), r.get("column", ""))
        if (not overwrite) and key in existing:
            continue

        messages = [
            {"role": "system", "content": r["system"]},
            {"role": "user", "content": r["user"]},
        ]

        meta = backend.generate_with_meta(messages, max_new_tokens=max_new_tokens, **gen_kwargs)
        final_text = (meta.get("text") or "").strip()
        raw_out = (meta.get("raw") or "").strip()
        thoughts = (meta.get("thoughts") or "").strip()

        short = _postprocess_one_sentence(final_text)

        out_rows.append({
            "db_id": r.get("db_id", ""),
            "table": r.get("table", ""),
            "column": r.get("column", ""),
            "short_profile_en": short,

            # Helpful for debugging:
            "extracted_final": final_text,
            "raw_model_output": raw_out,
            "thoughts": thoughts,

            "backend": backend.__class__.__name__,
            "max_new_tokens": max_new_tokens,
        })

        # flush in chunks (safer in long runs)
        if len(out_rows) >= 50:
            append_jsonl(output_jsonl, out_rows)
            out_rows = []

    if out_rows:
        append_jsonl(output_jsonl, out_rows)

    return Path(output_jsonl)

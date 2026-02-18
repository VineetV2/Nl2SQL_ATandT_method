"""
profiles_full_local.py - 2. comes after proflie_sqlite (long profile), next 3. short_profile_local
----------------------
Merge LONG profiles with developer metadata CSVs to create FULL profiles.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import csv
import json


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _norm_header(h: str) -> str:
    return "".join(ch.lower() for ch in (h or "").strip() if ch.isalnum() or ch == "_")


def _norm_colname(x: str) -> str:
    return (x or "").strip()


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
            fh.read(1024); fh.seek(0)  # probe for encoding errors
        except UnicodeDecodeError:
            fh = csv_path.open("r", encoding="latin-1", newline="")
        dict_reader = csv.DictReader(fh)
        field_norm_map = {_norm_header(fn): fn for fn in (dict_reader.fieldnames or [])}

        # robust detection of the key field
        col_field_norm_candidates = [
            "original_column_name",
            "originalcolumnname",
            "column_name",
            "columnname",
            "field_name",
            "fieldname",
            "column",
        ]
        col_field_norm = next((c for c in col_field_norm_candidates if c in field_norm_map), None)
        if col_field_norm is None:
            if debug:
                print(f"[WARN] {csv_path.name}: could not find original_column_name-like header. Headers={dict_reader.fieldnames}")
            continue
        col_key = field_norm_map[col_field_norm]

        # optional fields
        desc_field_norm = next((c for c in ["column_description","columndescription","description","column_desc","columndesc"] if c in field_norm_map), None)
        fmt_field_norm  = next((c for c in ["data_format","dataformat","format","datatype"] if c in field_norm_map), None)
        val_field_norm  = next((c for c in ["value_description","valuedescription","values","value_desc","valuedesc"] if c in field_norm_map), None)

        desc_key = field_norm_map.get(desc_field_norm) if desc_field_norm else None
        fmt_key  = field_norm_map.get(fmt_field_norm) if fmt_field_norm else None
        val_key  = field_norm_map.get(val_field_norm) if val_field_norm else None

        for raw in dict_reader:
            colname = _norm_colname(raw.get(col_key, "") or "")
            if not colname:
                continue

            dev = {
                "table": table,
                "column": colname,
                "column_description": (raw.get(desc_key, "") or "").strip() if desc_key else "",
                "data_format": (raw.get(fmt_key, "") or "").strip() if fmt_key else "",
                "value_description": (raw.get(val_key, "") or "").strip() if val_key else "",
                "raw_row": raw,
            }
            dev_map[(table, colname)] = dev

    return dev_map


def _render_full_profile(long_row: Dict[str, Any], dev: Optional[Dict[str, Any]]) -> str:
    parts = []
    parts.append("[PROFILE]\n" + (long_row.get("profile_long_en", "") or "").strip())

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


def build_full_profiles(
    long_profiles_jsonl: Path,
    desc_dir: Path,
    debug: bool = False
) -> List[Dict[str, Any]]:
    long_rows = read_jsonl(long_profiles_jsonl)
    dev_map = load_dev_docs_map(desc_dir, debug=debug)

    out: List[Dict[str, Any]] = []
    matched = 0
    for r in long_rows:
        key = (r.get("table",""), r.get("column",""))
        dev = dev_map.get(key)
        if dev:
            matched += 1

        rr = dict(r)
        rr["dev_doc"] = dev if dev else None
        rr["profile_full_en"] = _render_full_profile(r, dev)
        out.append(rr)

    if debug:
        print(f"[INFO] Full profile merge matched {matched}/{len(long_rows)} columns with dev docs.")
    return out


def build_and_write_full_profiles(
    long_profiles_jsonl: Path,
    desc_dir: Path,
    output_jsonl: Optional[Path] = None,
    debug: bool = False
) -> Path:
    long_profiles_jsonl = Path(long_profiles_jsonl)
    if output_jsonl is None:
        output_jsonl = long_profiles_jsonl.with_name(long_profiles_jsonl.name.replace(".long_profiles.jsonl", ".full_profiles.jsonl"))

    rows = build_full_profiles(long_profiles_jsonl, desc_dir, debug=debug)
    write_jsonl(output_jsonl, rows)
    return Path(output_jsonl)

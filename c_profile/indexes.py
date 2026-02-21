"""
indexes.py
----------
Phase 2: Build FAISS (semantic) + LSH (literal value) search indexes.

  FAISS index  — on long profile text -> semantic column retrieval
  LSH index    — on up to N=10,000 distinct column values -> literal matching

Usage:
  python indexes.py --db debit_card_specializing
  python indexes.py --all
  python indexes.py --all --overwrite

Outputs (saved next to profile files):
  <db_id>.faiss            FAISS binary index
  <db_id>.faiss_meta.json  column metadata for FAISS
  <db_id>.lsh.pkl          MinHash LSH pickle (required by datasketch)
  <db_id>.lsh_index.json   exact value index for LSH
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Set

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV" / "dev_databases"


# Embeddings (OpenAI text-embedding-3-small)

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small",
                   max_chars: int = 24000) -> List[List[float]]:
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")
    texts = [t[:max_chars] for t in texts]
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i: i + 100]
        response = openai.Embedding.create(model=model, input=batch)
        all_embeddings.extend(
            item["embedding"] for item in sorted(response["data"], key=lambda x: x["index"])
        )
    return all_embeddings


# FAISS Index

def build_faiss_index(db_id: str, db_dir: Path) -> Path:
    import faiss
    import numpy as np
    long_path = db_dir / f"{db_id}.long_profiles.jsonl"
    if not long_path.exists():
        raise FileNotFoundError(f"Long profiles not found: {long_path}")
    with open(long_path) as f:
        profiles = [json.loads(l) for l in f if l.strip()]
    texts = [r["profile_long_en"] for r in profiles]
    meta  = [{"db_id": r["db_id"], "table": r["table"], "column": r["column"],
               "decl_type": r.get("decl_type",""), "is_pk": r.get("is_pk", False),
               "profile_long_en": r["profile_long_en"]} for r in profiles]
    print(f"  [FAISS] Embedding {len(texts)} columns for {db_id}...")
    embeddings = get_embeddings(texts)
    dim  = len(embeddings[0])
    vecs = __import__('numpy').array(embeddings, dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss_path = db_dir / f"{db_id}.faiss"
    meta_path  = db_dir / f"{db_id}.faiss_meta.json"
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [FAISS] Saved: {faiss_path.name} ({len(texts)} vectors, dim={dim})")
    return faiss_path


def query_faiss(question: str, db_id: str, db_dir: Path, top_k: int = 10) -> List[Dict[str, Any]]:
    import faiss
    import numpy as np
    faiss_path = db_dir / f"{db_id}.faiss"
    meta_path  = db_dir / f"{db_id}.faiss_meta.json"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)
    q_emb = get_embeddings([question])[0]
    q_vec = __import__('numpy').array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            r = dict(meta[idx])
            r["faiss_score"] = float(score)
            results.append(r)
    return results


# LSH Index

def _normalize_value(v: Any) -> List[str]:
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    variants = [s, s.lower(), s.upper(), s.title()]
    no_punct = re.sub(r"[^\w\s]", "", s)
    if no_punct != s:
        variants.append(no_punct)
    return list(dict.fromkeys(variants))


def _char_shingles(text: str, k: int = 3) -> Set[str]:
    text = text.lower().strip()
    if not text:
        return set()
    if len(text) <= k:
        return {text}
    return {text[i: i + k] for i in range(len(text) - k + 1)}


def _fetch_distinct_values(sqlite_path: Path, table: str, col: str, n: int = 10000) -> List[str]:
    qt = '"' + table.replace('"', '""') + '"'
    qc = '"' + col.replace('"', '""') + '"'
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur  = conn.execute(f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?", (n,))
        rows = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None]
        conn.close()
        return rows
    except Exception:
        return []


def build_lsh_index(db_id: str, db_dir: Path, n_values: int = 10000) -> Path:
    from datasketch import MinHash, MinHashLSH
    long_path   = db_dir / f"{db_id}.long_profiles.jsonl"
    sqlite_path = db_dir / f"{db_id}.sqlite"
    if not long_path.exists():
        raise FileNotFoundError(f"Long profiles not found: {long_path}")
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite not found: {sqlite_path}")
    with open(long_path) as f:
        profiles = [json.loads(l) for l in f if l.strip()]
    num_perm = 128
    lsh = MinHashLSH(threshold=0.3, num_perm=num_perm)
    exact_index: Dict[str, Dict[str, Any]] = {}
    total_values = 0
    for r in profiles:
        table, col = r["table"], r["column"]
        key = f"{table}.{col}"
        raw_values = _fetch_distinct_values(sqlite_path, table, col, n=n_values)
        all_values: set = set()
        for v in raw_values:
            for variant in _normalize_value(v):
                all_values.add(variant)
        total_values += len(raw_values)
        exact_index[key] = {"db_id": db_id, "table": table, "column": col,
                             "decl_type": r.get("decl_type",""), "values": list(all_values)}
        if not all_values:
            continue
        m = MinHash(num_perm=num_perm)
        for val in all_values:
            for shingle in _char_shingles(val):
                m.update(shingle.encode("utf8"))
        try:
            lsh.insert(key, m)
        except Exception:
            pass
    lsh_pickle_path = db_dir / f"{db_id}.lsh.pkl"
    lsh_exact_path  = db_dir / f"{db_id}.lsh_index.json"
    with open(lsh_pickle_path, "wb") as f:
        pickle.dump(lsh, f)
    with open(lsh_exact_path, "w") as f:
        json.dump(exact_index, f, indent=2)
    print(f"  [LSH]   Saved: {lsh_pickle_path.name} ({len(profiles)} cols, ~{total_values} values)")
    return lsh_pickle_path


def query_lsh(literal: str, db_id: str, db_dir: Path) -> List[Dict[str, Any]]:
    from datasketch import MinHash
    lsh_pickle_path = db_dir / f"{db_id}.lsh.pkl"
    lsh_exact_path  = db_dir / f"{db_id}.lsh_index.json"
    if not lsh_pickle_path.exists():
        raise FileNotFoundError(f"LSH index not found: {lsh_pickle_path}")
    with open(lsh_pickle_path, "rb") as f:
        lsh = pickle.load(f)
    with open(lsh_exact_path) as f:
        exact_index = json.load(f)
    variants = _normalize_value(literal)
    # Exact match first
    exact_matches = []
    for key, col_data in exact_index.items():
        matched = [v for v in variants if v in set(col_data["values"])]
        if matched:
            exact_matches.append({"table": col_data["table"], "column": col_data["column"],
                                   "match_type": "exact", "matched_values": matched})
    if exact_matches:
        return exact_matches
    # Approximate LSH
    m = MinHash(num_perm=128)
    for v in variants:
        for shingle in _char_shingles(v):
            m.update(shingle.encode("utf8"))
    return [{"table": exact_index.get(k, {}).get("table",""),
             "column": exact_index.get(k, {}).get("column",""),
             "match_type": "approximate", "matched_values": variants}
            for k in lsh.query(m)]


# CLI

def build_indexes_for_db(db_id: str, overwrite: bool = False) -> None:
    db_dir = MINIDEV_ROOT / db_id
    print(f"\n{'='*60}\nBuilding indexes: {db_id}\n{'='*60}")
    faiss_path = db_dir / f"{db_id}.faiss"
    lsh_path   = db_dir / f"{db_id}.lsh.pkl"
    if not faiss_path.exists() or overwrite:
        build_faiss_index(db_id, db_dir)
    else:
        print(f"  [FAISS] Already exists (use --overwrite to rebuild)")
    if not lsh_path.exists() or overwrite:
        build_lsh_index(db_id, db_dir)
    else:
        print(f"  [LSH]   Already exists (use --overwrite to rebuild)")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Build FAISS + LSH indexes")
    parser.add_argument("--db",        type=str, default="debit_card_specializing")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.all:
        databases = sorted(d.name for d in MINIDEV_ROOT.iterdir() if d.is_dir())
        print(f"Building indexes for all {len(databases)} databases...")
        for db_id in databases:
            build_indexes_for_db(db_id, overwrite=args.overwrite)
    else:
        build_indexes_for_db(args.db, overwrite=args.overwrite)
    print("\nAll indexes built!")


if __name__ == "__main__":
    main()

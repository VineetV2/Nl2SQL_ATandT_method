"""
indexes.py
----------
Phase 2: Build search indexes from profiles.

1. FAISS index  - on long profile text → semantic similarity search
                  (answers: "which columns are semantically relevant to this question?")

2. LSH index    - on up to N=10000 distinct column values → literal value matching
                  (answers: "which columns contain the value 'CZK' or '2013'?")
                  Per paper Section 3: "Fetch N distinct values of the f, or as many
                  distinct values as exist... For the BIRD benchmark, we used N=10000."
                  Values are fetched directly from SQLite, NOT from the topk=5 profiles.

Usage:
  # Single database
  python build_indexes.py --db debit_card_specializing

  # All 11 databases
  python build_indexes.py --all

Outputs (saved next to profile files):
  <db_id>.faiss           ← FAISS binary index
  <db_id>.faiss_meta.json ← column metadata for FAISS (table, column, profile text)
  <db_id>.lsh_index.json  ← LSH index: column values for literal matching
  <db_id>.lsh_meta.json   ← column metadata for LSH

Install requirements:
  pip install faiss-cpu openai datasketch
"""

from __future__ import annotations

import json
import os
import argparse
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV" / "dev_databases"


# ─────────────────────────────────────────────
# Embedding via OpenAI (text-embedding-3-small)
# ─────────────────────────────────────────────

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Call OpenAI embeddings API for a list of texts."""
    import openai
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

    # Batch in chunks of 100 (API limit)
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = openai.Embedding.create(model=model, input=batch)
        batch_embeddings = [item["embedding"] for item in sorted(response["data"], key=lambda x: x["index"])]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


# ─────────────────────────────────────────────
# FAISS Index Builder
# ─────────────────────────────────────────────

def build_faiss_index(db_id: str, db_dir: Path) -> Path:
    """
    Build FAISS index from long profile text of every column.
    Each vector = embedding of the long profile description.
    """
    import faiss
    import numpy as np

    long_path = db_dir / f"{db_id}.long_profiles.jsonl"
    if not long_path.exists():
        raise FileNotFoundError(f"Long profiles not found: {long_path}")

    # Load profiles
    with open(long_path) as f:
        profiles = [json.loads(l) for l in f if l.strip()]

    # Build texts and metadata
    texts = [r["profile_long_en"] for r in profiles]
    meta = [
        {
            "db_id": r["db_id"],
            "table": r["table"],
            "column": r["column"],
            "decl_type": r.get("decl_type", ""),
            "is_pk": r.get("is_pk", False),
            "profile_long_en": r["profile_long_en"],
        }
        for r in profiles
    ]

    print(f"  [FAISS] Embedding {len(texts)} columns for {db_id}...")
    embeddings = get_embeddings(texts)

    # Build FAISS index (inner product = cosine similarity on normalized vectors)
    dim = len(embeddings[0])
    vecs = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vecs)  # normalize for cosine similarity

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Save index + metadata
    faiss_path = db_dir / f"{db_id}.faiss"
    meta_path = db_dir / f"{db_id}.faiss_meta.json"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [FAISS] Index saved: {faiss_path.name} ({len(texts)} vectors, dim={dim})")
    return faiss_path


# ─────────────────────────────────────────────
# LSH Index Builder
# ─────────────────────────────────────────────

def _kshingles(s: str, k: int = 3) -> List[str]:
    """
    Decompose a string into character k-shingles (overlapping n-grams).

    Paper footnote 6: uses kshingle (pypi.org/project/kshingle/0.1.0/)
    for character-level shingling before MinHash LSH.
    E.g. _kshingles("hello", 3) → ["hel", "ell", "llo"]
    """
    if len(s) < k:
        return [s] if s else []
    return [s[i:i + k] for i in range(len(s) - k + 1)]


def _normalize_value(v: Any) -> List[str]:
    """Generate string variants of a value for matching."""
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    variants = [s, s.lower(), s.upper(), s.title()]
    # Remove punctuation variant
    no_punct = re.sub(r"[^\w\s]", "", s)
    if no_punct != s:
        variants.append(no_punct)
    return list(dict.fromkeys(variants))  # deduplicate


def _fetch_distinct_values_from_db(
    sqlite_path: Path, table: str, col: str, n: int = 10000
) -> List[str]:
    """
    Fetch up to N distinct non-null string values for a column directly from SQLite.

    Paper Section 3: "Fetch N distinct values of the f, or as many distinct values
    as exist." N=10000 for BIRD benchmark.
    This is intentionally separate from the topk=5 used in profiling.
    """
    import sqlite3
    qt = '"' + table.replace('"', '""') + '"'
    qc = '"' + col.replace('"', '""') + '"'
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.execute(
            f"SELECT DISTINCT {qc} FROM {qt} WHERE {qc} IS NOT NULL LIMIT ?", (n,)
        )
        rows = [str(r[0]).strip() for r in cur.fetchall() if r[0] is not None]
        conn.close()
        return rows
    except Exception:
        return []


def build_lsh_index(db_id: str, db_dir: Path, n_values: int = 10000) -> Path:
    """
    Build LSH index by fetching up to N=10000 distinct values per column from SQLite.

    Per paper Section 3: values are fetched directly from the database, NOT limited
    to the topk=5 values stored during profiling.

    Stores: for each (table, column) → set of value variants
    Used for: "does column X contain literal value Y from the question?"

    Uses MinHash LSH from datasketch for approximate matching.
    Also stores exact value sets for direct lookup.
    """
    import sqlite3
    from datasketch import MinHash, MinHashLSH

    long_path = db_dir / f"{db_id}.long_profiles.jsonl"
    if not long_path.exists():
        raise FileNotFoundError(f"Long profiles not found: {long_path}")

    sqlite_path = db_dir / f"{db_id}.sqlite"
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite not found: {sqlite_path}")

    with open(long_path) as f:
        profiles = [json.loads(l) for l in f if l.strip()]

    # Build LSH index
    num_perm = 128  # number of permutations (higher = more accurate)
    lsh = MinHashLSH(threshold=0.3, num_perm=num_perm)

    # Also store exact value sets for direct lookup
    exact_index: Dict[str, Dict[str, Any]] = {}

    total_values = 0
    for r in profiles:
        table = r["table"]
        col = r["column"]
        key = f"{table}.{col}"

        # Fetch up to N=10000 distinct values directly from SQLite (paper Section 3)
        raw_values = _fetch_distinct_values_from_db(sqlite_path, table, col, n=n_values)

        # Build normalized variants for each raw value
        all_values: set = set()
        for v in raw_values:
            for variant in _normalize_value(v):
                all_values.add(variant)

        total_values += len(raw_values)

        # Store exact values
        exact_index[key] = {
            "db_id": db_id,
            "table": table,
            "column": col,
            "decl_type": r.get("decl_type", ""),
            "values": list(all_values),
        }

        if not all_values:
            continue

        # Build MinHash for this column using character k-shingles (paper footnote 6)
        m = MinHash(num_perm=num_perm)
        for val in all_values:
            for shingle in _kshingles(val.lower(), k=3):
                m.update(shingle.encode("utf8"))

        try:
            lsh.insert(key, m)
        except Exception:
            pass  # skip if duplicate key

    # Save LSH object (pickle) and exact index (JSON)
    lsh_pickle_path = db_dir / f"{db_id}.lsh.pkl"
    lsh_exact_path = db_dir / f"{db_id}.lsh_index.json"

    with open(lsh_pickle_path, "wb") as f:
        pickle.dump(lsh, f)
    with open(lsh_exact_path, "w") as f:
        json.dump(exact_index, f, indent=2)

    print(f"  [LSH]   Index saved: {lsh_pickle_path.name} + {lsh_exact_path.name} "
          f"({len(profiles)} columns, ~{total_values} total distinct values fetched)")
    return lsh_pickle_path


# ─────────────────────────────────────────────
# Query Functions (used in Phase 3)
# ─────────────────────────────────────────────

def query_faiss(question: str, db_id: str, db_dir: Path, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Find top-k columns semantically similar to the question.
    Returns list of {table, column, score, profile_long_en}
    """
    import faiss
    import numpy as np

    faiss_path = db_dir / f"{db_id}.faiss"
    meta_path = db_dir / f"{db_id}.faiss_meta.json"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)

    # Embed the question
    q_emb = get_embeddings([question])[0]
    q_vec = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q_vec)

    # Search
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        r = dict(meta[idx])
        r["faiss_score"] = float(score)
        results.append(r)

    return results


def query_lsh(literal: str, db_id: str, db_dir: Path) -> List[Dict[str, Any]]:
    """
    Find columns whose top-k values contain the given literal.
    Returns list of {table, column, matched_values}
    """
    from datasketch import MinHash

    lsh_pickle_path = db_dir / f"{db_id}.lsh.pkl"
    lsh_exact_path = db_dir / f"{db_id}.lsh_index.json"

    if not lsh_pickle_path.exists():
        raise FileNotFoundError(f"LSH index not found: {lsh_pickle_path}")

    with open(lsh_pickle_path, "rb") as f:
        lsh = pickle.load(f)
    with open(lsh_exact_path) as f:
        exact_index = json.load(f)

    # Generate variants of the literal
    variants = _normalize_value(literal)

    # 1. Exact match first
    exact_matches = []
    for key, col_data in exact_index.items():
        col_values = set(col_data["values"])
        matched = [v for v in variants if v in col_values]
        if matched:
            exact_matches.append({
                "table": col_data["table"],
                "column": col_data["column"],
                "match_type": "exact",
                "matched_values": matched,
            })

    if exact_matches:
        return exact_matches

    # 2. Approximate LSH match using character k-shingles (paper footnote 6)
    m = MinHash(num_perm=128)
    for v in variants:
        for shingle in _kshingles(v.lower(), k=3):
            m.update(shingle.encode("utf8"))

    approx_keys = lsh.query(m)
    approx_matches = []
    for key in approx_keys:
        col_data = exact_index.get(key, {})
        approx_matches.append({
            "table": col_data.get("table", ""),
            "column": col_data.get("column", ""),
            "match_type": "approximate",
            "matched_values": variants,
        })

    return approx_matches


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def build_indexes_for_db(db_id: str, overwrite: bool = False):
    db_dir = MINIDEV_ROOT / db_id

    faiss_path = db_dir / f"{db_id}.faiss"
    lsh_path = db_dir / f"{db_id}.lsh.pkl"

    print(f"\n{'='*60}")
    print(f"Building indexes: {db_id}")
    print(f"{'='*60}")

    if not faiss_path.exists() or overwrite:
        build_faiss_index(db_id, db_dir)
    else:
        print(f"  [FAISS] Already exists, skipping. (use --overwrite to rebuild)")

    if not lsh_path.exists() or overwrite:
        build_lsh_index(db_id, db_dir)
    else:
        print(f"  [LSH]   Already exists, skipping. (use --overwrite to rebuild)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="debit_card_specializing")
    parser.add_argument("--all", action="store_true", help="Build for all 11 databases")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.all:
        databases = sorted([d.name for d in MINIDEV_ROOT.iterdir() if d.is_dir()])
        print(f"Building indexes for all {len(databases)} databases...")
        for db_id in databases:
            build_indexes_for_db(db_id, overwrite=args.overwrite)
    else:
        build_indexes_for_db(args.db, overwrite=args.overwrite)

    print("\nAll indexes built!")


if __name__ == "__main__":
    main()

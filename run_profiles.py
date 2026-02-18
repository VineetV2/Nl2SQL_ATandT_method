"""
run_profiles.py
---------------
Run the full profile pipeline for one or all MINIDEV databases.

Pipeline:
  1. Long profiles  (from SQLite stats)
  2. Full profiles  (merge long + dev docs)
  3. Short profiles (LLM one-sentence summaries via OpenAI GPT-5.2)

Usage:
  # Single database
  python run_profiles.py --db debit_card_specializing

  # All 11 databases
  python run_profiles.py --all
"""

import argparse
import os
from pathlib import Path

from profiles_sqlite_local import build_and_write_long_profiles
from profiles_full_local import build_and_write_full_profiles
from short_profiles_local import build_short_profile_prompts, generate_short_profiles
from llm_backends_local import make_backend

MINIDEV_ROOT = Path(__file__).parent / "MINIDEV" / "dev_databases"


def run_for_db(db_id: str, backend, overwrite: bool = False):
    db_dir = MINIDEV_ROOT / db_id
    sqlite_path = db_dir / f"{db_id}.sqlite"
    desc_dir = db_dir / "database_description"

    if not sqlite_path.exists():
        print(f"[SKIP] SQLite not found: {sqlite_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {db_id}")
    print(f"{'='*60}")

    # Step 1: Long profiles
    long_path = db_dir / f"{db_id}.long_profiles.jsonl"
    if not long_path.exists() or overwrite:
        long_path = build_and_write_long_profiles(
            sqlite_path,
            db_id=db_id,
            sample_n=5,
            topk=5,
            distinct_limit=None,
        )
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
            prompts_path,
            backend=backend,
            max_new_tokens=64,
            overwrite=overwrite,
            gen_kwargs={"temperature": 0},
        )
        print(f"  [3/3] Short profiles written: {short_path.name}")
    else:
        print(f"  [3/3] Short profiles already exist: {short_path.name}")

    print(f"  Done: {db_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="debit_card_specializing",
                        help="Database name to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all 11 databases")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing profile files")
    args = parser.parse_args()

    # Initialize OpenAI backend
    print("Initializing OpenAI GPT-5.2 backend...")
    backend = make_backend("openai", model_id="gpt-5.2", cache=True)

    if args.all:
        databases = sorted([d.name for d in MINIDEV_ROOT.iterdir() if d.is_dir()])
        print(f"Processing all {len(databases)} databases: {databases}")
        for db_id in databases:
            run_for_db(db_id, backend, overwrite=args.overwrite)
    else:
        run_for_db(args.db, backend, overwrite=args.overwrite)

    print("\n All profiles generated!")


if __name__ == "__main__":
    main()

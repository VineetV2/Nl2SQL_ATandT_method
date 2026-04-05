"""
combine_schema_links.py
-----------------------
Combines per-database schema_links JSON files into one file for Phase 4.

Usage:
  python combine_schema_links.py --input_dir /path/to/results --output combined_schema_links.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing schema_links_*.json files")
    parser.add_argument("--output", default=None, help="Output file path (default: combined_schema_links_<ts>.json)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("schema_links_*.json"))

    # Exclude the already-combined file if present
    files = [f for f in files if "all_dbs" not in f.name]

    print(f"Found {len(files)} schema link files:")
    for f in files:
        print(f"  {f.name}")

    all_results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        results = data.get("results", [])
        all_results.extend(results)
        print(f"  {f.name}: {len(results)} questions")

    print(f"\nTotal: {len(all_results)} questions combined")

    output_path = Path(args.output) if args.output else \
        input_dir / f"combined_schema_links_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, "w") as fp:
        json.dump({"results": all_results}, fp, indent=2)

    print(f"Saved → {output_path}")

if __name__ == "__main__":
    main()

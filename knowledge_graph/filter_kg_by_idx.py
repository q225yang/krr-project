"""
Filter KG result JSON files to keep only entries whose idx is in
extracted_concepts.json.

Handles both plain .json and gzipped .json.gz inputs.
Output goes to <basename>_img_only.json next to the input.

Usage (from knowledge_graph/):
    python filter_kg_by_idx.py \\
        --idx_file ../extracted_concepts.json \\
        --inputs ../kg_v4.json \\
                 kg_concepts_v3.json.gz \\
                 kg_visual_v3.json.gz \\
                 kg_merged_v3.json.gz
"""

import argparse
import gzip
import json
import os


def load_json_any(path: str):
    """Load JSON whether the file is plain or gzipped."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def out_path_for(input_path: str) -> str:
    """e.g. 'kg_v4.json'           -> 'kg_v4_img_only.json'
            'kg_concepts_v3.json.gz' -> 'kg_concepts_v3_img_only.json'"""
    base = os.path.basename(input_path)
    if base.endswith(".json.gz"):
        stem = base[:-len(".json.gz")]
    elif base.endswith(".json"):
        stem = base[:-len(".json")]
    else:
        stem = base
    out_name = f"{stem}_img_only.json"
    return os.path.join(os.path.dirname(input_path) or ".", out_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx_file", required=True,
                        help="JSON list of samples (each with 'idx') — only those idx will be kept")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="One or more KG result JSON(.gz) files to filter")
    args = parser.parse_args()

    idx_data = load_json_any(args.idx_file)
    keep_idx = {str(item["idx"]) for item in idx_data if "idx" in item}
    print(f"Keep set: {len(keep_idx)} idx values from {args.idx_file}")

    for in_path in args.inputs:
        if not os.path.exists(in_path):
            print(f"  [SKIP] {in_path} (not found)")
            continue

        data = load_json_any(in_path)
        if not isinstance(data, dict):
            print(f"  [SKIP] {in_path} (expected dict keyed by idx, got {type(data).__name__})")
            continue

        before = len(data)
        filtered = {k: v for k, v in data.items() if str(k) in keep_idx}
        after = len(filtered)

        out_path = out_path_for(in_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        print(f"  {in_path}: {before} → {after}  →  {out_path}")


if __name__ == "__main__":
    main()

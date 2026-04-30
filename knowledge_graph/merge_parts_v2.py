"""
Merge chunk outputs from Slurm array jobs into single JSON files.

Usage:
    python merge_parts_v2.py --in_dir out_parts_v2 --out_dir .
"""

import json
import glob
import argparse
import os


def merge_json_parts(pattern: str) -> dict:
    merged = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="out_parts_v2")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    names = [
        #"kg_visual_v2", "kg_concepts_v2", "kg_merged_v2", "kg_planner_v2",
        #"kg_visual_v3", "kg_concepts_v3", "kg_merged_v3",
        "kg_v4",
    ]
    for name in names:
        pattern = os.path.join(args.in_dir, f"{name}_part*.json")
        files = glob.glob(pattern)
        if not files:
            continue
        merged = merge_json_parts(pattern)
        out_path = os.path.join(args.out_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Merged {len(files)} files → {out_path} ({len(merged)} samples)")


if __name__ == "__main__":
    main()

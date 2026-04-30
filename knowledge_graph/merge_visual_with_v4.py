"""
Merge kg_visual.json with kg_v4_img_only.json per idx.

For each idx:
  - Combine seeds from both files (dedup, preserve order)
  - Combine triples (dedup by (start, rel, end), sort by weight desc, cap at top-K)
"""

import json
import os

TOPK = 40
KG_VISUAL = "knowledge_graph/kg_visual.json"
KG_V4 = "kg_v4_img_only.json"
OUT_PATH = "knowledge_graph/kg_visual_plus_v4.json"


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kv_path = os.path.join(base, KG_VISUAL)
    v4_path = os.path.join(base, KG_V4)
    out_path = os.path.join(base, OUT_PATH)

    with open(kv_path, "r", encoding="utf-8") as f:
        kv = json.load(f)
    with open(v4_path, "r", encoding="utf-8") as f:
        v4 = json.load(f)

    out = {}
    all_idx = sorted(set(kv) | set(v4), key=lambda x: int(x))

    for idx in all_idx:
        a = kv.get(idx, {})
        b = v4.get(idx, {})

        # Merge seeds
        seeds_a = a.get("seeds", []) or []
        seeds_b = b.get("seeds", []) or []
        seen_seed = set()
        merged_seeds = []
        for s in seeds_a + seeds_b:
            if s not in seen_seed:
                seen_seed.add(s)
                merged_seeds.append(s)

        # Merge triples
        triples = (a.get("triples", []) or []) + (b.get("triples", []) or [])
        seen_triple = set()
        deduped = []
        for t in triples:
            key = (t["start"], t["rel"], t["end"])
            if key in seen_triple:
                continue
            seen_triple.add(key)
            deduped.append(t)
        deduped.sort(key=lambda t: t["weight"], reverse=True)
        deduped = deduped[:TOPK]

        out[idx] = {
            "idx": int(idx),
            "seeds": merged_seeds,
            "triples": deduped,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path} ({len(out)} samples, top-{TOPK} triples each)")


if __name__ == "__main__":
    main()

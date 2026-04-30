"""
KG retrieval pipeline — v4 (unconstrained, extracted_concepts_improved_full)
------------------------------------------------------------------------------
Same retrieval logic as the original run_retrieval_and_dump.py (v1):
  - No LLM planner
  - Direct ConceptNet retrieval per seed term
  - Filter by relation whitelist + min weight
  - Top-k by raw weight

Differences from v1:
  - Reads extracted_concepts_improved_full.json (10,002 samples)
  - Uses ONE seed group per sample (word_groups) — no visual/concepts split,
    because the input file is text-only.
  - Produces ONE output file: kg_v4.json
  - Supports chunked parallelism (--start/--end/--suffix) for cluster use.

Usage (single run, all samples):
    python run_retrieval_and_dump_v4.py

Usage (chunked, e.g. with xargs -P or Slurm array):
    python run_retrieval_and_dump_v4.py \\
        --start 0 --end 500 --suffix part0 \\
        --out_dir out_parts_v4 --no_seed_progress
"""

import json
import os
import argparse
import sqlite3
from typing import List, Tuple, Set, Dict, Any
from tqdm import tqdm


# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_DB = "conceptnet_en.db"
DEFAULT_INPUT = "../extracted_concepts_improved_full.json"
DEFAULT_OUT_DIR = "."

PHYS_RELS = {
    "HasProperty", "MadeOf", "PartOf", "HasA", "AtLocation",
    "UsedFor", "CapableOf", "ReceivesAction",
    "Causes", "HasPrerequisite", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent",
    "LocatedNear", "Entails",
    "MotivatedByGoal", "CausesDesire", "Desires", "NotDesires",
    "IsA",
}

# Retrieval budgets (same as v1)
MAX_SEEDS_PER_GROUP = 15
MAX_CANDIDATES_PER_SEED = 25
PER_CANDIDATE_EDGE_K = 30
MIN_WEIGHT = 1.5
TOPK_PER_GROUP = 40
ALLOW_CONTAINS = True
SEARCH_IN_END = False
MIN_TERM_LEN_FOR_CONTAINS = 3


# ----------------------------
# DB helpers (db_path injected at runtime)
# ----------------------------
def normalize(term: str) -> str:
    return str(term).strip().lower().replace(" ", "_")


def find_candidate_concepts_like(
    db_path: str,
    term: str,
    *,
    max_candidates: int = 30,
    allow_contains: bool = True,
    search_in_end: bool = False,
) -> List[Tuple[str, int]]:
    t = normalize(term)
    if not t:
        return []
    prefix_pattern = f"/c/en/{t}%"
    contains_pattern = f"%/c/en/%{t}%"
    do_contains = allow_contains and (len(t) >= MIN_TERM_LEN_FOR_CONTAINS)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    candidates: Dict[str, int] = {}

    cur.execute("""
        SELECT start, COUNT(*) as n
        FROM edges
        WHERE start LIKE ?
        GROUP BY start ORDER BY n DESC LIMIT ?
    """, (prefix_pattern, max_candidates))
    for uri, n in cur.fetchall():
        candidates[uri] = candidates.get(uri, 0) + int(n)

    if do_contains and len(candidates) < max_candidates:
        cur.execute("""
            SELECT start, COUNT(*) as n
            FROM edges
            WHERE start LIKE ?
            GROUP BY start ORDER BY n DESC LIMIT ?
        """, (contains_pattern, max_candidates))
        for uri, n in cur.fetchall():
            candidates[uri] = candidates.get(uri, 0) + int(n)

    if search_in_end and len(candidates) < max_candidates:
        cur.execute("""
            SELECT end, COUNT(*) as n
            FROM edges
            WHERE end LIKE ?
            GROUP BY end ORDER BY n DESC LIMIT ?
        """, (prefix_pattern, max_candidates))
        for uri, n in cur.fetchall():
            candidates[uri] = candidates.get(uri, 0) + int(n)
        if do_contains:
            cur.execute("""
                SELECT end, COUNT(*) as n
                FROM edges
                WHERE end LIKE ?
                GROUP BY end ORDER BY n DESC LIMIT ?
            """, (contains_pattern, max_candidates))
            for uri, n in cur.fetchall():
                candidates[uri] = candidates.get(uri, 0) + int(n)

    conn.close()
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:max_candidates]


def query_edges_for_concept_uri(db_path: str, cu: str, *, per_concept_limit: int = 200):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT start, rel, end, weight, surfaceText
        FROM edges
        WHERE start = ?
        ORDER BY weight DESC
        LIMIT ?
    """, (cu, per_concept_limit))
    rows = cur.fetchall()
    conn.close()
    return rows


def query_edges_like(db_path: str, term: str, *,
                     topk=TOPK_PER_GROUP, min_weight=MIN_WEIGHT,
                     allowed_rels=PHYS_RELS,
                     max_candidates=MAX_CANDIDATES_PER_SEED,
                     per_candidate_k=PER_CANDIDATE_EDGE_K,
                     allow_contains=ALLOW_CONTAINS,
                     search_in_end=SEARCH_IN_END):
    candidates = find_candidate_concepts_like(
        db_path, term,
        max_candidates=max_candidates,
        allow_contains=allow_contains,
        search_in_end=search_in_end,
    )
    all_edges = []
    seen: Set[Tuple[str, str, str]] = set()
    for cu, _ in candidates:
        for s, rel, e, w, surf in query_edges_for_concept_uri(
                db_path, cu, per_concept_limit=per_candidate_k):
            if allowed_rels and rel not in allowed_rels:
                continue
            w = float(w)
            if w < min_weight:
                continue
            key = (s, rel, e)
            if key in seen:
                continue
            seen.add(key)
            all_edges.append((s, rel, e, w, surf))
    all_edges.sort(key=lambda x: x[3], reverse=True)
    return all_edges[:topk]


def triples_to_jsonable(triples):
    return [
        {"start": s, "rel": rel, "end": e, "weight": w, "surfaceText": surf}
        for s, rel, e, w, surf in triples
    ]


def run_group(db_path: str, seeds: List[str], show_progress: bool = True
              ) -> List[Tuple[str, str, str, float, str]]:
    seeds = [str(x) for x in seeds if str(x).strip()]
    seeds = seeds[:MAX_SEEDS_PER_GROUP]

    all_triples = []
    seen = set()
    iterator = tqdm(seeds, desc="  Seeds", leave=False) if show_progress else seeds
    for term in iterator:
        for s, rel, e, w, surf in query_edges_like(db_path, term):
            key = (s, rel, e)
            if key in seen:
                continue
            seen.add(key)
            all_triples.append((s, rel, e, w, surf))
    all_triples.sort(key=lambda x: x[3], reverse=True)
    return all_triples[:TOPK_PER_GROUP]


# ----------------------------
# Data loading
# ----------------------------
def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of samples in {path}")
    return data


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="KG retrieval v4 (extracted_concepts_improved_full)")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no_seed_progress", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    samples = load_samples(args.input)
    samples = samples[args.start : args.end]
    print(f"Processing samples [{args.start}:{args.end}] → {len(samples)} items")

    show_seed_progress = not args.no_seed_progress

    tag = f"_{args.suffix}" if args.suffix else ""
    out_path = os.path.join(args.out_dir, f"kg_v4{tag}.json")

    out = {}
    for item in tqdm(samples, desc="Processing samples"):
        idx = item.get("idx")
        if idx is None:
            continue

        seeds = item.get("word_groups", []) or []
        triples = run_group(args.db, seeds, show_seed_progress)

        out[str(idx)] = {
            "idx": idx,
            "seeds": seeds,
            "triples": triples_to_jsonable(triples),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

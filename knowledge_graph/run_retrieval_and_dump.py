import json
import sqlite3
from typing import List, Tuple, Set, Dict, Any
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
DB_PATH = "conceptnet_en.db"
INPUT_JSON = "../combined_visual_and_concepts.json"

OUT_VISUAL = "kg_visual.json"
OUT_CONCEPTS = "kg_concepts.json"
OUT_MERGED = "kg_merged.json"

PHYS_RELS = {
    "HasProperty", "MadeOf", "PartOf", "HasA", "AtLocation",
    "UsedFor", "CapableOf", "ReceivesAction",
    "Causes", "HasPrerequisite", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent",
    "LocatedNear", "Entails",
    "MotivatedByGoal", "CausesDesire", "Desires", "NotDesires",
    "IsA"
}

# Retrieval budgets (baseline-friendly defaults)
MAX_SEEDS_PER_GROUP = 15         # per idx: only query first N seeds in each group
MAX_CANDIDATES_PER_SEED = 25     # LIKE candidate concepts per seed
PER_CANDIDATE_EDGE_K = 30        # edges fetched per candidate concept
MIN_WEIGHT = 1.5                 # filter low-quality edges
TOPK_PER_GROUP = 40              # final triples kept per group per idx
ALLOW_CONTAINS = True            # allow %term% matching (more recall, more noise)
SEARCH_IN_END = False            # usually noisy; keep False for baseline

# Optional: prevent LIKE explosion on tiny tokens
MIN_TERM_LEN_FOR_CONTAINS = 3    # only do contains match if len(term) >= 4

# ----------------------------
# Helpers: conceptnet querying
# ----------------------------
def normalize(term: str) -> str:
    return str(term).strip().lower().replace(" ", "_")

def concept_uri(term: str) -> str:
    return f"/c/en/{normalize(term)}"

def find_candidate_concepts_like(
    term: str,
    *,
    max_candidates: int = 30,
    allow_contains: bool = True,
    search_in_end: bool = False,
) -> List[Tuple[str, int]]:
    """
    Return candidate concept URIs that match the term by LIKE.
    Output: list of (concept_uri, freq) sorted by freq desc.
    """
    t = normalize(term)
    if not t:
        return []

    prefix_pattern = f"/c/en/{t}%"
    contains_pattern = f"%/c/en/%{t}%"

    # Safety: contains only for longer terms
    do_contains = allow_contains and (len(t) >= MIN_TERM_LEN_FOR_CONTAINS)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    candidates: Dict[str, int] = {}

    # 1) Prefix on start (fast)
    cur.execute("""
        SELECT start, COUNT(*) as n
        FROM edges
        WHERE start LIKE ?
        GROUP BY start
        ORDER BY n DESC
        LIMIT ?
    """, (prefix_pattern, max_candidates))
    for uri, n in cur.fetchall():
        candidates[uri] = candidates.get(uri, 0) + int(n)

    # 2) Contains on start (flexible, slower)
    if do_contains and len(candidates) < max_candidates:
        cur.execute("""
            SELECT start, COUNT(*) as n
            FROM edges
            WHERE start LIKE ?
            GROUP BY start
            ORDER BY n DESC
            LIMIT ?
        """, (contains_pattern, max_candidates))
        for uri, n in cur.fetchall():
            candidates[uri] = candidates.get(uri, 0) + int(n)

    # 3) Optional: search in end too (often noisy)
    if search_in_end and len(candidates) < max_candidates:
        cur.execute("""
            SELECT end, COUNT(*) as n
            FROM edges
            WHERE end LIKE ?
            GROUP BY end
            ORDER BY n DESC
            LIMIT ?
        """, (prefix_pattern, max_candidates))
        for uri, n in cur.fetchall():
            candidates[uri] = candidates.get(uri, 0) + int(n)

        if do_contains:
            cur.execute("""
                SELECT end, COUNT(*) as n
                FROM edges
                WHERE end LIKE ?
                GROUP BY end
                ORDER BY n DESC
                LIMIT ?
            """, (contains_pattern, max_candidates))
            for uri, n in cur.fetchall():
                candidates[uri] = candidates.get(uri, 0) + int(n)

    conn.close()

    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return ranked[:max_candidates]

def query_edges_for_concept_uri(cu: str, *, per_concept_limit: int = 200):
    conn = sqlite3.connect(DB_PATH)
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

def query_edges_like(
    term: str,
    *,
    topk: int = 20,
    min_weight: float = 1.5,
    allowed_rels=PHYS_RELS,
    max_candidates: int = 30,
    per_candidate_k: int = 30,
    allow_contains: bool = True,
    search_in_end: bool = False,
):
    """
    LIKE-based query:
      - find candidate concept URIs
      - pull edges for each candidate
      - filter by relation + weight
      - de-dup and return global topk by weight
    """
    candidates = find_candidate_concepts_like(
        term,
        max_candidates=max_candidates,
        allow_contains=allow_contains,
        search_in_end=search_in_end
    )

    all_edges = []
    seen: Set[Tuple[str, str, str]] = set()

    for cu, _freq in candidates:
        rows = query_edges_for_concept_uri(cu, per_concept_limit=per_candidate_k)
        for s, rel, e, w, surf in rows:
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
    # keep as dicts for readability
    out = []
    for s, rel, e, w, surf in triples:
        out.append({
            "start": s,
            "rel": rel,
            "end": e,
            "weight": w,
            "surfaceText": surf
        })
    return out

def merge_triples(triples_a, triples_b, topk=TOPK_PER_GROUP):
    seen = set()
    merged = []
    for t in triples_a + triples_b:
        key = (t[0], t[1], t[2])
        if key in seen:
            continue
        seen.add(key)
        merged.append(t)
    merged.sort(key=lambda x: x[3], reverse=True)
    return merged[:topk]

# ----------------------------
# Dataloader + main
# ----------------------------
def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of samples.")
    return data

def run_group(seeds: List[str]) -> List[Tuple[str, str, str, float, str]]:
    # baseline: don't clean, but apply retrieval budget
    seeds = [str(x) for x in seeds if str(x).strip()]
    seeds = seeds[:MAX_SEEDS_PER_GROUP]

    all_triples = []
    seen = set()

    for term in tqdm(seeds, desc="  Seeds", leave=False):#for term in seeds:
        triples = query_edges_like(
            term,
            topk=TOPK_PER_GROUP,              # per term will be cut again globally
            min_weight=MIN_WEIGHT,
            max_candidates=MAX_CANDIDATES_PER_SEED,
            per_candidate_k=PER_CANDIDATE_EDGE_K,
            allow_contains=ALLOW_CONTAINS,
            search_in_end=SEARCH_IN_END,
        )
        for s, rel, e, w, surf in triples:
            key = (s, rel, e)
            if key in seen:
                continue
            seen.add(key)
            all_triples.append((s, rel, e, w, surf))

    # global cap
    all_triples.sort(key=lambda x: x[3], reverse=True)
    return all_triples[:TOPK_PER_GROUP]

def main():
    samples = load_samples(INPUT_JSON)

    out_visual = {}
    out_concepts = {}
    out_merged = {}

    for item in tqdm(samples, desc="Processing samples (idx)"):#for item in samples:
        idx = item.get("idx")
        if idx is None:
            continue

        visual_entities = item.get("visual_entities", []) or []
        concepts = item.get("concepts", []) or []

        visual_triples = run_group(visual_entities)
        concept_triples = run_group(concepts)
        merged_triples = merge_triples(visual_triples, concept_triples, topk=TOPK_PER_GROUP)

        # store
        out_visual[str(idx)] = {
            "idx": idx,
            "seeds": visual_entities,
            "triples": triples_to_jsonable(visual_triples),
        }
        out_concepts[str(idx)] = {
            "idx": idx,
            "seeds": concepts,
            "triples": triples_to_jsonable(concept_triples),
        }
        out_merged[str(idx)] = {
            "idx": idx,
            "seeds": {
                "visual_entities": visual_entities,
                "concepts": concepts
            },
            "triples": triples_to_jsonable(merged_triples),
        }

    with open(OUT_VISUAL, "w", encoding="utf-8") as f:
        json.dump(out_visual, f, ensure_ascii=False, indent=2)

    with open(OUT_CONCEPTS, "w", encoding="utf-8") as f:
        json.dump(out_concepts, f, ensure_ascii=False, indent=2)

    with open(OUT_MERGED, "w", encoding="utf-8") as f:
        json.dump(out_merged, f, ensure_ascii=False, indent=2)

    print("Wrote:", OUT_VISUAL, OUT_CONCEPTS, OUT_MERGED)

if __name__ == "__main__":
    main()
import sqlite3
from typing import List, Tuple, Set

DB_PATH = "conceptnet_en.db"

PHYS_RELS = {
    "HasProperty", "MadeOf", "PartOf", "HasA", "AtLocation",
    "UsedFor", "CapableOf", "ReceivesAction",
    "Causes", "HasPrerequisite", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent",
    "LocatedNear", "Entails",
    "MotivatedByGoal", "CausesDesire", "Desires", "NotDesires",
    "IsA"
}

def normalize(term: str) -> str:
    # Keep it simple; you can expand later (lemmatize, etc.)
    return term.strip().lower().replace(" ", "_")

def concept_uri(term: str) -> str:
    return f"/c/en/{normalize(term)}"

def find_candidate_concepts_like(
    term: str,
    *,
    max_candidates: int = 30,
    allow_contains: bool = True,
    search_in_end: bool = True,
) -> List[Tuple[str, int]]:
    """
    Return candidate concept URIs that match the term by LIKE.
    Output: list of (concept_uri, freq) sorted by freq desc.
    """
    t = normalize(term)
    # Prefer prefix match first (can use index better)
    prefix_pattern = f"/c/en/{t}%"
    # Contains match (more flexible but slower)
    contains_pattern = f"%/c/en/%{t}%"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Collect candidates with frequency counts
    candidates = {}

    # 1) Prefix on start
    cur.execute("""
        SELECT start, COUNT(*) as n
        FROM edges
        WHERE start LIKE ?
        GROUP BY start
        ORDER BY n DESC
        LIMIT ?
    """, (prefix_pattern, max_candidates))
    for uri, n in cur.fetchall():
        candidates[uri] = candidates.get(uri, 0) + n

    # 2) (Optional) contains on start
    if allow_contains and len(candidates) < max_candidates:
        cur.execute("""
            SELECT start, COUNT(*) as n
            FROM edges
            WHERE start LIKE ?
            GROUP BY start
            ORDER BY n DESC
            LIMIT ?
        """, (contains_pattern, max_candidates))
        for uri, n in cur.fetchall():
            candidates[uri] = candidates.get(uri, 0) + n

    # 3) (Optional) search in end too
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
            candidates[uri] = candidates.get(uri, 0) + n

        if allow_contains:
            cur.execute("""
                SELECT end, COUNT(*) as n
                FROM edges
                WHERE end LIKE ?
                GROUP BY end
                ORDER BY n DESC
                LIMIT ?
            """, (contains_pattern, max_candidates))
            for uri, n in cur.fetchall():
                candidates[uri] = candidates.get(uri, 0) + n

    conn.close()

    # Sort by frequency desc and keep top max_candidates
    ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return ranked[:max_candidates]

def query_edges_for_concept_uri(
    cu: str,
    *,
    per_concept_limit: int = 200,
) -> List[Tuple[str, str, str, float, str]]:
    """Fetch raw edges for an exact concept URI."""
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
    search_in_end: bool = False,   # usually keep False; end-search can add noise
) -> List[Tuple[str, str, str, float, str]]:
    """
    LIKE-based query:
    1) find candidate concept URIs matching term
    2) pull edges for each candidate
    3) filter by rel/weight
    4) de-dup and return global topk by weight
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
            if w < min_weight:
                continue
            key = (s, rel, e)
            if key in seen:
                continue
            seen.add(key)
            all_edges.append((s, rel, e, float(w), surf))

    # Global ranking by weight
    all_edges.sort(key=lambda x: x[3], reverse=True)
    return all_edges[:topk]

if __name__ == "__main__":
    # exact
    print("=== exact knife ===")
    for s, rel, e, w, surf in query_edges_for_concept_uri(concept_uri("knife"))[:10]:
        print(s, rel, e, w, surf)

    # like
    print("\n=== like 'phone' ===")
    triples = query_edges_like("phone", topk=15, max_candidates=20, per_candidate_k=50)
    for s, rel, e, w, surf in triples:
        print(f"{s} --{rel}--> {e} (w={w}) | {surf}")

'''import sqlite3

DB_PATH = "conceptnet_en.db"

PHYS_RELS = {
    #"UsedFor", "CapableOf", "HasProperty", "MadeOf",
    #"AtLocation", "PartOf", "Causes", "HasSubevent", "IsA"
    "HasProperty", "MadeOf", "PartOf", "HasA", "AtLocation", 
    "UsedFor", "CapableOf", "ReceivesAction",
    "Causes", "HasPrerequisite", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent", 
    "LocatedNear", "Entails", "MotivatedByGoal", "CausesDesire", "Desires", "NotDesires",
    "IsA"
}

def concept_uri(term: str) -> str:
    term = term.strip().lower().replace(" ", "_")
    return f"/c/en/{term}"

def query_edges(term: str, topk: int = 20, min_weight: float = 1.5, allowed_rels=PHYS_RELS):
    cu = concept_uri(term)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT start, rel, end, weight, surfaceText
        FROM edges
        WHERE start = ?
        ORDER BY weight DESC
        LIMIT 500
    """, (cu,))
    rows = cur.fetchall()
    conn.close()

    out = []
    for s, rel, e, w, surf in rows:
        if allowed_rels and rel not in allowed_rels:
            continue
        if w < min_weight:
            continue
        out.append((s, rel, e, w, surf))

    # Keep topk after filtering
    return out[:topk]

if __name__ == "__main__":
    triples = query_edges("knife", topk=10)
    for s, rel, e, w, surf in triples:
        print(f"{s} --{rel}--> {e}  (w={w})  | {surf}")
'''
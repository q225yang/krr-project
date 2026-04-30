"""
KG retrieval pipeline — v2 (LLM-constrained)
----------------------------------------------
Copy of run_retrieval_and_dump.py that integrates the LLM query planner.

Supports Slurm array jobs via --start/--end/--suffix for chunk-based
parallel processing, matching the CLI interface of run_retrieval_and_dump_fast.py.

Changes from v1:
  1. Loads question stems + options from extracted_concepts.json
  2. Before querying ConceptNet, calls llm_query_planner.plan_queries()
     to get per-concept constraints (rewritten terms, relation subsets,
     exclude/include end-terms, and keep/skip decisions).
  3. Passes those constraints into query_cn_v2.query_edges_like().
  4. Saves the LLM planner output alongside KG results for debugging.

Usage (local):
    python run_retrieval_and_dump_v2.py --backend ollama

Usage (cluster with Slurm array jobs):
    python run_retrieval_and_dump_v2.py \\
        --backend hf --model mistralai/Mistral-7B-Instruct-v0.3 \\
        --db conceptnet_en.db \\
        --input ../combined_visual_and_concepts.json \\
        --concepts_input ../extracted_concepts.json \\
        --out_dir out_parts_v2 \\
        --start 0 --end 100 --suffix part0 \\
        --no_seed_progress
"""

import json
import os
import argparse
from typing import List, Tuple, Set, Dict, Any, Optional
from tqdm import tqdm

# Imports are done after argparse so --help is fast even without deps
def _import_query_module(db_path: str):
    """Import query_cn_v2 and set its DB_PATH before any queries."""
    import query_cn_v2
    query_cn_v2.DB_PATH = db_path
    return query_cn_v2

# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_DB = "conceptnet_en.db"
DEFAULT_INPUT = "../combined_visual_and_concepts.json"
DEFAULT_CONCEPTS_INPUT = "../extracted_concepts_improved.json"
DEFAULT_OUT_DIR = "."

PHYS_RELS_SET = {
    "HasProperty", "MadeOf", "PartOf", "HasA", "AtLocation",
    "UsedFor", "CapableOf", "ReceivesAction",
    "Causes", "HasPrerequisite", "HasSubevent",
    "HasFirstSubevent", "HasLastSubevent",
    "LocatedNear", "Entails",
    "MotivatedByGoal", "CausesDesire", "Desires", "NotDesires",
    "IsA",
}

# Retrieval budgets
MAX_SEEDS_PER_GROUP = 15
MAX_CANDIDATES_PER_SEED = 25
PER_CANDIDATE_EDGE_K = 30
MIN_WEIGHT = 1.5
TOPK_PER_GROUP = 40
ALLOW_CONTAINS = True
SEARCH_IN_END = False
MIN_TERM_LENGTH = 3              # skip short noise terms like "be", "do", "we"

# v2: include-term boost factor
INCLUDE_BOOST = 2.0


# ----------------------------
# Helpers
# ----------------------------
def triples_to_jsonable(triples):
    out = []
    for s, rel, e, w, surf in triples:
        out.append({
            "start": s,
            "rel": rel,
            "end": e,
            "weight": w,
            "surfaceText": surf,
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
# Data loading
# ----------------------------
def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of samples.")
    return data


def build_question_lookup(concepts_path: str) -> Dict[int, Dict[str, Any]]:
    """Build idx → {question_stem, options} lookup from extracted_concepts.json."""
    data = load_samples(concepts_path)
    lookup = {}
    for item in data:
        idx = item.get("idx")
        if idx is not None:
            lookup[idx] = {
                "question_stem": item.get("question_stem", ""),
                "options": item.get("options", []),
            }
    return lookup


# ----------------------------
# Core: constrained retrieval
# ----------------------------
def run_group_constrained(
    seeds: List[str],
    query_mod,
    constraints=None,
    show_progress: bool = True,
) -> List[Tuple[str, str, str, float, str]]:
    """Query ConceptNet for a list of seed concepts, applying per-concept
    constraints from the LLM planner.

    If constraints is None, falls back to unconstrained retrieval (v1 behavior).
    """
    seeds = [str(x) for x in seeds if str(x).strip() and len(str(x).strip()) >= MIN_TERM_LENGTH]
    seeds = seeds[:MAX_SEEDS_PER_GROUP]

    # Build a lookup: original concept → constraint
    constraint_map = {}
    if constraints:
        for cc in constraints:
            constraint_map[cc.original] = cc

    all_triples = []
    seen = set()

    iterator = tqdm(seeds, desc="  Seeds", leave=False) if show_progress else seeds
    for term in iterator:
        cc = constraint_map.get(term)

        # ── Apply constraints if available ──
        if cc is not None:
            if not cc.keep:
                continue
            query_term = cc.rewritten or term
            if cc.allowed_relations:
                allowed = set(cc.allowed_relations) & PHYS_RELS_SET
                if not allowed:
                    allowed = PHYS_RELS_SET
            else:
                allowed = PHYS_RELS_SET
            exclude_ends = cc.exclude_end_terms
            include_ends = cc.include_end_terms
        else:
            query_term = term
            allowed = PHYS_RELS_SET
            exclude_ends = None
            include_ends = None

        triples = query_mod.query_edges_like(
            query_term,
            topk=TOPK_PER_GROUP,
            min_weight=MIN_WEIGHT,
            allowed_rels=allowed,
            max_candidates=MAX_CANDIDATES_PER_SEED,
            per_candidate_k=PER_CANDIDATE_EDGE_K,
            allow_contains=ALLOW_CONTAINS,
            search_in_end=SEARCH_IN_END,
            exclude_end_terms=exclude_ends,
            include_end_terms=include_ends,
            include_boost=INCLUDE_BOOST,
        )
        for s, rel, e, w, surf in triples:
            key = (s, rel, e)
            if key in seen:
                continue
            seen.add(key)
            all_triples.append((s, rel, e, w, surf))

    all_triples.sort(key=lambda x: x[3], reverse=True)
    return all_triples[:TOPK_PER_GROUP]


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="KG retrieval v2 (LLM-constrained)")

    # ── Data paths (match run_retrieval_and_dump_fast.py interface) ──
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to conceptnet_en.db")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to combined_visual_and_concepts.json")
    parser.add_argument("--concepts_input", default=DEFAULT_CONCEPTS_INPUT,
                        help="Path to extracted_concepts.json (for question stems)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Output directory for result JSONs")

    # ── Slurm array job slicing ──
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--suffix", default="", help="Suffix for output filenames (e.g. part0)")
    parser.add_argument("--no_seed_progress", action="store_true",
                        help="Disable per-seed tqdm progress bars")

    # ── LLM planner ──
    parser.add_argument("--backend", default="hf",
                        choices=["hf", "ollama", "openai", "anthropic"],
                        help="LLM backend (default: hf for cluster)")
    parser.add_argument("--model", default=None,
                        help="Model name/path (default depends on backend)")
    parser.add_argument("--no_planner", action="store_true",
                        help="Disable LLM planner (run unconstrained, same as v1)")

    args = parser.parse_args()

    # ── Setup ──
    os.makedirs(args.out_dir, exist_ok=True)
    query_mod = _import_query_module(args.db)

    from llm_query_planner import plan_queries, constraints_to_json

    # Load data
    samples = load_samples(args.input)
    question_lookup = build_question_lookup(args.concepts_input)

    # Slice for this array task
    samples = samples[args.start : args.end]
    print(f"Processing samples [{args.start}:{args.end}] → {len(samples)} items")

    use_planner = not args.no_planner
    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model

    show_seed_progress = not args.no_seed_progress

    # ── Build output filenames ──
    tag = f"_{args.suffix}" if args.suffix else ""
    out_visual_path = os.path.join(args.out_dir, f"kg_visual_v2{tag}.json")
    out_concepts_path = os.path.join(args.out_dir, f"kg_concepts_v2{tag}.json")
    out_merged_path = os.path.join(args.out_dir, f"kg_merged_v2{tag}.json")
    out_planner_path = os.path.join(args.out_dir, f"kg_planner_v2{tag}.json")

    out_visual = {}
    out_concepts = {}
    out_merged = {}
    out_planner = {}

    for item in tqdm(samples, desc="Processing samples"):
        idx = item.get("idx")
        if idx is None:
            continue

        visual_entities = item.get("visual_entities", []) or []
        concepts = item.get("concepts", []) or []
        caption = item.get("caption", "") or ""

        q_info = question_lookup.get(idx, {})
        question_stem = q_info.get("question_stem", "")
        options = q_info.get("options", [])

        # ── Filter short noise terms before LLM planner ──
        visual_entities = [v for v in visual_entities if len(v.strip()) >= MIN_TERM_LENGTH]
        concepts = [c for c in concepts if len(c.strip()) >= MIN_TERM_LENGTH]

        # ── LLM planner step ──
        all_concepts = visual_entities + concepts
        constraints = None
        if use_planner and all_concepts:
            try:
                constraints = plan_queries(
                    caption=caption,
                    question=question_stem,
                    options=options,
                    concepts=all_concepts,
                    llm_backend=args.backend,
                    llm_kwargs=llm_kwargs,
                )
                out_planner[str(idx)] = {
                    "idx": idx,
                    "caption": caption,
                    "question": question_stem,
                    "options": options,
                    "input_concepts": all_concepts,
                    "constraints": constraints_to_json(constraints),
                }
            except Exception as e:
                print(f"[WARNING] Planner failed for idx={idx}: {e}")
                constraints = None

        # Build per-source constraint subsets
        visual_constraints = None
        concept_constraints = None
        if constraints:
            vis_set = set(visual_entities)
            con_set = set(concepts)
            visual_constraints = [cc for cc in constraints if cc.original in vis_set]
            concept_constraints = [cc for cc in constraints if cc.original in con_set]

        # ── Retrieve KG triples ──
        visual_triples = run_group_constrained(
            visual_entities, query_mod, visual_constraints, show_seed_progress)
        concept_triples = run_group_constrained(
            concepts, query_mod, concept_constraints, show_seed_progress)
        merged_triples = merge_triples(visual_triples, concept_triples, topk=TOPK_PER_GROUP)

        # Store results
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
            "seeds": {"visual_entities": visual_entities, "concepts": concepts},
            "triples": triples_to_jsonable(merged_triples),
        }

    # ── Write outputs ──
    for path, data in [
        (out_visual_path, out_visual),
        (out_concepts_path, out_concepts),
        (out_merged_path, out_merged),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {path}")

    if out_planner:
        with open(out_planner_path, "w", encoding="utf-8") as f:
            json.dump(out_planner, f, ensure_ascii=False, indent=2)
        print(f"Wrote planner debug: {out_planner_path}")


if __name__ == "__main__":
    main()

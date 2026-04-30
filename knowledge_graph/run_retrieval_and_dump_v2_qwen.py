"""
KG retrieval pipeline — v2 (LLM-constrained) on qwen-extracted entities
-------------------------------------------------------------------------
Same LLM-constrained retrieval logic as run_retrieval_and_dump_v2.py, but
reads the qwen-style input files:
  - img_entities_qwen.json   → visual_entities (main_entities, ≤3 words)
                                + caption (main_description)
  - text_entities_qwen.json  → concepts (entities)
  - extracted_concepts_improved.json → question_stem + options
                                       AND defines the 2,093 idx subset to run on.

Output filenames suffixed with _v2_qwen.

Usage (sequential, all 2,093 samples):
    python run_retrieval_and_dump_v2_qwen.py --backend ollama --model mistral

Usage (chunked for xargs parallelism):
    python run_retrieval_and_dump_v2_qwen.py \\
        --backend ollama --model mistral \\
        --start 0 --end 100 --suffix part0 \\
        --out_dir out_parts_v2_qwen --no_seed_progress
"""

import json
import os
import argparse
from typing import List, Tuple, Set, Dict, Any, Optional
from tqdm import tqdm


def _import_query_module(db_path: str):
    import query_cn_v2
    query_cn_v2.DB_PATH = db_path
    return query_cn_v2


# ----------------------------
# Config defaults
# ----------------------------
DEFAULT_DB = "conceptnet_en.db"
DEFAULT_IMG_INPUT = "../img_entities_qwen.json"
DEFAULT_TXT_INPUT = "../text_entities_qwen.json"
DEFAULT_QUESTIONS_INPUT = "../extracted_concepts_improved.json"
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
MIN_TERM_LENGTH = 3              # skip short noise terms (be, do, we)
MAX_WORDS_PER_ENTITY = 3         # filter long LLM-extracted noise

INCLUDE_BOOST = 2.0


# ----------------------------
# Helpers
# ----------------------------
def filter_entities(entities: List[str], max_words: int = MAX_WORDS_PER_ENTITY,
                    min_len: int = MIN_TERM_LENGTH) -> List[str]:
    out = []
    for e in entities or []:
        if not isinstance(e, str):
            continue
        e = e.strip()
        if len(e) < min_len:
            continue
        n_words = len(e.replace("_", " ").split())
        if n_words <= max_words:
            out.append(e)
    return out


def triples_to_jsonable(triples):
    return [
        {"start": s, "rel": rel, "end": e, "weight": w, "surfaceText": surf}
        for s, rel, e, w, surf in triples
    ]


def merge_triples(a, b, topk=TOPK_PER_GROUP):
    seen = set()
    merged = []
    for t in a + b:
        k = (t[0], t[1], t[2])
        if k in seen:
            continue
        seen.add(k)
        merged.append(t)
    merged.sort(key=lambda x: x[3], reverse=True)
    return merged[:topk]


# ----------------------------
# Data loading
# ----------------------------
def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of samples in {path}")
    return data


def build_idx_lookup(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for it in items:
        idx = it.get("idx")
        if idx is not None:
            out[idx] = it
    return out


# ----------------------------
# Constrained retrieval
# ----------------------------
def run_group_constrained(
    seeds: List[str],
    query_mod,
    constraints=None,
    show_progress: bool = True,
) -> List[Tuple[str, str, str, float, str]]:
    seeds = [str(x) for x in seeds if str(x).strip() and len(str(x).strip()) >= MIN_TERM_LENGTH]
    seeds = seeds[:MAX_SEEDS_PER_GROUP]

    constraint_map = {}
    if constraints:
        for cc in constraints:
            constraint_map[cc.original] = cc

    all_triples = []
    seen = set()
    iterator = tqdm(seeds, desc="  Seeds", leave=False) if show_progress else seeds
    for term in iterator:
        cc = constraint_map.get(term)
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
    parser = argparse.ArgumentParser(description="KG retrieval v2 on qwen entities")

    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--img_input", default=DEFAULT_IMG_INPUT,
                        help="img_entities_qwen.json (visual entities + caption)")
    parser.add_argument("--txt_input", default=DEFAULT_TXT_INPUT,
                        help="text_entities_qwen.json (text entities)")
    parser.add_argument("--questions_input", default=DEFAULT_QUESTIONS_INPUT,
                        help="extracted_concepts_improved.json (question stems + idx subset)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no_seed_progress", action="store_true")

    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "hf", "openai", "anthropic"],
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default=None,
                        help="Model name (e.g. mistral for ollama, mistralai/Mistral-7B-Instruct-v0.3 for hf)")
    parser.add_argument("--no_planner", action="store_true",
                        help="Disable LLM planner (unconstrained baseline)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    query_mod = _import_query_module(args.db)

    from llm_query_planner import plan_queries, constraints_to_json

    # Load all three inputs
    questions = load_json_list(args.questions_input)
    img_lookup = build_idx_lookup(load_json_list(args.img_input))
    txt_lookup = build_idx_lookup(load_json_list(args.txt_input))

    # The "questions" file defines the idx subset to run on (e.g. 2,093 samples)
    samples = questions[args.start : args.end]
    print(f"Processing samples [{args.start}:{args.end}] → {len(samples)} items")

    use_planner = not args.no_planner
    llm_kwargs = {}
    if args.model:
        llm_kwargs["model"] = args.model

    show_seed_progress = not args.no_seed_progress

    tag = f"_{args.suffix}" if args.suffix else ""
    out_visual_path = os.path.join(args.out_dir, f"kg_visual_v2_qwen{tag}.json")
    out_concepts_path = os.path.join(args.out_dir, f"kg_concepts_v2_qwen{tag}.json")
    out_merged_path = os.path.join(args.out_dir, f"kg_merged_v2_qwen{tag}.json")
    out_planner_path = os.path.join(args.out_dir, f"kg_planner_v2_qwen{tag}.json")

    out_visual = {}
    out_concepts = {}
    out_merged = {}
    out_planner = {}

    for q in tqdm(samples, desc="Processing samples"):
        idx = q.get("idx")
        if idx is None:
            continue

        question_stem = q.get("question_stem", "") or ""
        options = q.get("options", []) or []

        img_item = img_lookup.get(idx, {})
        txt_item = txt_lookup.get(idx, {})

        caption = img_item.get("main_description", "") or ""
        raw_visual = img_item.get("main_entities", []) or []
        raw_text = txt_item.get("entities", []) or []

        # Apply word-count + min-length filters
        visual_entities = filter_entities(raw_visual)
        text_entities = filter_entities(raw_text)

        # ── LLM planner step ──
        all_concepts = visual_entities + text_entities
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

        visual_constraints = None
        text_constraints = None
        if constraints:
            vis_set = set(visual_entities)
            txt_set = set(text_entities)
            visual_constraints = [cc for cc in constraints if cc.original in vis_set]
            text_constraints = [cc for cc in constraints if cc.original in txt_set]

        # ── Retrieve KG triples ──
        visual_triples = run_group_constrained(
            visual_entities, query_mod, visual_constraints, show_seed_progress)
        concept_triples = run_group_constrained(
            text_entities, query_mod, text_constraints, show_seed_progress)
        merged = merge_triples(visual_triples, concept_triples, topk=TOPK_PER_GROUP)

        out_visual[str(idx)] = {
            "idx": idx,
            "seeds": visual_entities,
            "triples": triples_to_jsonable(visual_triples),
        }
        out_concepts[str(idx)] = {
            "idx": idx,
            "seeds": text_entities,
            "triples": triples_to_jsonable(concept_triples),
        }
        out_merged[str(idx)] = {
            "idx": idx,
            "seeds": {"visual_entities": visual_entities, "concepts": text_entities},
            "triples": triples_to_jsonable(merged),
        }

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

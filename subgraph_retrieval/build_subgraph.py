#!/usr/bin/env python3

import argparse
import json
import os
import re
import sqlite3
from typing import Dict, List, Set, Any
from collections import defaultdict

import networkx as nx


# ============================================================
# Basic helpers
# ============================================================

def normalize_entity_text(text: str) -> str:
    """
    Normalize an entity phrase into a simple underscore phrase.
    Example:
        "wine glass" -> "wine_glass"
        "white web-like patterns inside" -> "white_web_like_patterns_inside"
    """
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s_]", "", text)
    text = text.strip()
    text = text.replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_text_to_concept(text: str) -> str:
    text = normalize_entity_text(text)
    return f"/c/en/{text}" if text else ""


def concept_to_plain(uri: str) -> str:
    """
    /c/en/wine_glass -> wine_glass
    """
    if uri.startswith("/c/en/"):
        return uri[len("/c/en/"):]
    return uri


def is_english_concept(uri: str) -> bool:
    return uri.startswith("/c/en/")


def rel_label(rel_uri: str) -> str:
    return rel_uri.split("/")[-1]


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = normalize_entity_text(x)
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ============================================================
# Filtering / cleaning
# ============================================================

BAD_ENTITIES = {
    "",
    "image",
    "video",
    "picture",
    "photo",
    "option",
    "options",
    "object",
    "objects",
    "thing",
    "things",
    "area",
    "scene",
    "content",
    "following",
    "corresponding",
    "phenomenon",
    "which",
    "what",
    "where",
    "when",
    "first",
    "happen",
    "happens",
    "shown",
    "shows",
    "visible",
    "appears",
    "appear",
    "placed",
    "displayed",
    "background",
    "dark_background",
    "black_background",
    "white_background",
}


def is_bad_entity(ent: str) -> bool:
    ent = normalize_entity_text(ent)

    if not ent:
        return True

    if ent in BAD_ENTITIES:
        return True

    # remove very long noisy phrases
    if len(ent.split("_")) > 6:
        return True

    # remove single-character junk
    if len(ent) <= 1:
        return True

    # remove pure numbers
    if ent.isdigit():
        return True

    return False


def clean_entity(ent: str) -> str:
    ent = normalize_entity_text(ent)

    # light cleanup for common determiners
    ent = re.sub(r"^(a|an|the)_", "", ent)
    ent = re.sub(r"_+", "_", ent)
    ent = ent.strip("_")

    if is_bad_entity(ent):
        return ""

    return ent


# ============================================================
# SQLite
# ============================================================

def connect_sqlite(db_path: str):
    return sqlite3.connect(db_path)


def db_has_edges(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")
    if cur.fetchone() is None:
        return False

    cur.execute("SELECT COUNT(*) FROM edges")
    return cur.fetchone()[0] > 0


def build_sqlite_index(csv_path, db_path, min_weight=0.5):
    conn = connect_sqlite(db_path)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        start TEXT,
        end TEXT,
        rel TEXT,
        weight REAL,
        source_json TEXT
    )
    """)

    if db_has_edges(conn):
        print("[INFO] SQLite already built")
        conn.close()
        return

    print("[INFO] Building SQLite index...")

    inserted = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue

            _, rel, start, end, meta = parts

            if not (is_english_concept(start) and is_english_concept(end)):
                continue

            try:
                meta_json = json.loads(meta)
                weight = float(meta_json.get("weight", 1.0))
            except Exception:
                weight = 1.0

            if weight < min_weight:
                continue

            conn.execute(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                (start, end, rel, weight, meta)
            )

            # Add reverse direction so neighbor lookup is easier.
            conn.execute(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                (end, start, rel, weight, meta)
            )

            inserted += 2

            if inserted % 500000 == 0:
                conn.commit()
                print(f"[INFO] Inserted {inserted} directed edges...")

    conn.commit()

    print("[INFO] Creating SQLite index on start...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_start ON edges(start)")
    conn.commit()

    conn.close()
    print(f"[INFO] SQLite built with {inserted} directed edges")


def get_neighbors(conn, node, limit=50):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT start, end, rel, weight, source_json
        FROM edges
        WHERE start=?
        ORDER BY weight DESC
        LIMIT ?
        """,
        (node, limit)
    )
    return cur.fetchall()


# ============================================================
# Load files
# ============================================================

def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # In case file is {idx: item}
        out = []
        for k, v in data.items():
            if isinstance(v, dict):
                if "idx" not in v:
                    v["idx"] = int(k)
                out.append(v)
        return out

    if not isinstance(data, list):
        raise ValueError(f"Expected list or dict JSON from {path}")

    return data


def index_by_idx(data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for item in data:
        if "idx" not in item:
            continue
        out[int(item["idx"])] = item
    return out


# ============================================================
# Entity extraction from different files
# ============================================================

def collect_entities_from_previous_item(item: Dict[str, Any]) -> List[str]:
    """
    Collect entities from the previous image-only file.
    Supports your older format:
        word_groups
        extracted.noun_phrases
        extracted.root_verbs
        extracted.entities
    """
    ents = []

    ents.extend(item.get("word_groups", []) or [])

    extracted = item.get("extracted", {}) or {}
    ents.extend(extracted.get("entities", []) or [])
    ents.extend(extracted.get("noun_phrases", []) or [])
    ents.extend(extracted.get("root_verbs", []) or [])

    # Sometimes previous files already contain "entities".
    ents.extend(item.get("entities", []) or [])

    return ents


def collect_entities_from_text_item(item: Dict[str, Any]) -> List[str]:
    """
    From text_entties_qwen.json:
        entities: [...]
    """
    ents = []
    ents.extend(item.get("entities", []) or [])
    return ents


def collect_entities_from_img_item(item: Dict[str, Any], include_states: bool = True) -> List[str]:
    """
    From img_entities_qwen.json:
        main_entities
        options[A/B/C/D].objects
        options[A/B/C/D].states
    """
    ents = []

    ents.extend(item.get("main_entities", []) or [])

    options = item.get("options", {}) or {}
    for opt_key, opt_val in options.items():
        if not isinstance(opt_val, dict):
            continue

        ents.extend(opt_val.get("objects", []) or [])

        if include_states:
            ents.extend(opt_val.get("states", []) or [])

    return ents


def build_combined_entities_for_idx(
    idx: int,
    previous_by_idx: Dict[int, Dict[str, Any]],
    text_by_idx: Dict[int, Dict[str, Any]],
    img_by_idx: Dict[int, Dict[str, Any]],
    include_states: bool = True
) -> List[str]:
    ents = []

    if idx in previous_by_idx:
        ents.extend(collect_entities_from_previous_item(previous_by_idx[idx]))

    if idx in text_by_idx:
        ents.extend(collect_entities_from_text_item(text_by_idx[idx]))

    if idx in img_by_idx:
        ents.extend(collect_entities_from_img_item(img_by_idx[idx], include_states=include_states))

    cleaned = []
    for e in ents:
        e = clean_entity(e)
        if e:
            cleaned.append(e)

    return dedup_keep_order(cleaned)


# ============================================================
# Anchor selection
# ============================================================

def entity_priority_score(ent: str) -> int:
    """
    Prefer concrete object phrases over vague states/actions.

    Higher score = earlier selected.
    """
    ent = normalize_entity_text(ent)

    score = 0

    # Prefer multi-word object names such as wine_glass, convex_lens.
    if "_" in ent:
        score += 2

    # Penalize likely state words.
    state_words = {
        "empty", "intact", "shattering", "shattered", "flying",
        "swirling", "closer", "sliding", "slide", "moving",
        "broken", "falling", "floating", "burning"
    }
    if ent in state_words:
        score -= 2

    # Prefer physical objects/materials.
    object_hints = {
        "glass", "ball", "cube", "block", "lens", "light", "smoke",
        "fragment", "fragments", "water", "liquid", "ramp", "table",
        "container", "bottle", "cup", "wheel", "stick", "rope"
    }
    for h in object_hints:
        if h in ent.split("_"):
            score += 3

    # Very long phrases are less likely to exist in ConceptNet.
    n_words = len(ent.split("_"))
    if n_words >= 5:
        score -= 3

    return score


def select_anchors_from_entities(entities: List[str], max_anchors: int = 8) -> List[str]:
    """
    Select final ConceptNet anchor phrases from combined entities.
    """
    cleaned = []
    for e in entities:
        e = clean_entity(e)
        if e:
            cleaned.append(e)

    cleaned = dedup_keep_order(cleaned)

    # Sort by priority, but keep deterministic ordering for ties.
    cleaned = sorted(
        cleaned,
        key=lambda x: (-entity_priority_score(x), cleaned.index(x))
    )

    return cleaned[:max_anchors]


# ============================================================
# Graph
# ============================================================

def build_graph(conn, anchors: List[str], max_leaves: int = 5, neighbor_limit: int = 50):
    G = nx.MultiDiGraph()

    for a in anchors:
        concept_uri = normalize_text_to_concept(a)
        if not concept_uri:
            continue

        G.add_node(concept_uri, label=a, node_type="anchor")

        neighbors = get_neighbors(conn, concept_uri, limit=neighbor_limit)

        count = 0
        for s, e, r, w, meta in neighbors:
            if count >= max_leaves:
                break

            # Skip edges to very abstract noisy nodes.
            end_plain = concept_to_plain(e)
            if is_bad_entity(end_plain):
                continue

            G.add_node(e, label=end_plain, node_type="neighbor")
            G.add_edge(
                s,
                e,
                rel=rel_label(r),
                weight=float(w),
                source_json=meta
            )
            count += 1

    return G


def graph_to_triples(G):
    triples = []

    for u, v, d in G.edges(data=True):
        try:
            meta = json.loads(d.get("source_json", "{}"))
            surface = meta.get("surfaceText", "")
        except Exception:
            surface = ""

        triples.append({
            "start": u,
            "start_label": concept_to_plain(u),
            "rel": d.get("rel", ""),
            "end": v,
            "end_label": concept_to_plain(v),
            "weight": float(d.get("weight", 1.0)),
            "surfaceText": surface
        })

    return triples


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    # This is your previous image-only data file.
    # It decides which idx values will be processed.
    parser.add_argument("--input_json", required=True)

    # New Qwen entity files.
    parser.add_argument("--img_entities_json", required=True)
    parser.add_argument("--text_entities_json", required=True)

    parser.add_argument("--conceptnet_csv", required=True)
    parser.add_argument("--sqlite_db", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    parser.add_argument("--max_anchors", type=int, default=8)
    parser.add_argument("--max_leaves", type=int, default=5)
    parser.add_argument("--neighbor_limit", type=int, default=50)
    parser.add_argument("--min_weight", type=float, default=0.5)

    parser.add_argument(
        "--include_states",
        action="store_true",
        help="Include option states from img_entities_qwen.json as possible entities."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load image-only previous file.
    # This file is used as the filter.
    # ------------------------------------------------------------
    previous_data = load_json_list(args.input_json)
    previous_by_idx = index_by_idx(previous_data)

    image_only_indices = sorted(previous_by_idx.keys())

    total = len(image_only_indices)
    start = args.start
    end = args.end if args.end is not None else total
    end = min(end, total)

    selected_indices = image_only_indices[start:end]

    print(f"[INFO] Image-only filter file: {args.input_json}")
    print(f"[INFO] Number of image-only samples: {total}")
    print(f"[INFO] Processing position range {start}-{end}, actual samples: {len(selected_indices)}")

    # ------------------------------------------------------------
    # Load Qwen entity files.
    # ------------------------------------------------------------
    img_data = load_json_list(args.img_entities_json)
    text_data = load_json_list(args.text_entities_json)

    img_by_idx = index_by_idx(img_data)
    text_by_idx = index_by_idx(text_data)

    print(f"[INFO] Loaded img entities: {len(img_by_idx)}")
    print(f"[INFO] Loaded text entities: {len(text_by_idx)}")

    # ------------------------------------------------------------
    # Build SQLite index once.
    # ------------------------------------------------------------
    build_sqlite_index(
        args.conceptnet_csv,
        args.sqlite_db,
        min_weight=args.min_weight
    )

    conn = connect_sqlite(args.sqlite_db)

    results = {}

    num_with_entities = 0
    num_with_anchors = 0
    num_with_triples = 0

    for pos, idx_int in enumerate(selected_indices):
        idx = str(idx_int)

        try:
            combined_entities = build_combined_entities_for_idx(
                idx_int,
                previous_by_idx=previous_by_idx,
                text_by_idx=text_by_idx,
                img_by_idx=img_by_idx,
                include_states=args.include_states
            )

            anchors_raw = select_anchors_from_entities(
                combined_entities,
                max_anchors=args.max_anchors
            )

            if combined_entities:
                num_with_entities += 1

            if anchors_raw:
                num_with_anchors += 1

            if len(anchors_raw) == 0:
                results[idx] = {
                    "idx": idx_int,
                    "seeds": combined_entities,
                    "anchors": [],
                    "triples": []
                }
                continue

            G = build_graph(
                conn,
                anchors_raw,
                max_leaves=args.max_leaves,
                neighbor_limit=args.neighbor_limit
            )

            triples = graph_to_triples(G)

            if triples:
                num_with_triples += 1

            results[idx] = {
                "idx": idx_int,
                "seeds": combined_entities,
                "anchors": anchors_raw,
                "triples": triples
            }

            if (pos + 1) % 100 == 0:
                print(
                    f"[INFO] Processed {pos + 1}/{len(selected_indices)} "
                    f"| idx={idx_int} "
                    f"| entities={len(combined_entities)} "
                    f"| anchors={len(anchors_raw)} "
                    f"| triples={len(triples)}"
                )

        except Exception as e:
            print(f"[ERROR] idx={idx}: {e}")
            results[idx] = {
                "idx": idx_int,
                "seeds": [],
                "anchors": [],
                "triples": [],
                "error": str(e)
            }

    conn.close()

    output_file = os.path.join(
        args.output_dir,
        f"kg_{start}_{end}.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("===== Summary =====")
    print(f"Image-only samples in this chunk: {len(selected_indices)}")
    print(f"Samples with combined entities:   {num_with_entities}")
    print(f"Samples with anchors:             {num_with_anchors}")
    print(f"Samples with triples:             {num_with_triples}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
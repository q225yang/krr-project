#!/usr/bin/env python3

import argparse
import json
import os
import re
import sqlite3
from typing import List, Set
from collections import defaultdict
import networkx as nx

# ============================================================
# Basic helpers
# ============================================================

def normalize_text_to_concept(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return f"/c/en/{text}"


def is_english_concept(uri: str) -> bool:
    return uri.startswith("/c/en/")


def rel_label(rel_uri: str) -> str:
    return rel_uri.split("/")[-1]


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

    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue

            _, rel, start, end, meta = parts

            if not (is_english_concept(start) and is_english_concept(end)):
                continue

            try:
                meta_json = json.loads(meta)
                weight = float(meta_json.get("weight", 1.0))
            except:
                weight = 1.0

            if weight < min_weight:
                continue

            conn.execute(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                (start, end, rel, weight, meta)
            )
            conn.execute(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
                (end, start, rel, weight, meta)
            )

    conn.commit()
    conn.close()
    print("[INFO] SQLite built")


def get_neighbors(conn, node, limit=50):
    cur = conn.cursor()
    cur.execute(
        "SELECT start, end, rel, weight, source_json FROM edges WHERE start=? LIMIT ?",
        (node, limit)
    )
    return cur.fetchall()


# ============================================================
# Graph
# ============================================================

def build_graph(conn, anchors: List[str], max_leaves=5):
    G = nx.MultiDiGraph()

    for a in anchors:
        G.add_node(a)

        neighbors = get_neighbors(conn, a, limit=50)

        count = 0
        for s, e, r, w, meta in neighbors:
            if count >= max_leaves:
                break

            G.add_edge(
                s, e,
                rel=rel_label(r),
                weight=w,
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
        except:
            surface = ""

        triples.append({
            "start": u,
            "rel": d.get("rel", ""),
            "end": v,
            "weight": float(d.get("weight", 1.0)),
            "surfaceText": surface
        })

    return triples


# ============================================================
# Anchor selection
# ============================================================

def clean_phrase(p: str):
    p = p.lower().strip()

    bad = {
        "which object", "we", "which area",
        "the picture", "the image", "all parts"
    }
    if p in bad:
        return ""
    return p


def select_anchors(item, max_anchors=5):
    anchors = []

    for w in item.get("word_groups", []):
        w = clean_phrase(w)
        if w:
            anchors.append(w)

    anchors = list(dict.fromkeys(anchors))[:max_anchors]
    return anchors


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--conceptnet_csv", required=True)
    parser.add_argument("--sqlite_db", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load data
    with open(args.input_json) as f:
        data = json.load(f)

    total = len(data)
    start = args.start
    end = args.end if args.end else total
    end = min(end, total)

    data_slice = data[start:end]

    print(f"[INFO] Processing {start}-{end} / {total}")

    # build sqlite once
    build_sqlite_index(args.conceptnet_csv, args.sqlite_db)

    conn = connect_sqlite(args.sqlite_db)

    results = {}

    for item in data_slice:
        idx = str(item["idx"])

        try:
            # seeds
            seeds = []
            seeds += item.get("word_groups", [])
            seeds += item["extracted"].get("noun_phrases", [])
            seeds += item["extracted"].get("root_verbs", [])

            seeds = [s.lower().replace(" ", "_") for s in seeds if s.strip()]
            seeds = list(dict.fromkeys(seeds))

            # anchors
            anchors_raw = select_anchors(item)
            anchors = [normalize_text_to_concept(a) for a in anchors_raw]

            if len(anchors) < 2:
                results[idx] = {"idx": int(idx), "seeds": seeds, "triples": []}
                continue

            G = build_graph(conn, anchors)
            triples = graph_to_triples(G)

            results[idx] = {
                "idx": int(idx),
                "seeds": seeds,
                "triples": triples
            }

        except Exception as e:
            print(f"[ERROR] {idx}: {e}")
            results[idx] = {"idx": int(idx), "seeds": [], "triples": []}

    conn.close()

    output_file = os.path.join(
        args.output_dir,
        f"kg_{start}_{end}.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved {output_file}")


if __name__ == "__main__":
    main()
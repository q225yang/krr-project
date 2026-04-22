#!/usr/bin/env python3
"""
Build a question-centered snowflake graph from a local ConceptNet 5.7 assertions file.

Graph structure:
- Anchor nodes = question concepts / important concepts
- Leaf nodes = property/material/affordance/type facts around each anchor
- Bridge nodes = short meaningful intermediate nodes connecting anchors

This is designed for VQA / physical reasoning use cases where:
- anchors should remain explicit
- each anchor should have its own supporting facts
- anchors may be connected if meaningful short paths exist
- disconnected anchors are allowed

Example:
python build_conceptnet_snowflake_graph.py \
    --conceptnet_csv /scratch/qyang129/conceptnet/conceptnet-assertions-5.7.0.csv \
    --anchors liquid particle flow area \
    --output_prefix liquid_snowflake

Outputs:
- liquid_snowflake.graphml
- liquid_snowflake.json
- liquid_snowflake_summary.txt
- conceptnet_en_neighbors.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx


# ============================================================
# Helpers
# ============================================================

def normalize_text_to_concept(text: str) -> str:
    text = text.strip()
    if text.startswith("/c/"):
        return text
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return f"/c/en/{text}"


def is_english_concept(uri: str) -> bool:
    return uri.startswith("/c/en/")


def short_label(uri: str) -> str:
    if not uri.startswith("/c/en/"):
        return uri
    parts = uri.split("/")
    if len(parts) >= 4:
        return parts[3].replace("_", " ")
    return uri


def rel_label(rel_uri: str) -> str:
    return rel_uri.split("/")[-1]


def is_generic_hub(uri: str) -> bool:
    generic = {
        "/c/en/object",
        "/c/en/entity",
        "/c/en/thing",
        "/c/en/stuff",
        "/c/en/item",
        "/c/en/person",
        "/c/en/place",
        "/c/en/location",
        "/c/en.activity",
        "/c/en.action",
    }
    return uri in generic


# ============================================================
# SQLite index
# ============================================================

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    rel TEXT NOT NULL,
    weight REAL NOT NULL,
    direction INTEGER NOT NULL,
    edge_uri TEXT,
    source_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_edges_start ON edges(start);
CREATE INDEX IF NOT EXISTS idx_edges_end ON edges(end);
CREATE INDEX IF NOT EXISTS idx_edges_start_rel ON edges(start, rel);
CREATE INDEX IF NOT EXISTS idx_edges_end_rel ON edges(end, rel);
CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight);
"""


@dataclass
class EdgeRecord:
    start: str
    end: str
    rel: str
    weight: float
    direction: int
    edge_uri: str
    source_json: str


def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn


def db_has_edges(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='edges'")
    if cur.fetchone() is None:
        return False
    cur.execute("SELECT COUNT(*) FROM edges")
    return cur.fetchone()[0] > 0


def build_sqlite_index(
    conceptnet_csv: str,
    db_path: str,
    english_only: bool = True,
    min_weight: float = 0.5,
    allowed_relations: Optional[Set[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = connect_sqlite(db_path)
    conn.executescript(SCHEMA_SQL)

    if db_has_edges(conn):
        print(f"[INFO] SQLite index already exists: {db_path}")
        conn.close()
        return

    print(f"[INFO] Building SQLite index from: {conceptnet_csv}")
    print("[INFO] One-time preprocessing. This may take a while.")

    insert_sql = """
        INSERT INTO edges (start, end, rel, weight, direction, edge_uri, source_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    batch_size = 50000
    seen = 0
    kept = 0

    with open(conceptnet_csv, "r", encoding="utf-8") as f:
        for line in f:
            seen += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue

            edge_uri, rel, start, end, meta_json = parts

            if english_only and not (is_english_concept(start) and is_english_concept(end)):
                continue

            if allowed_relations is not None and rel_label(rel) not in allowed_relations:
                continue

            try:
                meta = json.loads(meta_json)
                weight = float(meta.get("weight", 1.0))
            except Exception:
                weight = 1.0

            if weight < min_weight:
                continue

            batch.append((start, end, rel, weight, 1, edge_uri, meta_json))
            batch.append((end, start, rel, weight, -1, edge_uri, meta_json))
            kept += 1

            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

            if seen % 1_000_000 == 0:
                print(f"[INFO] processed={seen:,} kept={kept:,}")

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM edges")
    total = cur.fetchone()[0]
    print(f"[INFO] Index complete. Stored rows: {total:,}")
    conn.close()


# ============================================================
# Relation configuration
# ============================================================

PROPERTY_RELATIONS = {
    "HasProperty",
    "MadeOf",
    "UsedFor",
    "CapableOf",
    "IsA",
    "PartOf",
    "HasA",
}

BRIDGE_RELATIONS = {
    "AtLocation",
    "LocatedNear",
    "PartOf",
    "HasA",
    "UsedFor",
    "CapableOf",
    "IsA",
    "MadeOf",
    "HasProperty",
    "RelatedTo",
    "SimilarTo",
}

DEFAULT_ALLOWED_RELATIONS = PROPERTY_RELATIONS | BRIDGE_RELATIONS | {
    "Synonym",
    "DistinctFrom",
    "InstanceOf",
    "FormOf",
}


RELATION_QUALITY = {
    "HasProperty": 1.00,
    "MadeOf": 0.95,
    "UsedFor": 0.92,
    "CapableOf": 0.90,
    "PartOf": 0.88,
    "HasA": 0.86,
    "IsA": 0.80,
    "AtLocation": 0.72,
    "LocatedNear": 0.68,
    "SimilarTo": 0.45,
    "Synonym": 0.45,
    "RelatedTo": 0.30,
    "DistinctFrom": 0.20,
    "InstanceOf": 0.75,
    "FormOf": 0.35,
}


# ============================================================
# Retrieval
# ============================================================

def get_outgoing_neighbors(
    conn: sqlite3.Connection,
    node: str,
    top_k: int = 50,
    allowed_relations: Optional[Set[str]] = None,
) -> List[EdgeRecord]:
    cur = conn.cursor()

    if allowed_relations:
        placeholders = ",".join("?" for _ in allowed_relations)
        sql = f"""
            SELECT start, end, rel, weight, direction, edge_uri, source_json
            FROM edges
            WHERE start = ?
              AND substr(rel, 4) IN ({placeholders})
            ORDER BY weight DESC
            LIMIT ?
        """
        params = [node, *sorted(allowed_relations), top_k]
        cur.execute(sql, params)
    else:
        sql = """
            SELECT start, end, rel, weight, direction, edge_uri, source_json
            FROM edges
            WHERE start = ?
            ORDER BY weight DESC
            LIMIT ?
        """
        cur.execute(sql, (node, top_k))

    return [EdgeRecord(*row) for row in cur.fetchall()]


def relation_quality(rel_uri: str) -> float:
    return RELATION_QUALITY.get(rel_label(rel_uri), 0.25)


def score_leaf_edge(edge: EdgeRecord, anchor_set: Set[str]) -> float:
    rel = rel_label(edge.rel)
    score = 0.0
    score += 2.5 * relation_quality(edge.rel)
    score += 1.5 * min(edge.weight, 3.0)

    if edge.end in anchor_set:
        score -= 10.0  # not a leaf
    if is_generic_hub(edge.end):
        score -= 2.5
    if rel == "RelatedTo":
        score -= 1.5

    return score


def score_bridge_path(path_edges: List[EdgeRecord], anchor_set: Set[str]) -> float:
    score = 0.0
    for e in path_edges:
        score += 2.0 * relation_quality(e.rel)
        score += 1.2 * min(e.weight, 3.0)
        if rel_label(e.rel) == "RelatedTo":
            score -= 1.0
        if is_generic_hub(e.end):
            score -= 2.0

    # shorter is better
    score -= 1.5 * (len(path_edges) - 1)
    return score


# ============================================================
# Graph building
# ============================================================

def ensure_node(
    G: nx.MultiDiGraph,
    node: str,
    node_type: str,
    anchor: bool = False,
) -> None:
    if node not in G:
        G.add_node(
            node,
            label=short_label(node),
            node_type=node_type,
            anchor=bool(anchor),
        )
    else:
        if anchor:
            G.nodes[node]["anchor"] = True
            G.nodes[node]["node_type"] = "anchor"


def add_edge(
    G: nx.MultiDiGraph,
    edge: EdgeRecord,
    edge_role: str,
    score: float,
) -> None:
    ensure_node(G, edge.start, "support")
    ensure_node(G, edge.end, "support")
    G.add_edge(
        edge.start,
        edge.end,
        rel=rel_label(edge.rel),
        rel_uri=edge.rel,
        weight=edge.weight,
        edge_uri=edge.edge_uri,
        edge_role=edge_role,
        score=score,
    )


def attach_anchor_leaves(
    conn: sqlite3.Connection,
    G: nx.MultiDiGraph,
    anchor: str,
    anchor_set: Set[str],
    max_leaves: int = 5,
    top_k_query: int = 60,
) -> None:
    ensure_node(G, anchor, "anchor", anchor=True)

    neighbors = get_outgoing_neighbors(
        conn,
        node=anchor,
        top_k=top_k_query,
        allowed_relations=PROPERTY_RELATIONS,
    )

    scored = []
    seen_targets = set()
    for e in neighbors:
        if e.end == anchor:
            continue
        if e.end in seen_targets:
            continue
        seen_targets.add(e.end)

        s = score_leaf_edge(e, anchor_set)
        scored.append((s, e))

    scored.sort(key=lambda x: x[0], reverse=True)

    kept = 0
    for s, e in scored:
        if kept >= max_leaves:
            break
        if s <= 0:
            continue

        if e.end in anchor_set:
            # another anchor should not be treated as a leaf
            continue

        ensure_node(G, e.start, "anchor", anchor=(e.start in anchor_set))
        ensure_node(G, e.end, "leaf")
        add_edge(G, e, edge_role="leaf", score=s)
        kept += 1


def find_direct_anchor_edge(
    conn: sqlite3.Connection,
    a: str,
    b: str,
    top_k_query: int = 80,
) -> Optional[Tuple[float, EdgeRecord]]:
    neighbors = get_outgoing_neighbors(
        conn,
        node=a,
        top_k=top_k_query,
        allowed_relations=BRIDGE_RELATIONS,
    )
    candidates = []
    for e in neighbors:
        if e.end != b:
            continue
        s = score_bridge_path([e], anchor_set={a, b})
        candidates.append((s, e))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0]


def find_two_hop_bridge(
    conn: sqlite3.Connection,
    a: str,
    b: str,
    top_k_first: int = 80,
    top_k_second: int = 80,
) -> Optional[Tuple[float, EdgeRecord, EdgeRecord]]:
    first_hop = get_outgoing_neighbors(
        conn,
        node=a,
        top_k=top_k_first,
        allowed_relations=BRIDGE_RELATIONS,
    )

    best = None

    for e1 in first_hop:
        mid = e1.end
        if mid in {a, b}:
            continue
        if is_generic_hub(mid):
            continue

        second_hop = get_outgoing_neighbors(
            conn,
            node=mid,
            top_k=top_k_second,
            allowed_relations=BRIDGE_RELATIONS,
        )

        for e2 in second_hop:
            if e2.end != b:
                continue

            score = score_bridge_path([e1, e2], anchor_set={a, b})
            if best is None or score > best[0]:
                best = (score, e1, e2)

    return best


def connect_anchor_pairs(
    conn: sqlite3.Connection,
    G: nx.MultiDiGraph,
    anchors: Sequence[str],
    min_bridge_score: float = 3.5,
) -> None:
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            a = anchors[i]
            b = anchors[j]

            ensure_node(G, a, "anchor", anchor=True)
            ensure_node(G, b, "anchor", anchor=True)

            direct = find_direct_anchor_edge(conn, a, b)
            reverse_direct = find_direct_anchor_edge(conn, b, a)

            candidates = []
            if direct is not None:
                candidates.append(("direct", direct[0], [direct[1]]))
            if reverse_direct is not None:
                candidates.append(("direct", reverse_direct[0], [reverse_direct[1]]))

            two_hop = find_two_hop_bridge(conn, a, b)
            reverse_two_hop = find_two_hop_bridge(conn, b, a)

            if two_hop is not None:
                candidates.append(("two_hop", two_hop[0], [two_hop[1], two_hop[2]]))
            if reverse_two_hop is not None:
                candidates.append(("two_hop", reverse_two_hop[0], [reverse_two_hop[1], reverse_two_hop[2]]))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[1], reverse=True)
            best_type, best_score, best_edges = candidates[0]

            if best_score < min_bridge_score:
                continue

            for idx, e in enumerate(best_edges):
                ensure_node(G, e.start, "anchor" if e.start in anchors else "bridge", anchor=(e.start in anchors))
                end_type = "anchor" if e.end in anchors else "bridge"
                ensure_node(G, e.end, end_type, anchor=(e.end in anchors))
                add_edge(G, e, edge_role="bridge", score=best_score)


def build_snowflake_graph(
    conn: sqlite3.Connection,
    anchors: Sequence[str],
    max_leaves_per_anchor: int = 5,
    min_bridge_score: float = 3.5,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    anchor_set = set(anchors)

    for a in anchors:
        ensure_node(G, a, "anchor", anchor=True)

    for a in anchors:
        attach_anchor_leaves(
            conn=conn,
            G=G,
            anchor=a,
            anchor_set=anchor_set,
            max_leaves=max_leaves_per_anchor,
        )

    connect_anchor_pairs(
        conn=conn,
        G=G,
        anchors=anchors,
        min_bridge_score=min_bridge_score,
    )

    return G


# ============================================================
# Graph analysis
# ============================================================

def graph_mode_summary(G: nx.MultiDiGraph, anchors: Sequence[str]) -> Dict[str, float]:
    anchor_set = set(anchors)

    leaf_edges = 0
    bridge_edges = 0
    connected_anchor_pairs = 0
    total_anchor_pairs = 0

    for _, _, data in G.edges(data=True):
        if data.get("edge_role") == "leaf":
            leaf_edges += 1
        elif data.get("edge_role") == "bridge":
            bridge_edges += 1

    # anchor connectivity via bridge graph projection
    H = nx.Graph()
    for a in anchors:
        H.add_node(a)

    for u, v, data in G.edges(data=True):
        if data.get("edge_role") != "bridge":
            continue
        if u in anchor_set and v in anchor_set:
            H.add_edge(u, v)
        elif u in anchor_set or v in anchor_set:
            # bridge node exists; path counts via overall projection later
            pass

    P = nx.Graph()
    for n in G.nodes:
        P.add_node(n)
    for u, v, _ in G.edges(data=True):
        if u != v:
            P.add_edge(u, v)

    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            total_anchor_pairs += 1
            try:
                nx.shortest_path(P, anchors[i], anchors[j])
                connected_anchor_pairs += 1
            except nx.NetworkXNoPath:
                pass

    connectivity_ratio = 0.0
    if total_anchor_pairs > 0:
        connectivity_ratio = connected_anchor_pairs / total_anchor_pairs

    if connectivity_ratio < 0.3:
        mode = "property"
    elif connectivity_ratio > 0.7 and bridge_edges > 0:
        mode = "relational"
    else:
        mode = "hybrid"

    return {
        "leaf_edges": leaf_edges,
        "bridge_edges": bridge_edges,
        "connected_anchor_pairs": connected_anchor_pairs,
        "total_anchor_pairs": total_anchor_pairs,
        "connectivity_ratio": connectivity_ratio,
        "mode": mode,
    }


# ============================================================
# Export
# ============================================================
def build_display_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Convert MultiDiGraph to simple DiGraph for export.
    Keep the highest-score edge between each pair.
    """
    H = nx.DiGraph()

    for n, attrs in G.nodes(data=True):
        attrs2 = {
            k: (str(v) if isinstance(v, (list, dict, set, tuple)) else v)
            for k, v in attrs.items()
        }
        H.add_node(n, **attrs2)

    edge_counter = 0
    for u, v, attrs in G.edges(data=True):
        attrs2 = {
            k: (str(v) if isinstance(v, (list, dict, set, tuple)) else v)
            for k, v in attrs.items()
        }
        attrs2["edge_id"] = f"e{edge_counter}"
        edge_counter += 1

        if H.has_edge(u, v):
            old_score = float(H[u][v].get("score", 0.0))
            new_score = float(attrs2.get("score", 0.0))
            if new_score > old_score:
                H[u][v].clear()
                H[u][v].update(attrs2)
        else:
            H.add_edge(u, v, **attrs2)

    return H


def assign_gephi_positions(
    H: nx.DiGraph,
    anchors,
    outer_radius: float = 2000.0,
    middle_radius: float = 600.0,
    leaf_radius: float = 350.0,
):
    """
    Write x/y node positions for Gephi:
    - anchors on a large outside ring
    - bridge/support nodes in the middle
    - leaf nodes near their closest anchor
    """
    pos = {}
    anchor_set = set(anchors)
    anchor_list = [a for a in anchors if a in H.nodes]

    # 1) anchors on outer circle
    n_anchors = max(len(anchor_list), 1)
    for i, a in enumerate(anchor_list):
        theta = 2.0 * math.pi * i / n_anchors
        x = outer_radius * math.cos(theta)
        y = outer_radius * math.sin(theta)
        pos[a] = (x, y)

    # 2) middle nodes
    leaf_nodes = [n for n, d in H.nodes(data=True) if d.get("node_type") == "leaf" and n not in anchor_set]
    middle_nodes = [
        n for n, d in H.nodes(data=True)
        if n not in anchor_set and d.get("node_type") != "leaf"
    ]

    if middle_nodes:
        M = H.subgraph(middle_nodes).copy().to_undirected()
        if M.number_of_nodes() == 1:
            only = next(iter(M.nodes))
            pos[only] = (0.0, 0.0)
        else:
            spring_pos = nx.spring_layout(M, seed=42, k=1.2)
            for n in middle_nodes:
                x, y = spring_pos[n]
                pos[n] = (x * middle_radius, y * middle_radius)

    # 3) assign each leaf to a nearby anchor
    anchor_to_leaves = {a: [] for a in anchor_list}

    for leaf in leaf_nodes:
        direct_anchor_neighbors = [
            nbr for nbr in H.predecessors(leaf) if nbr in anchor_set
        ] + [
            nbr for nbr in H.successors(leaf) if nbr in anchor_set
        ]

        if direct_anchor_neighbors:
            chosen_anchor = direct_anchor_neighbors[0]
        else:
            chosen_anchor = None
            best_dist = float("inf")
            U = H.to_undirected()
            for a in anchor_list:
                try:
                    d = nx.shortest_path_length(U, source=leaf, target=a)
                    if d < best_dist:
                        best_dist = d
                        chosen_anchor = a
                except nx.NetworkXNoPath:
                    continue

        if chosen_anchor is not None:
            anchor_to_leaves[chosen_anchor].append(leaf)
        else:
            pos[leaf] = (0.0, 0.0)

    # 4) place leaves around each anchor
    for a, leaves in anchor_to_leaves.items():
        if not leaves:
            continue

        ax, ay = pos[a]
        anchor_angle = math.atan2(ay, ax)
        m = len(leaves)
        spread = math.pi / 2.0

        for i, leaf in enumerate(leaves):
            if m == 1:
                offset = anchor_angle
            else:
                offset = anchor_angle - spread / 2.0 + spread * i / (m - 1)

            lx = ax + leaf_radius * math.cos(offset)
            ly = ay + leaf_radius * math.sin(offset)
            pos[leaf] = (lx, ly)

    # 5) fallback
    for n in H.nodes:
        if n not in pos:
            pos[n] = (0.0, 0.0)

    # 6) write back to node attributes for Gephi
    for n, (x, y) in pos.items():
        H.nodes[n]["x"] = float(x)
        H.nodes[n]["y"] = float(y)

        # optional extras Gephi can also read
        H.nodes[n]["label"] = H.nodes[n].get("label", n.split("/")[-1].replace("_", " "))
        H.nodes[n]["size"] = (
            60.0 if n in anchor_set
            else 38.0 if H.nodes[n].get("node_type") == "bridge"
            else 28.0 if H.nodes[n].get("node_type") == "leaf"
            else 24.0
        )

def export_graph(G: nx.MultiDiGraph, output_prefix: str, anchors: Sequence[str]) -> None:
    graphml_path = f"{output_prefix}.graphml"
    json_path = f"{output_prefix}.json"
    summary_path = f"{output_prefix}_summary.txt"

    # Build a simple directed graph for export
    H = build_display_graph(G)

    # Assign explicit x/y positions for Gephi Lite
    assign_gephi_positions(
        H,
        anchors=anchors,
        outer_radius=2000.0,
        middle_radius=650.0,
        leaf_radius=320.0,
    )

    # Export GraphML with x/y attributes
    nx.write_graphml(H, graphml_path)

    json_data = {
        "anchors": list(anchors),
        "nodes": [
            {
                "id": n,
                "label": attrs.get("label", short_label(n)),
                "node_type": attrs.get("node_type", "support"),
                "anchor": bool(attrs.get("anchor", False)),
                "x": float(attrs.get("x", 0.0)),
                "y": float(attrs.get("y", 0.0)),
            }
            for n, attrs in H.nodes(data=True)
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "rel": attrs.get("rel", ""),
                "weight": float(attrs.get("weight", 1.0)),
                "edge_role": attrs.get("edge_role", ""),
                "score": float(attrs.get("score", 0.0)),
            }
            for u, v, attrs in H.edges(data=True)
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    summary = graph_mode_summary(G, anchors)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Anchors: {len(anchors)}\n")
        f.write("Anchor list:\n")
        for a in anchors:
            f.write(f"  - {a} ({short_label(a)})\n")
        f.write("\n")
        f.write(f"Leaf edges: {summary['leaf_edges']}\n")
        f.write(f"Bridge edges: {summary['bridge_edges']}\n")
        f.write(f"Connected anchor pairs: {summary['connected_anchor_pairs']}/{summary['total_anchor_pairs']}\n")
        f.write(f"Connectivity ratio: {summary['connectivity_ratio']:.4f}\n")
        f.write(f"Graph mode: {summary['mode']}\n")

    print(f"[INFO] Wrote: {graphml_path}")
    print(f"[INFO] Wrote: {json_path}")
    print(f"[INFO] Wrote: {summary_path}")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conceptnet_csv",
        type=str,
        required=True,
        help="Path to local conceptnet-assertions-5.7.0.csv",
    )
    parser.add_argument(
        "--sqlite_db",
        type=str,
        default="conceptnet_en_neighbors.sqlite",
        help="Path to SQLite cache/index",
    )
    parser.add_argument(
        "--anchors",
        nargs="+",
        required=True,
        help="Anchor concepts, usually from the question",
    )
    parser.add_argument(
        "--max_leaves_per_anchor",
        type=int,
        default=5,
        help="Max property leaves attached to each anchor",
    )
    parser.add_argument(
        "--min_bridge_score",
        type=float,
        default=3.5,
        help="Minimum score required to keep an anchor bridge",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.5,
        help="Minimum ConceptNet edge weight kept in index",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="snowflake_graph",
        help="Output file prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    anchors = [normalize_text_to_concept(x) for x in args.anchors]
    print("[INFO] Anchor concepts:")
    for a in anchors:
        print(f"  - {a}")

    build_sqlite_index(
        conceptnet_csv=args.conceptnet_csv,
        db_path=args.sqlite_db,
        english_only=True,
        min_weight=args.min_weight,
        allowed_relations=DEFAULT_ALLOWED_RELATIONS,
    )

    conn = connect_sqlite(args.sqlite_db)

    G = build_snowflake_graph(
        conn=conn,
        anchors=anchors,
        max_leaves_per_anchor=args.max_leaves_per_anchor,
        min_bridge_score=args.min_bridge_score,
    )

    print(f"[INFO] Snowflake graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    export_graph(G, args.output_prefix, anchors)

    conn.close()


if __name__ == "__main__":
    main()
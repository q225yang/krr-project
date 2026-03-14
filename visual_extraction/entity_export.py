"""Visual entity normalisation and export module.

Aggregates captions, detected objects, and scene-graph relations into a single
canonical visual entity list per image.  Entity strings are:

* lowercased and stripped of punctuation,
* lemmatised with spaCy,
* deduplicated,
* mapped to canonical ConceptNet node names via :data:`SYNONYM_MAP`.

Outputs
-------
``{output_dir}/visual_entities.jsonl``
    One JSON object per line — the handoff artefact for downstream
    Question Parsing and KG Retrieval modules::

        {
          "image_id": "42",
          "entities": ["ball", "ramp", "inclined_plane"],
          "relations": ["ball on_top_of ramp", "ball sliding_down ramp"],
          "caption": "A red ball sits at the top of a wooden ramp."
        }
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from .progress_utils import progress_percent, should_log_progress, should_warn_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConceptNet synonym map — maps surface forms to canonical node names
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, str] = {
    # Shapes / objects
    "sphere": "ball",
    "globe": "ball",
    "orb": "ball",
    "disc": "disk",
    "disk": "disk",
    "cuboid": "box",
    "cube": "box",
    "rectangular_box": "box",
    "container": "box",
    "wedge": "ramp",
    "slope": "ramp",
    "inclined_plane": "ramp",
    "incline": "ramp",
    "plank": "board",
    "slab": "board",
    "ledge": "shelf",
    # Materials
    "wooden": "wood",
    "timber": "wood",
    "metallic": "metal",
    "iron": "metal",
    "steel": "metal",
    "plastic": "plastic",
    "rubber": "rubber",
    # Physics concepts
    "liquid": "water",
    "fluid": "water",
    "gravity": "gravity",
    "friction": "friction",
    "force": "force",
    "velocity": "velocity",
    "acceleration": "acceleration",
    "momentum": "momentum",
    "pressure": "pressure",
    "weight": "weight",
    "mass": "mass",
    # General synonyms
    "automobile": "car",
    "vehicle": "car",
    "tube": "cylinder",
    "pipe": "cylinder",
    "rope": "string",
    "cord": "string",
    "chain": "chain",
    "hook": "hook",
    "pulley": "pulley",
    "lever": "lever",
    "spring": "spring",
    "pendulum": "pendulum",
}

# Minimum entity length to keep after normalisation
_MIN_ENTITY_LEN = 2

# spaCy model (lazy-loaded)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except ModuleNotFoundError:
            logger.warning(
                "spaCy is not installed. Lemmatisation will be skipped. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
            _nlp = None
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _nlp = None
    return _nlp


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_entity(text: str) -> str:
    """Normalise a single entity string.

    Steps:
    1. Lowercase and strip punctuation.
    2. Collapse whitespace; replace spaces with underscores.
    3. Lemmatise with spaCy (if available).
    4. Apply :data:`SYNONYM_MAP` lookup.

    Parameters
    ----------
    text:
        Raw entity string (e.g. ``"Wooden Ramps"``, ``"sphere"``).

    Returns
    -------
    str
        Normalised canonical form (e.g. ``"ramp"``).
    """
    s = text.strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", "_", s).strip("_")

    # Lemmatise each token
    nlp = _get_nlp()
    if nlp is not None:
        tokens = s.replace("_", " ")
        doc = nlp(tokens)
        lemmas = [tok.lemma_ for tok in doc if not tok.is_stop or len(doc) == 1]
        if lemmas:
            s = "_".join(lemmas)

    # Apply synonym map
    s = SYNONYM_MAP.get(s, s)
    return s


def _extract_entities_from_caption(caption: str) -> list[str]:
    """Extract noun/noun-chunk entities from a caption using spaCy."""
    nlp = _get_nlp()
    if nlp is None or not caption:
        return []
    doc = nlp(caption)
    entities: list[str] = []
    for chunk in doc.noun_chunks:
        root = chunk.root.lemma_.lower()
        entities.append(root)
    for ent in doc.ents:
        entities.append(ent.text.lower())
    return entities


def _relations_to_strings(nodes: list[dict], edges: list[dict]) -> list[str]:
    """Convert scene-graph edges to human-readable relation strings."""
    id_to_label: dict[int, str] = {n["id"]: n["label"] for n in nodes}
    relation_strings: list[str] = []
    for edge in edges:
        src = id_to_label.get(edge.get("source"), "")
        tgt = id_to_label.get(edge.get("target"), "")
        rel = edge.get("relation", "")
        if src and tgt and rel:
            relation_strings.append(f"{src} {rel} {tgt}")
    return relation_strings


# ---------------------------------------------------------------------------
# Per-image aggregation
# ---------------------------------------------------------------------------

def aggregate_entities(
    image_id: str,
    caption: str,
    objects: list[dict],
    scene_graph: dict,
) -> dict:
    """Build a normalised visual entity record for one image.

    Parameters
    ----------
    image_id:
        Unique identifier for the image.
    caption:
        Natural-language caption string.
    objects:
        List of object dicts from object detection.
    scene_graph:
        Scene graph dict with ``nodes`` and ``edges`` lists.

    Returns
    -------
    dict
        Record matching the ``visual_entities.jsonl`` schema.
    """
    raw_entities: list[str] = []

    # From objects
    for obj in objects:
        label = obj.get("label", "")
        if label:
            raw_entities.append(label)
        for attr_key in ("color", "shape", "material"):
            val = obj.get(attr_key)
            if val:
                raw_entities.append(val)

    # From caption (noun chunks)
    raw_entities.extend(_extract_entities_from_caption(caption))

    # From scene graph nodes
    nodes = scene_graph.get("nodes", [])
    edges = scene_graph.get("edges", [])
    for node in nodes:
        raw_entities.append(node.get("label", ""))

    # Normalise, deduplicate
    seen: set[str] = set()
    norm_entities: list[str] = []
    for raw in raw_entities:
        if not raw:
            continue
        norm = normalize_entity(raw)
        if len(norm) >= _MIN_ENTITY_LEN and norm not in seen:
            seen.add(norm)
            norm_entities.append(norm)

    # Relations as strings
    relations = _relations_to_strings(nodes, edges)

    return {
        "image_id": image_id,
        "entities": norm_entities,
        "relations": relations,
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_entities(
    manifest: list[dict],
    output_dir: Path,
    captions_map: dict[str, str],
    objects_map: dict[str, list[dict]],
    scene_graphs_map: dict[str, dict],
) -> list[dict]:
    """Aggregate and export visual entities for all images in *manifest*.

    Parameters
    ----------
    manifest:
        List of records from ``subset_manifest.json``.
    output_dir:
        Directory where ``visual_entities.jsonl`` will be written.
    captions_map:
        Mapping image_id → caption string.
    objects_map:
        Mapping image_id → list of object dicts.
    scene_graphs_map:
        Mapping image_id → scene graph dict.

    Returns
    -------
    list[dict]
        List of visual entity records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "visual_entities.jsonl"
    results: list[dict] = []
    total = len(manifest)
    suspicious_count = 0

    logger.info("Entity export on %d images. Progress will be logged periodically.", total)

    with out_path.open("w", encoding="utf-8") as fout:
        for index, rec in enumerate(manifest, start=1):
            image_id = str(rec["image_id"])
            try:
                caption = captions_map.get(image_id, "")
                objects = objects_map.get(image_id, [])
                scene_graph = scene_graphs_map.get(image_id, {"nodes": [], "edges": []})

                entry = aggregate_entities(image_id, caption, objects, scene_graph)
            except Exception as exc:
                logger.error(
                    "Entity export failed for image_id=%s: %s",
                    image_id,
                    exc,
                    exc_info=True,
                )
                entry = {
                    "image_id": image_id,
                    "entities": [],
                    "relations": [],
                    "caption": captions_map.get(image_id, ""),
                }

            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if not entry["entities"] and (
                entry["caption"] or objects or scene_graph.get("nodes") or scene_graph.get("edges")
            ):
                suspicious_count += 1
                if should_warn_count(suspicious_count):
                    logger.warning(
                        "Entity export produced no entities for image_id=%s despite upstream data. "
                        "caption_len=%d objects=%d edges=%d",
                        image_id,
                        len(entry["caption"]),
                        len(objects),
                        len(scene_graph.get("edges", [])),
                    )

            if should_log_progress(index, total):
                logger.info(
                    "Entity export progress: %d/%d (%.1f%%) | suspicious=%d",
                    index,
                    total,
                    progress_percent(index, total),
                    suspicious_count,
                )

            logger.debug(
                "Entities image_id=%s: %d entities, %d relations.",
                image_id,
                len(entry["entities"]),
                len(entry["relations"]),
            )

    logger.info("Visual entities written to %s (%d entries).", out_path, len(results))
    return results

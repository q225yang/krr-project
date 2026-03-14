"""Scene graph construction module.

Builds a lightweight scene graph for each image.  Nodes are the detected
objects (from :mod:`~visual_extraction.object_detection`); edges are spatial
or physical relations derived by prompting the VLM with the image and the
detected object list.

Outputs
-------
``{output_dir}/scene_graphs.jsonl``
    One JSON object per line::

        {
          "image_id": "42",
          "nodes": [
            {"id": 0, "label": "ball", "attributes": {"color": "red", ...}}
          ],
          "edges": [
            {"source": 0, "target": 1, "relation": "on_top_of"}
          ]
        }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from .model_backend import VLMBackend

logger = logging.getLogger(__name__)


def _build_nodes(objects: list[dict]) -> list[dict]:
    """Convert a list of object dicts into scene-graph node records."""
    nodes: list[dict] = []
    for idx, obj in enumerate(objects):
        label = obj.get("label", "object")
        attrs = {
            k: v
            for k, v in obj.items()
            if k != "label" and v is not None
        }
        nodes.append({"id": idx, "label": label, "attributes": attrs})
    return nodes


def _resolve_edge(
    source_label: str,
    target_label: str,
    nodes: list[dict],
) -> tuple[int | None, int | None]:
    """Return node ids for *source_label* and *target_label*."""
    label_to_id: dict[str, int] = {n["label"]: n["id"] for n in nodes}
    src_id = label_to_id.get(source_label)
    tgt_id = label_to_id.get(target_label)
    return src_id, tgt_id


def build_scene_graphs(
    manifest: list[dict],
    output_dir: Path,
    model: VLMBackend,
    objects_map: dict[str, list[dict]],
) -> list[dict]:
    """Build scene graphs for all images in *manifest*.

    Parameters
    ----------
    manifest:
        List of records from ``subset_manifest.json``.
    output_dir:
        Directory where ``scene_graphs.jsonl`` will be written.
    model:
        A loaded :class:`~visual_extraction.model_backend.VLMBackend` instance.
    objects_map:
        Mapping of image_id → list of object dicts (from object detection).

    Returns
    -------
    list[dict]
        List of scene graph records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "scene_graphs.jsonl"
    results: list[dict] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in manifest:
            image_id = str(rec["image_id"])
            image_path = Path(rec["image_path"])
            objects = objects_map.get(image_id, [])

            nodes = _build_nodes(objects)
            edges: list[dict] = []

            if not nodes:
                logger.debug("No objects for image_id=%s; skipping relation detection.", image_id)
            else:
                try:
                    image = Image.open(image_path).convert("RGB")
                    object_labels = [n["label"] for n in nodes]
                    raw_relations = model.detect_relations(image, object_labels)

                    for rel in raw_relations:
                        src_label = rel.get("source", "")
                        tgt_label = rel.get("target", "")
                        relation = rel.get("relation", "")
                        src_id, tgt_id = _resolve_edge(src_label, tgt_label, nodes)
                        if src_id is not None and tgt_id is not None and relation:
                            edges.append(
                                {
                                    "source": src_id,
                                    "target": tgt_id,
                                    "relation": relation,
                                }
                            )

                except FileNotFoundError:
                    logger.error(
                        "Image not found: %s (image_id=%s)", image_path, image_id
                    )
                except Exception as exc:
                    logger.error(
                        "Scene graph failed for image_id=%s: %s",
                        image_id,
                        exc,
                        exc_info=True,
                    )

            entry = {
                "image_id": image_id,
                "nodes": nodes,
                "edges": edges,
            }
            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(
                "Scene graph image_id=%s: %d nodes, %d edges.",
                image_id,
                len(nodes),
                len(edges),
            )

    logger.info("Scene graphs written to %s (%d entries).", out_path, len(results))
    return results


def load_scene_graphs(output_dir: Path) -> dict[str, dict]:
    """Load previously saved scene graphs from ``{output_dir}/scene_graphs.jsonl``.

    Returns
    -------
    dict[str, dict]
        Mapping of image_id → scene graph dict (nodes + edges).
    """
    path = output_dir / "scene_graphs.jsonl"
    result: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            result[rec["image_id"]] = rec
    return result

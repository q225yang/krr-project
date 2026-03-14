"""Object and attribute detection module.

For each image, extracts a list of visible objects with per-object attributes:
label, color, shape, material, and approximate size.

Implementation: Option A — structured VLM prompting.  The same
:class:`~visual_extraction.model_backend.VLMBackend` used for captioning is
prompted to return a JSON array of object descriptors.  Parsing failures fall
back to plain-text extraction.

Outputs
-------
``{output_dir}/objects.jsonl``
    One JSON object per line::

        {
          "image_id": "42",
          "objects": [
            {"label": "ball", "color": "red", "shape": "sphere",
             "material": "rubber", "size": "small"}
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

# Allowed size values (normalise anything else to null)
_VALID_SIZES = {"small", "medium", "large"}


def _normalise_size(raw: str | None) -> str | None:
    if raw is None:
        return None
    lower = raw.strip().lower()
    for v in _VALID_SIZES:
        if v in lower:
            return v
    return None


def _normalise_object(obj: dict) -> dict:
    """Normalise an object dict returned by the VLM."""

    def clean(val: str | None) -> str | None:
        if val is None:
            return None
        s = str(val).strip().lower()
        return s if s and s not in {"null", "none", "n/a", "unknown"} else None

    return {
        "label": clean(obj.get("label")) or "object",
        "color": clean(obj.get("color")),
        "shape": clean(obj.get("shape")),
        "material": clean(obj.get("material")),
        "size": _normalise_size(obj.get("size")),
    }


def detect_objects(
    manifest: list[dict],
    output_dir: Path,
    model: VLMBackend,
) -> list[dict]:
    """Detect objects and attributes for all images in *manifest*.

    Parameters
    ----------
    manifest:
        List of records from ``subset_manifest.json``.
    output_dir:
        Directory where ``objects.jsonl`` will be written.
    model:
        A loaded :class:`~visual_extraction.model_backend.VLMBackend` instance.

    Returns
    -------
    list[dict]
        List of ``{"image_id": str, "objects": list[dict]}`` records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "objects.jsonl"
    results: list[dict] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in manifest:
            image_id = str(rec["image_id"])
            image_path = Path(rec["image_path"])

            try:
                image = Image.open(image_path).convert("RGB")
                raw_objects = model.detect_objects(image)
                objects = [_normalise_object(o) for o in raw_objects]
            except FileNotFoundError:
                logger.error("Image not found: %s (image_id=%s)", image_path, image_id)
                objects = []
            except Exception as exc:
                logger.error(
                    "Object detection failed for image_id=%s: %s",
                    image_id,
                    exc,
                    exc_info=True,
                )
                objects = []

            entry = {"image_id": image_id, "objects": objects}
            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(
                "Detected %d objects for image_id=%s.", len(objects), image_id
            )

    logger.info("Objects written to %s (%d entries).", out_path, len(results))
    return results


def load_objects(output_dir: Path) -> dict[str, list[dict]]:
    """Load previously saved object lists from ``{output_dir}/objects.jsonl``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping of image_id → list of object dicts.
    """
    path = output_dir / "objects.jsonl"
    result: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            result[rec["image_id"]] = rec.get("objects", [])
    return result

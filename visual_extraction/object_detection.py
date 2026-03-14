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
import re
from pathlib import Path

from PIL import Image

from .model_backend import VLMBackend
from .progress_utils import progress_percent, should_log_progress, should_warn_count
from .text_cleanup import caption_primary_clause

logger = logging.getLogger(__name__)

# Allowed size values (normalise anything else to null)
_VALID_SIZES = {"small", "medium", "large"}
_GENERIC_OBJECT_LABELS = {"object", "thing", "item"}
_CAPTION_OBJECT_BLACKLIST = {
    "image",
    "picture",
    "photo",
    "scene",
    "type",
    "kind",
    "battery",
    "rider",
}
_nlp = None


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


def _object_warning_reason(objects: list[dict]) -> str | None:
    if not objects:
        return "no objects returned"

    labels = [str(obj.get("label", "")).strip().lower() for obj in objects]
    labels = [label for label in labels if label]
    if labels and all(label in _GENERIC_OBJECT_LABELS for label in labels):
        return "only generic object labels were returned"

    return None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except (ModuleNotFoundError, OSError):
            _nlp = None
    return _nlp


def _caption_mentions_label(caption: str, label: str) -> bool:
    caption_norm = f" {caption.lower()} "
    label_norm = re.sub(r"[_\-]+", " ", label).strip().lower()
    if not label_norm:
        return False
    if f" {label_norm} " in caption_norm:
        return True
    head = label_norm.split()[-1]
    return f" {head} " in caption_norm


def _objects_conflict_with_caption(objects: list[dict], caption: str) -> bool:
    """Return True when detected object labels are unsupported by the caption."""
    if not objects or not caption.strip():
        return False

    labels = [
        str(obj.get("label", "")).strip().lower()
        for obj in objects
        if str(obj.get("label", "")).strip()
    ]
    labels = [label for label in labels if label not in _GENERIC_OBJECT_LABELS]
    if not labels:
        return True

    return not any(_caption_mentions_label(caption, label) for label in labels)


def _simple_caption_object_fallback(caption: str) -> list[dict]:
    clause = caption_primary_clause(caption).lower()
    candidates = re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", clause)
    seen: set[str] = set()
    objects: list[dict] = []
    for token in candidates:
        if token in _CAPTION_OBJECT_BLACKLIST or token in _GENERIC_OBJECT_LABELS:
            continue
        if token in seen:
            continue
        seen.add(token)
        objects.append(
            {
                "label": token,
                "color": None,
                "shape": None,
                "material": None,
                "size": None,
            }
        )
    return objects[:5]


def _objects_from_caption(caption: str) -> list[dict]:
    """Derive object labels from a caption when structured detection fails."""
    clause = caption_primary_clause(caption)
    if not clause:
        return []

    nlp = _get_nlp()
    if nlp is None:
        return _simple_caption_object_fallback(clause)

    doc = nlp(clause)
    seen: set[str] = set()
    objects: list[dict] = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        if root.pos_ not in {"NOUN", "PROPN"}:
            continue
        label = root.lemma_.strip().lower()
        if not label or label in _CAPTION_OBJECT_BLACKLIST or label in _GENERIC_OBJECT_LABELS:
            continue
        if label in seen:
            continue
        seen.add(label)
        objects.append(
            {
                "label": label,
                "color": None,
                "shape": None,
                "material": None,
                "size": None,
            }
        )
    if objects:
        return objects
    return _simple_caption_object_fallback(clause)


def detect_objects(
    manifest: list[dict],
    output_dir: Path,
    model: VLMBackend,
    captions_map: dict[str, str] | None = None,
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
    total = len(manifest)
    suspicious_count = 0
    signature_counts: dict[str, int] = {}

    logger.info("Object detection on %d images. Progress will be logged periodically.", total)

    with out_path.open("w", encoding="utf-8") as fout:
        for index, rec in enumerate(manifest, start=1):
            image_id = str(rec["image_id"])
            image_path = Path(rec["image_path"])
            caption = captions_map.get(image_id, "") if captions_map else ""

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

            fallback_reason: str | None = None
            reason = _object_warning_reason(objects)
            if reason:
                fallback_reason = reason
            elif caption and _objects_conflict_with_caption(objects, caption):
                fallback_reason = "object labels conflict with the generated caption"

            if fallback_reason and caption:
                fallback_objects = _objects_from_caption(caption)
                if fallback_objects:
                    logger.warning(
                        "Using caption-derived object fallback for image_id=%s: %s. "
                        "original=%s fallback=%s",
                        image_id,
                        fallback_reason,
                        objects[:3],
                        fallback_objects[:3],
                    )
                    objects = fallback_objects

            entry = {"image_id": image_id, "objects": objects}
            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

            reason = _object_warning_reason(objects)
            if reason:
                suspicious_count += 1
                if should_warn_count(suspicious_count):
                    logger.warning(
                        "Suspicious object output for image_id=%s: %s. Objects=%s",
                        image_id,
                        reason,
                        objects[:3],
                    )

            signature = json.dumps(objects, sort_keys=True, separators=(",", ":"))
            if objects:
                repeat_count = signature_counts.get(signature, 0) + 1
                signature_counts[signature] = repeat_count
                if repeat_count >= 5 and should_warn_count(repeat_count):
                    logger.warning(
                        "Identical object output repeated %d times so far; latest image_id=%s. Objects=%s",
                        repeat_count,
                        image_id,
                        objects[:3],
                    )

            if should_log_progress(index, total):
                logger.info(
                    "Object detection progress: %d/%d (%.1f%%) | suspicious=%d",
                    index,
                    total,
                    progress_percent(index, total),
                    suspicious_count,
                )

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

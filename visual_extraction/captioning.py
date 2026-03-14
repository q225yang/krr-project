"""Image captioning module.

Generates a concise natural-language caption (1–3 sentences) for each image
in the subset manifest, using the shared :class:`VLMBackend`.

Outputs
-------
``{output_dir}/captions.jsonl``
    One JSON object per line::

        {"image_id": "42", "caption": "A red ball sits at the top of a ramp."}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from .model_backend import VLMBackend

logger = logging.getLogger(__name__)


def caption_images(
    manifest: list[dict],
    output_dir: Path,
    model: VLMBackend,
) -> list[dict]:
    """Generate captions for all images in *manifest* and save to JSONL.

    Parameters
    ----------
    manifest:
        List of records loaded from ``subset_manifest.json``.  Each record must
        have ``image_id`` and ``image_path``.
    output_dir:
        Directory where ``captions.jsonl`` will be written.
    model:
        A loaded :class:`~visual_extraction.model_backend.VLMBackend` instance.

    Returns
    -------
    list[dict]
        List of ``{"image_id": str, "caption": str}`` records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "captions.jsonl"
    results: list[dict] = []

    with out_path.open("w", encoding="utf-8") as fout:
        for rec in manifest:
            image_id = str(rec["image_id"])
            image_path = Path(rec["image_path"])

            try:
                image = Image.open(image_path).convert("RGB")
                caption = model.caption(image)
                entry = {"image_id": image_id, "caption": caption}
            except FileNotFoundError:
                logger.error("Image not found: %s (image_id=%s)", image_path, image_id)
                entry = {"image_id": image_id, "caption": ""}
            except Exception as exc:
                logger.error(
                    "Captioning failed for image_id=%s: %s", image_id, exc, exc_info=True
                )
                entry = {"image_id": image_id, "caption": ""}

            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug("Captioned image_id=%s: %s", image_id, entry["caption"][:80])

    logger.info("Captions written to %s (%d entries).", out_path, len(results))
    return results


def load_captions(output_dir: Path) -> dict[str, str]:
    """Load previously saved captions from ``{output_dir}/captions.jsonl``.

    Returns
    -------
    dict[str, str]
        Mapping of image_id → caption string.
    """
    path = output_dir / "captions.jsonl"
    captions: dict[str, str] = {}
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            captions[rec["image_id"]] = rec.get("caption", "")
    return captions

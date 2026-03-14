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
import re
from pathlib import Path

from PIL import Image

from .model_backend import CAPTION_PROMPT, VLMBackend
from .progress_utils import progress_percent, should_log_progress, should_warn_count
from .text_cleanup import clean_caption_text

logger = logging.getLogger(__name__)


def _normalise_caption(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _caption_warning_reason(caption: str) -> str | None:
    cleaned = caption.strip()
    if not cleaned:
        return "empty caption"

    prompt_norm = _normalise_caption(CAPTION_PROMPT)
    caption_norm = _normalise_caption(cleaned)
    if caption_norm == prompt_norm or caption_norm.startswith(prompt_norm):
        return "caption matches the instruction prompt"

    return None


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
    total = len(manifest)
    suspicious_count = 0
    caption_counts: dict[str, int] = {}

    logger.info("Captioning %d images. Progress will be logged periodically.", total)

    with out_path.open("w", encoding="utf-8") as fout:
        for index, rec in enumerate(manifest, start=1):
            image_id = str(rec["image_id"])
            image_path = Path(rec["image_path"])

            try:
                image = Image.open(image_path).convert("RGB")
                caption = clean_caption_text(model.caption(image))
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

            reason = _caption_warning_reason(entry["caption"])
            if reason:
                suspicious_count += 1
                if should_warn_count(suspicious_count):
                    logger.warning(
                        "Suspicious caption for image_id=%s: %s. Caption=%r",
                        image_id,
                        reason,
                        entry["caption"][:160],
                    )

            norm_caption = _normalise_caption(entry["caption"])
            if norm_caption:
                repeat_count = caption_counts.get(norm_caption, 0) + 1
                caption_counts[norm_caption] = repeat_count
                if repeat_count >= 5 and should_warn_count(repeat_count):
                    logger.warning(
                        "Caption output repeated %d times so far; latest image_id=%s. Caption=%r",
                        repeat_count,
                        image_id,
                        entry["caption"][:160],
                    )

            if should_log_progress(index, total):
                logger.info(
                    "Captioning progress: %d/%d (%.1f%%) | suspicious=%d",
                    index,
                    total,
                    progress_percent(index, total),
                    suspicious_count,
                )

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

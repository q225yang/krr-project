"""Dataset preparation for the visual extraction pipeline.

Loads PhysBench metadata, keeps every image-only sample in dataset order, and
writes a manifest that contains *no* answer or label fields. This matches the
coverage used by ``concept_extraction.py``.

Expected inputs
---------------
``{data_dir}/metadata.json``
    A JSON list where each record has at minimum the following keys::

        {
          "image_id":     str | int,
          "image_path":   str,          # relative or absolute path to the image
          "category":     str,          # task_type (e.g. "dynamics")
          "question_text": str,
          "mode":         str           # "image-only", "general", "image&video"
        }

    The script also accepts the raw PhysBench ``test.json`` combined with
    ``test_answer.json`` (answers column is stripped before writing).

Outputs
-------
``{output_dir}/subset_manifest.json``
    JSON list with fields: image_id, image_path, category, question_text.
    Answer / label fields are explicitly excluded.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

# Fields that must never appear in the manifest — label-leaking information.
_PROHIBITED_FIELDS: frozenset[str] = frozenset(
    {"answer", "label", "correct_answer", "ground_truth", "gt"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_prohibited(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *record* with all prohibited fields removed."""
    return {k: v for k, v in record.items() if k not in _PROHIBITED_FIELDS}


def _assert_no_leakage(records: list[dict[str, Any]]) -> None:
    """Raise *ValueError* if any record contains a prohibited field."""
    for rec in records:
        leaked = _PROHIBITED_FIELDS & rec.keys()
        if leaked:
            raise ValueError(
                f"Prohibited field(s) {leaked} found in manifest record {rec.get('image_id')}. "
                "Remove answer / label fields before writing the manifest."
            )


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    """Return the first existing path from *candidates*."""
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _resolve_raw_image_path(
    data_dir: Path,
    images_dir: Path,
    idx: Any,
    file_name_field: Any,
) -> str:
    """Resolve a robust image path for a raw PhysBench question record."""
    candidates: list[Path] = []
    search_stems: list[str] = []

    if file_name_field:
        fname = file_name_field[0] if isinstance(file_name_field, list) else file_name_field
        fname_path = Path(str(fname))
        search_stems.append(fname_path.stem)

        if fname_path.is_absolute():
            candidates.append(fname_path)
        else:
            candidates.extend(
                [
                    images_dir / fname_path,
                    data_dir / fname_path,
                    images_dir / fname_path.name,
                ]
            )

    if idx is not None:
        stem = str(idx)
        search_stems.append(stem)
        for ext in _IMAGE_EXTENSIONS:
            candidates.append(images_dir / f"{stem}{ext}")

    resolved = _resolve_existing_path(candidates)
    if resolved is not None:
        return str(resolved)

    seen_stems: set[str] = set()
    for stem in search_stems:
        if not stem or stem in seen_stems:
            continue
        seen_stems.add(stem)
        matches = sorted(p for p in images_dir.glob(f"{stem}.*") if p.is_file())
        if matches:
            return str(matches[0])

    if candidates:
        return str(candidates[0])
    return str(images_dir / f"{idx}.jpg")


def _load_raw_physbench(
    data_dir: Path,
    images_subdir: str = "images",
) -> list[dict[str, Any]]:
    """Build a metadata list from the raw PhysBench files on disk.

    Reads ``test.json`` (questions) and ``test_answer.json`` (category
    labels only — answers are stripped).  Falls back gracefully when either
    file is missing.

    Parameters
    ----------
    data_dir:
        Directory containing ``test.json`` and optionally ``test_answer.json``.
    images_subdir:
        Subdirectory under *data_dir* where images are stored (default
        ``"images"``; on the Sol cluster the unzipped folder is ``"image"``).
    """
    test_json = data_dir / "test.json"
    answer_json = data_dir / "test_answer.json"
    images_dir = data_dir / images_subdir

    if not test_json.exists():
        raise FileNotFoundError(
            f"Could not find {test_json}.  "
            "Pass --data_dir pointing to the PhysBench root directory."
        )

    questions: list[dict] = json.loads(test_json.read_text(encoding="utf-8"))

    # Build idx → category mapping from test_answer.json (no answer field)
    category_map: dict[int, str] = {}
    if answer_json.exists():
        answers = json.loads(answer_json.read_text(encoding="utf-8"))
        for rec in answers:
            idx = rec.get("idx")
            task_type = rec.get("task_type", "unknown")
            if idx is not None:
                category_map[idx] = task_type

    records: list[dict[str, Any]] = []
    for q in questions:
        idx = q.get("idx")
        mode = q.get("mode", "")
        if mode != "image-only":
            continue

        # Resolve the image path defensively because PhysBench exports vary.
        image_path = _resolve_raw_image_path(
            data_dir=data_dir,
            images_dir=images_dir,
            idx=idx,
            file_name_field=q.get("file_name"),
        )

        question_raw = q.get("question", "")
        # Strip option lines (A. / B. / …) from question text
        question_lines = [
            line for line in question_raw.splitlines()
            if line.strip() and not line.strip()[:2].rstrip(".").upper() in {"A", "B", "C", "D"}
        ]
        question_text = " ".join(question_lines).strip()

        records.append(
            {
                "image_id": str(idx),
                "image_path": image_path,
                "category": category_map.get(idx, "unknown"),
                "question_text": question_text,
                "mode": mode,
            }
        )

    return records

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_metadata(
    data_dir: Path,
    images_subdir: str = "images",
) -> list[dict[str, Any]]:
    """Load image metadata from *data_dir*.

    Tries ``{data_dir}/metadata.json`` first.  If that file does not exist,
    falls back to constructing metadata from the raw PhysBench files
    (``test.json`` + optional ``test_answer.json``).

    Returns a list of records with fields: image_id, image_path, category,
    question_text, mode.  Answer / label fields are always stripped.
    """
    metadata_path = data_dir / "metadata.json"

    if metadata_path.exists():
        logger.info("Loading metadata from %s", metadata_path)
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{metadata_path} must contain a JSON array.")
        # Strip any answer/label fields defensively
        records = [_strip_prohibited(r) for r in records]
    else:
        logger.info(
            "metadata.json not found; building from raw PhysBench files in %s", data_dir
        )
        records = _load_raw_physbench(data_dir, images_subdir=images_subdir)

    return records


def prepare_subset(
    data_dir: Path,
    output_dir: Path,
    subset_size: int | None = None,
    images_subdir: str = "images",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Load metadata, keep image-only records, and write the manifest.

    Parameters
    ----------
    data_dir:
        Root directory of the PhysBench dataset.
    output_dir:
        Directory where ``subset_manifest.json`` will be written.
    subset_size:
        Optional cap on the number of image-only records to keep, preserving
        dataset order. If omitted, all image-only records are kept.
    images_subdir:
        Subdirectory under *data_dir* where images live.
    seed:
        Deprecated and ignored. Present only for CLI compatibility.

    Returns
    -------
    list[dict]
        The sampled manifest records (same content as the written JSON file).
    """
    records = load_metadata(data_dir, images_subdir=images_subdir)

    # Filter to image-only mode
    image_only = [r for r in records if r.get("mode", "image-only") == "image-only"]
    logger.info(
        "Found %d image-only records out of %d total.", len(image_only), len(records)
    )

    selected_records = image_only
    if subset_size is not None:
        if subset_size <= 0:
            raise ValueError("subset_size must be a positive integer when provided.")
        selected_records = image_only[:subset_size]
        logger.info(
            "Using only the first %d image-only records for this run "
            "(full dataset available when subset_size is omitted).",
            subset_size,
        )
    if seed is not None:
        logger.info(
            "Ignoring deprecated seed=%s; dataset preparation no longer samples.",
            seed,
        )

    # Build final manifest — only allowed fields, preserving dataset order.
    manifest = [
        {
            "image_id": str(rec["image_id"]),
            "image_path": rec["image_path"],
            "category": rec.get("category", "unknown"),
            "question_text": rec.get("question_text", ""),
        }
        for rec in selected_records
    ]

    # Strict leakage check before writing
    _assert_no_leakage(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Manifest written to %s (%d images).", manifest_path, len(manifest))
    return manifest

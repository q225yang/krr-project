"""End-to-end orchestration for the visual extraction pipeline.

Run with::

    python -m visual_extraction.run_pipeline \\
        --data_dir /scratch/<asuid>/krr-data \\
        --images_subdir image \\
        --output_dir outputs \\
        --model_name blip2 \\
        --subset_size 300 \\
        --device cuda

Or using the entry-point installed by pyproject.toml::

    visual-extraction-run --data_dir ... --output_dir ...

Pipeline stages executed in order
----------------------------------
1. ``data_prep``        – load metadata, stratified-sample, write manifest.
2. ``captioning``       – generate captions for each image.
3. ``object_detection`` – extract objects and attributes.
4. ``scene_graph``      – build per-image scene graphs.
5. ``entity_export``    – normalise and export visual entities.

Failures in individual images are logged and skipped; the pipeline never
crashes due to a single bad image.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (must happen before other imports so child modules inherit it)
# ---------------------------------------------------------------------------


def _configure_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    logging.info("Logging to console and %s", log_path)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-extraction-run",
        description="Run the full visual extraction pipeline on PhysBench images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/physbench"),
        help="Root directory of the PhysBench dataset (must contain metadata.json "
        "or test.json).",
    )
    parser.add_argument(
        "--images_subdir",
        type=str,
        default="images",
        help="Subdirectory under --data_dir where images are stored. "
        "On the Sol cluster after unzipping: 'image'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where all output JSONL files and the pipeline log are written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="blip2",
        choices=["blip2", "blip"],
        help="VLM backend to use. 'blip2' requires a GPU with ~10 GB VRAM; "
        "'blip' runs on CPU.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=300,
        help="Number of images to sample (stratified across categories).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Torch device to use for the VLM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling.",
    )
    parser.add_argument(
        "--skip_captioning",
        action="store_true",
        help="Skip stage 2 (captioning) and load from existing captions.jsonl.",
    )
    parser.add_argument(
        "--skip_objects",
        action="store_true",
        help="Skip stage 3 (object detection) and load from existing objects.jsonl.",
    )
    parser.add_argument(
        "--skip_scene_graphs",
        action="store_true",
        help="Skip stage 4 (scene graph) and load from existing scene_graphs.jsonl.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _run_data_prep(args: argparse.Namespace) -> list[dict]:
    from .data_prep import prepare_subset

    logger.info("=== Stage 1 / 5: Dataset preparation ===")
    manifest = prepare_subset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subset_size=args.subset_size,
        images_subdir=args.images_subdir,
        seed=args.seed,
    )
    logger.info("Stage 1 complete — %d images in manifest.", len(manifest))
    return manifest


def _load_model(args: argparse.Namespace):
    from .model_backend import VLMBackend

    logger.info(
        "Loading VLM backend: model=%s, device=%s", args.model_name, args.device
    )
    model = VLMBackend(model_name=args.model_name, device=args.device)
    logger.info("VLM backend ready (backend=%s, device=%s).", model._backend, model.device)
    return model


def _run_captioning(
    args: argparse.Namespace, manifest: list[dict], model
) -> dict[str, str]:
    from .captioning import caption_images, load_captions

    if args.skip_captioning:
        logger.info("=== Stage 2 / 5: Captioning SKIPPED (loading from disk) ===")
        return load_captions(args.output_dir)

    logger.info("=== Stage 2 / 5: Image captioning ===")
    results = caption_images(manifest, args.output_dir, model)
    logger.info("Stage 2 complete — %d captions generated.", len(results))
    return {r["image_id"]: r["caption"] for r in results}


def _run_object_detection(
    args: argparse.Namespace, manifest: list[dict], model
) -> dict[str, list[dict]]:
    from .object_detection import detect_objects, load_objects

    if args.skip_objects:
        logger.info("=== Stage 3 / 5: Object detection SKIPPED (loading from disk) ===")
        return load_objects(args.output_dir)

    logger.info("=== Stage 3 / 5: Object and attribute detection ===")
    results = detect_objects(manifest, args.output_dir, model)
    logger.info("Stage 3 complete — %d images processed.", len(results))
    return {r["image_id"]: r["objects"] for r in results}


def _run_scene_graphs(
    args: argparse.Namespace,
    manifest: list[dict],
    model,
    objects_map: dict[str, list[dict]],
) -> dict[str, dict]:
    from .scene_graph import build_scene_graphs, load_scene_graphs

    if args.skip_scene_graphs:
        logger.info("=== Stage 4 / 5: Scene graph SKIPPED (loading from disk) ===")
        return load_scene_graphs(args.output_dir)

    logger.info("=== Stage 4 / 5: Scene graph construction ===")
    results = build_scene_graphs(manifest, args.output_dir, model, objects_map)
    logger.info("Stage 4 complete — %d scene graphs built.", len(results))
    return {r["image_id"]: r for r in results}


def _run_entity_export(
    args: argparse.Namespace,
    manifest: list[dict],
    captions_map: dict[str, str],
    objects_map: dict[str, list[dict]],
    scene_graphs_map: dict[str, dict],
) -> None:
    from .entity_export import export_entities

    logger.info("=== Stage 5 / 5: Visual entity normalisation and export ===")
    results = export_entities(
        manifest=manifest,
        output_dir=args.output_dir,
        captions_map=captions_map,
        objects_map=objects_map,
        scene_graphs_map=scene_graphs_map,
    )
    logger.info("Stage 5 complete — %d entity records exported.", len(results))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    _configure_logging(args.output_dir, level=log_level)

    logger.info("Visual Extraction Pipeline starting.")
    logger.info("Configuration: %s", vars(args))

    try:
        # Stage 1 — always runs
        manifest = _run_data_prep(args)

        if not manifest:
            logger.error("Empty manifest — no image-only samples found. Aborting.")
            return 1

        # Load the VLM once and reuse across stages
        needs_model = not (
            args.skip_captioning and args.skip_objects and args.skip_scene_graphs
        )
        model = _load_model(args) if needs_model else None

        # Stage 2
        captions_map = _run_captioning(args, manifest, model)

        # Stage 3
        objects_map = _run_object_detection(args, manifest, model)

        # Stage 4
        scene_graphs_map = _run_scene_graphs(args, manifest, model, objects_map)

        # Stage 5
        _run_entity_export(args, manifest, captions_map, objects_map, scene_graphs_map)

    except Exception as exc:
        logger.critical(
            "Pipeline failed with unhandled exception: %s", exc, exc_info=True
        )
        return 1

    logger.info(
        "Pipeline complete. Outputs are in: %s", args.output_dir.resolve()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

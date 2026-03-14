"""Visual extraction pipeline for PhysBench images.

Converts raw PhysBench images into structured visual evidence for KG retrieval
and VLM prompting. Operates on image content only — no answer labels or
supervision-leaking metadata are used at any stage.

Pipeline stages
---------------
1. data_prep       – keep all image-only records from the PhysBench dataset.
2. captioning      – generate natural-language captions per image.
3. object_detection – extract visible objects with attributes.
4. scene_graph     – build lightweight scene graphs from detected objects.
5. entity_export   – normalise and export visual entities for downstream use.
6. run_pipeline    – end-to-end CLI orchestration.
"""

__version__ = "0.1.0"

__all__ = [
    "data_prep",
    "captioning",
    "object_detection",
    "scene_graph",
    "entity_export",
    "run_pipeline",
]

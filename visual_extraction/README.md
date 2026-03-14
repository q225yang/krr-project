# visual_extraction

Converts raw [PhysBench](https://huggingface.co/datasets/USC-GVL/PhysBench) images into structured visual evidence for Knowledge Graph retrieval and VLM prompting.

Operates on image content **only** — no answer labels, ground-truth annotations, or other supervision-leaking metadata are used at any stage.

---

## Pipeline overview

| Stage | Module | Output file |
|-------|--------|-------------|
| 1. Dataset preparation | `data_prep.py` | `outputs/subset_manifest.json` |
| 2. Image captioning | `captioning.py` | `outputs/captions.jsonl` |
| 3. Object & attribute detection | `object_detection.py` | `outputs/objects.jsonl` |
| 4. Scene graph construction | `scene_graph.py` | `outputs/scene_graphs.jsonl` |
| 5. Entity normalisation & export | `entity_export.py` | `outputs/visual_entities.jsonl` |

The final `visual_entities.jsonl` is the **handoff artefact** consumed by the Question Parsing and KG Retrieval modules.

---

## Requirements

Install dependencies into the project conda environment:

```bash
# Add to environment.yml then run:
scripts/setup_env.sh

# Or install directly:
conda activate krr
pip install torch torchvision transformers pillow spacy tqdm
python -m spacy download en_core_web_sm
```

GPU requirements:
- **BLIP-2** (`--model_name blip2`): NVIDIA GPU with ≥ 10 GB VRAM (e.g. A100 on Sol cluster).
- **BLIP-base** (`--model_name blip`): Runs on CPU; slower but no GPU needed.

---

## Usage

### Running the full pipeline

```bash
# On Sol cluster (GPU, after downloading PhysBench to /scratch/<asuid>/krr-data)
python -m visual_extraction.run_pipeline \
    --data_dir /scratch/<asuid>/krr-data \
    --images_subdir image \
    --output_dir outputs \
    --model_name blip2 \
    --device cuda

# CPU fallback (local machine, default paths)
python -m visual_extraction.run_pipeline \
    --data_dir data/physbench \
    --model_name blip \
    --device cpu
```

All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `data/physbench` | PhysBench root directory |
| `--images_subdir` | `images` | Subdirectory under `data_dir` where images live (`image` on Sol cluster) |
| `--output_dir` | `outputs` | Where to write all output files |
| `--model_name` | `blip2` | VLM backend: `blip2` or `blip` |
| `--subset_size` | `None` | Optional cap on the number of `image-only` records to process; useful for smoke tests like `--subset_size 1` |
| `--device` | `cuda` | Torch device: `cuda` or `cpu` |
| `--seed` | `None` | Deprecated and ignored; dataset preparation no longer samples |
| `--skip_captioning` | off | Load `captions.jsonl` from disk, skip stage 2 |
| `--skip_objects` | off | Load `objects.jsonl` from disk, skip stage 3 |
| `--skip_scene_graphs` | off | Load `scene_graphs.jsonl` from disk, skip stage 4 |
| `--verbose` | off | Enable DEBUG logging |

### Re-running individual stages

```bash
# Re-run only entity export using previously computed intermediate files
python -m visual_extraction.run_pipeline \
    --data_dir /scratch/<asuid>/krr-data \
    --images_subdir image \
    --output_dir outputs \
    --model_name blip2 \
    --device cuda \
    --skip_captioning \
    --skip_objects \
    --skip_scene_graphs
```

### Quick smoke test

```bash
python -m visual_extraction.run_pipeline \
    --data_dir /scratch/<asuid>/krr-data \
    --images_subdir image \
    --output_dir outputs_smoketest \
    --model_name blip2 \
    --device cuda \
    --subset_size 1
```

---

## Input data formats

### `metadata.json` (preferred)

A JSON array of records with these fields (answer/label fields are always stripped):

```json
[
  {
    "image_id": "42",
    "image_path": "/path/to/images/42.jpg",
    "category": "dynamics",
    "question_text": "What happens when the ball reaches the bottom?",
    "mode": "image-only"
  }
]
```

### Raw PhysBench files (fallback)

If `metadata.json` is absent, the module constructs metadata automatically from:
- `{data_dir}/test.json` — questions, mode, etc.
- `{data_dir}/test_answer.json` — task categories (answers are stripped)

---

## Output formats

### `subset_manifest.json`

```json
[
  {
    "image_id": "42",
    "image_path": "/path/to/images/42.jpg",
    "category": "dynamics",
    "question_text": "What happens when the ball reaches the bottom?"
  }
]
```

### `captions.jsonl`

```jsonl
{"image_id": "42", "caption": "A red rubber ball rests at the top of a wooden ramp."}
```

### `objects.jsonl`

```jsonl
{"image_id": "42", "objects": [{"label": "ball", "color": "red", "shape": "sphere", "material": "rubber", "size": "small"}, {"label": "ramp", "color": "brown", "shape": null, "material": "wood", "size": "large"}]}
```

### `scene_graphs.jsonl`

```jsonl
{"image_id": "42", "nodes": [{"id": 0, "label": "ball", "attributes": {"color": "red"}}, {"id": 1, "label": "ramp", "attributes": {"material": "wood"}}], "edges": [{"source": 0, "target": 1, "relation": "on_top_of"}]}
```

### `visual_entities.jsonl`

```jsonl
{"image_id": "42", "entities": ["ball", "ramp", "wood", "red"], "relations": ["ball on_top_of ramp"], "caption": "A red rubber ball rests at the top of a wooden ramp."}
```

---

## Running tests

```bash
conda activate krr
pytest tests/test_visual_extraction.py -v
```

---

## Module structure

```
visual_extraction/
├── __init__.py          — package metadata
├── model_backend.py     — shared VLM wrapper (BLIP-2 / BLIP-base)
├── data_prep.py         — dataset loading, image-only filtering, manifest
├── captioning.py        — image captioning
├── object_detection.py  — object & attribute extraction
├── scene_graph.py       — scene graph construction
├── entity_export.py     — entity normalisation, synonym mapping, export
└── run_pipeline.py      — CLI orchestration entry point
```

---

## Notes on label leakage prevention

`data_prep.py` enforces a strict no-leakage policy:

1. `_strip_prohibited()` removes `answer`, `label`, `correct_answer`, `ground_truth`, and `gt` from every record before processing.
2. `_assert_no_leakage()` raises `ValueError` if any prohibited field survives into the final manifest.
3. Unit tests in `tests/test_visual_extraction.py` verify both checks.

The manifest keeps the historical filename `subset_manifest.json`, but it now contains all `image-only` records so it lines up with `concept_extraction.py`.

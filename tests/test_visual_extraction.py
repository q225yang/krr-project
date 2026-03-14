"""Unit tests for the visual_extraction package.

Covers:
- Entity normalisation (normalize_entity)
- Manifest field filtering / leakage prevention (data_prep)
- Manifest coverage / image-only filtering (data_prep)
- Scene graph edge resolution helpers (scene_graph)
- JSON extraction helper (model_backend)

Run with::

    pytest tests/test_visual_extraction.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# entity_export — normalize_entity
# ---------------------------------------------------------------------------


class TestNormalizeEntity:
    """Tests for visual_extraction.entity_export.normalize_entity."""

    def _norm(self, text: str) -> str:
        from visual_extraction.entity_export import normalize_entity
        return normalize_entity(text)

    def test_lowercase(self):
        assert self._norm("Ball") == self._norm("ball")

    def test_synonym_sphere_to_ball(self):
        result = self._norm("sphere")
        assert result == "ball", f"Expected 'ball', got '{result}'"

    def test_synonym_slope_to_ramp(self):
        result = self._norm("slope")
        assert result == "ramp", f"Expected 'ramp', got '{result}'"

    def test_synonym_wooden_to_wood(self):
        result = self._norm("wooden")
        assert result == "wood", f"Expected 'wood', got '{result}'"

    def test_no_change_for_canonical(self):
        # 'ball' is already canonical
        assert self._norm("ball") == "ball"

    def test_strips_punctuation(self):
        result = self._norm("ball!")
        assert "!" not in result

    def test_underscore_for_spaces(self):
        result = self._norm("inclined plane")
        # Should produce a normalised form without raw spaces
        assert " " not in result

    def test_empty_string(self):
        result = self._norm("")
        assert result == ""

    def test_metallic_to_metal(self):
        result = self._norm("metallic")
        assert result == "metal"

    def test_automobile_to_car(self):
        result = self._norm("automobile")
        assert result == "car"


# ---------------------------------------------------------------------------
# entity_export — aggregate_entities
# ---------------------------------------------------------------------------


class TestAggregateEntities:
    """Tests for visual_extraction.entity_export.aggregate_entities."""

    def _aggregate(self, image_id, caption, objects, scene_graph):
        from visual_extraction.entity_export import aggregate_entities
        return aggregate_entities(image_id, caption, objects, scene_graph)

    def test_basic_structure(self):
        result = self._aggregate(
            "1",
            "A red ball on a ramp.",
            [{"label": "ball", "color": "red", "shape": "sphere",
              "material": None, "size": "small"}],
            {"nodes": [{"id": 0, "label": "ball", "attributes": {}}],
             "edges": []},
        )
        assert result["image_id"] == "1"
        assert "entities" in result
        assert "relations" in result
        assert "caption" in result

    def test_entities_non_empty(self):
        result = self._aggregate(
            "2",
            "A wooden ramp sits on the floor.",
            [{"label": "ramp", "color": None, "shape": None,
              "material": "wood", "size": "large"}],
            {"nodes": [{"id": 0, "label": "ramp", "attributes": {}}],
             "edges": []},
        )
        assert len(result["entities"]) > 0

    def test_relations_derived_from_scene_graph(self):
        result = self._aggregate(
            "3",
            "",
            [{"label": "ball", "color": None, "shape": None,
              "material": None, "size": None},
             {"label": "ramp", "color": None, "shape": None,
              "material": None, "size": None}],
            {
                "nodes": [
                    {"id": 0, "label": "ball", "attributes": {}},
                    {"id": 1, "label": "ramp", "attributes": {}},
                ],
                "edges": [
                    {"source": 0, "target": 1, "relation": "on_top_of"}
                ],
            },
        )
        assert any("on_top_of" in r for r in result["relations"])

    def test_deduplication(self):
        # Same label appears in objects and caption — should appear once in entities
        result = self._aggregate(
            "4",
            "A ball.",
            [{"label": "ball", "color": None, "shape": None,
              "material": None, "size": None}],
            {"nodes": [{"id": 0, "label": "ball", "attributes": {}}],
             "edges": []},
        )
        count = result["entities"].count("ball")
        assert count == 1, f"'ball' appears {count} times; expected 1"


# ---------------------------------------------------------------------------
# data_prep — _strip_prohibited and _assert_no_leakage
# ---------------------------------------------------------------------------


class TestManifestLeakagePrevention:
    """Tests that answer / label fields cannot sneak into the manifest."""

    def test_strip_prohibited_removes_answer(self):
        from visual_extraction.data_prep import _strip_prohibited

        record = {
            "image_id": "1",
            "image_path": "img.jpg",
            "category": "dynamics",
            "question_text": "What happens?",
            "answer": "A",
        }
        clean = _strip_prohibited(record)
        assert "answer" not in clean
        assert "image_id" in clean

    def test_strip_prohibited_removes_all_prohibited(self):
        from visual_extraction.data_prep import _strip_prohibited

        record = {
            "image_id": "2",
            "answer": "B",
            "label": "dynamics",
            "ground_truth": "C",
            "question_text": "Q?",
        }
        clean = _strip_prohibited(record)
        for field in ("answer", "label", "ground_truth"):
            assert field not in clean, f"'{field}' should have been stripped"

    def test_assert_no_leakage_raises_on_answer(self):
        from visual_extraction.data_prep import _assert_no_leakage

        records = [
            {"image_id": "1", "answer": "A", "category": "dynamics"}
        ]
        with pytest.raises(ValueError, match="Prohibited field"):
            _assert_no_leakage(records)

    def test_assert_no_leakage_passes_clean_records(self):
        from visual_extraction.data_prep import _assert_no_leakage

        records = [
            {
                "image_id": "1",
                "image_path": "img.jpg",
                "category": "dynamics",
                "question_text": "Q?",
            }
        ]
        # Should not raise
        _assert_no_leakage(records)


# ---------------------------------------------------------------------------
# data_prep — image path resolution
# ---------------------------------------------------------------------------


class TestResolveRawImagePath:
    def test_prefers_existing_data_dir_file_name_path(self):
        from visual_extraction.data_prep import _resolve_raw_image_path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            images_dir = data_dir / "image"
            nested_dir = data_dir / "nested"
            images_dir.mkdir(parents=True)
            nested_dir.mkdir(parents=True)

            expected = nested_dir / "example.png"
            expected.write_bytes(b"fake")

            result = _resolve_raw_image_path(
                data_dir=data_dir,
                images_dir=images_dir,
                idx="42",
                file_name_field="nested/example.png",
            )

            assert result == str(expected)

    def test_falls_back_to_matching_stem_with_different_extension(self):
        from visual_extraction.data_prep import _resolve_raw_image_path

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            images_dir = data_dir / "image"
            images_dir.mkdir(parents=True)

            expected = images_dir / "42.png"
            expected.write_bytes(b"fake")

            result = _resolve_raw_image_path(
                data_dir=data_dir,
                images_dir=images_dir,
                idx="42",
                file_name_field=None,
            )

            assert result == str(expected)


# ---------------------------------------------------------------------------
# data_prep — prepare_subset writes correct manifest
# ---------------------------------------------------------------------------


class TestPrepareSubset:
    """Integration-style tests for prepare_subset with a fake metadata file."""

    def _make_metadata(self, n: int = 30) -> list[dict]:
        cats = ["dynamics", "scene", "relationships"]
        return [
            {
                "image_id": str(i),
                "image_path": f"/fake/images/{i}.jpg",
                "category": cats[i % 3],
                "question_text": f"Question {i}?",
                "mode": "image-only",
            }
            for i in range(n)
        ]

    def test_manifest_written(self):
        from visual_extraction.data_prep import prepare_subset

        meta = self._make_metadata(30)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            output_dir = tmp_path / "outputs"
            prepare_subset(data_dir, output_dir, subset_size=15)
            manifest_path = output_dir / "subset_manifest.json"
            assert manifest_path.exists()
            manifest = json.loads(manifest_path.read_text())
            assert len(manifest) == 30

    def test_manifest_keeps_all_image_only_records(self):
        from visual_extraction.data_prep import prepare_subset

        meta = [
            {
                "image_id": "0",
                "image_path": "/fake/images/0.jpg",
                "category": "dynamics",
                "question_text": "Question 0?",
                "mode": "image-only",
            },
            {
                "image_id": "1",
                "image_path": "/fake/images/1.jpg",
                "category": "scene",
                "question_text": "Question 1?",
                "mode": "general",
            },
            {
                "image_id": "2",
                "image_path": "/fake/images/2.jpg",
                "category": "relationships",
                "question_text": "Question 2?",
                "mode": "image-only",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            output_dir = tmp_path / "outputs"
            manifest = prepare_subset(data_dir, output_dir, subset_size=1, seed=7)

            assert [rec["image_id"] for rec in manifest] == ["0", "2"]

    def test_manifest_has_no_answer_field(self):
        from visual_extraction.data_prep import prepare_subset

        meta = self._make_metadata(20)
        # Inject answer fields (should be stripped)
        for rec in meta:
            rec["answer"] = "A"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            output_dir = tmp_path / "outputs"
            prepare_subset(data_dir, output_dir, subset_size=10)
            manifest = json.loads(
                (output_dir / "subset_manifest.json").read_text()
            )
            for rec in manifest:
                assert "answer" not in rec

    def test_manifest_required_fields(self):
        from visual_extraction.data_prep import prepare_subset

        meta = self._make_metadata(20)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            (data_dir / "metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            output_dir = tmp_path / "outputs"
            prepare_subset(data_dir, output_dir, subset_size=5)
            manifest = json.loads(
                (output_dir / "subset_manifest.json").read_text()
            )
            for rec in manifest:
                assert "image_id" in rec
                assert "image_path" in rec
                assert "category" in rec
                assert "question_text" in rec


# ---------------------------------------------------------------------------
# model_backend — _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    """Tests for the JSON extraction helper used by VLMBackend."""

    def _extract(self, text, fallback=None):
        from visual_extraction.model_backend import _extract_json
        return _extract_json(text, fallback=fallback)

    def test_clean_json_array(self):
        result = self._extract('[{"label":"ball","color":"red"}]')
        assert isinstance(result, list)
        assert result[0]["label"] == "ball"

    def test_json_embedded_in_text(self):
        text = 'The objects are: [{"label":"ramp","color":null}] in the scene.'
        result = self._extract(text, fallback=[])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_returns_fallback_on_garbage(self):
        result = self._extract("There is a ball and a ramp.", fallback=[])
        assert result == []

    def test_strips_answer_marker(self):
        text = 'Answer: [{"label":"box","color":"blue"}]'
        result = self._extract(text, fallback=None)
        assert isinstance(result, list)
        assert result[0]["color"] == "blue"


# ---------------------------------------------------------------------------
# model_backend — prompt echo stripping / caption fallback
# ---------------------------------------------------------------------------


class TestPromptEchoHandling:
    def test_strip_prompt_echo_removes_exact_prompt(self):
        from visual_extraction.model_backend import _strip_prompt_echo

        prompt = "Question: Describe this image. Answer:"
        assert _strip_prompt_echo(prompt, prompt) == ""

    def test_strip_prompt_echo_keeps_generated_answer(self):
        from visual_extraction.model_backend import _strip_prompt_echo

        prompt = "Question: Describe this image. Answer:"
        text = f"{prompt} A red ball sits on a ramp."
        assert _strip_prompt_echo(text, prompt) == "A red ball sits on a ramp."

    def test_caption_retries_without_prompt_when_prompted_output_is_empty(self):
        from visual_extraction.model_backend import CAPTION_PROMPT, VLMBackend

        backend = object.__new__(VLMBackend)
        calls = []

        def fake_generate(image, prompt, max_new_tokens):
            calls.append(prompt)
            return "" if prompt == CAPTION_PROMPT else "A red ball sits on a ramp."

        backend._generate = fake_generate

        result = backend.caption(None)

        assert result == "A red ball sits on a ramp."
        assert calls == [CAPTION_PROMPT, None]


# ---------------------------------------------------------------------------
# model_backend — detect_relations
# ---------------------------------------------------------------------------


class TestDetectRelations:
    def test_detect_relations_formats_prompt_and_parses_json(self):
        from visual_extraction.model_backend import VLMBackend

        backend = object.__new__(VLMBackend)

        def fake_generate(image, prompt, max_new_tokens):
            assert "Given these objects in the image: ball, ramp," in prompt
            assert '"source" (object label)' in prompt
            assert "Use short snake_case relation names." in prompt
            return '[{"source":"ball","target":"ramp","relation":"sliding_down"}]'

        backend._generate = fake_generate

        result = backend.detect_relations(None, ["ball", "ramp"])

        assert result == [
            {"source": "ball", "target": "ramp", "relation": "sliding_down"}
        ]


# ---------------------------------------------------------------------------
# scene_graph — _relations_to_strings
# ---------------------------------------------------------------------------


class TestRelationsToStrings:
    def test_basic_relation(self):
        from visual_extraction.entity_export import _relations_to_strings

        nodes = [
            {"id": 0, "label": "ball", "attributes": {}},
            {"id": 1, "label": "ramp", "attributes": {}},
        ]
        edges = [{"source": 0, "target": 1, "relation": "sliding_down"}]
        result = _relations_to_strings(nodes, edges)
        assert result == ["ball sliding_down ramp"]

    def test_empty_edges(self):
        from visual_extraction.entity_export import _relations_to_strings

        nodes = [{"id": 0, "label": "ball", "attributes": {}}]
        result = _relations_to_strings(nodes, [])
        assert result == []

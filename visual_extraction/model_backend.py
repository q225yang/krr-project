"""Shared VLM backend used by captioning, object detection, and scene graph modules.

Supports BLIP-2 (GPU, default) and BLIP-base (CPU fallback).  The same model
instance is reused across pipeline stages to avoid reloading weights.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CAPTION_PROMPT = (
    "Describe this image in 1 to 3 sentences, focusing on the physical objects "
    "present, their properties, and any ongoing physical processes."
)

OBJECT_PROMPT = (
    "List every visible physical object in this image. "
    "Return a JSON array where each element is an object with these keys: "
    '"label" (str), "color" (str or null), "shape" (str or null), '
    '"material" (str or null), "size" ("small", "medium", "large", or null). '
    "Use null when a field is unknown. Output only the JSON array, nothing else."
)

RELATION_PROMPT_TMPL = (
    "Given these objects in the image: {objects}, "
    "describe the spatial and physical relations between them. "
    "Return a JSON array of relation triples, each with keys: "
    '"source" (object label), "target" (object label), "relation" (e.g. on_top_of, '
    "next_to, connected_to, sliding_down, contains, below, above, leaning_against). "
    "Use short snake_case relation names. Output only the JSON array, nothing else."
)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str, fallback: Any) -> Any:
    """Try to extract a JSON value (array or object) from *text*.

    Attempts strict parsing first, then searches for a bracketed substring.
    Returns *fallback* when all parsing attempts fail.
    """
    text = text.strip()
    # Strip any leading prompt echo (BLIP-2 sometimes repeats the prompt)
    for marker in ("Answer:", "Output:", "```json", "```"):
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):].strip()
    text = text.rstrip("`").strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Search for first [...] or {...} block
    for pattern in (r"\[.*?\]", r"\{.*?\}"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass

    return fallback


def _normalise_whitespace(text: str) -> str:
    """Collapse repeated whitespace for prompt/output comparisons."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_prompt_echo(text: str, prompt: str | None) -> str:
    """Remove an echoed prompt prefix from generated text when present."""
    cleaned = text.strip()
    if not cleaned or not prompt:
        return cleaned

    prompt = prompt.strip()
    if cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):].lstrip(" \t\r\n:-")
    elif _normalise_whitespace(cleaned) == _normalise_whitespace(prompt):
        cleaned = ""

    return cleaned.strip()


# ---------------------------------------------------------------------------
# BLIP-2 loader
# ---------------------------------------------------------------------------

def _load_blip2(device: str):
    """Load BLIP-2 (Salesforce/blip2-opt-2.7b) model and processor."""
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    hf_id = "Salesforce/blip2-opt-2.7b"
    logger.info("Loading BLIP-2 from %s …", hf_id)
    processor = Blip2Processor.from_pretrained(hf_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, processor, "blip2"


# ---------------------------------------------------------------------------
# BLIP-base loader (CPU-friendly)
# ---------------------------------------------------------------------------

def _load_blip(device: str):
    """Load BLIP-base (Salesforce/blip-image-captioning-base) on *device*."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    hf_id = "Salesforce/blip-image-captioning-base"
    logger.info("Loading BLIP-base from %s …", hf_id)
    processor = BlipProcessor.from_pretrained(hf_id)
    model = BlipForConditionalGeneration.from_pretrained(hf_id)
    model = model.to(device)
    model.eval()
    return model, processor, "blip"


# ---------------------------------------------------------------------------
# Public model wrapper
# ---------------------------------------------------------------------------

class VLMBackend:
    """Thin wrapper around a vision-language model for captioning and VQA.

    Parameters
    ----------
    model_name:
        One of ``"blip2"`` (default) or ``"blip"``.  Any string containing
        ``"blip2"`` loads BLIP-2; everything else loads BLIP-base.
    device:
        ``"cuda"`` or ``"cpu"``.  Falls back to CPU + BLIP-base automatically
        when CUDA is not available.
    """

    def __init__(self, model_name: str = "blip2", device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        self._backend: str = ""
        self._load_model()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        import torch

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Falling back to CPU with BLIP-base."
            )
            self.device = "cpu"
            self.model_name = "blip"

        if "blip2" in self.model_name.lower():
            try:
                self._model, self._processor, self._backend = _load_blip2(self.device)
                logger.info("BLIP-2 loaded on %s.", self.device)
                return
            except Exception as exc:
                logger.warning(
                    "BLIP-2 load failed (%s). Falling back to BLIP-base on CPU.", exc
                )
                self.device = "cpu"
                self.model_name = "blip"

        self._model, self._processor, self._backend = _load_blip(self.device)
        logger.info("BLIP-base loaded on %s.", self.device)

    def _prepare_prompt(self, prompt: str | None) -> str | None:
        """Format prompts for backends that expect VQA-style text."""
        if not prompt:
            return None
        if self._backend == "blip2":
            return f"Question: {prompt} Answer:"
        return prompt

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(self, image: Image.Image, prompt: str | None, max_new_tokens: int) -> str:
        """Run the underlying model and return decoded text."""
        import torch

        img = image.convert("RGB")
        prepared_prompt = self._prepare_prompt(prompt)

        if self._backend == "blip2":
            inputs = self._processor(img, text=prepared_prompt, return_tensors="pt").to(
                self.device, torch.float16
            )
        else:
            # BLIP-base: text is the conditional prompt (can be None)
            if prepared_prompt:
                inputs = self._processor(img, text=prepared_prompt, return_tensors="pt").to(
                    self.device
                )
            else:
                inputs = self._processor(img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[0]
        input_ids = inputs.get("input_ids")
        if prepared_prompt and input_ids is not None:
            prompt_ids = input_ids[0]
            prompt_len = int(prompt_ids.shape[-1])
            if prompt_len and generated_ids.shape[-1] >= prompt_len:
                if generated_ids[:prompt_len].tolist() == prompt_ids.tolist():
                    generated_ids = generated_ids[prompt_len:]

        decoded = self._processor.decode(generated_ids, skip_special_tokens=True)
        return _strip_prompt_echo(decoded, prepared_prompt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def caption(self, image: Image.Image) -> str:
        """Return a 1–3 sentence natural-language caption for *image*."""
        caption = self._generate(image, CAPTION_PROMPT, max_new_tokens=128)
        if caption:
            return caption

        logger.debug(
            "Prompted caption generation returned empty text; retrying without a prompt."
        )
        return self._generate(image, None, max_new_tokens=128)

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Return free-form text conditioned on *image* and *prompt*."""
        return self._generate(image, prompt, max_new_tokens=max_new_tokens)

    def detect_objects(self, image: Image.Image) -> list[dict]:
        """Return a list of object dicts with label/color/shape/material/size."""
        raw = self._generate(image, OBJECT_PROMPT, max_new_tokens=256)
        if not raw:
            return []

        result = _extract_json(raw, fallback=None)
        if isinstance(result, list):
            objects = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                objects.append(
                    {
                        "label": str(item.get("label", "object")).strip().lower(),
                        "color": item.get("color"),
                        "shape": item.get("shape"),
                        "material": item.get("material"),
                        "size": item.get("size"),
                    }
                )
            return objects

        # Fallback: plain-text parsing — treat each comma/newline token as object
        logger.debug("JSON parse failed for object detection; using plain-text fallback.")
        return _parse_objects_from_text(raw)

    def detect_relations(
        self, image: Image.Image, object_labels: list[str]
    ) -> list[dict]:
        """Return a list of relation triples for the given object labels."""
        objects_str = ", ".join(object_labels) if object_labels else "unknown"
        prompt = RELATION_PROMPT_TMPL.format(objects=objects_str)
        raw = self._generate(image, prompt, max_new_tokens=256)
        result = _extract_json(raw, fallback=None)
        if isinstance(result, list):
            relations = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source", "")).strip().lower()
                tgt = str(item.get("target", "")).strip().lower()
                rel = str(item.get("relation", "")).strip().lower().replace(" ", "_")
                if src and tgt and rel:
                    relations.append({"source": src, "target": tgt, "relation": rel})
            return relations
        return []


# ---------------------------------------------------------------------------
# Plain-text object parsing fallback
# ---------------------------------------------------------------------------

def _parse_objects_from_text(text: str) -> list[dict]:
    """Best-effort extraction of object labels from unstructured VLM output."""
    tokens = re.split(r"[,\n;]+", text)
    objects = []
    seen: set[str] = set()
    for tok in tokens:
        label = tok.strip().lower()
        # Strip leading bullets / numbers
        label = re.sub(r"^[\-\*\d\.\)]+\s*", "", label)
        label = label.strip(" \"'")
        if label and label not in seen and len(label) < 60:
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
    return objects

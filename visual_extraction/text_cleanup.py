"""Shared text cleanup helpers for caption-derived pipeline steps."""

from __future__ import annotations

import re

_VISUAL_LEADIN_RE = re.compile(
    r"^(?:the|this)\s+(?:image|picture|photo|scene)\s+"
    r"(?:shows|depicts|captures|features|illustrates)\s+",
    re.IGNORECASE,
)
_COMMA_RELATIVE_CLAUSE_RE = re.compile(r",\s*(?:which|who|that)\b", re.IGNORECASE)


def clean_caption_text(text: str) -> str:
    """Normalise a caption and strip generic visual lead-ins."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = _VISUAL_LEADIN_RE.sub("", cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def caption_primary_clause(text: str) -> str:
    """Return the main visual clause of a caption for object/entity extraction."""
    cleaned = clean_caption_text(text)
    cleaned = _COMMA_RELATIVE_CLAUSE_RE.split(cleaned, maxsplit=1)[0].strip(" ,.;:")
    return cleaned

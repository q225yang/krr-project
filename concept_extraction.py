import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import spacy
from tqdm import tqdm


# ------------------------------------------------------------------------------
# Config
INPUT_JSON = Path("/scratch/qyang129/krr-data/test.json")
OUTPUT_JSON = Path("/scratch/qyang129/krr-data/extracted_concepts_image_only.json")

nlp = spacy.load("en_core_web_sm")

PLACEHOLDER_RE = re.compile(r"<\s*(image|video)\s*>", re.IGNORECASE)

# Option patterns:
# 1) Option line: "A. xxx"
OPTION_LINE_RE = re.compile(r"^\s*([A-D])\.\s*(.*)$", re.IGNORECASE)
# 2) Inline options: "A. xxx B. yyy C. zzz D. www" (same line)
INLINE_OPTION_RE = re.compile(r"\b([A-D])\.\s*", re.IGNORECASE)

# Object ref suffix: "glass_1" -> "glass"
OBJ_REF_SUFFIX_RE = re.compile(r"^(.*?)(?:_\d+)$")


# ------------------------------------------------------------------------------
# Helper functions
def normalize_phrase(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r.,;:!?\"'()[]{}")
    return s


def strip_obj_ref(s: str) -> str:
    s = (s or "").strip()
    m = OBJ_REF_SUFFIX_RE.match(s)
    return m.group(1) if m else s


def split_inline_options(line: str) -> Optional[List[str]]:
    if not line:
        return None

    # Quick check: must contain at least two option markers to be considered inline
    markers = list(INLINE_OPTION_RE.finditer(line))
    if len(markers) < 2:
        return None

    parts: List[str] = []
    for i, m in enumerate(markers):
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(line)
        chunk = line[start:end].strip()
        chunk = PLACEHOLDER_RE.sub(" ", chunk).strip()
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            parts.append(chunk)

    return parts if parts else None


def split_question_and_options(q: str) -> Tuple[str, List[str]]:
    """
    Splits raw question text into:
      stem_text: the question prompt (without option labels)
      options: list of option texts (A/B/C/D...) as separate strings

    Handles both:
      - options on separate lines: "A. ...\nB. ...\n"
      - inline options on one line: "A. ... B. ... C. ... D. ..."
    """
    q = q or ""
    lines = [ln.strip() for ln in q.splitlines() if ln.strip()]

    stem_lines: List[str] = []
    options: List[str] = []

    for ln in lines:
        # Inline splitting
        inline_opts = split_inline_options(ln)
        if inline_opts is not None:
            options.extend(inline_opts)
            continue

        # Normal line
        ln2 = PLACEHOLDER_RE.sub(" ", ln).strip()
        if not ln2:
            continue

        m = OPTION_LINE_RE.match(ln2)
        if m:
            opt_text = m.group(2).strip()
            opt_text = re.sub(r"\s+", " ", opt_text).strip()
            if opt_text:
                options.append(opt_text)
        else:
            stem_lines.append(ln2)

    stem_text = re.sub(r"\s+", " ", " ".join(stem_lines)).strip()
    return stem_text, options


# ------------------------------------------------------------------------------
# Text-to-concepts
def extract_concepts(text: str) -> Dict[str, Any]:
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    root_verbs: List[str] = []
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}:
            root_verbs.append(tok.lemma_)
            for c in tok.conjuncts:
                if c.pos_ == "VERB":
                    root_verbs.append(c.lemma_)

    all_verbs = [tok.lemma_ for tok in doc if tok.pos_ == "VERB"]

    amod_pairs = []
    for tok in doc:
        if tok.dep_ == "amod" and tok.head.pos_ in {"NOUN", "PROPN"}:
            amod_pairs.append((tok.lemma_, tok.head.lemma_))

    acomp_adjs = [tok.lemma_ for tok in doc if tok.dep_ == "acomp"]

    svos = []
    for tok in doc:
        if tok.pos_ == "VERB":
            subs = [w for w in tok.lefts if w.dep_ in {"nsubj", "nsubjpass"}]
            objs = [w for w in tok.rights if w.dep_ in {"dobj", "obj", "pobj"}]
            for s in subs:
                for o in objs:
                    svos.append((s.text, tok.lemma_, o.text))

    def uniq(seq):
        return list(dict.fromkeys(seq))

    return {
        "entities": uniq(entities),
        "noun_phrases": uniq(noun_phrases),
        "root_verbs": uniq(root_verbs),
        "all_verbs": uniq(all_verbs),
        "adjective_noun_pairs": uniq(amod_pairs),
        "adjective_complements": uniq(acomp_adjs),
        "svos": uniq(svos),
    }


def merge_extracted(extracted_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "entities": [],
        "noun_phrases": [],
        "root_verbs": [],
        "all_verbs": [],
        "adjective_noun_pairs": [],
        "adjective_complements": [],
        "svos": [],
    }

    for ex in extracted_list:
        for k in merged.keys():
            merged[k].extend(ex.get(k, []))

    def uniq(seq):
        return list(dict.fromkeys(seq))

    for k in list(merged.keys()):
        merged[k] = uniq(merged[k])

    return merged


# ------------------------------------------------------------------------------
# Build final word-groups
def make_word_groups(
    extracted: Dict[str, Any],
    obj_list: Optional[List[str]],
    options: Optional[List[str]],
) -> List[str]:
    groups: List[str] = []

    # noun phrases
    for np in extracted.get("noun_phrases", []):
        groups.append(np)

    # named entity strings
    for ent_text, _ in extracted.get("entities", []):
        groups.append(ent_text)

    # root verbs
    for v in extracted.get("root_verbs", []):
        groups.append(v)

    # adjective + noun pairs
    for adj, noun in extracted.get("adjective_noun_pairs", []):
        groups.append(f"{adj} {noun}")

    # add options explicitly as separate groups (prevents merged blob)
    for opt in options or []:
        groups.append(opt)

    # add sample objects, stripping trailing _digits
    if obj_list:
        for o in obj_list:
            if isinstance(o, str) and o.strip():
                groups.append(strip_obj_ref(o))

    # final normalize + dedup
    seen = set()
    final: List[str] = []
    for g in groups:
        ng = normalize_phrase(g)
        if not ng or ng in seen:
            continue
        seen.add(ng)
        final.append(ng)

    return final


# ------------------------------------------------------------------------------
def main() -> None:
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array (list of samples).")

    results: List[Dict[str, Any]] = []
    kept = 0

    for sample in tqdm(data, desc="Processing samples", unit="sample"):
        if not isinstance(sample, dict):
            continue

        # Keep only image-only datapoints
        if sample.get("mode") != "image-only":
            continue

        question_raw = sample.get("question") or ""
        stem_text, options = split_question_and_options(question_raw)

        # Extract from stem + each option separately, then merge
        extracted_parts: List[Dict[str, Any]] = []
        if stem_text:
            extracted_parts.append(extract_concepts(stem_text))
        for opt in options:
            extracted_parts.append(extract_concepts(opt))

        extracted = merge_extracted(extracted_parts) if extracted_parts else extract_concepts("")

        # Object is only None or list[str] per your constraint
        obj_list = sample.get("object")

        word_groups = make_word_groups(extracted, obj_list, options)

        results.append(
            {
                "idx": sample.get("idx"),
                "split": sample.get("split"),
                "mode": sample.get("mode"),
                "question_stem": stem_text,
                "options": options,
                "extracted": extracted,
                "word_groups": word_groups,
            }
        )
        kept += 1

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Kept image-only samples: {kept}")
    print(f"Wrote to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
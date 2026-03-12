import json
import re
from pathlib import Path
from typing import List

import spacy
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Config
INPUT_JSON = Path("/scratch/qyang129/krr-data/test.json")
OUTPUT_JSON = Path("/scratch/qyang129/krr-data/extracted_concepts.json")

nlp = spacy.load("en_core_web_sm")

PLACEHOLDER_RE = re.compile(r"<\s*(image|video)\s*>", re.IGNORECASE)
OPTION_LINE_RE = re.compile(r"^\s*[A-D]\.\s*", re.IGNORECASE)
OBJ_REF_SUFFIX_RE = re.compile(r"^(.*?)(?:_\d+)$")

# ------------------------------------------------------------------------------
# Helper functions
def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r.,;:!?\"'()[]{}")
    return s

def clean_question_text(q: str) -> str:
    q = PLACEHOLDER_RE.sub(" ", q)

    lines = []
    for line in q.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = OPTION_LINE_RE.sub("", stripped).strip()
        if not stripped:
            continue
        lines.append(stripped)

    q = " ".join(lines)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def strip_obj_ref(s: str) -> str:
    s = s.strip()
    m = OBJ_REF_SUFFIX_RE.match(s)
    return m.group(1) if m else s

# ------------------------------------------------------------------------------
# Text-to-concepts
def extract_concepts(text: str):
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    root_verbs = []
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
        "entities": entities,
        "noun_phrases": noun_phrases,
        "root_verbs": uniq(root_verbs),
        "all_verbs": uniq(all_verbs),
        "adjective_noun_pairs": amod_pairs,
        "adjective_complements": uniq(acomp_adjs),
        "svos": svos,
    }

# Build final concept list
def make_word_groups(extracted, obj_list):
    groups = []

    for np in extracted.get("noun_phrases", []):
        groups.append(np)

    for ent_text, _ in extracted.get("entities", []):
        groups.append(ent_text)

    for v in extracted.get("root_verbs", []):
        groups.append(v)

    for adj, noun in extracted.get("adjective_noun_pairs", []):
        groups.append(f"{adj} {noun}")

    if obj_list:
        for o in obj_list:
            if isinstance(o, str) and o.strip():
                groups.append(strip_obj_ref(o))

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

    results = []
    kept = 0

    for sample in tqdm(data, desc="Processing samples", unit="sample"):
        if sample.get("mode") != "image-only":
            continue

        question_raw = sample.get("question") or ""
        question_clean = clean_question_text(question_raw)
        extracted = extract_concepts(question_clean)

        obj_list = sample.get("object")

        word_groups = make_word_groups(extracted, obj_list)

        results.append({
            "idx": sample.get("idx"),
            "split": sample.get("split"),
            "mode": sample.get("mode"),
            "question_clean": question_clean,
            "extracted": extracted,
            "word_groups": word_groups,
        })
        kept += 1

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Kept image-only samples: {kept}")
    print(f"Wrote to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
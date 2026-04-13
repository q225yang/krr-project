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

# Option patterns
OPTION_LINE_RE = re.compile(r"^\s*([A-D])\.\s*(.*)$", re.IGNORECASE)
INLINE_OPTION_RE = re.compile(r"\b([A-D])\.\s*", re.IGNORECASE)

# Object ref suffix: "glass_1" -> "glass"
OBJ_REF_SUFFIX_RE = re.compile(r"^(.*?)(?:_\d+)$")

# ------------------------------------------------------------------------------
# Stopwords & blacklists
# ------------------------------------------------------------------------------

# Entities that are question scaffolding / function words — never useful as KG seeds
STOP_ENTITIES = frozenset({
    # Pronouns & determiners
    "we", "i", "you", "he", "she", "it", "they", "one", "ones",
    "this", "that", "these", "those", "which", "what", "who", "whom",
    "some", "any", "all", "each", "every", "both", "few", "many",
    "much", "several", "other", "another",
    # Function verbs
    "be", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "know", "think", "tell", "say", "see", "look", "show",
    "determine", "find", "identify", "select", "choose", "pick",
    "indicate", "mark", "denote", "signify", "represent",
    "exhibit", "display", "feature", "possess",
    "distinguish", "isolate", "capture", "enclose", "single",
    "infer", "guess", "estimate", "judge", "conclude",
    "pinpoint", "pinpoints",
    # Generic question words
    "option", "options", "follow", "following", "below", "above",
    "condition", "conditions", "case", "cases",
    "way", "manner", "method", "order",
    "many",  # from "how many" — the count target noun is kept separately
    # Image/picture references — never queryable in ConceptNet
    "image", "picture", "photo", "photograph", "scene", "figure",
    "diagram", "illustration", "screenshot", "frame",
    # Comparative scaffolding
    "close", "closest", "nearest", "farthest", "likely",
    "big", "biggest", "small", "smallest", "tall", "tallest",
    "heavy", "heaviest", "fast", "fastest", "slow", "slowest",
    # Abstract/generic nouns
    "thing", "things", "object", "objects", "stuff", "item", "items",
    "type", "types", "kind", "kinds", "variety", "sort",
    "set", "group", "number", "amount", "quantity",
    "area", "part", "parts", "portion", "region", "section", "side",
    "point", "points",  # as in "Point A", not physical points
    # Question scaffolding phrases (after lemmatization)
    "example", "instance", "result", "answer", "question",
})

# Phrases that should be removed wholesale (regex patterns)
SCAFFOLDING_PHRASES_RE = re.compile(
    r"\b("
    r"in the (?:image|picture|photo|photograph|scene|figure|diagram)"
    r"|from the (?:image|picture|photo|photograph)"
    r"|of the (?:image|picture|photo|photograph)"
    r"|as shown in the (?:image|picture|photo|photograph)"
    r"|in the following"
    r"|from the options below"
    r"|out of the following"
    r"|of the following"
    r"|among (?:these|the following)"
    r"|with option"
    r"|which (?:one|point) (?:is|are)"
    r"|which of the following (?:conditions|options|points)"
    r"|can you (?:tell me|see|infer)"
    r"|(?:we )?already know that"
    r"|how many"
    r")\b",
    re.IGNORECASE,
)

# Leading determiners to strip from noun phrases
DET_RE = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)

# Max word count for an option to be treated as a concept entity
# (longer options are sentences, not entities)
MAX_OPTION_WORDS = 5

# ------------------------------------------------------------------------------
# ConceptNet synonym map (shared with visual_extraction/entity_export.py)
# ------------------------------------------------------------------------------

SYNONYM_MAP: Dict[str, str] = {
    # Shapes / objects
    "sphere": "ball", "globe": "ball", "orb": "ball",
    "disc": "disk",
    "cuboid": "box", "rectangular_box": "box", "container": "box",
    "wedge": "ramp", "slope": "ramp", "inclined_plane": "ramp", "incline": "ramp",
    "plank": "board", "slab": "board",
    "ledge": "shelf",
    # Materials
    "wooden": "wood", "timber": "wood",
    "metallic": "metal", "iron": "metal", "steel": "metal",
    # General synonyms
    "automobile": "car", "vehicle": "car",
    "tube": "cylinder", "pipe": "cylinder",
    "rope": "string", "cord": "string",
}

# ------------------------------------------------------------------------------
# Physics concept injection
# ------------------------------------------------------------------------------
# Maps question keywords/patterns to physics concepts that should be injected
# as KG seeds. These are the "implicit" concepts the question is really about.

PHYSICS_KEYWORD_MAP: Dict[str, List[str]] = {
    # Motion & dynamics
    "fastest": ["velocity", "speed", "flow_rate"],
    "slowest": ["velocity", "speed", "flow_rate"],
    "speed": ["velocity", "speed"],
    "velocity": ["velocity", "speed"],
    "accelerat": ["acceleration", "force"],
    "moving": ["motion", "velocity"],
    "motion": ["motion", "velocity"],
    "fall": ["gravity", "free_fall", "acceleration"],
    "falling": ["gravity", "free_fall", "acceleration"],
    "drop": ["gravity", "free_fall"],
    "roll": ["rolling", "friction", "inclined_plane"],
    "rolling": ["rolling", "friction", "inclined_plane"],
    "slide": ["sliding", "friction"],
    "sliding": ["sliding", "friction"],
    "bounc": ["elasticity", "collision", "restitution"],
    "collid": ["collision", "momentum", "impact"],
    "collision": ["collision", "momentum", "impact"],
    # Fluid
    "flow": ["fluid_flow", "velocity", "viscosity"],
    "liquid": ["liquid", "fluid", "viscosity"],
    "fluid": ["fluid", "liquid", "viscosity"],
    "float": ["buoyancy", "density"],
    "sink": ["buoyancy", "density"],
    "buoyan": ["buoyancy", "density", "fluid"],
    "pressure": ["pressure", "force", "area"],
    # Force & energy
    "force": ["force", "newton"],
    "friction": ["friction", "surface", "coefficient"],
    "gravity": ["gravity", "weight", "mass"],
    "energy": ["energy", "kinetic_energy", "potential_energy"],
    "momentum": ["momentum", "mass", "velocity"],
    "torque": ["torque", "rotation", "force"],
    "elastic": ["elasticity", "deformation", "spring"],
    "spring": ["spring", "elasticity", "hooke_law"],
    "pendulum": ["pendulum", "oscillation", "gravity"],
    # Thermodynamics
    "heat": ["heat", "temperature", "thermal_energy"],
    "temperature": ["temperature", "heat", "thermal"],
    "melt": ["melting", "phase_change", "heat"],
    "freeze": ["freezing", "phase_change", "temperature"],
    "boil": ["boiling", "phase_change", "heat"],
    "evaporat": ["evaporation", "phase_change"],
    # Properties
    "volume": ["volume", "size"],
    "mass": ["mass", "weight"],
    "weight": ["weight", "mass", "gravity"],
    "heav": ["weight", "mass", "density"],
    "light": [],  # ambiguous — skip (could be light source or lightweight)
    "dense": ["density", "mass", "volume"],
    "density": ["density", "mass", "volume"],
    "tall": ["height", "size"],
    "height": ["height", "size"],
    "big": ["size", "volume"],
    "small": ["size", "volume"],
    "large": ["size", "volume"],
    "largest": ["size", "volume"],
    "biggest": ["size", "volume"],
    "smallest": ["size", "volume"],
    # Spatial
    "closest": ["distance", "proximity"],
    "nearest": ["distance", "proximity"],
    "farthest": ["distance"],
    "distance": ["distance"],
    "behind": ["spatial_relation", "depth"],
    "front": ["spatial_relation", "depth"],
    "depth": ["depth", "distance"],
    "on top": ["support", "stacking", "gravity"],
    "below": ["spatial_relation", "gravity"],
    "above": ["spatial_relation"],
    # Optics & light
    "light source": ["light", "illumination", "shadow"],
    "shadow": ["shadow", "light", "optics"],
    "reflect": ["reflection", "light", "optics"],
    "refract": ["refraction", "light", "optics"],
    "mirror": ["mirror", "reflection", "optics"],
    "lens": ["lens", "refraction", "optics"],
    # Manipulation
    "pick up": ["grasp", "manipulation", "force"],
    "affordance": ["affordance", "grasp", "manipulation"],
    "grasp": ["grasp", "manipulation", "force"],
    "push": ["force", "push", "friction"],
    "pull": ["force", "pull", "tension"],
    # Fracture / structural
    "broken": ["fracture", "impact", "force"],
    "break": ["fracture", "impact", "force"],
    "crack": ["fracture", "stress"],
    "shatter": ["fracture", "impact", "brittleness"],
    "stable": ["stability", "equilibrium", "balance"],
    "balance": ["balance", "equilibrium", "center_of_mass"],
    "topple": ["stability", "center_of_mass", "torque"],
    "tip": ["stability", "center_of_mass"],
}


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def normalize_phrase(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)  # keep hyphens for compounds like "ping-pong"
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" \t\n\r-")  # strip trailing/leading hyphens
    return s


def strip_determiners(s: str) -> str:
    """Remove leading determiners (the, a, an, this, etc.)."""
    return DET_RE.sub("", s).strip()


# spaCy mis-lemmatizes spatial words (e.g., "left" → "leave")
_SPATIAL_LEMMA_FIXES = {
    "leave": "left",  # "left side" should NOT become "leave side"
}


def lemmatize_phrase(phrase: str) -> str:
    """Lemmatize a phrase, keeping content words only.

    Preserves hyphenated compounds (e.g., 'ping-pong' stays as 'ping-pong').
    Fixes known spaCy mis-lemmatizations for spatial terms.
    """
    doc = nlp(phrase)
    lemmas = []
    for tok in doc:
        # Keep content words; drop stop words unless it's the only token
        if tok.is_stop and len(doc) > 1:
            continue
        lemma = tok.lemma_.lower()
        # Fix spatial mis-lemmatizations
        if lemma in _SPATIAL_LEMMA_FIXES and tok.text.lower() != lemma:
            lemma = _SPATIAL_LEMMA_FIXES[lemma]
        lemmas.append(lemma)
    result = " ".join(lemmas) if lemmas else phrase
    # Re-join hyphenated tokens that spaCy may have split
    result = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", result)
    return result


def strip_obj_ref(s: str) -> str:
    s = (s or "").strip()
    m = OBJ_REF_SUFFIX_RE.match(s)
    return m.group(1) if m else s


def apply_synonym(entity: str) -> str:
    """Apply synonym normalization (underscore-joined form)."""
    key = entity.replace(" ", "_")
    mapped = SYNONYM_MAP.get(key)
    if mapped:
        return mapped.replace("_", " ")
    return entity


def is_stop_entity(entity: str) -> bool:
    """Check if entity is a stop word or scaffolding term."""
    words = entity.split()
    # Single-word check
    if len(words) == 1 and entity in STOP_ENTITIES:
        return True
    # Multi-word: reject if ALL content words are stop entities
    if all(w in STOP_ENTITIES for w in words):
        return True
    # Multi-word: reject if it's mostly stop words with only generic remnants
    content_words = [w for w in words if w not in STOP_ENTITIES]
    if len(content_words) == 0:
        return True
    # "follow condition", "follow point" — if >50% stop words, likely scaffolding
    if len(words) >= 2 and len(content_words) <= 1:
        # The one remaining content word might still be generic
        if content_words and content_words[0] in {
            "condition", "point", "color", "part", "area", "side",
            "section", "region", "portion", "manner", "way",
        }:
            return True
    return False


def is_numeric_option(text: str) -> bool:
    """Check if option is a bare number like '1', '2', '3.5'."""
    try:
        float(text.strip())
        return True
    except ValueError:
        return False


def clean_stem_text(stem: str) -> str:
    """Remove scaffolding phrases from question stem for cleaner extraction."""
    cleaned = SCAFFOLDING_PHRASES_RE.sub(" ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ------------------------------------------------------------------------------
# Question parsing (unchanged from original — works well)
# ------------------------------------------------------------------------------

def split_inline_options(line: str) -> Optional[List[str]]:
    if not line:
        return None
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
    """Split raw question text into stem_text and list of option texts."""
    q = q or ""
    lines = [ln.strip() for ln in q.splitlines() if ln.strip()]

    stem_lines: List[str] = []
    options: List[str] = []

    for ln in lines:
        inline_opts = split_inline_options(ln)
        if inline_opts is not None:
            options.extend(inline_opts)
            continue

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
# Physics concept detection
# ------------------------------------------------------------------------------

def detect_physics_concepts(stem: str, options: List[str]) -> List[str]:
    """Detect physics concepts implied by question keywords.

    Scans the question stem (and options) for keywords that map to
    physics concepts useful for KG retrieval.
    """
    text = (stem + " " + " ".join(options)).lower()
    concepts: List[str] = []
    seen = set()

    for keyword, mapped_concepts in PHYSICS_KEYWORD_MAP.items():
        if keyword in text:
            for c in mapped_concepts:
                if c not in seen:
                    seen.add(c)
                    concepts.append(c)

    return concepts


# ------------------------------------------------------------------------------
# Core entity extraction (improved)
# ------------------------------------------------------------------------------

def extract_entities_from_text(text: str) -> List[Tuple[str, str]]:
    """Extract (entity, source_type) pairs from a text span.

    Returns clean, lemmatized, determiner-stripped entities with their source:
    - "noun_phrase": from spaCy noun chunks
    - "named_entity": from spaCy NER
    - "adj_noun": adjective-noun compounds
    - "verb": physically relevant verbs
    """
    doc = nlp(text)
    entities: List[Tuple[str, str]] = []

    # --- Noun chunks (the main entity source) ---
    for chunk in doc.noun_chunks:
        raw = chunk.text.strip()
        clean = strip_determiners(raw)
        clean = normalize_phrase(clean)
        if not clean:
            continue

        # Lemmatize
        lemmatized = lemmatize_phrase(clean)
        lemmatized = apply_synonym(lemmatized)

        if lemmatized and not is_stop_entity(lemmatized):
            entities.append((lemmatized, "noun_phrase"))

    # --- Named entities ---
    for ent in doc.ents:
        raw = ent.text.strip()
        clean = strip_determiners(raw)
        clean = normalize_phrase(clean)
        if not clean:
            continue

        lemmatized = lemmatize_phrase(clean)
        lemmatized = apply_synonym(lemmatized)

        if lemmatized and not is_stop_entity(lemmatized):
            entities.append((lemmatized, "named_entity"))

    # --- Adjective-noun pairs ---
    for tok in doc:
        if tok.dep_ == "amod" and tok.head.pos_ in {"NOUN", "PROPN"}:
            adj = tok.lemma_.lower()
            noun = tok.head.lemma_.lower()
            # Skip if the adjective is scaffolding (e.g., "close source")
            if adj in STOP_ENTITIES or noun in STOP_ENTITIES:
                continue
            pair = f"{adj} {noun}"
            pair = apply_synonym(pair)
            if not is_stop_entity(pair):
                entities.append((pair, "adj_noun"))

    # --- Physically relevant verbs (not function verbs) ---
    PHYSICAL_VERBS = {
        "flow", "fall", "drop", "roll", "slide", "bounce", "collide",
        "float", "sink", "melt", "freeze", "boil", "evaporate",
        "push", "pull", "lift", "throw", "catch", "hit", "strike",
        "spin", "rotate", "oscillate", "vibrate", "swing",
        "compress", "stretch", "bend", "break", "crack", "shatter",
        "accelerate", "decelerate", "stop", "move", "travel",
        "support", "balance", "topple", "stack", "lean",
        "pour", "fill", "empty", "overflow", "drip",
        "burn", "ignite", "explode", "dissolve", "mix",
        "reflect", "refract", "absorb", "emit", "glow",
    }
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() in PHYSICAL_VERBS:
            entities.append((tok.lemma_.lower(), "verb"))

    return entities


def extract_option_entities(options: List[str]) -> List[Tuple[str, str]]:
    """Extract entities from answer options.

    Options are treated specially:
    - Short options (≤ MAX_OPTION_WORDS words) are kept as whole entities
    - Numeric options are kept as-is (useful for counting questions)
    - Long sentence-options are parsed for noun phrases only
    """
    entities: List[Tuple[str, str]] = []

    for opt in options:
        clean = normalize_phrase(opt)
        if not clean:
            continue

        # Numeric options: "1", "2", "3", "4"
        if is_numeric_option(clean):
            entities.append((clean, "option_numeric"))
            continue

        # "Point A", "Point B" — keep as-is (useful for spatial questions)
        if re.match(r"^point\s+[a-d]$", clean, re.IGNORECASE):
            entities.append((clean, "option_point"))
            continue

        word_count = len(clean.split())

        if word_count <= MAX_OPTION_WORDS:
            # Short option = likely a concrete entity like "red cube", "cyan ball"
            clean = strip_determiners(clean)
            lemmatized = lemmatize_phrase(clean)
            lemmatized = apply_synonym(lemmatized)
            if lemmatized and not is_stop_entity(lemmatized):
                entities.append((lemmatized, "option_entity"))
        else:
            # Long option = sentence; extract noun phrases from it
            for ent, src in extract_entities_from_text(clean):
                entities.append((ent, "option_parsed"))

    return entities


def extract_stem_subject(stem: str) -> Optional[str]:
    """Extract the main subject entity from the question stem.

    For questions like "Which object is the cyan ball closest to?",
    the subject is "cyan ball". For "In which area does the liquid flow fastest?",
    the subject is "liquid flow" or "liquid".
    """
    doc = nlp(stem)

    # Look for the subject of the root verb
    for tok in doc:
        if tok.dep_ == "ROOT":
            # Check for subject noun phrases
            for child in tok.children:
                if child.dep_ in {"nsubj", "nsubjpass", "attr"}:
                    # Get the full noun chunk containing this token
                    for chunk in doc.noun_chunks:
                        if child.i >= chunk.start and child.i < chunk.end:
                            subj = strip_determiners(chunk.text)
                            subj = normalize_phrase(subj)
                            if subj and not is_stop_entity(subj):
                                return lemmatize_phrase(subj)

    return None


# ------------------------------------------------------------------------------
# Build final word-groups (the main output)
# ------------------------------------------------------------------------------

def make_word_groups(
    stem: str,
    options: List[str],
    obj_list: Optional[List[str]],
) -> List[str]:
    """Build scored, deduplicated entity list from question stem + options.

    Scoring (higher = more important, appears earlier in output):
      - option_entity:    5  (answer candidates — most important for KG)
      - physics_concept:  4  (domain concepts implied by question)
      - stem_subject:     4  (what the question asks about)
      - option_point:     3  (spatial reference points)
      - noun_phrase:      2  (from question stem)
      - adj_noun:         2  (compound descriptors)
      - verb:             1  (physical actions)
      - option_numeric:   1  (counting answers)
      - option_parsed:    2  (entities from long option sentences)
      - object_ref:       3  (explicit object list from dataset)
    """
    SCORE_MAP = {
        "option_entity": 5,
        "physics_concept": 4,
        "stem_subject": 4,
        "option_point": 3,
        "object_ref": 3,
        "noun_phrase": 2,
        "adj_noun": 2,
        "option_parsed": 2,
        "named_entity": 2,
        "verb": 1,
        "option_numeric": 1,
    }

    # Collect all (entity, source_type) pairs
    all_entities: List[Tuple[str, str]] = []

    # 1. Clean the stem and extract entities from it
    cleaned_stem = clean_stem_text(stem)
    stem_entities = extract_entities_from_text(cleaned_stem)
    all_entities.extend(stem_entities)

    # 2. Extract the question subject for boosting
    subject = extract_stem_subject(stem)
    if subject:
        all_entities.append((subject, "stem_subject"))

    # 3. Extract entities from options
    option_entities = extract_option_entities(options)
    all_entities.extend(option_entities)

    # 4. Detect physics concepts from the question
    physics_concepts = detect_physics_concepts(stem, options)
    for pc in physics_concepts:
        all_entities.append((pc.replace("_", " "), "physics_concept"))

    # 5. Add explicit object references from dataset
    if obj_list:
        for o in obj_list:
            if isinstance(o, str) and o.strip():
                ref = strip_obj_ref(o)
                ref = normalize_phrase(ref)
                ref = lemmatize_phrase(ref)
                ref = apply_synonym(ref)
                if ref and not is_stop_entity(ref):
                    all_entities.append((ref, "object_ref"))

    # --- Deduplicate and score ---
    entity_scores: Dict[str, int] = {}
    entity_order: Dict[str, int] = {}  # preserve first-seen order within score tier

    for idx, (entity, source) in enumerate(all_entities):
        # Underscore form for dedup
        key = entity.replace(" ", "_")
        if not key:
            continue
        # Allow short numeric options ("1", "2") but filter other single-char junk
        if len(key) < 2 and source not in ("option_numeric", "option_point"):
            continue

        score = SCORE_MAP.get(source, 1)
        # Keep highest score for each entity
        if key not in entity_scores or score > entity_scores[key]:
            entity_scores[key] = score
        if key not in entity_order:
            entity_order[key] = idx

    # Sort by score (descending), then by first appearance
    sorted_entities = sorted(
        entity_scores.keys(),
        key=lambda k: (-entity_scores[k], entity_order[k]),
    )

    # Convert back to space-separated form for readability
    result = [e.replace("_", " ") for e in sorted_entities]
    return result


# ------------------------------------------------------------------------------
# Top-level extraction per sample (preserves output format)
# ------------------------------------------------------------------------------

def extract_concepts(text: str) -> Dict[str, Any]:
    """Extract raw NLP features from text (kept for backward compat in output)."""
    doc = nlp(text)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    root_verbs: List[str] = []
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}:
            root_verbs.append(tok.lemma_)
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
    merged: Dict[str, list] = {
        "entities": [], "noun_phrases": [], "root_verbs": [],
        "all_verbs": [], "adjective_noun_pairs": [],
        "adjective_complements": [], "svos": [],
    }
    for ex in extracted_list:
        for k in merged:
            merged[k].extend(ex.get(k, []))

    def uniq(seq):
        return list(dict.fromkeys(seq))
    for k in merged:
        merged[k] = uniq(merged[k])
    return merged


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

        if sample.get("mode") != "image-only":
            continue

        question_raw = sample.get("question") or ""
        stem_text, options = split_question_and_options(question_raw)

        # --- Raw NLP extraction (kept for the `extracted` field) ---
        extracted_parts: List[Dict[str, Any]] = []
        if stem_text:
            extracted_parts.append(extract_concepts(stem_text))
        for opt in options:
            extracted_parts.append(extract_concepts(opt))
        extracted = merge_extracted(extracted_parts) if extracted_parts else extract_concepts("")

        # --- Improved word_groups (the main improvement) ---
        obj_list = sample.get("object")
        word_groups = make_word_groups(stem_text, options, obj_list)

        results.append({
            "idx": sample.get("idx"),
            "split": sample.get("split"),
            "mode": sample.get("mode"),
            "question_stem": stem_text,
            "options": options,
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

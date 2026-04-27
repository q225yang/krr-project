"""
Unified Qwen2.5-VL inference script.

Pick one KG source via KG_SOURCE:
    None              — no KG (baseline)
    "snowflake"       — subgraph_retrieval/kg_snowflake/kg_trad_question.json
                        (ConceptNet subgraphs from traditional NLP entity extraction)
    "snowflake_llm"   — subgraph_retrieval/kg_snowflake/kg_llm_mixed.json
                        (ConceptNet subgraphs from LLM-based entity extraction)
    "vlm"             — qwen_entity_kg_inference/final_vlm_kr.json
                        (VLM-extracted KR: list of {idx, kr: [[h, r, t], ...]})

Everything else (KG handling, prompts, metrics, output structure) is identical to
internvl_inference.py so accuracy comparisons are clean apples-to-apples.
"""

import json
import re
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ===================== config =====================
KG_SOURCE = "snowflake_llm"   # "snowflake" | "snowflake_llm" | "vlm" | None

QUESTION_FILE = "/data/xinyua11/krr-project/data/test.json"
ANSWER_FILE   = "/data/xinyua11/krr-project/test_answer.json"
IMAGE_DIR     = "/data/xinyua11/krr-project/data/image"
RESULTS_DIR   = "/data/xinyua11/krr-project/results"

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_SAMPLES = None         # None = full set; set an int for a quick smoke test


# ===================== unified KG renderer =====================
# One render function, one cap, applied to both KG sources. Per-source registry
# below only varies file path, schema-load adapter, and output tag.
KG_MAX_LINES = 8

# Map snowflake (ConceptNet) CamelCase rels → natural-language phrase, so a
# snowflake triple lands in the same `head rel-phrase tail` shape as a VLM triple.
_REL_TO_PHRASE = {
    "IsA":          "is a type of",
    "AtLocation":   "is typically found at/in",
    "CapableOf":    "can",
    "Causes":       "causes",
    "HasA":         "has",
    "PartOf":       "is part of",
    "UsedFor":      "is used for",
    "HasProperty":  "has the property",
    "MadeOf":       "is made of",
    "RelatedTo":    "is related to",
    "DistinctFrom": "is distinct from",
    "Synonym":      "is a synonym of",
    "Antonym":      "is an antonym of",
}
# Pure-morphology snowflake relations that add no semantic signal — drop.
_SNOWFLAKE_NOISY_RELS = {"FormOf", "DerivedFrom"}

# VLM filtering rules — also safe to apply to snowflake.
_WEAK_VERBS = ["contains", "shows", "has", "depicts", "includes", "is shown", "appears"]
_CAUSAL_KW = [
    "cause", "lead", "result", "produce",
    "increase", "decrease", "trigger", "affect",
    "generate", "break",
]


def _clean_concept(uri):
    """'/c/en/liquid_flow/n' -> 'liquid flow' (snowflake URIs)."""
    if not uri:
        return ""
    s = uri[len("/c/en/"):] if uri.startswith("/c/en/") else uri
    s = re.sub(r"/[a-z]$", "", s)
    return s.replace("_", " ").strip()


def _camel_to_phrase(rel):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", rel).lower()


def _extract_triples(kg_item):
    """Normalize either KG schema into a list of (head, rel_phrase, tail) string tuples."""
    if not kg_item or not isinstance(kg_item, dict):
        return []

    # VLM schema: kr is already [[h, r, t], ...] with natural-language rels.
    if "kr" in kg_item:
        return [
            (str(h), str(r), str(t))
            for triple in kg_item["kr"]
            if isinstance(triple, (list, tuple)) and len(triple) == 3
            for h, r, t in [triple]
        ]

    # Snowflake schema: ConceptNet URIs + CamelCase rels.
    if "triples" in kg_item:
        out = []
        for tr in kg_item["triples"]:
            rel = tr.get("rel", "")
            if rel in _SNOWFLAKE_NOISY_RELS:
                continue
            h = _clean_concept(tr.get("start", ""))
            t = _clean_concept(tr.get("end", ""))
            if not h or not t:
                continue
            phrase = _REL_TO_PHRASE.get(rel) or _camel_to_phrase(rel)
            out.append((h, phrase, t))
        return out

    return []


def render_kg(kg_item, max_lines=KG_MAX_LINES):
    """Drop weak-verb triples, push causal-rel triples first, dedup, render as bullets."""
    triples = _extract_triples(kg_item)
    if not triples:
        return ""

    triples = [
        t for t in triples
        if not any(w in " ".join(t).lower() for w in _WEAK_VERBS)
    ]
    if not triples:
        return ""

    causal = [t for t in triples if any(k in t[1].lower() for k in _CAUSAL_KW)]
    others = [t for t in triples if t not in causal]
    ordered = causal + others

    seen, lines = set(), []
    for h, r, t in ordered:
        s = f"{h} {r} {t}"
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(s)
        if len(lines) >= max_lines:
            break

    return "\n".join(f"- {s}" for s in lines) if lines else ""


# ----- registry: only what varies between sources -----
KG_REGISTRY = {
    "snowflake": {
        "file":   "/data/xinyua11/krr-project/subgraph_retrieval/kg_snowflake/kg_trad_question.json",
        "tag":    "kg_trad",
        "to_map": lambda data: data,                       # already dict-by-idx
    },
    "snowflake_llm": {
        "file":   "/data/xinyua11/krr-project/subgraph_retrieval/kg_snowflake/kg_llm_mixed.json",
        "tag":    "kg_llm",
        "to_map": lambda data: data,                       # same schema as snowflake
    },
    "vlm": {
        "file":   "/data/xinyua11/krr-project/qwen_entity_kg_inference/final_vlm_kr.json",
        "tag":    "kg_vlm",
        "to_map": lambda data: {x["idx"]: x for x in data},  # list -> dict-by-idx
    },
}

cfg = KG_REGISTRY.get(KG_SOURCE)
TAG = cfg["tag"] if cfg else "no_kg"
OUTPUT_FILE = f"{RESULTS_DIR}/predictions_qwen_{TAG}.jsonl"
RESULT_FILE = f"{RESULTS_DIR}/results_qwen_{TAG}.json"


# ===================== helpers =====================
def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def extract_choice(text):
    text = (text or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text)
    return m.group(1) if m else None


def strip_image_placeholders(question):
    """Dataset questions carry InternVL-style `<image>` tokens; Qwen sees them as
    literal text, which corrupts generation. Drop them — Qwen gets images via
    the chat-template content list instead."""
    return re.sub(r"<image>\s*", "", question).strip()


# ===================== prompts =====================
# Same structure as internvl_inference.py, minus the `<image>` placeholders —
# Qwen receives images as separate message-content entries, not inline tokens.
_ANSWER_INSTRUCTION = "Answer with only one letter: A, B, C, or D."
_KG_HEADER = "Relevant physical knowledge:"
_KG_USAGE = "Use the knowledge only if it helps answer the question."


def build_prompt_single(question, kg_text):
    if not kg_text:
        return f"{question}\n\n{_ANSWER_INSTRUCTION}"
    return (
        f"{question}\n\n"
        f"{_KG_HEADER}\n{kg_text}\n\n"
        f"{_KG_USAGE}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )


def build_prompt_four(question, kg_text):
    header = (
        "The four images above correspond to options A, B, C, and D respectively."
    )
    if not kg_text:
        return f"{header}\n\nQuestion:\n{question}\n\n{_ANSWER_INSTRUCTION}"
    return (
        f"{header}\n\n"
        f"Question:\n{question}\n\n"
        f"{_KG_HEADER}\n{kg_text}\n\n"
        f"{_KG_USAGE}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )


def build_messages(image_paths, prompt_text):
    """Qwen's chat template needs a path/URL string per image entry to emit the
    `<|image_pad|>` placeholders correctly. Actual pixel data goes to the
    processor's `images=` arg separately."""
    content = [{"type": "image", "image": str(p)} for p in image_paths]
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def get_kg_item(kg_map, idx):
    return kg_map.get(idx) or kg_map.get(str(idx))


# ===================== load model =====================
processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()


# ===================== load data =====================
with open(QUESTION_FILE, encoding="utf-8") as f:
    questions = json.load(f)
with open(ANSWER_FILE, encoding="utf-8") as f:
    answers = json.load(f)

if cfg:
    with open(cfg["file"], encoding="utf-8") as f:
        kg_map = cfg["to_map"](json.load(f))
    print(f"loaded KG '{KG_SOURCE}' from {cfg['file']}  ({len(kg_map)} entries)")
else:
    kg_map = {}
    print("running baseline (no KG)")

answer_dict = {a["idx"]: a["answer"] for a in answers}
meta_dict   = {a["idx"]: a for a in answers}


generation_kwargs = dict(max_new_tokens=8, do_sample=False, use_cache=True)


@torch.inference_mode()
def run_inference(image_paths, pil_images, prompt_text):
    messages = build_messages(image_paths, prompt_text)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=pil_images, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, **generation_kwargs)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ===================== inference loop =====================
results = []
correct = 0
total = 0

open(OUTPUT_FILE, "w", encoding="utf-8").close()

for sample in questions:
    if sample.get("mode") != "image-only":
        continue
    if MAX_SAMPLES is not None and total >= MAX_SAMPLES:
        break

    idx = sample["idx"]
    question = strip_image_placeholders(sample["question"])
    files = sample["file_name"]
    gt = answer_dict[idx]

    if cfg:
        kg_text = render_kg(get_kg_item(kg_map, idx))
    else:
        kg_text = ""

    image_paths = [f"{IMAGE_DIR}/{x}" for x in files]
    num_images = len(image_paths)

    try:
        images = [load_image(p) for p in image_paths]

        if num_images == 1:
            prompt = build_prompt_single(question, kg_text)
        elif num_images == 4:
            prompt = build_prompt_four(question, kg_text)
        else:
            print(f"idx {idx} skipped: unsupported number of images = {num_images}")
            continue

        output_text = run_inference(image_paths, images, prompt)

    except Exception as e:
        print(f"idx {idx} error: {e}")
        output_text = ""

    pred = extract_choice(output_text)
    is_correct = (pred == gt)

    correct += int(is_correct)
    total += 1

    meta = meta_dict.get(idx, {})
    result = {
        "idx": idx,
        "prediction": pred,
        "answer": gt,
        "correct": is_correct,
        "raw_output": output_text,
        "question": question,
        "files": files,
        "kg_text": kg_text,
        "n_kg_lines": (kg_text.count("\n") + 1) if kg_text else 0,
        "task_type": meta.get("task_type"),
        "sub_type": meta.get("sub_type"),
        "ability_type": meta.get("ability_type"),
    }
    results.append(result)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"idx {idx} | pred {pred} | gt {gt} | correct {is_correct} "
        f"| n_kg={result['n_kg_lines']} | raw={repr(output_text)}"
    )


# ===================== metrics =====================
acc = correct / total if total > 0 else 0.0
print("\nAccuracy:", acc)


def _accuracy_by(results, key):
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        k = r.get(key)
        buckets[k]["total"] += 1
        buckets[k]["correct"] += int(bool(r.get("correct")))
    return {
        str(k): {"correct": v["correct"], "total": v["total"],
                 "accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0}
        for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
    }


def _kg_bucket(n):
    if n == 0: return "0"
    if n <= 2: return "1-2"
    if n <= 5: return "3-5"
    return "6-8"


for r in results:
    r["_kg_bucket"] = _kg_bucket(r["n_kg_lines"])

pred_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "invalid": 0}
for r in results:
    p = r.get("prediction")
    pred_dist[p if p in pred_dist else "invalid"] += 1

summary = {
    "kg_source": KG_SOURCE,
    "kg_file": cfg["file"] if cfg else None,
    "kg_max_lines": KG_MAX_LINES if cfg else 0,
    "model": MODEL_NAME,
    "mode_filter": "image-only",
    "total": total,
    "correct": correct,
    "accuracy": acc,
    "prediction_distribution": pred_dist,
    "accuracy_by_task_type":   _accuracy_by(results, "task_type"),
    "accuracy_by_sub_type":    _accuracy_by(results, "sub_type"),
    "accuracy_by_ability_type":_accuracy_by(results, "ability_type"),
    "accuracy_by_kg_bucket":   _accuracy_by(results, "_kg_bucket"),
}

for r in results:
    r.pop("_kg_bucket", None)

print("\nPrediction distribution:", pred_dist)
print("Accuracy by ability_type:", summary["accuracy_by_ability_type"])
print("Accuracy by kg_bucket:   ", summary["accuracy_by_kg_bucket"])

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

print("Saved jsonl to:", OUTPUT_FILE)
print("Saved full results to:", RESULT_FILE)

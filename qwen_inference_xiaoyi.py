"""
Unified Qwen2.5-VL inference script.

Pick one KG source via KG_SOURCE:
    None              — no KG baseline
    "star"            — image-only KG
    "snowflake"       — ConceptNet traditional NLP KG
    "snowflake_llm"   — ConceptNet LLM KG
    "vlm"             — VLM-extracted KG
"""

import json
import re
import traceback
from collections import defaultdict

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ===================== config =====================
KG_SOURCE = "star"   # "star" | "snowflake" | "snowflake_llm" | "vlm" | None

QUESTION_FILE = "/mnt/disk11/user/xiaoyih1/xh/krr/test.json"
ANSWER_FILE   = "/mnt/disk11/user/xiaoyih1/xh/krr/new/krr-project/test_answer.json"
IMAGE_DIR     = "/mnt/disk11/user/xiaoyih1/xh/krr/image"
RESULTS_DIR   = "/mnt/disk11/user/xiaoyih1/xh/krr/new/krr-project/results"

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

MAX_SAMPLES = None   # None = full set; set int for smoke test


# ===================== KG rendering =====================
KG_MAX_LINES = 8

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

_SNOWFLAKE_NOISY_RELS = {"FormOf", "DerivedFrom"}

_WEAK_VERBS = [
    "contains", "shows", "has", "depicts", "includes",
    "is shown", "appears"
]

_CAUSAL_KW = [
    "cause", "lead", "result", "produce",
    "increase", "decrease", "trigger", "affect",
    "generate", "break",
]


def _clean_concept(uri):
    if not uri:
        return ""
    s = uri[len("/c/en/"):] if uri.startswith("/c/en/") else uri
    s = re.sub(r"/[a-z]$", "", s)
    return s.replace("_", " ").strip()


def _camel_to_phrase(rel):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", rel).lower()


def _extract_triples(kg_item):
    if not kg_item or not isinstance(kg_item, dict):
        return []

    if "kr" in kg_item:
        return [
            (str(h), str(r), str(t))
            for triple in kg_item["kr"]
            if isinstance(triple, (list, tuple)) and len(triple) == 3
            for h, r, t in [triple]
        ]

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


KG_REGISTRY = {
    "star": {
        "file": "/mnt/disk11/user/xiaoyih1/xh/krr/new/krr-project/knowledge_graph/kg_visual_v3.json",
        "tag": "kg_star_v3_visual",
        "to_map": lambda data: data,
    },
    "snowflake": {
        "file": "/data/xinyua11/krr-project/subgraph_retrieval/kg_snowflake/kg_trad_question.json",
        "tag": "kg_trad",
        "to_map": lambda data: data,
    },
    "snowflake_llm": {
        "file": "/data/xinyua11/krr-project/subgraph_retrieval/kg_snowflake/kg_llm_mixed.json",
        "tag": "kg_llm",
        "to_map": lambda data: data,
    },
    "vlm": {
        "file": "/data/xinyua11/krr-project/qwen_entity_kg_inference/final_vlm_kr.json",
        "tag": "kg_vlm",
        "to_map": lambda data: {x["idx"]: x for x in data},
    },
}

cfg = KG_REGISTRY.get(KG_SOURCE)
TAG = cfg["tag"] if cfg else "no_kg"

OUTPUT_FILE = f"{RESULTS_DIR}/predictions_qwen25vl_{TAG}.jsonl"
RESULT_FILE = f"{RESULTS_DIR}/results_qwen25vl_{TAG}.json"


# ===================== prompt =====================
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
        "These four images correspond to options A, B, C, and D respectively."
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


def get_kg_item(kg_map, idx):
    return kg_map.get(idx) or kg_map.get(str(idx))


def extract_choice(text):
    text = (text or "").strip().upper()

    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)

    m = re.search(r"([ABCD])", text)
    return m.group(1) if m else None


# ===================== load model =====================
print("Loading Qwen2.5-VL model:", MODEL_NAME)

# processor = AutoProcessor.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True,
# )

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    min_pixels=224 * 224,
    max_pixels=336 * 336,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
).eval()


# ===================== Qwen inference =====================
def qwen_infer(image_paths, prompt):
    content = []

    for p in image_paths:
        content.append({
            "type": "image",
            "image": p,
        })

    content.append({
        "type": "text",
        "text": prompt,
    })

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


# ===================== load data =====================
with open(QUESTION_FILE) as f:
    questions = json.load(f)

with open(ANSWER_FILE) as f:
    answers = json.load(f)

if cfg:
    with open(cfg["file"]) as f:
        kg_map = cfg["to_map"](json.load(f))
    print(f"loaded KG '{KG_SOURCE}' from {cfg['file']} ({len(kg_map)} entries)")
else:
    kg_map = {}
    print("running baseline no KG")

answer_dict = {a["idx"]: a["answer"] for a in answers}
meta_dict = {a["idx"]: a for a in answers}


# ===================== inference loop =====================
results = []
correct = 0
total = 0

open(OUTPUT_FILE, "w").close()

for sample in questions:
    if sample.get("mode") != "image-only":
        continue

    if MAX_SAMPLES is not None and total >= MAX_SAMPLES:
        break

    idx = sample["idx"]
    question = sample["question"]
    files = sample["file_name"]
    gt = answer_dict[idx]

    if cfg:
        kg_text = render_kg(get_kg_item(kg_map, idx))
    else:
        kg_text = ""

    image_paths = [f"{IMAGE_DIR}/{x}" for x in files]
    num_images = len(image_paths)

    try:
        if num_images == 1:
            prompt = build_prompt_single(question, kg_text)
            output_text = qwen_infer(image_paths, prompt)

        elif num_images == 4:
            prompt = build_prompt_four(question, kg_text)
            output_text = qwen_infer(image_paths, prompt)

        else:
            print(f"idx {idx} skipped: unsupported number of images = {num_images}")
            continue

    except Exception as e:
        print(f"idx {idx} error: {repr(e)}")
        traceback.print_exc()
        output_text = ""

    pred = extract_choice(output_text)
    is_correct = pred == gt

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

    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"idx {idx} | pred {pred} | gt {gt} | correct {is_correct} | {sample["mode"]} | total {total}"
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
        str(k): {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
    }


def _kg_bucket(n):
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6-8"


for r in results:
    r["_kg_bucket"] = _kg_bucket(r["n_kg_lines"])

pred_dist = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0,
    "invalid": 0,
}

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
    "accuracy_by_task_type": _accuracy_by(results, "task_type"),
    "accuracy_by_sub_type": _accuracy_by(results, "sub_type"),
    "accuracy_by_ability_type": _accuracy_by(results, "ability_type"),
    "accuracy_by_kg_bucket": _accuracy_by(results, "_kg_bucket"),
}

for r in results:
    r.pop("_kg_bucket", None)

print("\nPrediction distribution:", pred_dist)
print("Accuracy by ability_type:", summary["accuracy_by_ability_type"])
print("Accuracy by kg_bucket:", summary["accuracy_by_kg_bucket"])

with open(RESULT_FILE, "w") as f:
    json.dump(
        {
            "summary": summary,
            "results": results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("Saved jsonl to:", OUTPUT_FILE)
print("Saved full results to:", RESULT_FILE)
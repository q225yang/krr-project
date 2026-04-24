import json
import re
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

USE_KG = True              # set to False to run the same prompt without retrieved KG facts

QUESTION_FILE = "/data/xinyua11/krr-project/data/test.json"
ANSWER_FILE   = "/data/xinyua11/krr-project/test_answer.json"
KG_FILE       = "/data/xinyua11/krr-project/subgraph_retrieval/kg_snowflake/kg_trad_question.json"
IMAGE_DIR     = "/data/xinyua11/krr-project/data/image"

_TAG = "kg_trad" if USE_KG else "no_kg"
OUTPUT_FILE = f"/data/xinyua11/krr-project/results/predictions_internvl_{_TAG}.jsonl"
RESULT_FILE = f"/data/xinyua11/krr-project/results/results_internvl_{_TAG}.json"

MODEL_NAME = "OpenGVLab/InternVL2_5-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_SAMPLES = None         # None = full set; set an int to cap for quick smoke tests
MAX_KG_LINES = 20          # cap KG lines fed into the prompt

# ===== image preprocess =====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 448

transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image_tensor(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def extract_choice(text: str):
    text = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text)
    return m.group(1) if m else None


# ===== KG rendering =====
_REL_TEMPLATES = {
    "IsA":          "{a} is a type of {b}",
    "AtLocation":   "{a} is typically found at/in {b}",
    "CapableOf":    "{a} can {b}",
    "Causes":       "{a} causes {b}",
    "HasA":         "{a} has {b}",
    "PartOf":       "{a} is part of {b}",
    "UsedFor":      "{a} is used for {b}",
    "HasProperty":  "{a} has the property {b}",
    "MadeOf":       "{a} is made of {b}",
    "RelatedTo":    "{a} is related to {b}",
    "DistinctFrom": "{a} is distinct from {b}",
    "FormOf":       "{a} is a form of {b}",
    "Synonym":      "{a} is a synonym of {b}",
    "Antonym":      "{a} is an antonym of {b}",
    "DerivedFrom":  "{a} is derived from {b}",
}

# Relations that add almost no semantic signal for physics MCQs — drop them.
_NOISY_RELS = {"FormOf", "DerivedFrom"}


def _clean_concept(uri: str) -> str:
    """'/c/en/liquid_flow/n' -> 'liquid flow'"""
    if not uri:
        return ""
    s = uri
    if s.startswith("/c/en/"):
        s = s[len("/c/en/"):]
    # strip trailing POS tag like '/n', '/v', '/a'
    s = re.sub(r"/[a-z]$", "", s)
    return s.replace("_", " ").strip()


def _clean_surface(text: str) -> str:
    if not text:
        return ""
    return text.replace("[[", "").replace("]]", "").strip()


def _triple_to_sentence(triple: dict) -> str:
    rel = triple.get("rel", "")
    if rel in _NOISY_RELS:
        return ""

    surf = _clean_surface(triple.get("surfaceText", ""))
    if surf:
        return surf

    a = _clean_concept(triple.get("start", ""))
    b = _clean_concept(triple.get("end", ""))
    if not a or not b:
        return ""

    tmpl = _REL_TEMPLATES.get(rel)
    if tmpl:
        return tmpl.format(a=a, b=b)
    # Unknown relation: fall back to readable form
    pretty_rel = re.sub(r"(?<!^)(?=[A-Z])", " ", rel).lower()
    return f"{a} {pretty_rel} {b}"


def build_kg_text(kg_item, max_lines=MAX_KG_LINES):
    """Render a subgraph item into a deduped, bulleted list of facts."""
    if not kg_item:
        return ""

    seen = set()
    lines = []
    for t in kg_item.get("triples", []):
        s = _triple_to_sentence(t)
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(s)
        if len(lines) >= max_lines:
            break

    if not lines:
        return ""
    return "\n".join(f"- {s}" for s in lines)


def get_kg_item(kg_data, idx):
    return kg_data.get(str(idx)) or kg_data.get(idx)


# ===== prompt building =====
_ANSWER_INSTRUCTION = "Answer with only one letter: A, B, C, or D."
_KG_HEADER = "Relevant facts about concepts in this question (use if helpful):"


def build_prompt_single(question: str, kg_text: str) -> str:
    # No KG: exactly match inference_internvl.py baseline so accuracy is comparable.
    if not kg_text:
        return (
            f"<image>\n"
            f"{question}\n\n"
            f"{_ANSWER_INSTRUCTION}"
        )
    # With KG: image → question → supporting facts → answer instruction.
    return (
        f"<image>\n"
        f"{question}\n\n"
        f"{_KG_HEADER}\n{kg_text}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )


def build_prompt_four(question: str, kg_text: str) -> str:
    header = (
        "Image-1: <image>\nImage-2: <image>\nImage-3: <image>\nImage-4: <image>\n\n"
        "These four images correspond to options A, B, C, and D respectively."
    )
    if not kg_text:
        return (
            f"{header}\n\n"
            f"Question:\n{question}\n\n"
            f"{_ANSWER_INSTRUCTION}"
        )
    return (
        f"{header}\n\n"
        f"Question:\n{question}\n\n"
        f"{_KG_HEADER}\n{kg_text}\n\n"
        f"{_ANSWER_INSTRUCTION}"
    )


# ===== load model =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True,
).eval().to(DEVICE)


# ===== load data =====
with open(QUESTION_FILE, "r") as f:
    questions = json.load(f)

with open(ANSWER_FILE, "r") as f:
    answers = json.load(f)

if USE_KG:
    with open(KG_FILE, "r") as f:
        kg_data = json.load(f)
else:
    kg_data = {}

answer_dict = {a["idx"]: a["answer"] for a in answers}
meta_dict = {a["idx"]: a for a in answers}


generation_config = dict(
    max_new_tokens=8,
    do_sample=False,
)


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

    if USE_KG:
        kg_item = get_kg_item(kg_data, idx)
        kg_text = build_kg_text(kg_item)
    else:
        kg_text = ""

    image_paths = [f"{IMAGE_DIR}/{x}" for x in files]
    num_images = len(image_paths)

    try:
        if num_images == 1:
            pixel_values = load_image_tensor(image_paths[0]).to(DTYPE).to(DEVICE)
            prompt = build_prompt_single(question, kg_text)
            output_text = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
            )

        elif num_images == 4:
            pixel_values_list = []
            num_patches_list = []
            for p in image_paths:
                pv = load_image_tensor(p)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])

            pixel_values = torch.cat(pixel_values_list, dim=0).to(DTYPE).to(DEVICE)
            prompt = build_prompt_four(question, kg_text)
            output_text = model.chat(
                tokenizer,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list,
            )

        else:
            print(f"idx {idx} skipped: unsupported number of images = {num_images}")
            continue

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
        "n_kg_lines": 0 if not kg_text else kg_text.count("\n- ") + 1,
        "task_type": meta.get("task_type"),
        "sub_type": meta.get("sub_type"),
        "ability_type": meta.get("ability_type"),
    }
    results.append(result)

    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"idx {idx} | pred {pred} | gt {gt} | correct {is_correct} "
        f"| n_kg={result['n_kg_lines']} | raw={repr(output_text)}"
    )


if total > 0:
    acc = correct / total
    print("\nAccuracy:", acc)
else:
    acc = 0.0
    print("\nNo valid samples were processed.")


def _accuracy_by(results, key):
    from collections import defaultdict
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
    if n <= 3: return "1-3"
    if n <= 10: return "4-10"
    return "11+"


for r in results:
    r["_kg_bucket"] = _kg_bucket(r["n_kg_lines"])

pred_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "invalid": 0}
for r in results:
    p = r.get("prediction")
    if p in pred_dist:
        pred_dist[p] += 1
    else:
        pred_dist["invalid"] += 1

summary = {
    "use_kg": USE_KG,
    "kg_file": KG_FILE if USE_KG else None,
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
print("Accuracy by kg_bucket:   ", summary["accuracy_by_kg_bucket"])

with open(RESULT_FILE, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

print("Saved jsonl to:", OUTPUT_FILE)
print("Saved full results to:", RESULT_FILE)

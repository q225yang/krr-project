import json
import re
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

QUESTION_FILE = "/mnt/disk11/user/xiaoyih1/xh/krr/test.json"
ANSWER_FILE = "/mnt/disk11/user/xiaoyih1/xh/krr/krr-project/test_answer.json"
IMAGE_DIR = "/mnt/disk11/user/xiaoyih1/xh/krr/image"

OUTPUT_FILE = "/mnt/disk11/user/xiaoyih1/xh/krr/krr-project/results/predictions_internvl_questions_only_50.jsonl"
RESULT_FILE = "/mnt/disk11/user/xiaoyih1/xh/krr/krr-project/results/results_internvl_questions_only_50.json"

MODEL_NAME = "OpenGVLab/InternVL2_5-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
    pixel_values = transform(img).unsqueeze(0)   # [1, 3, H, W]
    return pixel_values


def extract_choice(text: str):
    text = text.strip().upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text)
    return m.group(1) if m else None


# ===== load model =====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True,
).eval().to(DEVICE)


# ===== load dataset =====
with open(QUESTION_FILE, "r") as f:
    questions = json.load(f)

with open(ANSWER_FILE, "r") as f:
    answers = json.load(f)

answer_dict = {a["idx"]: a["answer"] for a in answers}


generation_config = dict(
    max_new_tokens=8,
    do_sample=False,
)


results = []
correct = 0
total = 0

# 每次运行前清空 jsonl，避免重复追加旧结果
open(OUTPUT_FILE, "w").close()

for sample in questions:
    if sample.get("mode") != "image-only":
        continue
    if total >= 50:   # 只测试前 50 个样本，避免一次跑太久
        break

    idx = sample["idx"]
    question = sample["question"]
    files = sample["file_name"]
    gt = answer_dict[idx]

    image_paths = [f"{IMAGE_DIR}/{x}" for x in files]
    num_images = len(image_paths)

    try:
        if num_images == 1:
            pixel_values = load_image_tensor(image_paths[0]).to(DTYPE).to(DEVICE)

            prompt = f"""<image>
{question}

Answer with only one letter: A, B, C, or D."""

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
                num_patches_list.append(pv.shape[0])  # 这里固定为 1

            pixel_values = torch.cat(pixel_values_list, dim=0).to(DTYPE).to(DEVICE)

            prompt = f"""Image-1: <image>
Image-2: <image>
Image-3: <image>
Image-4: <image>

These four images correspond to options A, B, C, and D respectively.

Question:
{question}

Answer with only one letter: A, B, C, or D."""

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

    result = {
        "idx": idx,
        "prediction": pred,
        "answer": gt,
        "correct": is_correct,
        "raw_output": output_text,
        "question": question,
        "files": files,
    }
    results.append(result)

    # 实时保存一条 jsonl
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"idx {idx} | pred {pred} | gt {gt} | correct {is_correct} | raw={repr(output_text)}")

# ===== 最终统计 =====
if total > 0:
    acc = correct / total
    print("\nAccuracy:", acc)
else:
    acc = 0
    print("\nNo valid samples were processed.")

# ===== 保存完整 results =====
with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved jsonl to:", OUTPUT_FILE)
print("Saved full results to:", RESULT_FILE)
import json
from pathlib import Path

visual_path = Path("/home/qyang129/krr-project/visual_entities_output/visual_entities.jsonl")
concepts_path = Path("/home/qyang129/krr-project/extracted_concepts.json")
output_path = Path("/home/qyang129/krr-project/combined_visual_and_concepts.json")

def dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# Load visual entities jsonl
visual_map = {}
with visual_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        image_id = int(row["image_id"])
        visual_map[image_id] = row

# Load extracted concepts json
with concepts_path.open("r", encoding="utf-8") as f:
    concepts_data = json.load(f)

combined = []

for item in concepts_data:
    idx = item["idx"]
    visual_item = visual_map.get(idx, {})

    visual_entities = visual_item.get("entities", [])
    word_groups = item.get("word_groups", [])

    # replace spaces with underscores in word_groups
    concept_groups = [wg.replace(" ", "_") for wg in word_groups]

    merged_terms = dedup_keep_order(visual_entities + concept_groups)

    combined.append({
        "idx": idx,
        "image_id": str(idx),
        "split": item.get("split"),
        "mode": item.get("mode"),
        "question_clean": item.get("question_clean"),
        "visual_entities": visual_entities,
        "concepts": concept_groups,
        "combined_entities_concepts": merged_terms,
        "caption": visual_item.get("caption", ""),
        "relations": visual_item.get("relations", [])
    })

with output_path.open("w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"Saved to: {output_path}")
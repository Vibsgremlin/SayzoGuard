import json
from pathlib import Path

BASE_DATASET = Path("sayzoguard/model_training/rules_train.jsonl")
FEEDBACK = Path("sayzoguard/model_training/feedback_dump.jsonl")
OUT = Path("sayzoguard/model_training/rules_train_merged.jsonl")

assert BASE_DATASET.exists(), "rules_train.jsonl not found"
assert FEEDBACK.exists(), "feedback_dump.jsonl not found"

base_samples = []
feedback_samples = []

# Load base training data
with open(BASE_DATASET, "r", encoding="utf-8") as f:
    for line in f:
        base_samples.append(json.loads(line))

# Load feedback (human-labeled)
with open(FEEDBACK, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        feedback_samples.append({
            "text": item["text"],
            "label": item["label"],
            "source": "human_review"
        })

# Write merged dataset
with open(OUT, "w", encoding="utf-8") as f:
    for row in base_samples:
        row["source"] = "synthetic"
        f.write(json.dumps(row) + "\n")

    for row in feedback_samples:
        f.write(json.dumps(row) + "\n")

print("Merge complete")
print(f"Base samples    : {len(base_samples)}")
print(f"Feedback samples: {len(feedback_samples)}")
print(f"Output file     : {OUT}")

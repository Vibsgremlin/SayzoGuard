import json
from pathlib import Path

FILE = Path("sayzoguard/model_training/feedback_dump.jsonl")

assert FILE.exists(), "feedback_dump.jsonl not found"

valid = 0
invalid = 0

with open(FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            data = json.loads(line)

            assert "text" in data
            assert "label" in data
            assert data["label"] in (0, 1)

            assert isinstance(data["text"], str)
            assert len(data["text"].strip()) > 0

            valid += 1

        except Exception as e:
            print(f"Invalid entry at line {i}: {e}")
            invalid += 1

print("Validation complete")
print(f"Valid samples  : {valid}")
print(f"Invalid samples: {invalid}")

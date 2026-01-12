import redis
import json
import time
from pathlib import Path

REDIS_KEY = "sayzoguard:feedback"
OUTPUT_FILE = Path("sayzoguard/model_training/feedback_dump.jsonl")

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

items = r.lrange(REDIS_KEY, 0, -1)

if not items:
    print("No feedback found.")
    exit(0)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    for raw in items:
        data = json.loads(raw)
        f.write(json.dumps(data) + "\n")

print(f"Exported {len(items)} feedback samples to {OUTPUT_FILE}")

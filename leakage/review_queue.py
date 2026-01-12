import redis
import json
import time

REDIS_HOST = "localhost"
REDIS_PORT = 6379

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

REVIEW_QUEUE = "sayzoguard:review_queue"
FEEDBACK_SET = "sayzoguard:feedback"


def enqueue_for_review(payload: dict):
    item = {
        "timestamp": time.time(),
        "data": payload
    }
    r.rpush(REVIEW_QUEUE, json.dumps(item))


def dequeue_for_review():
    item = r.lpop(REVIEW_QUEUE)
    if not item:
        return None
    return json.loads(item)


def mark_review(text: str, label: int):
    """
    label: 1 = leakage, 0 = safe
    """
    record = {
        "text": text,
        "label": int(label),
        "timestamp": time.time()
    }
    r.rpush(FEEDBACK_SET, json.dumps(record))
    return record

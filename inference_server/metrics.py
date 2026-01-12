from collections import Counter
from threading import Lock

_metrics = Counter()
_lock = Lock()

def incr(key: str):
    with _lock:
        _metrics[key] += 1

def snapshot():
    with _lock:
        return dict(_metrics)

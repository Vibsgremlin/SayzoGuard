from sayzoguard.leakage.review_queue import enqueue_for_review, dequeue_for_review

enqueue_for_review({
    "text": "budget is 5000",
    "ml_probability": 0.35,
    "llm_confidence": 0.4
})

item = dequeue_for_review()
print(item)

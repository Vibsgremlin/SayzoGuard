from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "finetuned"
)

_classifier = None  # singleton cache


def load_model():
    global _classifier

    if _classifier is not None:
        return _classifier

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32
    )

    model.eval()
    torch.set_grad_enabled(False)

    _classifier = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,          # smaller = faster + safer
        do_sample=False,
        return_full_text=False
    )

    return _classifier


# EXPORT FOR FASTAPI
classifier = load_model()

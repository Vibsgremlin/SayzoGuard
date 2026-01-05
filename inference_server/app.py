# Inference API (FastAPI)
# Includes escrow gating + regex + LLM + stitching

import sys
import os

# Ensure project root is in PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import FastAPI
from pydantic import BaseModel
import json, re

# Correct absolute imports using project name
from sayzoguard.leakage.normaliser import normalize
from sayzoguard.leakage.rules import basic_rule_score, contains_forbidden_link
from sayzoguard.leakage.stitcher import add_message, get_stitched
from sayzoguard.leakage.escrow import is_escrow_funded

from sayzoguard.inference_server.model_loader import classifier

app = FastAPI()

class Request(BaseModel):
    text: str
    task_id: str = None
    session_id: str = None

@app.post("/classify")
def classify(req: Request):

    text = req.text

    # ESCROW GATING
    if contains_forbidden_link(text):
        if not is_escrow_funded(req.task_id):
            return {
                "blocked": True,
                "reason": "escrow_required",
                "message": "Please add money to escrow before sharing restricted links."
                
            }

    # MULTI-TURN CONTEXT
    if req.session_id:
        add_message(req.session_id, text)
        stitched = get_stitched(req.session_id)
        text = stitched + "\n" + text

    # NORMALIZE + REGEX FIREWALL
    norm = normalize(text)
    rule_score = basic_rule_score(norm)

    # LLM CLASSIFICATION
    prompt = (
        f"Classify leakage:\nText:\n{norm}\n"
        f"Return JSON with fields leakage, confidence, reason.\n"
    )

    raw = classifier(prompt)[0]["generated_text"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"error": "model_parse_error"}

    data = json.loads(match.group(0))

    return {
        "blocked": data.get("leakage", False) or rule_score >= 3,
        "rule_score": rule_score,
        "llm_result": data,
    }

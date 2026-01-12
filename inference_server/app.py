# Inference API (FastAPI)
# Escrow gating + Regex firewall + Hard credential gate
# + ML rules + LLM + Rate limiting + Human review + Metrics

import sys
import os
import json
import re
from fastapi import FastAPI, Request as FastAPIRequest
from pydantic import BaseModel

# -----------------------------
# PATH SETUP
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# -----------------------------
# RATE LIMITING
# -----------------------------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)

# -----------------------------
# INTERNAL IMPORTS
# -----------------------------
from sayzoguard.leakage.normaliser import normalize
from sayzoguard.leakage.rules import basic_rule_score, contains_forbidden_link
from sayzoguard.leakage.rules_llm import ml_rule_score
from sayzoguard.leakage.stitcher import add_message, get_stitched
from sayzoguard.leakage.escrow import is_escrow_funded
from sayzoguard.leakage.review_queue import enqueue_for_review
from sayzoguard.inference_server.model_loader import classifier

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -----------------------------
# METRICS (IN-MEMORY)
# -----------------------------
METRICS = {
    "requests": 0,
    "llm_called": 0,
    "sent_for_review": 0,
    "blocked_credential": 0,
    "blocked_regex": 0,
}

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class Request(BaseModel):
    text: str
    task_id: str | None = None
    session_id: str | None = None

# -----------------------------
# SAFE LLM JSON PARSER
# -----------------------------
def safe_parse_llm(raw: str) -> dict:
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError
        return json.loads(match.group(0))
    except Exception:
        return {
            "leakage": False,
            "confidence": 0.5,
            "reason": "llm_non_json_fallback"
        }

# -----------------------------
# CLASSIFY ENDPOINT
# -----------------------------
@app.post("/classify")
@limiter.limit("30/minute")
def classify(request: FastAPIRequest, req: Request):

    METRICS["requests"] += 1

    text = req.text

    # 1️⃣ ESCROW GATING (HARD BLOCK)
    if contains_forbidden_link(text):
        if not is_escrow_funded(req.task_id):
            return {
                "blocked": True,
                "reason": "escrow_required"
            }

    # 2️⃣ MULTI-TURN STITCHING
    if req.session_id:
        add_message(req.session_id, text)
        text = get_stitched(req.session_id) + "\n" + text

    # 3️⃣ NORMALIZATION
    norm = normalize(text)

    # -----------------------------
    # 🔒 HARD CREDENTIAL GATE
    # -----------------------------
    HARD_CREDENTIAL_TERMS = [
        "otp", "password", "passcode", "pin",
        "verificationcode", "one-time",
        "aadhaar", "accountnumber", "bankaccount"
    ]

    if any(k in norm for k in HARD_CREDENTIAL_TERMS):
        METRICS["blocked_credential"] += 1
        return {
            "blocked": True,
            "reason": "credential_leakage_detected"
        }

    # -----------------------------
    # 4️⃣ REGEX FIREWALL
    # -----------------------------
    regex_score = basic_rule_score(norm)
    if regex_score >= 3:
        METRICS["blocked_regex"] += 1
        return {
            "blocked": True,
            "reason": "regex_rule_triggered",
            "regex_score": regex_score
        }

    # -----------------------------
    # 5️⃣ INTENT-AWARE ML RULES
    # -----------------------------
    ml_prob = ml_rule_score(norm)

    LEAKAGE_TERMS = [
        "phone", "email", "contact",
        "whatsapp", "telegram", "call", "dm"
    ]

    has_leak_terms = any(k in norm for k in LEAKAGE_TERMS)

    # ML cannot hard-block without intent
    if not has_leak_terms:
        ml_prob = min(ml_prob, 0.40)

    if ml_prob >= 0.98 and has_leak_terms:
        return {
            "blocked": True,
            "reason": "ml_rules_high_confidence",
            "ml_probability": round(ml_prob, 3)
        }

    if ml_prob <= 0.10:
        return {
            "blocked": False,
            "reason": "low_signal_safe",
            "ml_probability": round(ml_prob, 3)
        }

    # -----------------------------
    # 6️⃣ LLM (AMBIGUOUS ONLY)
    # -----------------------------
    METRICS["llm_called"] += 1

    prompt = (
        "You are a security classifier.\n"
        "Decide whether the following text leaks contact details, credentials, "
        "or off-platform communication.\n\n"
        f"Text:\n{norm}\n\n"
        "Return STRICT JSON:\n"
        "{ \"leakage\": true|false, \"confidence\": 0-1, \"reason\": \"...\" }"
    )

    raw = classifier(prompt)[0]["generated_text"]
    data = safe_parse_llm(raw)
    llm_conf = float(data.get("confidence", 0.5))

    # -----------------------------
    # NUMERIC NEGOTIATION DAMPENING
    # -----------------------------
    NEGOTIATION_TERMS = [
        "price", "cost", "budget", "rate",
        "deal", "amount", "payment", "invoice", "quote"
    ]

    has_digits = any(c.isdigit() for c in norm)
    has_negotiation_context = any(k in norm for k in NEGOTIATION_TERMS)

    if has_digits and has_negotiation_context and not has_leak_terms:
        ml_prob *= 0.3
        llm_conf *= 0.5

    # -----------------------------
    # FINAL CONFIDENCE
    # -----------------------------
    final_confidence = round((0.6 * ml_prob) + (0.4 * llm_conf), 3)

    # -----------------------------
    # FINAL DECISION
    # -----------------------------
    if final_confidence >= 0.85:
        decision = True
        reason = "high_confidence_leakage"

    elif final_confidence <= 0.20:
        decision = False
        reason = "low_signal_safe"

    else:
        if has_negotiation_context and not has_leak_terms:
            decision = False
            reason = "negotiation_allowed"
        else:
            METRICS["sent_for_review"] += 1
            enqueue_for_review({
                "text": text,
                "normalized": norm,
                "ml_probability": round(ml_prob, 3),
                "llm_confidence": round(llm_conf, 3)
            })
            decision = True
            reason = "sent_for_human_review"

    return {
        "blocked": decision,
        "reason": reason,
        "final_confidence": final_confidence,
        "ml_probability": round(ml_prob, 3),
        "llm_confidence": round(llm_conf, 3),
        "llm_result": data
    }

# -----------------------------
# METRICS ENDPOINT
# -----------------------------
@app.get("/metrics")
def metrics():
    return METRICS

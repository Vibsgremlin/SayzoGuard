import joblib

MODEL_PATH = "sayzoguard/models/rules_ml.joblib"

# Load once at startup
_clf = None
_vectorizer = None

def _load():
    global _clf, _vectorizer
    if _clf is None:
        _clf, _vectorizer = joblib.load(MODEL_PATH)

def ml_rule_score(text: str) -> float:
    """
    Returns probability of contact leakage (0.0 – 1.0)
    """
    _load()
    X = _vectorizer.transform([text])
    prob = _clf.predict_proba(X)[0][1]
    return float(prob)

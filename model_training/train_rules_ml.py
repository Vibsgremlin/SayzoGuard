import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DATASET = Path(__file__).parent.parent / "dataset" / "leakage_train.jsonl"
OUTPUT = Path(__file__).parent.parent / "models" / "rules_ml.joblib"

texts, labels = [], []

with open(DATASET, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(row["label"])

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    min_df=2
)

X = vectorizer.fit_transform(texts)

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X, labels)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump((clf, vectorizer), OUTPUT)

print(f"Saved ML rules model to {OUTPUT}")

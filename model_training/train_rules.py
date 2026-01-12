import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts, labels = [], []

with open("sayzoguard/dataset/leakage_train.jsonl", "r", encoding="utf-8") as f:
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

joblib.dump((clf, vectorizer), "sayzoguard/models/rules_ml.joblib")
print("rules ML model saved")

"""
Trains the spam/promotion classifier used by the detector.

Pipeline: word (1-2 gram) + character (3-5 gram) TF-IDF -> Logistic Regression.
Saves model/spam_model.joblib and prints a held-out evaluation report.

    python ml/build_dataset.py   # (re)generate data/comments.csv
    python ml/train.py
"""
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "comments.csv"
MODEL = ROOT / "model" / "spam_model.joblib"


def load():
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])


def main():
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    pipe = build_pipeline().fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    print(classification_report(y_te, pred, target_names=["genuine", "spam"]))
    metrics = {"accuracy": round(accuracy_score(y_te, pred), 4), "f1_spam": round(f1_score(y_te, pred), 4),
               "train_size": len(X_tr), "test_size": len(X_te)}

    final = build_pipeline().fit(X, y)  # refit on everything for deployment
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL)
    (MODEL.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL}\n{metrics}")


if __name__ == "__main__":
    main()

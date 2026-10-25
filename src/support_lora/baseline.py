"""Train-only TF-IDF baseline, with abstention threshold selected on validation data."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import RESPONSES, dataset_hashes, read_rows, validate_splits


def fit(directory, output):
    validate_splits(directory)
    train = read_rows(Path(directory) / "train.jsonl")
    validation = read_rows(Path(directory) / "validation.jsonl")
    models = {}
    for label in ("category", "urgency"):
        pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        )
        pipeline.fit([r["text"] for r in train], [r[label] for r in train])
        models[label] = pipeline
    probabilities = models["category"].predict_proba([r["text"] for r in validation])
    confidence = probabilities.max(axis=1)
    correct = models["category"].classes_[probabilities.argmax(axis=1)] == [
        r["category"] for r in validation
    ]
    candidates = []
    for threshold in np.arange(0.25, 1.01, 0.05):
        retained = confidence >= threshold
        if retained.any() and correct[retained].mean() >= 0.9:
            candidates.append((float(retained.mean()), float(threshold)))
    threshold = max(candidates, default=(0.0, 1.0))[1]
    models["threshold"] = threshold
    models["dataset_hashes"] = dataset_hashes(directory)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output)
    print(json.dumps({"baseline_saved": str(output), "validation_selected_threshold": threshold}))


def predict(models, text):
    probabilities = models["category"].predict_proba([text])[0]
    category = models["category"].classes_[probabilities.argmax()]
    urgency = models["urgency"].predict([text])[0]
    confidence = float(probabilities.max())
    language = "ar" if any("\u0600" <= c <= "\u06ff" for c in text) else "en"
    escalated = confidence < models["threshold"]
    response = RESPONSES[language][category]
    if escalated:
        response = (
            "سيتم تحويل طلبك للمراجعة البشرية."
            if language == "ar"
            else "Your request needs human review."
        )
    return {
        "category": str(category),
        "urgency": str(urgency),
        "response": response,
        "confidence": confidence,
        "escalated": escalated,
        "backend": "tfidf-baseline",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="artifacts/baseline.joblib")
    args = parser.parse_args()
    fit(args.data, args.output)


if __name__ == "__main__":
    main()

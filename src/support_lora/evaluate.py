"""Evaluate only the held-out test split; never use test results to select a threshold."""

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score

from .baseline import predict
from .data import CATEGORIES, URGENCIES, dataset_hashes, read_rows, validate_splits


def metrics(rows, predictions):
    result = {
        "samples": len(rows),
        "valid_output_rate": sum(p.get("valid_json", True) for p in predictions) / len(rows),
    }
    for label, labels in (("category", CATEGORIES), ("urgency", URGENCIES)):
        truth = [r[label] for r in rows]
        predicted = [p.get(label, "invalid") for p in predictions]
        result[label + "_macro_f1"] = f1_score(
            truth, predicted, labels=labels, average="macro", zero_division=0
        )
        result[label + "_accuracy"] = accuracy_score(truth, predicted)
    accepted = [i for i, p in enumerate(predictions) if not p.get("escalated", False)]
    result["coverage"] = len(accepted) / len(rows)
    result["accepted_category_accuracy"] = (
        sum(rows[i]["category"] == predictions[i].get("category") for i in accepted) / len(accepted)
        if accepted
        else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--kind", choices=["baseline", "causal"], default="baseline")
    parser.add_argument("--baseline", default="artifacts/baseline.joblib")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="artifacts/lora")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args()
    validate_splits(args.data)
    rows = read_rows(Path(args.data) / "test.jsonl")
    if args.limit:
        # Round-robin categories so smoke subsets are not a single-class benchmark.
        groups = [[r for r in rows if r["category"] == label] for label in CATEGORIES]
        rows = [r for group in zip(*groups, strict=True) for r in group][: args.limit]
    if args.kind == "baseline":
        model = joblib.load(args.baseline)
        variants = {"tfidf": lambda text: predict(model, text)}
    else:
        from .inference import CausalPredictor

        predictor = CausalPredictor(args.model, args.adapter)
        variants = {
            v: (lambda text, variant=v: predictor.predict(text, variant)) for v in ("base", "lora")
        }
    report = {
        "dataset": "original synthetic fixture; not a real-world benchmark",
        "dataset_hashes": dataset_hashes(args.data),
        "response_quality": "Not automatically scored. Use docs/RESPONSE_REVIEW.md for human review.",
        "variants": {},
    }
    for name, function in variants.items():
        started = time.perf_counter()
        predictions = [function(row["text"]) for row in rows]
        report["variants"][name] = {
            **metrics(rows, predictions),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "predictions": [{"id": r["id"], **p} for r, p in zip(rows, predictions, strict=True)],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "output": str(output),
                "metrics": {
                    k: {key: value for key, value in v.items() if key != "predictions"}
                    for k, v in report["variants"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

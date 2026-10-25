import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from support_lora.api import create_app
from support_lora.baseline import fit
from support_lora.data import encode_example, parse_prediction, read_rows, validate_splits
from support_lora.evaluate import metrics

DATA = Path(__file__).resolve().parents[1] / "data"
TOKENS = {"test-api-token-00000000000000": {"tenant": "demo", "user": "alice"}}
AUTH = {"Authorization": "Bearer " + next(iter(TOKENS))}


def test_dataset_is_disjoint_and_balanced():
    assert validate_splits(DATA) == {"train": 32, "validation": 16, "test": 24}


def test_duplicate_split_rejected(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_bytes((DATA / "train.jsonl").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        validate_splits(tmp_path)


def test_response_only_mask_and_no_truncation():
    class Tokenizer:
        eos_token = "!"

        def apply_chat_template(self, *args, **kwargs):
            return [1, 2, 3]

        def __call__(self, text, **kwargs):
            return {"input_ids": [4, 5, 6, 7]}

    row = read_rows(DATA / "train.jsonl")[0]
    encoded = encode_example(Tokenizer(), row, 8)
    assert encoded["labels"] == [-100, -100, -100, 4, 5, 6, 7]
    with pytest.raises(ValueError, match="exceeds"):
        encode_example(Tokenizer(), row, 6)


def test_schema_and_invalid_output_penalty():
    with pytest.raises(ValueError):
        parse_prediction('{"category":"admin","urgency":"high","response":"x"}')
    rows = [{"category": "refund", "urgency": "normal"}]
    result = metrics(rows, [{"valid_json": False, "escalated": True}])
    assert result["category_accuracy"] == 0
    assert result["valid_output_rate"] == 0
    assert result["coverage"] == 0


def test_train_save_reload_and_api(tmp_path):
    output = tmp_path / "baseline.joblib"
    fit(DATA, output)
    client = TestClient(create_app(TOKENS, output))
    assert client.post("/predict", json={"text": "refund please"}).status_code == 401
    result = client.post("/predict", headers=AUTH, json={"text": "أريد استرداد المبلغ"})
    assert result.status_code == 200
    assert result.json()["backend"] == "tfidf-baseline"
    assert 0 <= result.json()["confidence"] <= 1
    assert (
        client.post("/predict", headers=AUTH, json={"text": "test", "variant": "lora"}).status_code
        == 503
    )


def test_missing_artifact_is_not_a_fake_model(tmp_path):
    client = TestClient(create_app(TOKENS, tmp_path / "missing.joblib"))
    assert not client.get("/health").json()["baseline_ready"]
    assert client.post("/predict", headers=AUTH, json={"text": "test"}).status_code == 503


def test_causal_comparison_routing(tmp_path):
    class FakePredictor:
        has_adapter = True

        def predict(self, text, variant):
            return {"backend": variant, "valid_json": True}

    client = TestClient(create_app(TOKENS, tmp_path / "missing", FakePredictor()))
    assert (
        client.post("/predict", headers=AUTH, json={"text": "test", "variant": "base"}).json()[
            "backend"
        ]
        == "base"
    )


def test_json_roundtrip():
    row = read_rows(DATA / "test.jsonl")[0]
    output = {k: row[k] for k in ("category", "urgency", "response")}
    assert parse_prediction(json.dumps(output)) == output

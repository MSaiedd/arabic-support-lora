# Validation status

Validated locally on 2026-10-25 with Python 3.12.3. Exact commands and results are recorded
below. Portfolio fixtures are synthetic; passing checks do not establish production readiness.

## Automated checks

- `ruff check .` — passed.
- `ruff format --check .` — passed.
- `pytest --cov=src --cov-report=term-missing` — see the recorded test log in `reports/tests.txt`.
- `python scripts/browser_smoke.py` — passed: real Chromium desktop flow, 390 px mobile layout,
  no uncaught browser errors. Screenshots are in `docs/screenshots/`.

GitHub Actions repeats lint, tests on Python 3.11/3.12, and browser validation after publishing.
Local evidence is separate from hosted CI status.

## Boundaries

Static bearer tokens are development identities. No public hosting or production rollout was performed.
Optional external providers and CUDA-dependent paths require their own runtime validation.

## Packaging

`python -m build --wheel --outdir dist` — passed. The wheel includes HTML/CSS/JavaScript assets.

## Baseline

`python -m support_lora.baseline` and `python -m support_lora.evaluate --output reports/baseline.json` — passed.
On 24 original synthetic test examples: category macro-F1 0.7542, urgency macro-F1 0.7483,
category accuracy 0.75, coverage 0.625, accepted category accuracy 1.0. The abstention threshold
was selected only on validation examples. See the report for individual predictions.
These results are not real-world accuracy estimates or fine-tuned language model scores.

The CPU integration test performs an actual LoRA parameter update and adapter save/reload.
The GPU QLoRA path was not run because the NVIDIA driver is unavailable.

## Full trainer CLI smoke

`python -m support_lora.train --model artifacts/tiny-base --output artifacts/tiny-lora --max-steps 8 --grad-accum 1 --max-length 512` — passed.
A tiny Qwen architecture with random initial weights and the cached Qwen tokenizer was used
to verify the entire training path without downloading pretrained weights. The run performed
eight real optimizer steps and saved an adapter and manifest. See `reports/tiny-training-smoke.json`.

`python -m support_lora.evaluate --kind causal --model artifacts/tiny-base --adapter artifacts/tiny-lora --limit 4 --output reports/tiny-causal-smoke.json` — passed as an engineering check.
Both random-model variants failed JSON generation on all four examples; this expected failure
is recorded rather than presented as useful model quality. Full pretrained Qwen and CUDA QLoRA
quality are not established by this smoke test.

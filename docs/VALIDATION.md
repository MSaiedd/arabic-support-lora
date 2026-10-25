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

## Hosted CI

[Initial GitHub Actions run](https://github.com/MSaiedd/arabic-support-lora/actions/runs/40206904141) — passed:
Python 3.11 tests/lint, Python 3.12 tests/lint, and Chromium browser validation.
The default CI skips the optional training integration test when torch/PEFT are absent;
local validation included these dependencies and passed the actual adapter test.

[Optional hosted CPU adapter smoke](https://github.com/MSaiedd/arabic-support-lora/actions/runs/40207014548) — passed.
This installs torch/PEFT and verifies real adapter updates and save/reload parity without model downloads.

## Pretrained CPU LoRA run

`HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 python -m support_lora.train --max-steps 8 --grad-accum 1 --max-length 512` — passed.
This run adapted pretrained `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775` with 540,672 trainable parameters.
It used eight optimizer steps / eight training examples, not full convergence.
Validation target-token NLL: original 1.51249, adapter 1.25266.
This validation loss change is not evidence of held-out response correctness.
The downloaded base weights' SHA-256 matched the Hugging Face content hash.
See `reports/pretrained-training.json` for seed, dataset hashes, losses, and versions.

## Container

`docker build -t arabic-support-lora:portfolio .` — passed after retrying a package download timeout.
`python scripts/container_smoke.py` — passed: non-root image, original data included,
baseline training in a disposable named volume, and authenticated inference in a separate
container using the saved artifact. See `reports/container.json`.

## Held-out pretrained model comparison

`python -m support_lora.evaluate --kind causal --output reports/causal.json` — passed.
Both variants were evaluated on the same 24 held-out synthetic tickets, with malformed JSON
counting as a failed output. Original model generation took 248.09 seconds; adapter generation
168.93 seconds on this CPU. These single-run timings are not a latency benchmark.

| Variant | Category macro-F1 | Category accuracy | Urgency macro-F1 | Valid JSON |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF baseline | 0.7542 | 0.7500 | 0.7483 | 1.0000 (structured code output) |
| Pretrained Qwen | 0.3889 | 0.3333 | 0.3000 | 0.7500 |
| Eight-step LoRA | 0.4554 | 0.5000 | 0.3143 | 0.9583 |

The adapter improved output formatting and classification relative to its base model on
this fixture, while the TF-IDF baseline remained stronger at classification. The language
models generate responses; baseline responses use templates. Response correctness and policy
compliance have not been human-reviewed. No real-world performance claim follows from this
small synthetic comparison. The short adapter is available as a reproducible smoke artifact.

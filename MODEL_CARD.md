# Model card

## Identity

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Method: rank-8 LoRA on query/value projections,
or optional NF4 QLoRA. The base model retains its upstream license.
The repository contains training code and original synthetic data; generated adapter files
are excluded from Git. Run reports identify the exact upstream revision used.

## Intended use

Engineering education and portfolio demonstrations of Arabic/English ticket classification
and policy-aware response drafting. Categories: account, billing, refund, shipping.
Urgency: normal/high, using this fixture's annotation convention.

## Data provenance

All 72 examples were authored for this project. No real customers, proprietary datasets,
or personal information are included. Splits have independent phrasing and no exact normalized
duplicates. This does not establish the absence of semantic overlap. Responses are deliberately
short policy templates. The mixed-language test examples represent a distribution shift.

## Training and evaluation

Only response tokens receive supervision. `training_report.json` records data hashes, model
revision, seed, optimizer steps, losses, validation token NLL, and dependency versions.
Held-out evaluation measures category/urgency macro-F1, JSON validity, coverage, and latency.
Invalid generated JSON counts as a failed classification. Response quality needs human review.
See `docs/VALIDATION.md` and `reports/` for actual runs and their limits.

## Limitations

The fixture is far too small for production use. Model output can be incorrect, biased,
unsafe, malformed, or influenced by prompt injection. Arabic dialects, code switching, and
unseen intents are underrepresented. A low token loss does not demonstrate correct routing
or safe answers. Synthetic test metrics do not estimate real customer performance.
Short smoke training does not establish useful fine-tuning quality.

Do not use for live financial decisions, autonomous account changes, or unsupervised customer
communication. No action execution is connected. Always retain human review for sensitive issues.

## Reproduction

Follow the README commands. Pin the base model commit with `--revision` and retain the
training report. The optional GPU QLoRA path requires a CUDA setup and separate validation.

## Recorded smoke adapter

The [CPU smoke release](https://github.com/MSaiedd/arabic-support-lora/releases/tag/v0.1.0)
contains an actual adapter trained for eight CPU optimizer steps with 540,672 trainable parameters.
On the 24 synthetic test tickets, category macro-F1 changed from 0.3889 (base) to 0.4554
(adapter), and valid JSON changed from 0.75 to 0.9583. The TF-IDF baseline scored 0.7542
category macro-F1. These are fixture results, not production quality estimates.
The trained adapter's weights are distributed under Apache-2.0 alongside the base model license.
Code and original fixture data remain MIT licensed.

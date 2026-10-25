# Architecture decisions

- TF-IDF and logistic regression provide a fast, measured control; baseline responses use templates.
- LoRA trains only rank-8 adapters on query/value projections of a causal instruction model.
- QLoRA optionally quantizes base weights with NF4. CPU LoRA does not require bitsandbytes.
- Prompt tokens are masked with -100. Complete JSON response tokens, including EOS, receive supervision.
- Training shuffles examples with a fixed seed. Reports record model revision, dataset hashes, and versions.
- A validation-selected confidence threshold governs the baseline abstention policy. Test data does not select it.
- Base/adapter generation shares underlying weights; adapter disabling provides a direct comparison.
- Invalid generated JSON counts as a failed output and incorrect classification, rather than being dropped.
- Responses require separate human review: classification F1 and token NLL do not establish response quality.
- Fine-tuned artifacts are not loaded from arbitrary uploads. joblib deserialization is trusted-local only.

For real deployment, add larger consented datasets, semantic duplicate checks, per-language metrics,
human safety review, workload testing, model/artifact registries, monitoring, and an identity gateway.

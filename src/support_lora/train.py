"""Response-only causal SFT using LoRA, or NF4 QLoRA on a CUDA GPU."""

import argparse
import importlib.metadata
import json
import random
import time
from pathlib import Path

from .data import dataset_hashes, encode_example, read_rows, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main", help="Use a commit hash for repeatable runs")
    parser.add_argument("--output", default="artifacts/lora")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    if (
        min(args.max_steps, args.max_length, args.grad_accum, args.threads) < 1
        or args.learning_rate <= 0
    ):
        parser.error("Training limits and learning rate must be positive")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    counts = validate_splits(args.data)
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.qlora and not torch.cuda.is_available():
        parser.error("QLoRA requires a working CUDA GPU; omit --qlora for CPU LoRA")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"revision": args.revision, "torch_dtype": torch.float32}
    if args.qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            device_map={"": 0},
            torch_dtype=compute_dtype,
        )
    base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    revision = getattr(base.config, "_commit_hash", None)
    if args.qlora:
        base = prepare_model_for_kbit_training(base)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    if not args.qlora:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False
    train = [
        encode_example(tokenizer, r, args.max_length)
        for r in read_rows(Path(args.data) / "train.jsonl")
    ]
    validation = [
        encode_example(tokenizer, r, args.max_length)
        for r in read_rows(Path(args.data) / "validation.jsonl")
    ]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    rng = random.Random(args.seed)
    order = list(range(len(train)))
    rng.shuffle(order)
    position = 0
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.grad_accum):
            if position == len(order):
                rng.shuffle(order)
                position = 0
            batch = {
                k: torch.tensor([v], device=model.device) for k, v in train[order[position]].items()
            }
            position += 1
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(total_loss)
        print(json.dumps({"step": step + 1, "loss": total_loss}), flush=True)

    def validation_nll():
        loss_sum, target_tokens = 0.0, 0
        for row in validation:
            batch = {k: torch.tensor([v], device=model.device) for k, v in row.items()}
            count = sum(token != -100 for token in row["labels"][1:])
            loss_sum += model(**batch).loss.item() * count
            target_tokens += count
        return loss_sum / target_tokens

    model.eval()
    with torch.no_grad():
        adapter_nll = validation_nll()
        with model.disable_adapter():
            base_nll = validation_nll()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    report = {
        "model": args.model,
        "resolved_revision": revision,
        "seed": args.seed,
        "method": "QLoRA-NF4" if args.qlora else "LoRA",
        "device": str(model.device),
        "optimizer_steps": args.max_steps,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset_counts": counts,
        "dataset_hashes": dataset_hashes(args.data),
        "training_losses": losses,
        "validation_base_token_nll": base_nll,
        "validation_adapter_token_nll": adapter_nll,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "accelerate")
        },
        "limitation": "Small synthetic fixture. Validation NLL does not establish held-out task quality or production suitability.",
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

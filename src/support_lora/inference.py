"""Load a local adapter once; compare it with its exact underlying base model."""

import threading

from .data import messages, parse_prediction


class CausalPredictor:
    def __init__(self, model_id, adapter=None, revision="main"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        self.torch = torch
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter or model_id, revision=None if adapter else revision
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if adapter:
            from peft import PeftConfig, PeftModel

            config = PeftConfig.from_pretrained(adapter)
            if config.base_model_name_or_path != model_id:
                raise ValueError("Adapter base model does not match MODEL_ID")
            # Train reports pin the exact original weights, rather than following a moving main.
            import json
            from pathlib import Path

            report_path = Path(adapter) / "training_report.json"
            if report_path.exists():
                revision = json.loads(report_path.read_text()).get("resolved_revision") or revision
            base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
            self.model = PeftModel.from_pretrained(base, adapter)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.has_adapter = bool(adapter)

    def predict(self, text, variant="lora", max_new_tokens=192):
        from contextlib import nullcontext

        if variant == "lora" and not self.has_adapter:
            raise ValueError("No adapter configured")
        with self.lock, self.torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                messages(text),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
            context = (
                self.model.disable_adapter()
                if variant == "base" and self.has_adapter
                else nullcontext()
            )
            with context:
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            raw = self.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        try:
            return {**parse_prediction(raw), "backend": variant, "valid_json": True, "raw": raw}
        except (ValueError, TypeError):
            return {
                "backend": variant,
                "valid_json": False,
                "raw": raw,
                "escalated": True,
                "response": "Model output was invalid; human review required.",
            }

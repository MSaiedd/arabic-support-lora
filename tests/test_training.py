"""Optional CPU integration: actual adapter update and save/reload, without downloads."""

import pytest


def test_real_lora_update_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    base = Qwen2ForCausalLM(config)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
        ),
    )
    before = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    labels = ids.clone()
    labels[:, :2] = -100
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(before[k], p) for k, p in model.named_parameters() if k in before)
    output = tmp_path / "adapter"
    model.save_pretrained(output)
    model.eval()
    reloaded = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(base_dir), output).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, reloaded(input_ids=ids).logits)

"""Evaluate the Param2 LoRA adapter with the native Transformers model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers.utils import import_utils

if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel


def default_rope(config, device=None):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    base = getattr(config, "rope_theta", 10000.0)
    positions = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (positions / dim)), 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", default_rope)
_expanded_tied = PreTrainedModel.get_expanded_tied_weights_keys


def compat_tied_weights(self, all_submodels=False):
    if isinstance(getattr(self, "_tied_weights_keys", None), list):
        self._tied_weights_keys = None
    return _expanded_tied(self, all_submodels=all_submodels)


PreTrainedModel.get_expanded_tied_weights_keys = compat_tied_weights

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": 0},
    )
    if hasattr(model, "model") and hasattr(model.model, "word_embeddings") and hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.word_embeddings.weight
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    tools = [{
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                           "required": ["a", "b"], "additionalProperties": False},
        },
    }]
    cases = [
        ("direct", [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with only the word READY."}], None),
        ("native_tool", [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": "Use the add tool to compute 17 + 25."}], tools),
    ]
    records = []
    for case_id, messages, case_tools in cases:
        prompt = tokenizer.apply_chat_template(messages, tools=case_tools, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        started = time.monotonic()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)
        generated = output[0, inputs["input_ids"].shape[-1]:]
        records.append({"case_id": case_id, "messages": messages, "tools": case_tools,
                        "rendered_prompt": prompt, "raw_text": tokenizer.decode(generated, skip_special_tokens=False),
                        "prompt_tokens": int(inputs["input_ids"].shape[-1]), "completion_tokens": int(generated.shape[-1]),
                        "latency_seconds": time.monotonic() - started})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n")
    print(json.dumps({"output": str(output), "cases": len(records)}), flush=True)


if __name__ == "__main__":
    main()

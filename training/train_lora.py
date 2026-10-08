"""Conservative assistant-only LoRA SFT for the Param2 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Each DDP rank imports the custom Param2 modules independently. A private
# cache avoids concurrent dynamic-module writes corrupting one rank's import.
_rank = os.environ.get("LOCAL_RANK", "0")
os.environ.setdefault("HF_HOME", f"/workspace/cache/hf_home_{_rank}")
os.environ.setdefault("HF_DATASETS_CACHE", f"/workspace/cache/hf_datasets_{_rank}")
os.environ.setdefault("HF_MODULES_CACHE", f"/workspace/cache/hf_modules_{_rank}")
from datasets import load_dataset
from transformers.utils import import_utils
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

def _default_rope(config, device=None):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
    base = getattr(config, "rope_theta", 10000.0)
    positions = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (positions / dim)), 1.0

ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope)
_expanded_tied = PreTrainedModel.get_expanded_tied_weights_keys
def _compat_tied_weights(self, all_submodels=False):
    if isinstance(getattr(self, "_tied_weights_keys", None), list):
        self._tied_weights_keys = None
    return _expanded_tied(self, all_submodels=all_submodels)
PreTrainedModel.get_expanded_tied_weights_keys = _compat_tied_weights

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def target_message(row: dict[str, Any]) -> dict[str, Any]:
    target = dict(row["target"])
    if target.get("content"):
        return {"role": "assistant", "content": target["content"]}
    payload: dict[str, Any] = {"role": "assistant"}
    metadata = target.get("metadata") or {}
    tool_call = target.get("tool_call") or metadata.get("tool_call") or target.get("tool_use") or metadata.get("tool_use")
    if tool_call is not None:
        # Param2's native template expects a compact {name, arguments} object
        # inside <tool_call>. The transcript export stores extra provenance
        # fields and uses native_name, which must not be trained verbatim.
        name = tool_call.get("name") or tool_call.get("native_name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call is missing a callable name")
        arguments = tool_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        payload["content"] = json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    else:
        payload["content"] = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return payload


def format_row(row: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    context = list(row["context"])
    target = target_message(row)
    prompt_text = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
    target_text = tokenizer.apply_chat_template([target], tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    room = max_length - len(target_ids)
    if room <= 0:
        target_ids = target_ids[:max_length]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-room:]
    input_ids = prompt_ids + target_ids
    return {"input_ids": input_ids, "labels": [-100] * len(prompt_ids) + target_ids, "attention_mask": [1] * len(input_ids)}


@dataclass
class Collator:
    pad_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(x["input_ids"]) for x in features)
        ids, labels, masks = [], [], []
        for x in features:
            pad = width - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * pad)
            labels.append(x["labels"] + [-100] * pad)
            masks.append(x["attention_mask"] + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(masks)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw = load_dataset("json", data_files=args.data, split="train")
    raw = raw.filter(lambda x: x.get("loss_mask", False) and x.get("split") == "train")
    encoded = raw.map(lambda x: format_row(x, tokenizer, args.max_length), remove_columns=raw.column_names)
    if args.dry_run:
        sample = encoded[0]
        print(json.dumps({"rows": len(encoded), "tokens": len(sample["input_ids"]), "supervised_tokens": sum(v != -100 for v in sample["labels"])}, indent=2))
        return

    from peft import LoraConfig, get_peft_model

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Keep one full copy on host memory until Trainer performs its device move.
    model = model.to("cpu")
    if hasattr(model, "model") and hasattr(model.model, "word_embeddings") and hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.word_embeddings.weight
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["query_key_value", "dense"]))
    model.print_trainable_parameters()
    training = TrainingArguments(output_dir=args.output, per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accumulation, learning_rate=args.learning_rate, num_train_epochs=args.epochs, max_steps=args.max_steps, bf16=True, logging_steps=1, save_steps=50, save_total_limit=2, report_to="none", remove_unused_columns=False)
    trainer = Trainer(model=model, args=training, train_dataset=encoded, data_collator=Collator(tokenizer.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()

"""Merge a Param2 LoRA adapter into a fresh model directory."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},
    )
    if hasattr(model, "model") and hasattr(model.model, "word_embeddings") and hasattr(model, "lm_head"):
        model.lm_head.weight = model.model.word_embeddings.weight
    peft_model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    merged = peft_model.merge_and_unload()
    if hasattr(merged, "model") and hasattr(merged.model, "word_embeddings") and hasattr(merged, "lm_head"):
        # The custom Param2 class does not expose compatible tied-weight metadata.
        # Clone the tied values so safe serialization can write both tensors.
        merged.lm_head.weight = torch.nn.Parameter(merged.model.word_embeddings.weight.detach().clone())
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    metadata = {
        "base_model": args.model,
        "adapter": args.adapter,
        "output": str(output),
        "dtype": "bfloat16",
        "transformers": __import__("transformers").__version__,
        "peft": __import__("peft").__version__,
    }
    (output / "merge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

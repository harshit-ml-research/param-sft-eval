"""Check native Param loading, LoRA attachment, and one backward pass."""

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
def _compat_tied_weights(self, all_submodels=False):
    if isinstance(getattr(self, "_tied_weights_keys", None), list):
        self._tied_weights_keys = None
    return _expanded_tied(self, all_submodels=all_submodels)
PreTrainedModel.get_expanded_tied_weights_keys = _compat_tied_weights

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/model", trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager"
)
if hasattr(model, "model") and hasattr(model.model, "word_embeddings") and hasattr(model, "lm_head"):
    model.lm_head.weight = model.model.word_embeddings.weight
print("MODEL_LOADED", flush=True)
model.config.use_cache = False
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense"],
    ),
)
model.print_trainable_parameters()
print("LORA_ATTACHED", flush=True)
model.to("cuda:0")
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Call the available tool to add 17 and 25."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
            },
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)
tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0][-128:]
inputs = tokens.unsqueeze(0).to("cuda:0")
labels = inputs.clone()
labels[:, :-16] = -100
model.train()
loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
print("FORWARD_LOSS", float(loss.detach().cpu()), flush=True)
loss.backward()
print("BACKWARD_OK", flush=True)

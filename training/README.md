# Param2 tool calling post training

This package trains a parameter efficient adapter for the internal Param2 17B SFT checkpoint.

The example below uses the earlier action export at `artifacts/sft/runs/derived_tool_sft_full_final/sft_actions.jsonl`. That file is not the next training release. The next corpus is the undecided tool-calling set from four or five teachers.
Each row becomes one causal language modeling example. The context is prompt only and the target assistant action is the only text receiving loss.

The first path uses Hugging Face Transformers plus PEFT LoRA. It preserves the checkpoint's native tokenizer and chat template and does not assume that the model is compatible with TRL's conversational dataset helpers.

Example:

```sh
python training/train_lora.py \
  --model /model \
  --data /workspace/artifacts/sft/runs/derived_tool_sft_full_final/sft_actions.jsonl \
  --output /workspace/runs/param2-tool-sft-lora-v1 \
  --max-length 4096 \
  --max-steps 100
```

The script supports a `--dry-run` mode that loads the tokenizer, formats one example, and reports token counts without loading model weights.

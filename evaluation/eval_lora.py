"""Run fixed native-template smoke prompts with a vLLM LoRA adapter."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    add_tool = [{
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }]
    cases = [
        ("direct", [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with only the word READY."}], None),
        ("native_tool", [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": "Use the add tool to compute 17 + 25."}], add_tool),
    ]
    prompts = [tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                              add_generation_prompt=True)
               for _, messages, tools in cases]
    llm = LLM(model=args.model, trust_remote_code=True, dtype="bfloat16",
              max_model_len=32768, gpu_memory_utilization=0.85,
              enforce_eager=True, enable_lora=True)
    lora = LoRARequest("param2-tool-sft-full-ddp-v1", 1, args.adapter)
    results = []
    for (case_id, messages, tools), prompt in zip(cases, prompts):
        started = time.monotonic()
        result = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=2048),
                              lora_request=lora)[0]
        candidate = result.outputs[0]
        results.append({
            "case_id": case_id,
            "messages": messages,
            "tools": tools,
            "rendered_prompt": prompt,
            "raw_text": candidate.text,
            "token_ids": list(candidate.token_ids),
            "prompt_tokens": len(result.prompt_token_ids),
            "finish_reason": candidate.finish_reason,
            "stop_reason": candidate.stop_reason,
            "latency_seconds": time.monotonic() - started,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n")
    print(json.dumps({"output": str(output), "cases": len(results)}), flush=True)


if __name__ == "__main__":
    main()

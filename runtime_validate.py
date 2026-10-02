"""Validate the checkpoint with its native template and preserve raw outputs."""

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import transformers
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    model = Path('/model')
    output = Path('/workspace/results') / os.environ['RUN_ID']
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'checkpoint': os.environ['CHECKPOINT_SOURCE'],
        'image': os.environ['IMAGE_ID'],
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'vllm': vllm.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'worker_node': os.environ.get('WORKER_NODE'),
        'physical_gpu': os.environ.get('PHYSICAL_GPU'),
        'files': {name: hashlib.sha256((model / name).read_bytes()).hexdigest()
                  for name in ['config.json', 'tokenizer_config.json', 'chat_template.jinja',
                               'model.safetensors.index.json']},
        'settings': {'temperature': 0, 'max_tokens': 2048, 'max_model_len': 32768,
                     'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.85},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    tools = [{'type': 'function', 'function': {
        'name': 'add', 'description': 'Add two integers.',
        'parameters': {'type': 'object', 'properties': {
            'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
            'required': ['a', 'b'], 'additionalProperties': False}}}]
    cases = [
        ('direct', 'Reply with only the word READY.', None),
        ('native_tool', 'Use the add tool to compute 17 + 25.', tools),
    ]
    llm = LLM(model=str(model), trust_remote_code=True, dtype='bfloat16',
              tensor_parallel_size=1, max_model_len=32768,
              gpu_memory_utilization=0.85, enforce_eager=True)
    for case_id, prompt, case_tools in cases:
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}]
        rendered = tokenizer.apply_chat_template(messages, tools=case_tools,
                                                 tokenize=False, add_generation_prompt=True)
        start = time.monotonic()
        result = llm.generate([rendered], SamplingParams(temperature=0, max_tokens=2048))[0]
        candidate = result.outputs[0]
        record = {'case_id': case_id, 'messages': messages, 'tools': case_tools,
                  'rendered_prompt': rendered, 'raw_text': candidate.text,
                  'token_ids': list(candidate.token_ids),
                  'prompt_tokens': len(result.prompt_token_ids),
                  'finish_reason': candidate.finish_reason,
                  'stop_reason': candidate.stop_reason,
                  'latency_seconds': time.monotonic() - start}
        with (output / 'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    print('VALIDATION_FINISHED', flush=True)
    if os.environ.get('RUN_ITBENCH') == '1':
        from eval_smoke.itbench_lite import run_scenarios
        from eval_smoke.models.backend import Generation
        from eval_smoke.models.param2 import parse_output

        class Backend:
            def generate(self, messages, tools, temperature, max_tokens, model, timeout):
                rendered = tokenizer.apply_chat_template(
                    messages, tools=tools or None, tokenize=False, add_generation_prompt=True)
                started = time.monotonic()
                result = llm.generate([rendered], SamplingParams(
                    temperature=temperature, max_tokens=max_tokens))[0]
                candidate = result.outputs[0]
                raw = {'raw_text': candidate.text, 'rendered_prompt': rendered,
                       'finish_reason': candidate.finish_reason,
                       'stop_reason': candidate.stop_reason,
                       'token_ids': list(candidate.token_ids)}
                try:
                    message = parse_output(candidate.text)
                except Exception as exc:
                    message = {'role': 'assistant', 'content': candidate.text}
                    raw['parser_error'] = f'{type(exc).__name__}: {exc}'
                return Generation(message, time.monotonic() - started,
                                  {'prompt_tokens': len(result.prompt_token_ids),
                                   'completion_tokens': len(candidate.token_ids),
                                   'total_tokens': len(result.prompt_token_ids) + len(candidate.token_ids)}, raw)

        scenario_ids = [int(value) for value in os.environ.get(
            'ITBENCH_SCENARIOS', '1,2,4,5,6,7,8,9,11,12').split(',')]
        if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError('Scenario selection must be nonempty and unique')
        manifest['benchmark'] = {
            'name': 'ITBench Lite SRE', 'dataset_revision':
            'd0916b08ba421ce5e672e9ad68aa947d938dfef0',
            'scenarios': scenario_ids,
            'attempts': 1, 'max_tokens': 2048, 'max_turns': 9,
            'timeout_seconds': 600, 'scoring': 'root cause entity proxy',
            'timeout_note': 'Checked between generations; one generation may exceed remaining time',
            'code_identities': json.loads(Path('/workspace/code-identities.json').read_text())}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(run_scenarios(
            Backend(), 'internal-step_20000', Path('/workspace/data/sre'),
            output / 'trajectory.jsonl', len(scenario_ids), 2048, 9, 600,
            scenario_ids=scenario_ids)), flush=True)
        print('ITBENCH_FINISHED', flush=True)


if __name__ == "__main__":
    main()

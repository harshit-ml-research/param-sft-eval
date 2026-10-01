# Param SFT evaluation

Evaluate the internal Param17B `step_20000` SFT checkpoint, inspect its failures, and identify why it goes wrong. Benchmark scores, complete trajectories, and manual review provide evidence for the diagnosis.

## Current state

Infrastructure preparation is complete enough to begin runtime validation. An AWS BharatGen HyperPod worker with H100 GPUs has been inspected, and a cached Docker container starts successfully. The checkpoint architecture is `Param2MoEForCausalLM`; its config records Transformers 5.3.0 and a 32,768 token position range. Model loading, native tool calling, and the full evaluation have not yet been validated on this worker.

No new benchmark score is available. The earlier public Param2 Thinking checkpoint experiments belong to the separate [smoke evaluation repository](https://github.com/harshit-ml-research/agent-eval-smoke-test).

## Evaluation scope

ITBench is the primary benchmark. Additional tests should cover tool use and reasoning. GDPval is a candidate; the additional suites and their scoring protocols are not finalized.

Preserve raw model output, parsed calls, observations, final answers, automatic scores, manual verdicts, token usage, latency, and failures. Record immutable model, image, data, and code identities. Separate serving failures, protocol failures, evidence skipping, reasoning errors, and malformed final output.

The historical offline ITBench Lite harness uses six SRE tools and a deterministic entity proxy. Those results are not official ITBench judge scores. Any forced tool use or evidence requirement is a separate experimental condition.

## References

The reference repositories are Git submodules:

- `references/itbench-agent`: ITBench SRE tools and reference agent.
- `references/itbench-evaluations`: official evaluator reference.

```sh
git clone --recurse-submodules https://github.com/harshit-ml-research/param-sft-eval.git
```

Operational access instructions, private checkpoint locations, and worker inventories live in the separate private internal repository. This repository currently documents preparation; it does not yet provide a validated model-serving or benchmark runner.

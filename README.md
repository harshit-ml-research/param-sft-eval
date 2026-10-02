# Param SFT evaluation

Evaluate the internal Param17B `step_20000` SFT checkpoint, inspect its failures, and identify why it goes wrong. Benchmark scores, complete trajectories, and manual review provide evidence for the diagnosis.

## Current state

The internal SFT checkpoint has completed all 35 pinned ITBench-Lite SRE cases, using eight independent model processes. Every container exited successfully and none reported GPU OOM. The current agent configuration produced zero valid final diagnoses and zero root-cause entity proxy hits.

| Failure category | Cases |
| --- | ---: |
| Tool call after the final-turn budget | 17 |
| Malformed output | 7 |
| Invalid diagnosis structure | 6 |
| Output token limit | 5 |

This is an end-to-end failure under the current protocol, not an official ITBench judge score or a conclusion about every intermediate hypothesis. The 35-case dataset revision is `d0916b08ba421ce5e672e9ad68aa947d938dfef0`. There was one attempt per incident, temperature 0, 2,048 output tokens per generation, and eight investigation turns followed by a diagnosis turn.

The earlier public Param2 Thinking checkpoint experiments belong to the separate [smoke evaluation repository](https://github.com/harshit-ml-research/agent-eval-smoke-test).

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

Operational access instructions, private checkpoint locations, and worker inventories live in the separate private internal repository. The validated Docker entry point is [`runtime_validate.py`](runtime_validate.py). It uses the mounted checkpoint at `/model`, project workspace at `/workspace`, and the checkpoint's native chat template. It preserves generation checks and can run selected SRE incidents through the separate smoke harness.

`RUN_ITBENCH=1` enables incident evaluation; `ITBENCH_SCENARIOS` is a comma-separated list of exact scenario IDs. `RUN_ID`, `CHECKPOINT_SOURCE` and `IMAGE_ID` identify the run. Worker and physical GPU identities are supplied through environment variables and recorded in manifests. Dependencies, snapshots and the smoke harness must be mounted or staged before launch.

The private internal repository retains the launch scripts, detailed `current_status.md`, compact completion records and operational recovery notes. Raw traces stay in experiment storage. The first automatic coordinator stalled after preparation; the full run was recovered by direct launch.

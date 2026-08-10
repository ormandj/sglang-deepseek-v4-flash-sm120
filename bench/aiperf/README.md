# Engine-performance benchmark

This directory provides the current performance gate for DeepSeek V4 Flash on
the two-card SM120 system. It measures an already-running server; it never
starts, stops, or restarts the serving process.

The benchmark separates three effects that must not be collapsed into one
number:

1. engine forward-work rate at fixed concurrency and context shape;
2. useful output-token rate and speculative acceptance;
3. cold-prefill throughput and TTFT.

Agentic traffic, GSM8K, and long-output stability remain separate workload and
quality checks. They are not substitutes for the engine gate.

[`STATISTICAL-DESIGN.md`](STATISTICAL-DESIGN.md) explains the measurement units,
prompt panel, and variability controls.

## Why the decode metric changed

One long generated response is one model trajectory. Repeated byte-identical C1
inputs can produce different useful-token rates as DSPark acceptance changes.
Temperature zero reduces sampling choices but does not make the engine
numerically deterministic.

The engine-specific analyzers therefore report two decode rates over the same
validated plateau:

- `engine_work.forward_passes_per_second_ols`: slope of the live speculative
  decode-step counter. SGLang uses
  `sglang:cuda_graph_passes{mode=decode_cuda_graph|decode_none,tp_rank=0}`;
  vLLM uses the observation count of
  `vllm:iteration_tokens_total{engine=0}`. The running r33 source updates that
  histogram exactly once per engine step.
- `decode.tokens_per_second_ols`: slope of the live useful decode-token
  counter: `sglang:realtime_tokens{mode=decode,tp_rank=0}` or
  `vllm:generation_tokens{engine=0}`. This is retained with accepted tokens per
  step so a fast or slow sampled path cannot be mistaken for a kernel or
  scheduler change.

The analyzer compares the same average context-length interval on every run,
holds exact request occupancy, requires an empty queue, and rejects any prefill
inside the decode window. For vLLM, average context is derived from live
prompt-token and generation-token deltas, including the actual chat-template
expansion; no fixed template offset is assumed. AIPerf client throughput is not
used for the plateau because it includes ramp and drain.

## Gate sizes

All modes keep one server process running and warm each measured shape once.
Panel members then execute sequentially without another warmup or restart.

| Mode | Decode panel | Cold prefill | Intended use |
|---|---|---|---|
| `quick` | C1 x3, C8 x2, plus C32 x1 where supported | 8K, 64K, 128K C1 | routine per-change regression signal |
| `decode-supplement` | C2 x6, C4 x4, C16 x2 | none | add the intermediate concurrency cells to an existing matched quick comparison without repeating it |
| `qualification` | C1 x8, C2 x6, C4 x4, C8 x3, C16 x2, plus C32 x2 where supported | 8K C1/C2/C4, 64K C1, 128K C1 | complete candidate characterization |

The vLLM r33 deployment is configured for at most 16 sequences and cannot run
C32. Its summary records only supported cells; cross-engine comparison records
C32 as SGLang-only instead of treating missing capacity as a zero throughput
measurement. The common near-limit prefill target is 130,816 synthetic content
tokens. On r33, a nominal 131,008-token input expanded to at least 131,072
prompt tokens before the one-token output and was correctly rejected by the
131,072-token context limit. The retained headroom admits engine chat-template
expansion without changing the nominal workload between engines.

Lower concurrency receives more prompt paths because it averages fewer paths
within a single batch. The seed sequence is fixed in
`run-engine-gate-in-pod.sh`; each run retains AIPerf's exact `inputs.json`.

## Network and client placement

All measured requests originate inside the selected serving pod and target
`127.0.0.1:8000`. `run-in-pod.sh` rejects execution outside Kubernetes and
rejects non-loopback inference or metrics URLs. The workstation may stage the
pinned source and retrieve artifacts but is never in the timed path.

AIPerf is pinned by `aiperf.lock.json`. The current install is built from the
audited local checkout revision recorded there. Each cell captures the AIPerf
environment, exact config hash, server command, image provenance supplied by
the caller, CPU, GPU, and driver. Credentials and arbitrary environment
variables are not captured.

Authentication is optional. `BENCH_API_KEY` takes precedence; otherwise the
runner consumes `VLLM_API_KEY` when the container provides it. With neither
variable, requests are keyless; pinned AIPerf adds the `Authorization` header
only when `endpoint.api_key` is nonempty. AIPerf redacts configured API keys,
environment capture never records either variable, and the committed configs
retain only the `${BENCH_API_KEY:}` placeholder.

The pinned Linux `uv` executable is staged at
`/models/.bench-tools/uv-0.12.3-linux-x86_64/uv`; its release artifact and
digests are recorded in `aiperf.lock.json`. This keeps execution identical in
serving images that do and do not bundle `uv` themselves.

Download the archive URL recorded in the lock file, copy it and this directory
into the serving pod, then perform the one-time immutable install before
preparing AIPerf:

```bash
./stage-uv-in-pod.sh /tmp/uv-x86_64-unknown-linux-gnu.tar.gz \
  /models/.bench-tools/uv-0.12.3-linux-x86_64
./prepare-in-pod.sh /tmp/aiperf \
  /models/.bench-tools/aiperf-0.12.0-03c9c6dd
```

## Warmup and cache rules

Warmup is mandatory once after a server start, image or configuration change,
or profiler run that perturbs the process. The gate covers every measured
decode batch size and prefill length before recording results. It does not
repeat warmup between panel members on the same healthy process.

Decode uses exact occupancy and ignores EOS. Cold prefill uses temperature
zero, one output token, and a first-token cache-bust marker. SGLang's supported
cache endpoint is flushed at each measured cell. The r33 deployment does not
mount vLLM's development-only cache-reset endpoint, so its coldness is enforced
without enabling unsafe development APIs: the marker prevents a reusable first
block and `vllm:prompt_tokens_cached` must remain exactly zero.
`analyze_prefill.py` rejects cached prompts, wrong token shapes, incomplete
request sets, and failure to reach requested concurrency. The r33 analyzer
allows up to 128 tokens of engine-reported template expansion;
SGLang retains the 16-token default. Every actual input length remains in the
result artifacts and is used for throughput rather than the nominal target.

## Run

First verify that no image build, compiler work, storage maintenance, profiler,
or unrelated GPU workload is active. Then invoke the staged runner inside the
serving pod:

```bash
BENCH_IMAGE_REF='public-image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='03c9c6ddc5e6227782e53ded177f1227d332af48' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
BENCH_ENGINE='sglang' \
./run-engine-gate-in-pod.sh rc3-vllm-r33 rc3 quick
```

Set `BENCH_ENGINE=vllm` for vLLM. Add `BENCH_API_KEY` only when the endpoint
requires a key; a container-provided `VLLM_API_KEY` is discovered automatically.

Arguments are campaign ID, build ID, and mode. Output is immutable under
`/models/bench/results/aiperf-greenfield/engine-gates/` by default. Override
`AIPERF_CAMPAIGN_ROOT` only to select another retained artifact volume.

Use `decode-supplement` after matched quick gates when a cross-engine report
needs C2/C4/C16 but does not need to repeat already valid C1/C8/prefill cells.
It is a separate checksum-bound artifact, not a replacement for the larger
`qualification` panel.

The runner produces `summary.json` plus every raw AIPerf export, analyzer JSON,
input file, environment capture, and a relative-path checksum inventory.

Compare matching summaries without hiding any prompt-pair value:

```bash
uv run compare_engine_gates.py baseline/summary.json candidate/summary.json
```

For SGLang versus the C16-limited vLLM r33 deployment, add
`--allow-decode-cell-mismatch`; the output then lists common and engine-only
cells explicitly.

## Metric definitions

Every prompt-path value remains visible. Compare builds using the same mode,
seed panel, image configuration, and hardware state.

- Speculative decode steps/sec measures execution of the fixed decode work
  shape. The JSON field retains its original
  `forward_passes_per_second` name for schema continuity.
- Useful tokens/sec answers how much output the production speculative stack
  delivered.
- Useful tokens per forward explains how much of that result came from
  acceptance rather than engine execution rate.
- Prefill throughput and TTFT remain independent release dimensions.

Quick and qualification repetitions are prompt-path subsamples from one
serving process, not independent deployment replicates. They are designed for
fast engineering decisions and transparent effect sizes, not a p-value claim.
If a small effect needs formal inference, collect independent process blocks
only for that accepted candidate and size that experiment from the observed
paired variance.

## Other workloads

- `agentx-mvp.yaml`: production-shaped multi-turn workload at its explicitly
  declared sampling settings.
- `gsm8k.yaml`: deterministic full-dataset correctness gate.
- `run-performance-block-in-pod.sh` and the paired-analysis utilities: retained
  for experiments that specifically require independent deployment blocks;
  they are not the routine per-change gate.

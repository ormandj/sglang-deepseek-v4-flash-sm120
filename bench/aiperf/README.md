# AIPerf benchmark harness

This directory measures an already-running DeepSeek-V4-Flash-0731 server. The
client runs inside the serving pod against `127.0.0.1:8000`; the harness never
starts, stops, or restarts the server.

The method deliberately separates three questions:

1. Does the engine execute a fixed decode shape faster?
2. How many output tokens does the speculative stack produce per verifier step?
3. How does the server behave on cold prefill and production-shaped agentic
   traffic?
4. Does the scheduler keep replacement requests batched while short requests
   continuously finish at fixed client concurrency?

[`STATISTICAL-DESIGN.md`](STATISTICAL-DESIGN.md) records why the workloads and
sample counts were selected.

## Pinned implementation

AIPerf is pinned by [`aiperf.lock.json`](aiperf.lock.json). The harness uses
the exact source revision in that file, not a moving package release. Install
that checkout once in the serving pod's persistent tools directory:

```bash
./stage-uv-in-pod.sh /tmp/uv-x86_64-unknown-linux-gnu.tar.gz \
  /models/.bench-tools/uv-0.12.3-linux-x86_64
./prepare-in-pod.sh /tmp/aiperf \
  /models/.bench-tools/aiperf-0.12.0-6ed4823d
```

The committed configs accept both authenticated and keyless endpoints.
`BENCH_API_KEY` takes precedence, then a container-provided `VLLM_API_KEY`;
with neither set, no authorization header is sent. Keys, arbitrary environment
variables, internal endpoint names, and registry credentials are not captured.

## Controlled engine gate

Every decode cell uses the same workload:

- OpenAI chat-completions endpoint with streaming;
- synthetic coding prompt, 16,384 requested input tokens;
- 4,096 output tokens, `ignore_eos=true` and `min_tokens=4096`;
- temperature 0, top-p 1, and a fixed seed panel;
- exact occupancy at C1, C2, C4, C8, C16, and C32 where supported;
- analysis over the same 17,408–20,480 average-context interval.

The primary engine rate is the OLS slope of server-side decode forward passes.
Synthetic fixed-window output rate and output tokens per forward are reported
beside it, so speculative-acceptance variation remains visible. “Synthetic
fixed-window output tok/s” is the OLS slope of the engine's
generated-output-token counter over the fixed 17,408–20,480 average-context
interval in this synthetic workload. It is not client end-to-end, production,
interactive, or application throughput. It combines verifier rate with output
tokens produced per step, so the accepted-draft-length distribution of each
fixed prompt/seed path directly affects it and gives it a wider observed spread
than verifier steps/s. Every
repetition also records the mean, median, minimum, and maximum DSpARK acceptance
rate and accepted draft length; the summary describes those run-level values
without pooling them. SGLang uses its acceptance gauges. vLLM uses the deltas
of its cumulative draft, draft-token, and accepted-token counters between
adjacent server-metric scrapes; the per-run record also retains each
whole-window counter ratio. The analyzer rejects a window with queueing,
prefill work,
counter resets, wrong occupancy, or an insufficient equal-context interval.

Client metrics are retained in the machine-readable summary, not substituted
for the server-side plateau. AIPerf computes ITL for each request as
`(request latency - TTFT) / (output tokens - 1)`. It is the average
post-first-token time per generated token, including speculative acceptance and
scheduler effects; it is not a distribution of literal streamed-chunk gaps.
Request latency is the end-to-end completion time for the fixed 16K-input/4K-
output request and is a secondary integrated workload measurement.

TTFT is retained as raw input to the prefill calculation and for bounds
validation. It is not scored or published as a separate performance result.
For these long prompts it primarily restates prefill speed, while decode-sweep
TTFT additionally mixes prefill and concurrent scheduling.

For each latency metric, the summary preserves each run's average and
p50/p90/p99, then summarizes those run-level values without pooling requests
across repetitions.

Cold prefill uses one output token, temperature 0, top-p 1, an explicit cache
bust, and 8K/32K/64K/130,816-token input targets. SGLang is flushed at the cell
boundary. The vLLM r33 analyzer instead requires its cached-prompt counter to
remain zero because that deployment does not expose the development cache-reset
API. Throughput is observed input tokens divided by TTFT. Only the resulting
prompt-token rate is used for comparison and publication.

## Gate sizes

All cells run sequentially on one unchanged server process. Each distinct shape
is warmed once before any timed cell. There is no restart or per-repetition
warmup.

| Mode | Decode panel | Cold-prefill panel | Use |
|---|---|---|---|
| `quick` | C1/C4/C8 x3 | 8K/32K/64K/128K x3 | fast candidate screen |
| `decode-supplement` | C2/C16 x3 | none | fill scale guardrails after a quick run |
| `prefill-quick` | none | 8K/32K/64K/128K x3 | matched prefill-only comparison |
| `qualification` | C1/C2/C4/C8 x5; C16/C32 x3 | all lengths x5 | release decision |
| `publication` | every supported C x5 | all lengths x5 | uniform public table |

C1 is the primary single-user programming workload. C2/C4/C8 cover ordinary
sub-agent fan-out. C16/C32 are scale and regression guardrails. vLLM r33 is
configured for at most 16 sequences, so C32 is recorded as unsupported rather
than assigned a synthetic value.

Run an engine gate inside the selected pod:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
BENCH_ENGINE=sglang \
./run-engine-gate-in-pod.sh <campaign> <build> qualification
```

Set `BENCH_ENGINE=vllm` for vLLM. `BENCH_DP_SIZE` defaults to one and should
match a genuinely data-parallel deployment; it is not the tensor-parallel
size.

The output includes raw AIPerf request/server/GPU exports, exact inputs and
config, the analyzer results, `summary.json`, environment provenance, and a
checksum inventory. Performance value is never an exclusion rule.

## Turnover and refill gate

[`run-turnover-gate-in-pod.sh`](run-turnover-gate-in-pod.sh) runs unique,
cache-busted synthetic coding requests: 16 at C1/C2/C4 and 32 at C8. This gives
at least four replacement waves per cell without turning the structural
scheduler check into another long throughput sweep. Every request has a
256-token input target and a forced 256-token output at temperature 0. AIPerf
keeps the client side closed-loop: when one request finishes, another is
submitted until the cell's request set is complete.

The analyzer requires the complete successful request set, exact input/output
shape, sustained target client occupancy while requests remain to be admitted,
the requested peak server occupancy, bounded queue depth, and valid server
metrics. Terminal drain is reported but is not misclassified as a loss of
client load. The gate reports aggregate output tokens/s, median TTFT, median
ITL, running/queued occupancy, total prefill passes, and effective requests per
prefill pass. The batching ratio is valid for this short, one-pass prompt shape
and must not be compared with a different prompt or chunking method.

The `release-screen` mode runs C8 three times and is the routine turnover check
for an engine-changing public release. The broader `screen` mode runs three
repetitions at C1/C2/C4/C8. `qualification` and `publication` run five at every
concurrency. Use the full `publication` panel when the candidate changes
scheduler, admission, batching, or refill behavior; integrates a new
upstream-main source baseline; or produces a suspicious C8 release screen. A
packaging- or documentation-only release may reuse evidence only when the exact
immutable engine candidate and runtime configuration were already qualified.
Run turnover immediately after the engine panel without restarting the server
so the short screen reuses the existing serving session while retaining
separate metrics and artifacts.

Run it inside the selected pod with the same provenance variables as the engine
gate:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
./run-turnover-gate-in-pod.sh <campaign> <build> release-screen
```

Turnover is its own regression dimension. A gain here cannot erase a decode or
prefill regression, and a clean decode plateau cannot excuse a turnover
regression. Compare two matched summaries with
[`compare_turnover_gates.py`](compare_turnover_gates.py); it rejects different
modes, cell sets, request shapes, and repetition IDs.

## Production-shaped AgentX gate

The controlled engine gate is intentionally deterministic. It does not claim
to reproduce a coding-agent session. For that separate question,
[`agentx-mvp.yaml`](configs/agentx-mvp.yaml) uses AIPerf's date-pinned
InferenceX AgentX MVP scenario with its public Weka coding traces:

- temperature 1.0 and top-p 0.95;
- fixed random and sampling seeds;
- C1 and C8 as separate 900-second minimum runs;
- built-in warmup, replay timing, cache behavior, and submission-validity
  checks.

Run it inside the pod:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
./run-agentx-gate-in-pod.sh <campaign> <build>
```

AgentX results are reported separately. They are not averaged into the fixed
engine gate and cannot turn an engine regression into a pass.

## Execution rules

- Run only when the host has no image build, compiler, profiler, maintenance,
  or unrelated GPU workload.
- Verify the immutable image digest, exact server command, model revision,
  driver, and clocks/power state before comparing builds.
- Keep one healthy process unchanged while its panel runs sequentially.
- Warm every shape once after a process start, configuration change, or
  profiler attachment. Do not repeatedly restart between samples.
- Retain every valid measurement. Exclude only objective validation failures.
- Compare like-for-like summaries. Do not splice cells from different modes or
  old methods into one table.

Publication uses only a fresh `publication` panel for each engine. It includes
all five run values, medians and dispersion, and states unsupported cells
directly.

## Release quality and stability checks

These checks are reported separately from the engine-performance panel. They
do not count as additional decode repetitions.

### GSM8K

[`gsm8k.yaml`](configs/gsm8k.yaml) runs the complete 1,319-question GSM8K test
set once at concurrency 16, temperature 0, seed 42, and a 16,384-token response
cap. Accuracy is `correct / 1,319`; this is a dataset accuracy check, not an
`n=5` timing cell.

The pinned AIPerf GSM8K grader prefers the dataset's `####` answer marker. A
response without that marker is recorded as `unparsed` when the documented
last-number fallback is used. A fallback-extracted answer can still be correct,
so the correct count and fallback count are separate facts.

Run it from the serving pod with the same provenance variables as the engine
gate:

```bash
AIPERF_ARTIFACT_ROOT=/path/to/quality-results \
GSM8K_REQUESTS=1319 \
GSM8K_CONCURRENCY=16 \
GSM8K_MAX_TOKENS=16384 \
./run-in-pod.sh configs/gsm8k.yaml gsm8k-full
```

### Near-context and AgentX

[`../near_context_bench.py`](../near_context_bench.py) sends one persisted-corpus
request near the configured context limit. Its authorization header is omitted
when the selected API-key environment variable is unset.

[`run-agentx-gate-in-pod.sh`](run-agentx-gate-in-pod.sh) runs the pinned AgentX
MVP scenario for 900 seconds each at C1 and C8 with temperature 1.0 and top-p
0.95. It requires AIPerf's `submission_valid` result and writes a checksum
inventory on success.

## Optional targeted diagnostics

[`../long_write_quality.py`](../long_write_quality.py) issues sequential
single-file HTML/JavaScript requests and validates their structure and inline
JavaScript. It is not part of the recurring release or publication protocol;
run it only when investigating long-generation behavior. See the tool's
`--help` output for its run and validation commands.

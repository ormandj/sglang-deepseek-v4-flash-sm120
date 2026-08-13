# DeepSeek-V4-Flash benchmark tools

These tools measure an already-running OpenAI-compatible server. They do not
start, stop, restart, or reconfigure the server.

## Engine and prefill measurements

[`aiperf/run-engine-gate-in-pod.sh`](aiperf/run-engine-gate-in-pod.sh) is the
canonical runner. It enforces execution inside Kubernetes against
`127.0.0.1:8000`, uses the pinned AIPerf revision in
[`aiperf.lock.json`](aiperf/aiperf.lock.json), and records image, deployment,
project, and model revisions without recording credentials.

The runner separates:

- server-side decode steps per second;
- synthetic fixed-window output tokens per second;
- output tokens per step per request;
- DSpARK acceptance rate and accepted length;
- AIPerf inter-token latency;
- cache-cold prompt tokens per second.

TTFT and request latency remain in raw AIPerf output. TTFT is not scored or
published separately because it is the timing input used to calculate prefill
throughput for these long prompts. Request latency combines prefill, decode,
and scheduling and is treated as a secondary integrated measurement.

The complete workload, warmup, validity, repetition, and reporting contract is
in [`aiperf/README.md`](aiperf/README.md). The rationale is in
[`aiperf/STATISTICAL-DESIGN.md`](aiperf/STATISTICAL-DESIGN.md).

The synthetic fixed-window output rate is an engine-comparison measurement,
not an estimate of expected production, interactive, or application
throughput. It varies with the accepted-draft-length distribution of each fixed
prompt/seed path; verifier steps/s is reported separately from that acceptance
effect.

Publication uses ten sequential fixed-seed paths at C1, five repetitions at
every other supported decode concurrency, and five requests at each prefill
length.

## Release quality checks

- [`aiperf/configs/gsm8k.yaml`](aiperf/configs/gsm8k.yaml): full 1,319-question
  GSM8K accuracy run.
- [`near_context_bench.py`](near_context_bench.py): one persisted-corpus
  near-context admission request.
- [`aiperf/run-agentx-gate-in-pod.sh`](aiperf/run-agentx-gate-in-pod.sh): pinned
  AgentX MVP replay at C1 and C8.

Quality results remain separate from engine-performance measurements.

[`long_write_quality.py`](long_write_quality.py) remains available as an
optional targeted diagnostic. It is not part of the recurring release or
publication protocol.

Both authenticated and keyless endpoints are supported. `BENCH_API_KEY` is
used when set; otherwise no authorization header is sent. Optional diagnostic
tools also accept their documented API-key environment variables. No key is
written to output.

## Results

Machine-readable publication summaries are in [`results/`](results/). The
human-readable tables and exact system configuration are in
[`../BENCHMARKS.md`](../BENCHMARKS.md).

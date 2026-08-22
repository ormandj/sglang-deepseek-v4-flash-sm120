# Current benchmark panel

Last updated: 2026-08-22 CDT.

This page reports the qualified SGLang `0.8.1-rc4` panel beside the retained
vLLM r33 panel. Performance, capacity, and quality are separate measurements;
the tables do not combine them into an overall score. Machine-readable
summaries retain every measured repetition, distribution statistics, and raw
metric inputs for the published release panels.

## Results

### Fixed-window decode

Every supported cell contains five sequential fixed-seed repetitions on one
unchanged process. Forward passes/s is the primary engine-execution statistic.
Synthetic output tok/s is the generated-token counter slope over the same
window and also reflects speculative acceptance. ITL is mean post-first-token
time per generated token.

| C | SGLang forward/s | vLLM forward/s | SGLang synthetic tok/s | vLLM synthetic tok/s | SGLang ITL ms | vLLM ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.783 | 66.575 | 331.1 | 255.2 | 3.113 | 3.886 |
| 2 | 48.468 | 46.337 | 489.1 | 421.0 | 4.512 | 5.215 |
| 4 | 33.715 | 33.074 | 672.9 | 616.0 | 6.324 | 7.082 |
| 8 | 23.318 | 23.454 | 924.0 | 795.8 | 9.533 | 11.122 |
| 16 | 17.995 | 15.865 | 1408.1 | 1097.2 | 14.245 | 17.111 |
| 32 | 13.380 | — | 2062.6 | — | 21.855 | — |

The paired acceptance statistics explain why synthetic output rate can move
independently of forward rate. They are workload/configuration evidence, not a
second engine-speed metric.

| C | SGLang acceptance | vLLM acceptance | SGLang output/forward/request | vLLM output/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 0.793 | 0.532 | 5.034 | 3.952 |
| 2 | 0.781 | 0.714 | 5.070 | 4.638 |
| 4 | 0.786 | 0.761 | 4.899 | 4.814 |
| 8 | 0.785 | 0.656 | 4.895 | 4.241 |
| 16 | 0.756 | 0.674 | 4.817 | 4.352 |
| 32 | 0.750 | — | 4.785 | — |

vLLM r33 was not measured at C32. Its documented TP2 profile set
`max_num_seqs=16` and reported 143,599 KV tokens; the C32 workload needs about
655,000 tokens. No value is imputed, and this is a limit of that profile rather
than a claim about every possible vLLM deployment.

### Cache-cold prefill

Each cell contains five requests with one output token and explicit cache
busting. Prompt throughput is observed input tokens divided by TTFT; TTFT is
therefore not repeated as an independent result.

| Input target | SGLang prompt tok/s | vLLM prompt tok/s |
|---:|---:|---:|
| 8K | 7948.1 | 7689.8 |
| 32K | 8823.9 | 8784.7 |
| 64K | 8484.1 | 8518.7 |
| 128K | 7966.7 | 7953.6 |

### Request turnover

The closed-loop turnover gate continuously replaces completed 256-input,
256-output coding requests at fixed client concurrency. It is separate from the
clean decode and cold-prefill panels. This release changed refill scheduling,
so it completed the full five-repetition C1/C2/C4/C8 panel.

| C | Median output tok/s | Median TTFT ms | Median ITL ms | Median requests / prefill pass |
|---:|---:|---:|---:|---:|
| 1 | 212.7 | 193.7 | 3.853 | 1.00 |
| 2 | 299.0 | 178.0 | 5.549 | 1.00 |
| 4 | 432.3 | 191.0 | 8.014 | 1.23 |
| 8 | 607.9 | 358.5 | 11.096 | 2.00 |

In the matched `0.8.1-rc4` C8 control with automatic refill delay disabled,
output throughput was 591.3 tokens/s, ITL was 11.887 ms, and requests per
prefill pass was 1.33. Enabling the automatic policy changed those five-run
means to 610.9 tokens/s, 11.132 ms, and 2.06 while increasing mean TTFT from
241.5 ms to 352.7 ms. This is a
throughput/first-token tradeoff for repeated request turnover, not a
clean-decode or cold-prefill result. The causal-control artifact remains in the
private qualification record; the machine-readable public turnover summary is
the enabled-policy release panel.

### TP2 capacity

These are startup reports from the exact measured profiles. The profiles use
different context and scheduler limits, so this table describes deployable
capacity rather than isolating engine memory efficiency.

| Setting | SGLang 0.8.1-rc4 | vLLM r33 |
|---|---:|---:|
| Reported KV cache | 801,536 tokens | 143,599 tokens |
| Declared context limit | 786,432 | 131,072 |
| Scheduler sequence limit | 48 | 16 |
| Static/GPU memory fraction | 0.93 | 0.975 |
| C32 fixed-window workload | completed, n=5 | not reachable in profile |

At the shipped SGLang envelope, a cold 780,000-token request completed in 255.8
seconds and four cold concurrent 250,000-token requests completed in 140.8
seconds without process restarts. The four-request shape recomputed all
1,000,000 prompt tokens. See
[`RUN.md`](RUN.md) for capacity guidance and the prior higher-fraction failure.

### Production-shaped coding replay

The public Weka coding-trace AgentX scenario ran for 900 seconds each at C1 and
C8 with temperature 1.0, top-p 0.95, fixed seeds, and HiCache disabled. This is
a separate workload-validity check, not a replacement for the fixed decode and
cold-prefill panels.

| C | Completed requests | Output tok/s | Prompt-cache read share | Request errors | Submission valid |
|---:|---:|---:|---:|---:|---:|
| 1 | 116 | 118.5 | 96.85% | 0 | yes |
| 8 | 157 | 206.8 | 90.07% | 0 | yes |

### Quality

Each engine ran the full GSM8K set once at C16, temperature 0, seed 42, and a
16,384-token response cap.

| Result | SGLang 0.8.1-rc4 | vLLM r33 |
|---|---:|---:|
| Questions | 1,319 | 1,319 |
| Correct | 1,260 | 1,243 |
| Accuracy | 95.53% | 94.24% |
| Request errors | 0 | 0 |

All 1,319 SGLang responses used the grader's documented last-number fallback
because the model did not emit the dataset's `####` marker. Correctness and
fallback use are recorded separately.

## Test system and configurations

Both panels used:

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs, SM120, TP2 over PCIe Gen 4 x16;
- a 300 W power limit per GPU and one active engine at a time;
- clients inside the selected serving pod against `127.0.0.1:8000`;
- `deepseek-ai/DeepSeek-V4-Flash-0731` at model revision
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- DSpARK speculative decoding with block size 5.

The SGLang panel used the README launch contract, including FP8 KV, HC prenorm,
fused MHC post+pre, FP8 W_o_A, the FP4 indexer, eligible PCIe-IPC all-reduce,
NCCL fallback, and prefill/decode warmups. PCIe-IPC all-gather and TRT/MNNVL
fusion were absent. Its qualified source identity is:

```text
SGLang base                   cce0a1244b5a3ee60411fb403f65f856a1d012f0
SGLang effective tree         bedec1d0ee735ddbd246f2e2b91d724ceefc5136
FlashInfer base               fb28d7242b3506a2348265962041acc1fb56cca4
FlashInfer effective tree     d6d4f3d7daf66a57947049cf1cc7866b1678549b
FlashInfer version            0.6.18.dev20260819
DeepGEMM base                 80b2c44b9ae95b90c1e0a1626a05b6c4f7f09f1f
DeepGEMM effective tree       b7e23a6fb5ac6571046cc12e85352d43af63f27d
DeepGEMM version              0.0.0+sm120jit5
Compilation-cache schema      v37
```

The vLLM panel used local-inference-lab r33 at:

```text
docker.io/voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

Its profile used TP2, FP8 KV, DSpARK block size 5, `max_num_seqs=16`,
`max_num_batched_tokens=8192`, `max_model_len=131072`, and
`gpu_memory_utilization=0.975`.

## Frozen method

AIPerf 0.12.0 is pinned at revision:

```text
6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23
```

The SGLang engine and turnover panels used benchmark-project revision
`2e9ac4d54d618543856cde9e018c2a4758827822`; the retained vLLM panel used
`6fc08101a6c309b998a2f393b30065942e98a0b4`. Both revisions use the same
request shapes, fixed context interval, five repetitions per supported cell,
and 20-scrape minimum. The SGLang analyzer accepts a valid interval of at least
one second; the retained vLLM panel used the earlier seven-second minimum.

Decode requests use streaming OpenAI chat completions, 16,384 requested input
tokens, 4,096 output tokens with EOS ignored, temperature 0, top-p 1, and five
fixed-seed paths. Every concurrency is warmed once before measurement. The
analyzer fits the OLS slope of server counters over the same
17,408–20,480-token average-context interval and rejects a window unless it has
exact occupancy, an empty queue, no prefill work, monotonic counters, the
required duration, and at least 20 metric scrapes.

Cold prefill uses 8K, 32K, 64K, and 130,816-token input targets, one output
token, temperature 0, top-p 1, and explicit cache busting. Each length is
warmed once before its five measured requests.

All engine and turnover cells ran sequentially on one unchanged server process.
The
five repetitions are fixed prompt/seed-path subsamples, not independent
deployment replicates. First-use compilation is excluded from steady-state
measurements.

## Machine-readable results

- [`sglang-v0.8.1-rc.4-publication-summary.json`](bench/results/sglang-v0.8.1-rc.4-publication-summary.json)
- [`sglang-v0.8.1-rc.4-turnover-summary.json`](bench/results/sglang-v0.8.1-rc.4-turnover-summary.json)
- [`sglang-v0.8.1-rc.4-gsm8k.json`](bench/results/sglang-v0.8.1-rc.4-gsm8k.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`vllm-r33-gsm8k.json`](bench/results/vllm-r33-gsm8k.json)

Historical public panels remain available in repository history rather than on
current main.

## Reproduce the engine gate

Install the exact AIPerf revision from
[`bench/aiperf/aiperf.lock.json`](bench/aiperf/aiperf.lock.json), stage this
repository's `bench/` directory inside the serving pod, and run against the
pod's loopback endpoint:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='9e165c30e2704aec5d9d593cce3eebd58bbef1cb' \
BENCH_ENGINE=sglang \
  ./bench/aiperf/run-engine-gate-in-pod.sh campaign build publication
```

Use `BENCH_ENGINE=vllm` for vLLM. Set `BENCH_API_KEY` only when the endpoint
requires authentication; credentials are not written to output. The runner
rejects non-Kubernetes or non-loopback execution, wrong shapes, incomplete
request sets, occupancy/queue/prefill contamination, counter resets, cached
tokens in a cold-prefill cell, and insufficient equal-context intervals.

The executable contract is in [`bench/aiperf/README.md`](bench/aiperf/README.md)
and its rationale is in
[`bench/aiperf/STATISTICAL-DESIGN.md`](bench/aiperf/STATISTICAL-DESIGN.md).

For an ordinary engine-changing release, run the short C8 turnover screen after
the engine panel on the same server process:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='9e165c30e2704aec5d9d593cce3eebd58bbef1cb' \
  ./bench/aiperf/run-turnover-gate-in-pod.sh campaign build release-screen
```

Use `publication` instead for scheduler, admission, batching, refill, or new
upstream-main integration releases and after a suspicious C8 screen.

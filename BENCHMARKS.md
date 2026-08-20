# Current benchmark panel

Last updated: 2026-08-20 CDT.

This page reports the qualified SGLang `0.7.0-rc1` panel beside the retained
vLLM r33 panel. Performance, capacity, and quality are separate measurements;
the tables do not combine them into an overall score. Machine-readable
summaries retain every measured repetition, distribution statistics, and raw
metric inputs.

## Results

### Fixed-window decode

Every supported cell contains five sequential fixed-seed repetitions on one
unchanged process. Forward passes/s is the primary engine-execution statistic.
Synthetic output tok/s is the generated-token counter slope over the same
window and also reflects speculative acceptance. ITL is mean post-first-token
time per generated token.

| C | SGLang forward/s | vLLM forward/s | SGLang synthetic tok/s | vLLM synthetic tok/s | SGLang ITL ms | vLLM ITL ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 65.106 | 66.575 | 345.3 | 255.2 | 3.154 | 3.886 |
| 2 | 48.797 | 46.337 | 457.3 | 421.0 | 4.544 | 5.215 |
| 4 | 33.670 | 33.074 | 660.0 | 616.0 | 6.688 | 7.082 |
| 8 | 23.226 | 23.454 | 939.4 | 795.8 | 9.485 | 11.122 |
| 16 | 17.853 | 15.865 | 1397.0 | 1097.2 | 14.318 | 17.111 |
| 32 | 13.336 | — | 2101.7 | — | 21.552 | — |

The paired acceptance statistics explain why synthetic output rate can move
independently of forward rate. They are workload/configuration evidence, not a
second engine-speed metric.

| C | SGLang acceptance | vLLM acceptance | SGLang output/forward/request | vLLM output/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 0.870 | 0.532 | 5.324 | 3.952 |
| 2 | 0.741 | 0.714 | 4.661 | 4.638 |
| 4 | 0.779 | 0.761 | 4.794 | 4.814 |
| 8 | 0.814 | 0.656 | 4.964 | 4.241 |
| 16 | 0.779 | 0.674 | 4.874 | 4.352 |
| 32 | 0.767 | — | 4.887 | — |

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
| 8K | 7925.9 | 7689.8 |
| 32K | 8825.5 | 8784.7 |
| 64K | 8461.4 | 8518.7 |
| 128K | 7953.3 | 7953.6 |

### TP2 capacity

These are startup reports from the exact measured profiles. The profiles use
different context and scheduler limits, so this table describes deployable
capacity rather than isolating engine memory efficiency.

| Setting | SGLang 0.7.0-rc1 | vLLM r33 |
|---|---:|---:|
| Reported KV cache | 801,536 tokens | 143,599 tokens |
| Declared context limit | 786,432 | 131,072 |
| Scheduler sequence limit | 48 | 16 |
| Static/GPU memory fraction | 0.93 | 0.975 |
| C32 fixed-window workload | completed, n=5 | not reachable in profile |

At the shipped SGLang envelope, a cold 780,000-token request and four cold
concurrent 250,000-token requests completed without process restarts. The
four-request shape recomputed all 1,000,000 prompt tokens. See
[`RUN.md`](RUN.md) for capacity guidance and the prior higher-fraction failure.

### Quality

Each engine ran the full GSM8K set once at C16, temperature 0, seed 42, and a
16,384-token response cap.

| Result | SGLang 0.7.0-rc1 | vLLM r33 |
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
SGLang base                   7f8f030000b628ea2cb033e7457a13dd0ac80f99
SGLang effective tree         6696e98a1f037f16774eeea793c05d3eb1316d6d
FlashInfer base               5366177a074e27df7db527f5b744c77dfd748484
FlashInfer effective tree     917a439a4cd74f5f0fa4f7dbb13543c606ffe346
FlashInfer version            0.6.18.dev20260819
DeepGEMM base                 80b2c44b9ae95b90c1e0a1626a05b6c4f7f09f1f
DeepGEMM effective tree       b7e23a6fb5ac6571046cc12e85352d43af63f27d
DeepGEMM version              0.0.0+sm120jit5
Compilation-cache schema      v26
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

The SGLang panel used benchmark-project revision
`27bf5b07924c0d001ac142b92138cfdf4b162626`; the retained vLLM panel used
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

All cells for an engine ran sequentially on one unchanged server process. The
five repetitions are fixed prompt/seed-path subsamples, not independent
deployment replicates. First-use compilation is excluded from steady-state
measurements.

## Machine-readable results

- [`sglang-v0.7.0-rc.1-publication-summary.json`](bench/results/sglang-v0.7.0-rc.1-publication-summary.json)
- [`sglang-v0.7.0-rc.1-gsm8k.json`](bench/results/sglang-v0.7.0-rc.1-gsm8k.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`vllm-r33-gsm8k.json`](bench/results/vllm-r33-gsm8k.json)

Historical public panels remain available in immutable release tags rather
than on current main.

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

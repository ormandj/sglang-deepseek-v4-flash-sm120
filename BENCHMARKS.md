# v0.5.0-rc.1 and vLLM r33 measurements

Last updated: 2026-08-18 CDT.

This document presents the current SGLang `v0.5.0-rc.1` publication panel and
the retained vLLM r33 panel. It reports the two engines' values without a
winner/loser interpretation. The executable harness, analyzers, scoring logic,
quality graders, and machine-readable summaries are under [`bench/`](bench/).

## System and software

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs;
- SM120, TP2 over PCIe Gen 4 x16;
- 300 W power limit per GPU;
- one engine active at a time;
- clients executed inside the serving pod against `127.0.0.1:8000`;
- model `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- speculative decoding: DSpARK block size 5;
- SGLang HC prenorm, fused MHC post+pre, and FP8 W_o_A enabled;
- SGLang FlashInfer PCIe-IPC all-reduce enabled for eligible decode
  reductions, legacy custom all-reduce disabled, and NCCL fallback retained;
- SGLang TRT/MNNVL fusion disabled and PCIe-IPC all-gather absent.

The qualified SGLang source identity is:

```text
SGLang main                  87a09494fa3fbd685bd7c88d6a2dbdd3135de602
SGLang effective tree        dc71e9b9bb380e96bcb2c0cb4aed120f79478c3e
FlashInfer main              7aa0cd3b64f84c50c18ee958e24f708afb2103c1
FlashInfer effective tree    489e9318e4d21d3ecddc8d2ec8f138dde93784b5
FlashInfer version           0.6.18.dev20260818
DeepGEMM base                75f60622bc6d317306a41c1f38dc9d888b3ec841
DeepGEMM effective tree      ff371cf8cc7186c8dab8e07cc7cda6c28baa092f
DeepGEMM version             0.0.0+sm120jit3
cache schema                 v20
```

The performance and quality measurements used an image built from those exact
three effective trees and the documented serving options. The public build
reconstructs the same source trees from the locked upstream bases and patches.

vLLM used local-inference-lab r33:

```text
docker.io/voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

Its configuration used TP2, DSpARK block size 5, `max_num_seqs=16`,
`max_num_batched_tokens=8192`, FP8 KV cache, `max_model_len=131072`, and
`gpu_memory_utilization=0.975`.

## Benchmark implementation

AIPerf 0.12.0 is pinned at:

```text
6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23
```

The SGLang panel used benchmark project revision:

```text
15f5dcc169f64b629523803a9c0d8ded1728aabf
```

The retained vLLM r33 panel used benchmark project revision:

```text
6fc08101a6c309b998a2f393b30065942e98a0b4
```

Both revisions use the same request shapes, fixed context interval, five
repetitions per supported cell, and 20-scrape minimum. The current SGLang
analyzer calibration accepts an interval of at least 1 second; the retained
vLLM panel used the earlier 7-second minimum.

Each engine used one unchanged process. All cells ran sequentially in the
selected serving pod against localhost. Every distinct decode concurrency and
prefill length was warmed once before measurement; there was no restart or
per-repetition warmup.

Decode used OpenAI chat completions with streaming, 16,384 requested input
tokens, 4,096 output tokens with EOS ignored, temperature 0, top-p 1, and five
fixed-seed paths at every supported concurrency. The analyzer fits the OLS
slope of server-side counters over the same 17,408-20,480 average-context
interval. A valid interval requires exact occupancy, an empty queue, no
prefill work, monotonic counters, at least 1 second, and at least 20 metric
scrapes.

Forward passes/s is the primary engine-execution statistic. Synthetic output
tok/s is the generated-output-token counter slope over the same interval. It
combines verifier rate with speculative acceptance and is not expected
production, interactive, or application throughput. Acceptance is reported
separately so a generated-path change is not represented as an engine-clock
change.

AIPerf calculates per-request ITL as:

```text
(request latency - TTFT) / (output tokens - 1)
```

ITL is average post-first-token time per generated token, including
speculative acceptance and scheduling. TTFT remains in raw output but is not a
separate headline because the long-prompt value primarily restates prefill.

Cold prefill used one output token, temperature 0, top-p 1, explicit cache
busting, and 8K, 32K, 64K, and 130,816-token input targets. Prompt throughput
is observed input tokens divided by TTFT. Every cell used five requests.

The repetitions are same-process prompt-path subsamples, not independent
deployment replicates. The machine-readable summary retains every run, its
median, min/max, sample standard deviation, and coefficient of variation. No
valid result was excluded because its performance was surprising.

vLLM r33 was not measured at C32 because its deployment lacked enough KV
capacity to admit that workload. No value is imputed.

## Fixed-window decode

### SGLang run values

| C | Forward passes/s, five runs | Median | Sample CV | Synthetic output tok/s, five runs | Median | Median ITL |
|---:|---|---:|---:|---|---:|---:|
| 1 | 64.723, 67.402, 65.719, 65.476, 64.840 | 65.476 | 1.64% | 291.9, 404.4, 325.7, 320.0, 334.3 | 325.7 | 3.160 ms |
| 2 | 47.477, 48.199, 48.848, 48.699, 48.852 | 48.699 | 1.22% | 486.8, 480.3, 469.5, 546.3, 472.3 | 480.3 | 4.337 ms |
| 4 | 33.538, 33.292, 33.389, 34.900, 33.627 | 33.538 | 1.94% | 645.4, 657.0, 680.6, 749.5, 653.3 | 657.0 | 6.291 ms |
| 8 | 23.389, 22.879, 23.330, 23.515, 23.037 | 23.330 | 1.13% | 939.0, 917.8, 879.6, 968.0, 904.3 | 917.8 | 9.478 ms |
| 16 | 17.587, 17.728, 17.392, 17.518, 17.316 | 17.518 | 0.92% | 1,399.1, 1,460.6, 1,373.5, 1,368.7, 1,365.9 | 1,373.5 | 14.583 ms |
| 32 | 12.919, 13.062, 13.012, 13.071, 13.013 | 13.013 | 0.46% | 1,970.7, 2,132.6, 2,048.0, 2,033.4, 2,047.9 | 2,047.9 | 22.349 ms |

### Current engine table

| Engine | C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1 | 5 | 65.476 | 325.7 | 3.160 |
| vLLM r33 | 1 | 5 | 66.580 | 255.2 | 3.890 |
| SGLang v0.5.0-rc.1 | 2 | 5 | 48.699 | 480.3 | 4.337 |
| vLLM r33 | 2 | 5 | 46.340 | 421.0 | 5.220 |
| SGLang v0.5.0-rc.1 | 4 | 5 | 33.538 | 657.0 | 6.291 |
| vLLM r33 | 4 | 5 | 33.070 | 616.0 | 7.060 |
| SGLang v0.5.0-rc.1 | 8 | 5 | 23.330 | 917.8 | 9.478 |
| vLLM r33 | 8 | 5 | 23.450 | 795.8 | 11.080 |
| SGLang v0.5.0-rc.1 | 16 | 5 | 17.518 | 1,373.5 | 14.583 |
| vLLM r33 | 16 | 5 | 15.860 | 1,097.2 | 17.590 |
| SGLang v0.5.0-rc.1 | 32 | 5 | 13.013 | 2,047.9 | 22.349 |
| vLLM r33 | 32 | 0 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — | — |

## DSpARK acceptance

Each SGLang value is the median of the five per-run means. Acceptance explains
synthetic output-rate variation; it is not used as an engine-speed result.

| Engine | C | Acceptance rate | Output tokens/forward/request |
|---|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1 | 0.787 | 4.939 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.5.0-rc.1 | 2 | 0.813 | 4.977 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.5.0-rc.1 | 4 | 0.778 | 4.985 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.5.0-rc.1 | 8 | 0.798 | 5.016 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.5.0-rc.1 | 16 | 0.782 | 4.947 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.5.0-rc.1 | 32 | 0.765 | 4.881 |
| vLLM r33 | 32 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — |

> **vLLM C=32 capacity.** The vLLM side runs the upstream-documented TP2 profile
> unmodified ("Gilded Gnosis v20 r33, documented TP2 fixed-K5",
> [`local-inference-lab/rtx6kpro`](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/ds4dspark-v20-r33.md)),
> which sets `max_num_seqs=16`. Independently, vLLM self-reported at startup:
> `Available KV cache memory: 8.07 GiB`, `GPU KV cache size: 143,599 tokens`,
> `Maximum concurrency for 131,072 tokens per request: 1.10x`. At 16,384-token
> input plus 4,096 output, 32 streams need roughly 655,000 KV tokens against the
> 143,599 available. Both limits belong to that profile, not to vLLM as an
> engine. Contexts also differed (131,072 vs 774,656), so the pools are not
> directly comparable.


## Cold prefill

### SGLang run values

| Target | Prompt tok/s, five requests | Aggregate prompt tok/s |
|---:|---|---:|
| 8K | 7,415.1, 7,679.3, 7,740.8, 7,800.7, 7,766.2 | 7,665.1 |
| 32K | 8,647.9, 8,649.3, 8,640.1, 8,564.0, 8,540.3 | 8,602.4 |
| 64K | 8,409.8, 8,368.0, 8,267.4, 8,214.0, 8,181.9 | 8,284.2 |
| 128K | 7,980.2, 7,824.8, 7,710.3, 7,713.1, 7,714.4 | 7,785.1 |

### Current engine table

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 7,665.1 | 8,602.4 | 8,284.2 | 7,785.1 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

## Quality results

Each engine ran the full GSM8K set once at concurrency 16, temperature 0,
seed 42, and a 16,384-token response cap.

| Engine | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.5.0-rc.1 | 1,319 | 1,261 | 95.60% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

All 1,319 SGLang responses used the grader's documented last-number fallback
because the model did not emit the dataset's `####` marker. Correctness and
fallback counts are separate facts.

## Machine-readable results

- [`sglang-v0.5.0-rc.1-publication-summary.json`](bench/results/sglang-v0.5.0-rc.1-publication-summary.json)
- [`sglang-v0.5.0-rc.1-gsm8k.json`](bench/results/sglang-v0.5.0-rc.1-gsm8k.json)
- [`vllm-r33-publication-summary.json`](bench/results/vllm-r33-publication-summary.json)
- [`vllm-r33-gsm8k.json`](bench/results/vllm-r33-gsm8k.json)

The summaries retain TTFT and request latency because AIPerf records them, but
neither is presented as an independent engine-performance result.

## Reproducing the engine gate

Install the exact AIPerf revision recorded by
[`bench/aiperf/aiperf.lock.json`](bench/aiperf/aiperf.lock.json), then stage
this repository's `bench/` directory inside the serving pod. Run against the
pod's localhost endpoint:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='6ed4823d127b3a6d12c63fb8c2ca5eff13f9ba23' \
BENCH_MODEL_REVISION='9e165c30e2704aec5d9d593cce3eebd58bbef1cb' \
BENCH_ENGINE=sglang \
./bench/aiperf/run-engine-gate-in-pod.sh campaign build publication
```

Use `BENCH_ENGINE=vllm` for vLLM. Set `BENCH_API_KEY` when authentication is
required; omit it for a keyless endpoint. The key is not written to output.

The runner rejects execution outside Kubernetes, a non-loopback endpoint,
wrong token shape, incomplete or failed request sets, wrong occupancy, queued
work, prefill in a decode interval, counter resets, cached tokens in a cold
prefill cell, and an insufficient equal-context interval.

See [`bench/aiperf/README.md`](bench/aiperf/README.md) and
[`bench/aiperf/STATISTICAL-DESIGN.md`](bench/aiperf/STATISTICAL-DESIGN.md) for
the complete method and gate commands.

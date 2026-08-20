# v0.6.0-rc.3 and vLLM r33 measurements

Last updated: 2026-08-20 CDT.

This document presents the current SGLang `v0.6.0-rc.3` publication panel and
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
SGLang main                  5f128395910dafb98c34083dc26cb790c7674d34
SGLang effective tree        3d6254585f7baf4aa4c78db37c50d90e63156342
FlashInfer main              05e5d927399d62a2479c430ad3e167738254d760
FlashInfer effective tree    06cbd9e30d319454ecca57bf51bed915d86c9d52
FlashInfer version           0.6.18.dev20260819
DeepGEMM base                80b2c44b9ae95b90c1e0a1626a05b6c4f7f09f1f
DeepGEMM effective tree      ed1efbc5588a673b39a78cfdfafaac4eb282365a
DeepGEMM version             0.0.0+sm120jit4
cache schema                 v24
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
| 1 | 63.426, 66.942, 66.505, 64.372, 63.511 | 64.372 | 2.57% | 355.9, 401.7, 295.5, 373.7, 352.6 | 355.9 | 2.967 ms |
| 2 | 48.533, 50.103, 49.042, 53.849, 49.527 | 49.527 | 4.21% | 459.5, 524.0, 410.9, 614.6, 450.1 | 459.5 | 4.392 ms |
| 4 | 32.943, 33.353, 33.253, 33.833, 33.199 | 33.253 | 0.98% | 635.8, 668.5, 540.6, 670.7, 646.5 | 646.5 | 6.668 ms |
| 8 | 23.443, 23.184, 23.045, 23.508, 23.467 | 23.443 | 0.87% | 943.2, 928.8, 856.7, 861.3, 870.3 | 870.3 | 9.849 ms |
| 16 | 18.429, 17.863, 17.987, 17.792, 17.823 | 17.863 | 1.46% | 1,389.6, 1,384.6, 1,433.7, 1,436.4, 1,301.6 | 1,389.6 | 14.065 ms |
| 32 | 13.438, 13.328, 13.344, 13.376, 13.243 | 13.344 | 0.53% | 2,033.5, 2,049.6, 2,092.4, 2,097.5, 2,043.2 | 2,049.6 | 21.899 ms |

### Current engine table

| Engine | C | n | Forward passes/s | Synthetic output tok/s | ITL ms/token |
|---|---:|---:|---:|---:|---:|
| SGLang v0.6.0-rc.3 | 1 | 5 | 64.372 | 355.9 | 2.967 |
| vLLM r33 | 1 | 5 | 66.580 | 255.2 | 3.890 |
| SGLang v0.6.0-rc.3 | 2 | 5 | 49.527 | 459.5 | 4.392 |
| vLLM r33 | 2 | 5 | 46.340 | 421.0 | 5.220 |
| SGLang v0.6.0-rc.3 | 4 | 5 | 33.253 | 646.5 | 6.668 |
| vLLM r33 | 4 | 5 | 33.070 | 616.0 | 7.060 |
| SGLang v0.6.0-rc.3 | 8 | 5 | 23.443 | 870.3 | 9.849 |
| vLLM r33 | 8 | 5 | 23.450 | 795.8 | 11.080 |
| SGLang v0.6.0-rc.3 | 16 | 5 | 17.863 | 1,389.6 | 14.065 |
| vLLM r33 | 16 | 5 | 15.860 | 1,097.2 | 17.590 |
| SGLang v0.6.0-rc.3 | 32 | 5 | 13.344 | 2,049.6 | 21.899 |
| vLLM r33 | 32 | 0 | not reachable: vLLM-reported KV 143,599 tok, `max_num_seqs=16` (upstream TP2 recipe) | — | — |

## DSpARK acceptance

Each SGLang value is the median of the five per-run means. Acceptance explains
synthetic output-rate variation; it is not used as an engine-speed result.

| Engine | C | Acceptance rate | Output tokens/forward/request |
|---|---:|---:|---:|
| SGLang v0.6.0-rc.3 | 1 | 0.940 | 5.614 |
| vLLM r33 | 1 | 0.588 | 3.938 |
| SGLang v0.6.0-rc.3 | 2 | 0.750 | 4.702 |
| vLLM r33 | 2 | 0.731 | 4.655 |
| SGLang v0.6.0-rc.3 | 4 | 0.769 | 4.885 |
| vLLM r33 | 4 | 0.764 | 4.821 |
| SGLang v0.6.0-rc.3 | 8 | 0.729 | 4.598 |
| vLLM r33 | 8 | 0.649 | 4.245 |
| SGLang v0.6.0-rc.3 | 16 | 0.769 | 4.837 |
| vLLM r33 | 16 | 0.672 | 4.359 |
| SGLang v0.6.0-rc.3 | 32 | 0.765 | 4.806 |
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
| 8K | 7,674.2, 8,055.2, 7,956.3, 7,988.3, 7,983.0 | 7,916.2 |
| 32K | 8,839.8, 8,890.8, 8,832.7, 8,771.8, 8,782.6 | 8,818.5 |
| 64K | 8,655.5, 8,577.4, 8,488.9, 8,380.3, 8,359.7 | 8,487.6 |
| 128K | 8,154.4, 7,998.4, 7,883.0, 7,859.1, 7,869.4 | 7,949.3 |

### Current engine table

| Engine | 8K prompt tok/s | 32K prompt tok/s | 64K prompt tok/s | 128K prompt tok/s |
|---|---:|---:|---:|---:|
| SGLang v0.6.0-rc.3 | 7,916.2 | 8,818.5 | 8,487.6 | 7,949.3 |
| vLLM r33 | 7,689.8 | 8,784.7 | 8,518.7 | 7,953.6 |

## Quality results

Each engine ran the full GSM8K set once at concurrency 16, temperature 0,
seed 42, and a 16,384-token response cap.

| Engine | Questions | Correct | Accuracy | Request errors |
|---|---:|---:|---:|---:|
| SGLang v0.6.0-rc.3 | 1,319 | 1,263 | 95.75% | 0 |
| vLLM r33 | 1,319 | 1,243 | 94.24% | 0 |

All 1,319 SGLang responses used the grader's documented last-number fallback
because the model did not emit the dataset's `####` marker. Correctness and
fallback counts are separate facts.

## Machine-readable results

- [`sglang-v0.6.0-rc.3-publication-summary.json`](bench/results/sglang-v0.6.0-rc.3-publication-summary.json)
- [`sglang-v0.6.0-rc.3-gsm8k.json`](bench/results/sglang-v0.6.0-rc.3-gsm8k.json)
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

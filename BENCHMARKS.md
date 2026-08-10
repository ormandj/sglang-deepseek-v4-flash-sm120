# RC3 and vLLM r33 measurements

Last updated: 2026-08-10 CDT.

This document contains the current SGLang `v0.1.0-rc.3` and vLLM r33
measurements. The executable benchmark is in [`bench/aiperf`](bench/aiperf).

## System and software

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs;
- SM120, TP=2 over PCIe Gen 4 x16;
- 300 W power limit per GPU;
- one engine active at a time;
- clients executed inside the serving pod against `127.0.0.1:8000`;
- model: `deepseek-ai/DeepSeek-V4-Flash-0731`;
- speculative decoding: DSpark K=5.

SGLang used the `v0.1.0-rc.3` source composition in this repository:

```text
SGLang main                 5a8e360e705fc7b8046f6b060ba4fc557ff606c7
SGLang effective tree       70fa46c3f950ff80c3bb3b9160a69ff531935dc5
FlashInfer main              4fbac49f30e1f40a0dcddd90512b8c56d68037f7
FlashInfer effective tree    616094d4a8b4a2bc94f3d43a832312c335924696
FlashInfer version           0.6.18.dev20260810
cache schema                 v2
```

vLLM used the local-inference-lab r33 image and documented DSpark
configuration:

```text
docker.io/voipmonitor/vllm@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

The r33 deployment used TP=2, DSpark K=5, `max_num_seqs=16`,
`max_num_batched_tokens=8192`, FP8 KV cache, `max_model_len=131072`, and
`gpu_memory_utilization=0.975`.

## Benchmark implementation

The client is AIPerf `0.12.0` pinned at commit:

```text
03c9c6ddc5e6227782e53ded177f1227d332af48
```

[`bench/aiperf/aiperf.lock.json`](bench/aiperf/aiperf.lock.json) records the
source and runtime pins. [`bench/aiperf/README.md`](bench/aiperf/README.md)
documents installation, execution, warmup, cache handling, metric definitions,
and keyed or keyless endpoint operation.

All panel members ran sequentially on one unchanged server process for each
engine. Each measured decode shape and prefill length was warmed once before
the measured panel began. Fixed prompts and seeds were used with temperature
0, top-p 1, and ignored EOS.

Decode used a synthetic coding prompt with a nominal 256-token input. Output
length and average-context analysis windows vary by concurrency as encoded in
[`run-engine-gate-in-pod.sh`](bench/aiperf/run-engine-gate-in-pod.sh). The
analyzers require exact request occupancy, an empty request queue, no prefill in
the decode interval, monotonic counters, and the configured context window.

Cold-prefill requests generated one output token. SGLang was flushed at each
measured cell. The vLLM analyzer required its cached-prompt-token counter to
remain zero. The common nominal near-limit input was 130,816 tokens; throughput
uses the actual engine-reported token count.

## Repetition panel

The reported campaign combines the `quick` and `decode-supplement` modes:

| C | repetitions |
|---:|---:|
| 1 | 3 |
| 2 | 6 |
| 4 | 4 |
| 8 | 2 |
| 16 | 2 |
| 32 | 1 SGLang only |

The vLLM deployment was not run at C32 because it was configured for at most
16 sequences and its KV capacity did not admit that cell.

## Decode engine rate

Forward passes per second is calculated from each engine's live decode-step
counter over the accepted context interval. The change column is the geometric
mean of matched per-seed RC3/r33 ratios minus one.

| C | vLLM r33 median forward/s | SGLang rc.3 median forward/s | rc.3 paired change |
|---:|---:|---:|---:|
| 1 | 65.622 | 59.614 | -9.28% |
| 2 | 47.801 | 46.783 | -0.65% |
| 4 | 34.072 | 34.129 | -0.35% |
| 8 | 23.862 | 22.828 | -4.34% |
| 16 | 16.618 | 17.322 | +4.24% |
| 32 | not run | 12.854 | — |

## Useful decode throughput and acceptance

Useful tokens per second is calculated from each engine's live generation-token
counter over the same interval. Accepted tokens per forward per request is the
ratio of useful-token rate to forward rate and concurrency.

| C | vLLM r33 median useful tok/s | SGLang rc.3 median useful tok/s | vLLM accepted tokens/forward/request | rc.3 accepted tokens/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 353.0 | 303.4 | 5.405 | 5.082 |
| 2 | 484.6 | 534.4 | 5.128 | 5.702 |
| 4 | 632.6 | 706.0 | 4.610 | 5.266 |
| 8 | 938.1 | 985.5 | 4.872 | 5.378 |
| 16 | 1,098.3 | 1,358.9 | 4.118 | 4.871 |
| 32 | not run | 1,890.0 | not run | 4.573 |

## Decode time to first token

Each entry is the median of the per-run AIPerf p50 TTFT values. Brackets contain
the minimum and maximum per-run p50 values.

| C | vLLM r33 median TTFT [min, max] | SGLang rc.3 median TTFT [min, max] |
|---:|---:|---:|
| 1 | 506.5 ms [501.2, 511.5] | 231.8 ms [223.7, 248.9] |
| 2 | 774.9 ms [769.1, 808.9] | 391.8 ms [379.6, 404.7] |
| 4 | 1,139.2 ms [1,128.9, 1,369.9] | 407.7 ms [399.1, 427.9] |
| 8 | 1,533.3 ms [1,255.6, 1,811.0] | 640.0 ms [447.2, 832.8] |
| 16 | 1,559.0 ms [1,555.1, 1,562.9] | 1,365.2 ms [570.9, 2,159.5] |
| 32 | not run | 965.4 ms [965.4, 965.4] |

## Cold prefill

| target | vLLM r33 prompt tok/s | SGLang rc.3 prompt tok/s | rc.3 change | vLLM median TTFT | rc.3 median TTFT |
|---:|---:|---:|---:|---:|---:|
| 8K C1 | 7,722.7 | 7,516.2 | -2.67% | 1,068.4 ms | 1,086.3 ms |
| 64K C1 | 8,647.4 | 8,343.9 | -3.51% | 7,564.6 ms | 7,828.1 ms |
| near-128K C1 | 8,001.0 | 7,780.1 | -2.76% | 16,358.0 ms | 16,811.6 ms |

The near-limit cell used the same nominal 130,816-token input for both engines.
Actual engine-reported input medians were 130,900.5 for r33 and 130,821 for
RC3.

## Running the benchmark

Stage the pinned AIPerf checkout and this repository's `bench/aiperf` directory
inside the serving pod. Run the warmup and measured panel against the pod's
localhost endpoint:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment-config revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='03c9c6ddc5e6227782e53ded177f1227d332af48' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
BENCH_ENGINE='sglang' \
./bench/aiperf/run-engine-gate-in-pod.sh rc3-vllm-r33 rc3 quick
```

Run `decode-supplement` with the same campaign and build identity for C2, C4,
and C16. Set `BENCH_ENGINE=vllm` for r33. Set `BENCH_API_KEY` when the endpoint
requires a key; omit it for a keyless endpoint. A container-provided
`VLLM_API_KEY` is also accepted.

The runner creates `summary.json`. Compare two matching summaries with:

```bash
uv run bench/aiperf/compare_engine_gates.py \
  path/to/vllm-r33/summary.json \
  path/to/sglang-rc3/summary.json \
  --allow-decode-cell-mismatch
```

The mismatch flag reports C32 as an engine-only cell instead of assigning a
value to the unmeasured vLLM cell.

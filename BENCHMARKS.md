# RC3 and vLLM r33 measurements

Last updated: 2026-08-10 CDT.

This document contains only the current SGLang `v0.1.0-rc.3` and vLLM r33
publication measurements. The executable benchmark is in
[`bench/aiperf`](bench/aiperf).

## System and software

- two NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs;
- SM120, TP=2 over PCIe Gen 4 x16;
- 300 W power limit per GPU;
- one engine active at a time;
- clients executed inside the serving pod against `127.0.0.1:8000`;
- model: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- speculative decoding: DSpark K=5.

SGLang used the `v0.1.0-rc.3` source composition in this repository:

```text
SGLang main                  5a8e360e705fc7b8046f6b060ba4fc557ff606c7
SGLang effective tree        70fa46c3f950ff80c3bb3b9160a69ff531935dc5
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

The benchmark method used project harness revision:

```text
24a33782e1f894e6e4ae2c4f9d01a394b1027663
```

This public repository contains the same executable harness. Its
[`aiperf.lock.json`](bench/aiperf/aiperf.lock.json) records the source and
runtime pins. [`bench/aiperf/README.md`](bench/aiperf/README.md) documents
installation, execution, warmup, cache handling, metric definitions, and keyed
or keyless endpoint operation.

## Publication method

Each engine used one fresh server process. The process was not restarted or
reconfigured while its panel ran. All clients ran inside the selected serving
pod against localhost, and panel members ran sequentially.

The `publication` mode is fixed as follows:

| workload | SGLang rc.3 | vLLM r33 |
|---|---:|---:|
| decode C1 | 5 repetitions | 5 repetitions |
| decode C2 | 5 repetitions | 5 repetitions |
| decode C4 | 5 repetitions | 5 repetitions |
| decode C8 | 5 repetitions | 5 repetitions |
| decode C16 | 5 repetitions | 5 repetitions |
| decode C32 | 5 repetitions | not measured |
| cold prefill 8K C1 | 5 requests | 5 requests |
| cold prefill 64K C1 | 5 requests | 5 requests |
| cold prefill near-128K C1 | 5 requests | 5 requests |

vLLM r33 was not measured at C32 because that deployment cannot admit C32. No
value is assigned to the unmeasured cell.

Before measurement, the runner warms every supported decode concurrency and
each prefill length once. It then records all decode cells and all prefill cells
sequentially on the unchanged process. A publication table is generated from
one fresh `publication` campaign per engine; quick, supplement,
qualification, and earlier publication cells are not combined with it.

Decode uses fixed synthetic coding prompts and five fixed seeds, temperature
0, top-p 1, ignored EOS, a nominal 256-token input, exact occupancy, and an
empty request queue. Output lengths and equal-context analysis windows vary by
concurrency as encoded in
[`run-engine-gate-in-pod.sh`](bench/aiperf/run-engine-gate-in-pod.sh).
The analyzers reject wrong occupancy, queued work, prefill in the decode
interval, counter resets, and an insufficient analysis window.

Cold prefill generates one output token per request. Every request is
cache-busted. SGLang is flushed at each measured cell. The vLLM analyzer
requires its cached-prompt-token counter to remain zero. The common nominal
near-limit input is 130,816 tokens; throughput uses the actual engine-reported
input count.

## Decode engine rate

Forward passes per second is calculated from each engine's live decode-step
counter over the accepted context interval. Each value is the median of five
repetitions.

| C | vLLM r33 median forward/s | SGLang rc.3 median forward/s |
|---:|---:|---:|
| 1 | 65.410 | 60.013 |
| 2 | 46.940 | 46.479 |
| 4 | 33.637 | 32.887 |
| 8 | 23.914 | 23.329 |
| 16 | 16.488 | 17.444 |
| 32 | not measured | 12.989 |

## Useful decode throughput and acceptance

Useful tokens per second is calculated from each engine's live
generation-token counter over the same interval. Useful tokens per forward per
request is the ratio of useful-token rate to forward rate and concurrency.
Each value is the median of five repetitions.

| C | vLLM r33 median useful tok/s | SGLang rc.3 median useful tok/s | vLLM useful tokens/forward/request | rc.3 useful tokens/forward/request |
|---:|---:|---:|---:|---:|
| 1 | 355.7 | 326.8 | 5.179 | 5.331 |
| 2 | 511.2 | 498.3 | 5.404 | 5.492 |
| 4 | 678.6 | 687.5 | 4.937 | 5.175 |
| 8 | 863.1 | 956.5 | 4.401 | 5.067 |
| 16 | 1,122.5 | 1,410.6 | 4.189 | 4.993 |
| 32 | not measured | 1,952.9 | not measured | 4.690 |

## Decode time to first token

Each entry is the median of five per-run AIPerf p50 TTFT values. Brackets
contain the minimum and maximum per-run p50 values.

| C | vLLM r33 median TTFT [min, max] | SGLang rc.3 median TTFT [min, max] |
|---:|---:|---:|
| 1 | 502.4 ms [497.8, 504.1] | 239.6 ms [230.3, 240.9] |
| 2 | 784.1 ms [779.7, 797.5] | 402.1 ms [386.3, 403.5] |
| 4 | 1,148.2 ms [1,119.8, 1,379.5] | 379.1 ms [361.8, 400.2] |
| 8 | 1,291.7 ms [1,268.1, 1,787.6] | 371.2 ms [350.1, 412.6] |
| 16 | 1,559.3 ms [1,527.1, 1,595.5] | 466.4 ms [460.2, 474.8] |
| 32 | not measured | 1,050.6 ms [706.3, 2,656.5] |

## Cold prefill

Each length used five requests.

| target | vLLM r33 prompt tok/s | SGLang rc.3 prompt tok/s | vLLM median TTFT | rc.3 median TTFT |
|---:|---:|---:|---:|---:|
| 8K C1 | 7,699.7 | 7,479.6 | 1,072.1 ms | 1,087.1 ms |
| 64K C1 | 8,572.4 | 8,264.6 | 7,649.8 ms | 7,905.2 ms |
| near-128K C1 | 7,941.9 | 7,758.4 | 16,551.1 ms | 16,996.9 ms |

The common nominal near-limit input was 130,816 tokens. Actual
engine-reported input medians were 130,899 for r33 and 130,821 for RC3.

## Running the benchmark

Stage the pinned AIPerf checkout and this repository's `bench/aiperf`
directory inside the serving pod. Use one fresh, otherwise idle server process
per engine and run against the pod's localhost endpoint:

```bash
BENCH_IMAGE_REF='image@sha256:...' \
BENCH_GITOPS_REVISION='<deployment-config revision>' \
BENCH_PROJECT_REVISION='<this repository revision>' \
AIPERF_REVISION='03c9c6ddc5e6227782e53ded177f1227d332af48' \
BENCH_MODEL_REVISION='<model snapshot revision>' \
BENCH_ENGINE='sglang' \
./bench/aiperf/run-engine-gate-in-pod.sh rc3-vllm-r33 rc3 publication
```

Set `BENCH_ENGINE=vllm` and use a distinct build ID for r33. Set
`BENCH_API_KEY` when the endpoint requires a key; omit it for a keyless
endpoint. A container-provided `VLLM_API_KEY` is also accepted. Neither key is
written to benchmark output.

The runner rejects execution outside Kubernetes or against a non-loopback
endpoint. It also rejects an existing output directory, incomplete request
sets, and any publication cell that does not contain exactly five
repetitions or five cold-prefill requests.

Compare the two completed summaries with:

```bash
uv run bench/aiperf/compare_engine_gates.py \
  path/to/vllm-r33/summary.json \
  path/to/sglang-rc3/summary.json \
  --allow-decode-cell-mismatch
```

The mismatch flag reports C32 as an engine-only cell instead of assigning a
value to vLLM.

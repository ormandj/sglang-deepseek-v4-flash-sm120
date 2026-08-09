# Reproducible RC2 versus vLLM r33 benchmark

This is the canonical cross-engine performance record for this repository. It
replaces every table from older SGLang images and older vLLM images.

## Scope

The measurements were collected on 2026-08-09 on one host, one engine at a
time, using the same model snapshot and the same two NVIDIA RTX PRO 6000
Blackwell Max-Q GPUs:

- compute capability: SM120;
- topology: TP=2 over PCIe Gen 4 x16;
- power limit: 300 W per GPU;
- model: `deepseek-ai/DeepSeek-V4-Flash-0731` revision
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`;
- speculative decoding: DSpark K=5 on both engines.

The SGLang side is release `v0.1.0-rc.2` from this repository's exact
`release.json`, `stack.lock.json`, and patch set. The corresponding public image
is:

```text
ghcr.io/ormandj/sglang-deepseek-v4-flash-sm120:v0.1.0-rc.2@sha256:71766e76fd1ffd2bcac4ba79cbeb2317b326144558fc2e7bb49e2bbea05e8cbb
```

The tested private mirror reported the same release identity recorded by the
public lock:

```text
release                    v0.1.0-rc.2
SGLang main                dc9624deb2f03ebe5e52bd03337addf91386c041
SGLang effective tree      cee50a9f372b26b64eec189b861b16f2c23c3244
FlashInfer main             29196cf437778906c72630dc5d9850de547501de
FlashInfer effective tree   a089f6c4beaf103306775014a7a3d42eed1214c5
cache schema                v2
```

Its non-secret server arguments and tuning environment matched
[`examples/serve-dsv4-0731.sh`](examples/serve-dsv4-0731.sh), including the
DeepGEMM indexer selection, fused MHC pre-norm selection, 8,192-token prefill
chunk, FP8 KV cache, memory fraction, graph sizes, DSpark K=5, and TP=2. The
private registry address and deployment-specific authentication are not part of
the published configuration.

The vLLM side is the local-inference-lab r33 configuration documented in
[`ds4dspark-v20-r33.md`](https://github.com/local-inference-lab/rtx6kpro/blob/3ca3c00c2b23273ed3df5ece5fdf8fd20c5619d8/models/ds4dspark-v20-r33.md):

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33@sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

Its relevant runtime settings were TP=2, DSpark K=5, B12x W4A8 MoE,
`max_num_seqs=16`, `max_num_batched_tokens=8192`, FP8 KV cache,
`max_model_len=131072`, and `gpu_memory_utilization=0.975`.

No other workload ran on these GPUs during an accepted measurement. Every
decode cell reported the requested effective concurrency, zero average queue,
zero errors, no admission timeout, and no capacity-limit flag.

Both engines mounted the same read-only model directory. Small-file hashes from
that snapshot are:

```text
config.json                    6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023
generation_config.json         5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0
model.safetensors.index.json   98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
tokenizer.json                 8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf
```

## Benchmark source

The client is the unmodified public
[`local-inference-lab/llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench/blob/86cf05c2f42f4d21b909b6e684424ca1aab89fd5/llm_decode_bench.py)
`main` branch at:

```text
commit  86cf05c2f42f4d21b909b6e684424ca1aab89fd5
file    llm_decode_bench.py
sha256  fa227030012a8b55545af6b6a50fa4adcbdff8d003bb16469dc8e2de024ed0c0
version 0.4.29
```

This repository does not carry a modified benchmark. The raw reports in
[`benchmarks/2026-08-09`](benchmarks/2026-08-09) were produced by that exact
file.

## Sampling and request behavior

- Decode used the benchmark's built-in encyclopedia prompt.
- `temperature=1.0` was explicit.
- Sustained decode does not expose a general top-p option. The request omitted
  `top_p`; the model's `generation_config.json` and both engines resolve it to
  `top_p=1.0`.
- `--max-tokens` was not passed. The pinned source's decode default is 8,192.
- `--respect-eos` was not passed, so the benchmark kept its default
  `ignore_eos` behavior for sustained measurement.
- Decode used a 20-second measured window for every cell.
- Prefill used exact tokenizer targeting at 8,192, 65,536, and 131,008 input
  tokens and the benchmark's client metric: reported prompt tokens divided by
  TTFT.
- Both clients ran inside the selected serving container against `localhost`.
  Authentication, when enabled by the server, changes only the request header
  and is omitted from the published command examples.

The report shows 8,194, 65,538, and 131,010 prompt tokens because the
OpenAI-compatible chat request adds two fixed control tokens after exact input
text calibration. Both engines used the same requested targets and the same
reported-token formula.

## Warmup and repetition policy

Warmup is mandatory:

1. After each engine reached readiness, one complete C=1,2,4,8,16 decode sweep
   ran at three seconds per cell and was discarded.
2. One exact 8K/64K/131K prefill sweep ran with a one-second duration and was
   discarded.
3. Every measured decode repetition retained the benchmark's hidden C=1
   three-second warmup and its per-cell three-second admission-stability gate.
   The latter also warmed SGLang's C=32 cell before its timed window.
4. Five sequential measured decode repetitions and five sequential measured
   prefill repetitions were retained for each engine.

The median is computed across the five per-run values. Prefill samples within a
run are first reduced by the benchmark; the published n=5 median is not a pooled
sample median.

For future campaigns, the discarded decode sweep must include every supported
cell, including SGLang C=32. The hidden and admission-stability warmups remain
required as well.

## Exact commands

The SGLang decode command was:

```bash
uv run --quiet --no-project python -B llm_decode_bench.py \
  --host localhost --port 8000 \
  --model deepseek-v4-flash \
  --temperature 1.0 \
  --concurrency 1,2,4,8,16,32 \
  --contexts 0 \
  --duration 20 \
  --skip-prefill \
  --display-mode plain \
  --output run-N/decode.json
```

The vLLM command was identical except for its supported concurrency set and a
client-side token-budget override:

```bash
uv run --quiet --no-project python -B llm_decode_bench.py \
  --host localhost --port 8000 \
  --model deepseek-v4-flash \
  --temperature 1.0 \
  --concurrency 1,2,4,8,16 \
  --contexts 0 \
  --duration 20 \
  --max-total-tokens 2000000 \
  --skip-prefill \
  --display-mode plain \
  --output run-N/decode.json
```

`--max-total-tokens` is only the benchmark client's static cell-skip guard. It
is not sent in the generation request and does not alter vLLM's KV cache,
server configuration, admission, or timing. At the pinned benchmark revision,
automatic budget inspection was too conservative and skipped valid
zero-context cells before sending a request. With the client guard overridden,
vLLM admitted C=1 through C=16 with no queue. For consistency, future
zero-context comparisons will pass this same client-only value to both engines;
it has no effect on cells that the server admits.

The measured prefill command was the same for both engines:

```bash
uv run --quiet --no-project python -B llm_decode_bench.py \
  --host localhost --port 8000 \
  --model deepseek-v4-flash \
  --concurrency 1 \
  --contexts 0 \
  --prefill-only \
  --standalone-prefill \
  --prefill-contexts 8k,64k,131008 \
  --token-targeting exact \
  --prefill-duration 10 \
  --display-mode plain \
  --output run-N/prefill.json
```

An authenticated server additionally needs its normal `--api-key` argument;
the key is not part of this repository or the published reports.

## Sustained decode results

Aggregate throughput is OpenAI continuous stream-usage completion tokens per
measured window. TTFT is the median of the five per-run p50 TTFT values.

| C | SGLang rc.2 tok/s | vLLM r33 tok/s | SGLang vs vLLM | SGLang p50 TTFT | vLLM p50 TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 178.4 | 185.5 | -3.8% | 0.190 s | 0.495 s |
| 2 | 275.9 | 271.3 | +1.7% | 0.375 s | 0.772 s |
| 4 | 405.8 | 392.1 | +3.5% | 0.366 s | 1.122 s |
| 8 | 566.2 | 573.7 | -1.3% | 0.382 s | 1.287 s |
| 16 | 846.1 | 814.2 | +3.9% | 0.399 s | 1.548 s |
| 32 | 1,234.9 | — | — | 0.447 s | — |

vLLM r33 was intentionally limited to C=16. Its deployment is configured for
16 sequences and its KV envelope does not support the C=32 cell; the blank is
not a zero.

### Per-run aggregate tok/s

| engine | run | C1 | C2 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SGLang rc.2 | 1 | 177.4 | 266.3 | 415.9 | 542.5 | 847.0 | 1,245.0 |
| SGLang rc.2 | 2 | 178.4 | 275.9 | 399.3 | 566.2 | 846.1 | 1,234.7 |
| SGLang rc.2 | 3 | 185.8 | 283.2 | 399.9 | 580.8 | 840.2 | 1,234.9 |
| SGLang rc.2 | 4 | 173.6 | 285.1 | 408.6 | 551.8 | 836.4 | 1,211.4 |
| SGLang rc.2 | 5 | 193.6 | 275.3 | 405.8 | 573.1 | 862.4 | 1,239.7 |
| vLLM r33 | 1 | 183.9 | 268.0 | 389.1 | 578.9 | 796.3 | — |
| vLLM r33 | 2 | 181.0 | 279.4 | 406.6 | 585.9 | 824.1 | — |
| vLLM r33 | 3 | 189.1 | 271.3 | 390.0 | 565.7 | 814.4 | — |
| vLLM r33 | 4 | 187.2 | 275.6 | 409.0 | 573.7 | 787.1 | — |
| vLLM r33 | 5 | 185.5 | 256.6 | 392.1 | 569.8 | 814.2 | — |

## Exact cold-prefill results

| target | reported prompt tokens | SGLang rc.2 tok/s | vLLM r33 tok/s | SGLang vs vLLM | SGLang TTFT | vLLM TTFT |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,194 | 7,395 | 7,659 | -3.4% | 1.108 s | 1.070 s |
| 65,536 | 65,538 | 8,367 | 8,755 | -4.4% | 7.833 s | 7.486 s |
| 131,008 | 131,010 | 7,797 | 8,062 | -3.3% | 16.802 s | 16.250 s |

### Per-run client tok/s

| engine | run | 8K | 64K | 131K |
|---|---:|---:|---:|---:|
| SGLang rc.2 | 1 | 6,938 | 8,389 | 7,849 |
| SGLang rc.2 | 2 | 7,396 | 8,320 | 7,776 |
| SGLang rc.2 | 3 | 7,260 | 8,344 | 7,782 |
| SGLang rc.2 | 4 | 7,456 | 8,376 | 7,821 |
| SGLang rc.2 | 5 | 7,395 | 8,367 | 7,797 |
| vLLM r33 | 1 | 7,595 | 8,755 | 8,047 |
| vLLM r33 | 2 | 7,703 | 8,752 | 8,032 |
| vLLM r33 | 3 | 7,683 | 8,875 | 8,194 |
| vLLM r33 | 4 | 7,659 | 8,731 | 8,062 |
| vLLM r33 | 5 | 7,450 | 8,772 | 8,064 |

## What the image changes

RC2 is not a stock SGLang image or a launch-script-only change. The complete
source composition is in [`stack.lock.json`](stack.lock.json), and the
remaining patch is reproduced and tree-checked by
[`scripts/verify-patches.sh`](scripts/verify-patches.sh). The runtime changes
required by this validated composition include:

- the SM120 DeepSeek-V4 stack supplies the FP4 MoE, batched sparse-MLA prefill,
  and DeepGEMM paged-MQA indexer paths;
- the DSpark draft keeps separate shared-expert modules after upstream made
  fusion decisions runner-local; without the gate, the draft creates an extra
  fused expert slot and rejects the bundled shared-expert weights;
- the compressed-state planner rewrites the actual DSpark verifier width rather
  than a fixed four-row window;
- the SM120-capable DeepGEMM indexer is selected instead of the forced TileLang
  fallback;
- the DeepGEMM fused MHC pre-norm path is enabled on SM120 instead of the eager
  float32 fallback;
- all-reduce workspace sizing and dispatch are bounded so prefill-sized
  collectives stay off the latency-optimized decode path;
- sliding-window KV eviction runs on the DFLASH/DSPARK speculative path so long
  generations do not exhaust the smaller SWA pool while full-attention KV is
  still available;
- the image uses cache schema `v2`, separated from earlier image caches.

The validated serving flags are in
[`examples/serve-dsv4-0731.sh`](examples/serve-dsv4-0731.sh). No benchmark-only
server flag was used.

## Published raw reports

The 20 measured JSON reports are under
[`benchmarks/2026-08-09`](benchmarks/2026-08-09). Upstream v0.4.29 records the
entire server environment in `startup_diagnostics`, which can include API keys.
Before publication we removed only:

- `startup_diagnostics.env`;
- `startup_diagnostics.hostname`;
- `startup_diagnostics.uname`.

The exact filter is [`sanitize.jq`](benchmarks/2026-08-09/sanitize.jq). No
result, request sample, timing, hardware summary, engine version, methodology,
or event-log field was changed. [`SHA256SUMS`](benchmarks/2026-08-09/SHA256SUMS)
pins every published report and the filter itself.

Verify the published bytes from the repository root:

```bash
sha256sum --check benchmarks/2026-08-09/SHA256SUMS
```

Recompute the decode medians for either `sglang` or `vllm-r33` directly from
the five reports:

```bash
engine=sglang
jq -s '
  [range(0; (.[0].results | length)) as $i
   | {concurrency: .[0].results[$i].concurrency,
      aggregate_tps_median:
        ([.[].results[$i].aggregate_tps] | sort | .[2]),
      ttft_p50_median:
        ([.[].results[$i].ttft_p50] | sort | .[2])}]
' benchmarks/2026-08-09/$engine/run-*-decode.json
```

Recompute its exact-prefill medians the same way:

```bash
engine=sglang
jq -s '
  ["8192", "65536", "131008"] as $targets
  | [$targets[] as $target
     | {target: ($target | tonumber),
        prompt_tokens: .[0].prefill[$target].prompt_tokens,
        client_tps_median:
          ([.[].prefill[$target].client_tok_per_sec] | sort | .[2]),
        ttft_median:
          ([.[].prefill[$target].client_ttft_seconds] | sort | .[2])}]
' benchmarks/2026-08-09/$engine/run-*-prefill.json
```

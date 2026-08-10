# Historical paired-block calibration: 2026-08-10

This preserves the calibration for the independent paired-block experiment.
That experiment is no longer the routine engine-performance gate; do not reuse
its sample counts or stochastic single-trajectory decode cells for per-change
decisions. The current design and runner are in `README.md` and
`run-engine-gate-in-pod.sh`.

The observations below remain useful provenance for the retained artifacts.
They are single-build calibration values, not RC3 performance claims.

## Provenance

- AIPerf: version `0.12.0`, commit
  `03c9c6ddc5e6227782e53ded177f1227d332af48`.
- image: private RC3 manifest
  `sha256:3ff7ff01bd2bcbbae4f74136511e9dc6e590847adb98a7474a62e06004ee1684`;
- SGLang main: `5a8e360e705fc7b8046f6b060ba4fc557ff606c7`;
- SGLang effective tree: `70fa46c3f950ff80c3bb3b9160a69ff531935dc5`;
- FlashInfer main: `4fbac49f30e1f40a0dcddd90512b8c56d68037f7`;
- FlashInfer effective tree: `616094d4a8b4a2bc94f3d43a832312c335924696`;
- client placement: inside the serving pod against `127.0.0.1:8000`;
- private artifact root:
  `/models/bench/results/aiperf-greenfield/calibration-20260810-clean/`.

All tables below were read from the retained analyzer JSON. Each listed cell
passed its analyzer validity checks.

## AIPerf worker count

C32 temperature-zero request-bounded cohorts were run with one, two, and four
AIPerf workers. All three held exact occupancy for 100% of the analyzed window,
had an empty server queue, and admitted no new prefill during the plateau.

| workers | decode OLS tok/s | plateau |
|---:|---:|---:|
| 1 | 2,072.074 | 38.369 s |
| 2 | 1,964.612 | 37.026 s |
| 4 | 1,996.970 | 35.359 s |

One worker is the smallest tested count and produced the maximum measured
rate. Additional client workers did not increase server throughput. The paired
campaign therefore fixes `AIPERF_WORKERS=1` and
`AIPERF_RECORD_PROCESSORS=1`.

Sources are the `plateau.json` files in
`rc3-decode-c32-temp0-w{1,2,4}-calibration/`.

## Decode cohort lengths

The production-shaped calibration explicitly set temperature 1.0 and top-p
0.95. Each concurrency uses one stream per lane and the output lengths below.
After a 15-second settle exclusion and three-second terminal tail exclusion,
every cell supplied more than the required 30-second exact-occupancy plateau.

| C | output tokens per request | OLS tok/s | delta tok/s | plateau | mean acceptance | mean accepted length |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18,432 | 301.390 | 300.047 | 45.234 s | 81.982% | 5.099 |
| 2 | 16,384 | 471.611 | 473.470 | 51.028 s | 81.153% | 5.058 |
| 4 | 12,288 | 777.832 | 777.224 | 43.357 s | 90.626% | 5.531 |
| 8 | 8,192 | 871.442 | 859.346 | 57.711 s | 75.079% | 4.754 |
| 16 | 6,144 | 1,417.442 | 1,414.214 | 45.699 s | 76.989% | 4.849 |
| 32 | 4,096 | 1,840.449 | 1,833.508 | 43.697 s | 67.983% | 4.399 |

Every row had exact occupancy for the full analyzed window, an empty queue,
and unchanged prefill-compute and prefill-cache counters. The variation in
acceptance across this one seed is one reason the final design pairs committed,
distinct sampling seeds and uses at least five final pairs rather than treating
this curve as a release comparison.

Sources are `decode-analysis.json` in the corresponding
`rc3-decode-c*-production-pilot` directories; C1 uses the `-v2` directory and
C4 uses `rc3-decode-c4-w1-calibration`.

## Cold-prefill duration

Cold-prefill calibration uses temperature zero, one forced output token, an
explicit cache flush, a unique first-turn prefix, and server-reported token
counts. The displayed input length is the configured synthetic target; the
analyzer retains actual request token counts and permits only the documented
chat-template tolerance.

| target | C | calibration requests | aggregate prompt tok/s | median TTFT | p90 TTFT | time at target C |
|---:|---:|---:|---:|---:|---:|---:|
| 8,192 | 1 | 20 | 7,634.902 | 1,061.139 ms | 1,084.183 ms | 99.831% |
| 8,192 | 2 | 20 | 8,837.115 | 1,846.482 ms | 1,876.642 ms | 99.734% |
| 8,192 | 4 | 20 | 9,628.379 | 3,273.123 ms | 3,455.531 ms | 91.728% |
| 65,536 | 1 | 10 | 8,130.351 | 7,835.101 ms | 8,210.118 ms | 99.957% |
| 131,072 | 1 | 5 | 7,759.324 | 16,630.109 ms | 17,945.000 ms | 99.979% |

The measured block fixes 40 requests for each 8K cell, 10 for 64K C1, and five
for 128K C1. At the calibrated rates those counts give approximately 34--43
seconds for each 8K phase and approximately 80--84 seconds for each long-input
phase. The analyzer requires the exact completion count, one output token,
target concurrency, zero cache-attributed tokens, and bounded agreement between
server prefill-compute tokens and observed request tokens.

Sources are `prefill-analysis.json` in
`rc3-prefill-{8k-c1-pilot20,8k-c2-pilot20-v2,8k-c4-pilot20,64k-c1-pilot10,128k-c1-pilot5}/`.

## Frozen block shape

Each fresh-process block performs coverage warmup for decode C1/C2/C4/C8/C16/
C32 and prefill 8K C1/C2/C4, 64K C1, and 128K C1. It then runs the six measured
decode cells and five measured prefill cells above. The executable definition
is `run-performance-block-in-pod.sh`; its manifest hashes the three configs,
seed panel, and AIPerf lock before measurement.

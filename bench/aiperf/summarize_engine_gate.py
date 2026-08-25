#!/usr/bin/env python3
"""Validate and summarize one same-process engine-performance gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class SummaryError(RuntimeError):
    """Raised when an engine-gate artifact set is incomplete or invalid."""


EXPECTED_DECODE = {
    "sglang": {
        "exploratory-decode": {1: 3, 2: 3, 4: 3, 8: 3},
        "quick": {1: 3, 4: 3, 8: 3},
        "prefill-quick": {},
        "decode-supplement": {2: 3, 16: 3},
        "repeat-c2-c4": {2: 5, 4: 5},
        "repeat-c8": {8: 5},
        "qualification": {1: 5, 2: 5, 4: 5, 8: 5, 16: 3, 32: 3},
        "publication": {1: 5, 2: 5, 4: 5, 8: 5, 16: 5, 32: 5},
    },
    "vllm": {
        "exploratory-decode": {1: 3, 2: 3, 4: 3, 8: 3},
        "quick": {1: 3, 4: 3, 8: 3},
        "prefill-quick": {},
        "decode-supplement": {2: 3, 16: 3},
        "repeat-c2-c4": {2: 5, 4: 5},
        "repeat-c8": {8: 5},
        "qualification": {1: 5, 2: 5, 4: 5, 8: 5, 16: 3},
        "publication": {1: 5, 2: 5, 4: 5, 8: 5, 16: 5},
    },
}
EXPECTED_PREFILL = {
    "exploratory-decode": {},
    "quick": {"8k-c1": 3, "32k-c1": 3, "64k-c1": 3, "128k-c1": 3},
    "prefill-quick": {"8k-c1": 3, "32k-c1": 3, "64k-c1": 3, "128k-c1": 3},
    "decode-supplement": {},
    "repeat-c2-c4": {},
    "repeat-c8": {},
    "qualification": {
        "8k-c1": 5,
        "32k-c1": 5,
        "64k-c1": 5,
        "128k-c1": 5,
    },
    "publication": {"8k-c1": 5, "32k-c1": 5, "64k-c1": 5, "128k-c1": 5},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(EXPECTED_PREFILL), required=True)
    parser.add_argument("--engine", choices=tuple(EXPECTED_DECODE), default="sglang")
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise SummaryError(f"{path} does not contain a JSON object")
    return document


def _finite_positive(value: Any, source: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{source} is not numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise SummaryError(f"{source} is not positive and finite")
    return numeric


def _finite_unit_interval(value: Any, source: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{source} is not numeric") from exc
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise SummaryError(f"{source} is not finite and within [0, 1]")
    return numeric


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    mean = statistics.fmean(values)
    sample_stddev = statistics.stdev(values) if len(values) > 1 else None
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "max": max(values),
        "sample_stddev": sample_stddev,
        "sample_cv": sample_stddev / mean if sample_stddev is not None else None,
    }


def _client_latency_metric(
    document: dict[str, Any], metric: str, source: str
) -> dict[str, float]:
    value = document.get(metric, {})
    if value.get("unit") != "ms":
        raise SummaryError(f"{source} {metric} does not use milliseconds")
    return {
        statistic: _finite_positive(
            value.get(statistic), f"{source} {metric} {statistic}"
        )
        for statistic in ("avg", "p50", "p90", "p99")
    }


def summarize(
    root: Path, *, mode: str, build_id: str, engine: str = "sglang"
) -> dict[str, Any]:
    expected_decode = EXPECTED_DECODE[engine][mode]
    decode_documents: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.glob("decode/c*/r*/decode-analysis.json")):
        try:
            concurrency = int(path.parents[1].name.removeprefix("c"))
        except ValueError as exc:
            raise SummaryError(f"invalid decode directory: {path}") from exc
        decode_documents[concurrency].append((path.parent.name, _load(path)))

    if set(decode_documents) != set(expected_decode):
        raise SummaryError(
            f"decode cells are {sorted(decode_documents)}; "
            f"expected {sorted(expected_decode)}"
        )

    decode: dict[str, Any] = {}
    for concurrency, expected_repetitions in expected_decode.items():
        documents = decode_documents[concurrency]
        if len(documents) != expected_repetitions:
            raise SummaryError(
                f"C{concurrency} has {len(documents)} repetitions; "
                f"expected {expected_repetitions}"
            )
        forward_rates: list[float] = []
        token_rates: list[float] = []
        accepted_lengths: list[float] = []
        speculative_values: dict[str, dict[str, list[float]]] = {
            metric: {statistic: [] for statistic in ("mean", "median", "min", "max")}
            for metric in ("accept_rate", "accept_length")
        }
        latency_values: dict[str, dict[str, list[float]]] = {
            metric: {statistic: [] for statistic in ("avg", "p50", "p90", "p99")}
            for metric in (
                "time_to_first_token",
                "inter_token_latency",
                "request_latency",
            )
        }
        repetitions: list[dict[str, Any]] = []
        for repetition, document in documents:
            if document.get("validation", {}).get("valid") is not True:
                raise SummaryError(f"C{concurrency}/{repetition} is invalid")
            if document.get("decode", {}).get("target_concurrency") != concurrency:
                raise SummaryError(f"C{concurrency}/{repetition} has the wrong shape")
            forward_rate = _finite_positive(
                document.get("engine_work", {}).get("forward_passes_per_second_ols"),
                f"C{concurrency}/{repetition} forward rate",
            )
            token_rate = _finite_positive(
                document.get("decode", {}).get("tokens_per_second_ols"),
                f"C{concurrency}/{repetition} token rate",
            )
            accepted_length = _finite_positive(
                document.get("engine_work", {}).get(
                    "useful_tokens_per_forward_per_request"
                ),
                f"C{concurrency}/{repetition} accepted length",
            )
            forward_rates.append(forward_rate)
            token_rates.append(token_rate)
            accepted_lengths.append(accepted_length)
            server_cross_checks = document.get("server_cross_checks", {})
            speculative = {}
            for metric, validator in (
                ("accept_rate", _finite_unit_interval),
                ("accept_length", _finite_positive),
            ):
                metric_document = server_cross_checks.get(f"spec_{metric}", {})
                speculative[metric] = {
                    statistic: validator(
                        metric_document.get(statistic),
                        f"C{concurrency}/{repetition} speculative {metric} {statistic}",
                    )
                    for statistic in speculative_values[metric]
                }
                for statistic, value in speculative[metric].items():
                    speculative_values[metric][statistic].append(value)
            client_document = _load(
                root
                / "decode"
                / f"c{concurrency}"
                / repetition
                / "profile_export_aiperf.json"
            )
            client_latency = {
                metric: _client_latency_metric(
                    client_document, metric, f"C{concurrency}/{repetition}"
                )
                for metric in latency_values
            }
            for metric, statistics_by_name in client_latency.items():
                for statistic, value in statistics_by_name.items():
                    latency_values[metric][statistic].append(value)
            repetitions.append(
                {
                    "id": repetition,
                    "forward_passes_per_second": forward_rate,
                    "synthetic_decode_tokens_per_second": token_rate,
                    "output_tokens_per_forward_per_request": accepted_length,
                    "speculative": speculative,
                    "client_latency_ms": client_latency,
                }
            )
        decode[f"c{concurrency}"] = {
            "repetitions": repetitions,
            "engine_forward_passes_per_second": _distribution(forward_rates),
            "synthetic_decode_tokens_per_second": _distribution(token_rates),
            "output_tokens_per_forward_per_request": _distribution(accepted_lengths),
            "speculative": {
                metric: {
                    f"{statistic}_per_run": _distribution(values)
                    for statistic, values in statistics_by_name.items()
                }
                for metric, statistics_by_name in speculative_values.items()
            },
            "client_latency_ms": {
                metric: {
                    f"{statistic}_per_run": _distribution(values)
                    for statistic, values in statistics_by_name.items()
                }
                for metric, statistics_by_name in latency_values.items()
            },
        }

    prefill_paths = sorted(root.glob("prefill/*/prefill-analysis.json"))
    labels = {path.parent.name for path in prefill_paths}
    expected_prefill = EXPECTED_PREFILL[mode]
    if labels != set(expected_prefill):
        raise SummaryError(
            f"prefill cells are {sorted(labels)}; "
            f"expected {sorted(expected_prefill)}"
        )
    prefill: dict[str, Any] = {}
    for path in prefill_paths:
        document = _load(path)
        if document.get("validation", {}).get("valid") is not True:
            raise SummaryError(f"{path.parent.name} is invalid")
        requests = document.get("requests", {})
        completed = requests.get("completed")
        expected_requests = expected_prefill[path.parent.name]
        if completed != expected_requests:
            raise SummaryError(
                f"{path.parent.name} has {completed} completed requests; "
                f"expected {expected_requests}"
            )
        prefill[path.parent.name] = {
            "prompt_tokens_per_second": _finite_positive(
                requests.get("aggregate_prompt_tokens_per_second"),
                f"{path.parent.name} prefill rate",
            ),
            "median_ttft_ms": _finite_positive(
                requests.get("time_to_first_token_ms", {}).get("median"),
                f"{path.parent.name} TTFT",
            ),
            "requests": completed,
        }

    return {
        "schema_version": "1.2",
        "engine": engine,
        "build_id": build_id,
        "mode": mode,
        "interpretation": (
            "same-process engineering regression signal; synthetic fixed-window "
            "output rate is not expected production, interactive, or application "
            "throughput and includes path-dependent speculative acceptance; "
            "repetitions are prompt-path subsamples, not independent deployment "
            "replicates"
        ),
        "decode": decode,
        "prefill": prefill,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = summarize(
            args.root, mode=args.mode, build_id=args.build_id, engine=args.engine
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise SummaryError(f"output already exists: {args.output}")
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (SummaryError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

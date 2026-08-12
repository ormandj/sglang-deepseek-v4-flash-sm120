#!/usr/bin/env python3
"""Validate and summarize a cold-prefill AIPerf cell.

The primary observations come from AIPerf's per-request records.  Live SGLang
or vLLM prompt counters are independent controls: cache work must remain zero,
while compute work must account for the requested prompts without a large
amount of unrelated prefill traffic.
"""

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

NANOSECONDS = 1_000_000_000
MILLISECONDS_TO_NANOSECONDS = 1_000_000


class AnalysisError(RuntimeError):
    """Raised when required evidence is missing or malformed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--server-summary", type=Path, required=True)
    parser.add_argument("--target-isl", type=int, required=True)
    parser.add_argument("--target-concurrency", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--engine", choices=("sglang", "vllm"), default="sglang")
    parser.add_argument("--isl-tolerance", type=int, default=16)
    parser.add_argument("--maximum-compute-ratio", type=float, default=1.10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise AnalysisError(f"{path} does not contain a JSON object")
    return document


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record["metadata"], dict):
                    raise TypeError("metadata is not an object")
                if not isinstance(record["metrics"], dict):
                    raise TypeError("metrics is not an object")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise AnalysisError(f"invalid record at {path}:{line_number}") from exc
            records.append(record)
    return records


def _metric(record: dict[str, Any], name: str, unit: str) -> float:
    try:
        metric = record["metrics"][name]
        value = float(metric["value"])
        actual_unit = str(metric["unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"record has no valid {name} metric") from exc
    if actual_unit != unit:
        raise AnalysisError(f"{name} uses {actual_unit!r}, expected {unit!r}")
    if not math.isfinite(value):
        raise AnalysisError(f"{name} is not finite")
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _summary_metric(document: dict[str, Any], name: str, unit: str) -> float:
    try:
        metric = document[name]
        if metric["unit"] != unit:
            raise ValueError(f"unexpected unit {metric['unit']!r}")
        value = float(metric["avg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"profile summary has no valid {name}") from exc
    if not math.isfinite(value):
        raise AnalysisError(f"profile summary {name} is not finite")
    return value


def _counter_stats(
    document: dict[str, Any],
    names: Sequence[str],
    *,
    required_labels: dict[str, str],
) -> tuple[str, float, float]:
    try:
        metrics = document["metrics"]
    except (KeyError, TypeError) as exc:
        raise AnalysisError("server summary has no metrics object") from exc
    for name in names:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            continue
        for series in metric.get("series", []):
            labels = {str(k): str(v) for k, v in series.get("labels", {}).items()}
            if any(labels.get(key) != value for key, value in required_labels.items()):
                continue
            try:
                total = float(series["stats"]["total"])
                rate = float(series["stats"]["rate"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisError(f"server counter {name} has invalid stats") from exc
            if not math.isfinite(total) or not math.isfinite(rate):
                raise AnalysisError(f"server counter {name} is not finite")
            return name, total, rate
    raise AnalysisError(
        f"server summary has no counter {names[0]} with labels {required_labels}"
    )


def _sglang_counter_stats(
    document: dict[str, Any],
    names: Sequence[str],
    *,
    mode: str,
    absent_mode_is_zero: bool = False,
) -> tuple[str, float, float]:
    try:
        metrics = document["metrics"]
    except (KeyError, TypeError) as exc:
        raise AnalysisError("server summary has no metrics object") from exc

    present_family: str | None = None
    for name in names:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            continue
        if present_family is None:
            present_family = name
        matches: list[tuple[dict[str, str], float, float]] = []
        for series in metric.get("series", []):
            labels = {str(k): str(v) for k, v in series.get("labels", {}).items()}
            if labels.get("mode") != mode:
                continue
            try:
                total = float(series["stats"]["total"])
                rate = float(series["stats"]["rate"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisError(f"server counter {name} has invalid stats") from exc
            if not math.isfinite(total) or not math.isfinite(rate):
                raise AnalysisError(f"server counter {name} is not finite")
            matches.append((labels, total, rate))

        if not matches:
            continue
        if not any("dp_rank" in labels for labels, _, _ in matches):
            return _counter_stats(
                document,
                (name,),
                required_labels={"mode": mode, "tp_rank": "0"},
            )

        by_owner: dict[tuple[tuple[str, str], ...], tuple[float, float]] = {}
        for labels, total, rate in matches:
            if "dp_rank" not in labels:
                continue
            owner_key = tuple(
                sorted(
                    (key, value)
                    for key, value in labels.items()
                    if key not in {"tp_rank", "pp_rank", "moe_ep_rank"}
                )
            )
            previous = by_owner.get(owner_key)
            if previous is None or total > previous[0]:
                by_owner[owner_key] = (total, rate)
        if by_owner:
            return (
                name,
                sum(total for total, _ in by_owner.values()),
                sum(rate for _, rate in by_owner.values()),
            )

    if absent_mode_is_zero and present_family is not None:
        return present_family, 0.0, 0.0
    raise AnalysisError(f"server summary has no counter {names[0]} for mode {mode}")


def _server_counters(
    document: dict[str, Any], engine: str
) -> tuple[tuple[str, float, float], tuple[str, float, float]]:
    if engine == "sglang":
        family = ("sglang:realtime_tokens", "sglang:realtime_tokens_total")
        return (
            _sglang_counter_stats(document, family, mode="prefill_compute"),
            _sglang_counter_stats(
                document,
                family,
                mode="prefill_cache",
                absent_mode_is_zero=True,
            ),
        )
    if engine == "vllm":
        return (
            _counter_stats(
                document,
                ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
                required_labels={"engine": "0"},
            ),
            _counter_stats(
                document,
                (
                    "vllm:prompt_tokens_cached",
                    "vllm:prompt_tokens_cached_total",
                ),
                required_labels={"engine": "0"},
            ),
        )
    raise AnalysisError(f"unsupported engine: {engine}")


def _concurrency_summary(
    intervals: Sequence[tuple[int, int]], target_concurrency: int
) -> dict[str, Any]:
    events: dict[int, list[int]] = defaultdict(list)
    for start_ns, end_ns in intervals:
        events[start_ns].append(1)
        events[end_ns].append(-1)
    timestamps = sorted(events)
    if len(timestamps) < 2:
        raise AnalysisError("prefill intervals have no time span")

    occupancy = 0
    occupancy_ns: dict[int, int] = defaultdict(int)
    max_occupancy = 0
    # Treat intervals as [start, end): departures at a timestamp happen before
    # arrivals at that same timestamp.
    previous = timestamps[0]
    for timestamp in timestamps:
        occupancy_ns[occupancy] += timestamp - previous
        occupancy += sum(delta for delta in events[timestamp] if delta < 0)
        occupancy += sum(delta for delta in events[timestamp] if delta > 0)
        if occupancy < 0:
            raise AnalysisError("prefill concurrency became negative")
        max_occupancy = max(max_occupancy, occupancy)
        previous = timestamp
    if occupancy != 0:
        raise AnalysisError("prefill concurrency did not return to zero")

    duration_ns = timestamps[-1] - timestamps[0]
    if duration_ns <= 0:
        raise AnalysisError("prefill intervals have no positive duration")
    active_area = sum(level * duration for level, duration in occupancy_ns.items())
    return {
        "window_start_ns": timestamps[0],
        "window_end_ns": timestamps[-1],
        "duration_seconds": duration_ns / NANOSECONDS,
        "max": max_occupancy,
        "time_weighted_mean": active_area / duration_ns,
        "target": target_concurrency,
        "fraction_of_window_at_target": occupancy_ns[target_concurrency] / duration_ns,
    }


def analyze(
    summary_path: Path,
    records_path: Path,
    server_summary_path: Path,
    *,
    target_isl: int,
    target_concurrency: int,
    expected_requests: int,
    isl_tolerance: int,
    maximum_compute_ratio: float,
    engine: str = "sglang",
) -> dict[str, Any]:
    for name, value in (
        ("target ISL", target_isl),
        ("target concurrency", target_concurrency),
        ("expected requests", expected_requests),
    ):
        if value < 1:
            raise AnalysisError(f"{name} must be positive")
    if isl_tolerance < 0:
        raise AnalysisError("ISL tolerance cannot be negative")
    if maximum_compute_ratio < 1:
        raise AnalysisError("maximum compute ratio must be at least one")

    summary = _load_json(summary_path)
    records = _load_records(records_path)
    server_summary = _load_json(server_summary_path)
    failures: list[str] = []

    if summary.get("was_cancelled") is not False:
        failures.append("AIPerf run was cancelled or has no cancellation status")
    if summary.get("error_summary") not in ([], None):
        failures.append("AIPerf error summary is not empty")
    summary_count = _summary_metric(summary, "request_count", "requests")
    if summary_count != expected_requests:
        failures.append(
            f"profile summary has {summary_count:g} requests; "
            f"expected {expected_requests}"
        )
    if len(records) != expected_requests:
        failures.append(
            f"record export has {len(records)} requests; expected {expected_requests}"
        )

    ttft_ms: list[float] = []
    input_tokens: list[float] = []
    intervals: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        metadata = record["metadata"]
        if metadata.get("benchmark_phase") != "profiling":
            failures.append(f"record {index} is not from the profiling phase")
        if metadata.get("was_cancelled") is not False:
            failures.append(f"record {index} was cancelled")
        if record.get("error"):
            error_type = record["error"].get("type", "unknown")
            failures.append(f"record {index} failed with {error_type}")
            continue
        try:
            start_ns = int(metadata["request_start_ns"])
            end_ns = int(metadata["request_end_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                f"record {index} has invalid request timestamps"
            ) from exc
        if end_ns <= start_ns:
            raise AnalysisError(f"record {index} has a nonpositive request interval")

        try:
            ttft = _metric(record, "time_to_first_token", "ms")
            isl = _metric(record, "input_sequence_length", "tokens")
            osl = _metric(record, "output_sequence_length", "tokens")
        except AnalysisError as exc:
            failures.append(f"record {index}: {exc}")
            continue
        if ttft <= 0:
            failures.append(f"record {index} has nonpositive TTFT")
        if abs(isl - target_isl) > isl_tolerance:
            failures.append(
                f"record {index} ISL is {isl:g}; target {target_isl} "
                f"+/- {isl_tolerance}"
            )
        if osl != 1:
            failures.append(f"record {index} OSL is {osl:g}; expected 1")
        generation_start_ns = start_ns + round(ttft * MILLISECONDS_TO_NANOSECONDS)
        if generation_start_ns > end_ns + MILLISECONDS_TO_NANOSECONDS:
            failures.append(f"record {index} TTFT extends beyond request end")
        ttft_ms.append(ttft)
        input_tokens.append(isl)
        intervals.append((start_ns, min(generation_start_ns, end_ns)))

    if not ttft_ms:
        raise AnalysisError("record export has no successful prefill requests")
    concurrency = _concurrency_summary(intervals, target_concurrency)
    if concurrency["max"] != target_concurrency:
        failures.append(
            f"client reached prefill concurrency {concurrency['max']}; "
            f"expected {target_concurrency}"
        )

    compute_counter, cache_counter = _server_counters(server_summary, engine)
    counter_family, compute_total, compute_rate = compute_counter
    cache_family, cache_total, cache_rate = cache_counter
    if cache_total != 0:
        failures.append(f"server attributed {cache_total:g} tokens to prefill cache")
    observed_prompt_tokens = sum(input_tokens)
    compute_ratio = compute_total / observed_prompt_tokens
    if compute_ratio < 0.95:
        failures.append(
            f"server compute counter covers only {compute_ratio:.3f}x observed prompts"
        )
    if compute_ratio > maximum_compute_ratio:
        failures.append(
            f"server compute counter is {compute_ratio:.3f}x observed prompts; "
            f"maximum is {maximum_compute_ratio:.3f}x"
        )

    duration_seconds = float(concurrency["duration_seconds"])
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "engine": engine,
        "source": {
            "summary": str(summary_path),
            "records": str(records_path),
            "server_summary": str(server_summary_path),
        },
        "cell": {
            "target_isl": target_isl,
            "isl_tolerance": isl_tolerance,
            "target_concurrency": target_concurrency,
            "expected_requests": expected_requests,
        },
        "requests": {
            "records_exported": len(records),
            "completed": len(ttft_ms),
            "input_tokens": _distribution(input_tokens),
            "time_to_first_token_ms": _distribution(ttft_ms),
            "aggregate_prompt_tokens": observed_prompt_tokens,
            "aggregate_prompt_tokens_per_second": observed_prompt_tokens
            / duration_seconds,
        },
        "prefill_concurrency": concurrency,
        "server_controls": {
            "counter_family": counter_family,
            "compute": {
                "total_tokens": compute_total,
                "tokens_per_second": compute_rate,
                "ratio_to_observed_prompt_tokens": compute_ratio,
                "maximum_allowed_ratio": maximum_compute_ratio,
            },
            "cache": {
                "counter_family": cache_family,
                "total_tokens": cache_total,
                "tokens_per_second": cache_rate,
            },
        },
        "validation": {"valid": not failures},
        "failures": failures,
    }
    return result


def main() -> int:
    args = _parse_args()
    try:
        result = analyze(
            args.summary,
            args.records,
            args.server_summary,
            target_isl=args.target_isl,
            target_concurrency=args.target_concurrency,
            expected_requests=args.expected_requests,
            isl_tolerance=args.isl_tolerance,
            maximum_compute_ratio=args.maximum_compute_ratio,
            engine=args.engine,
        )
    except (AnalysisError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

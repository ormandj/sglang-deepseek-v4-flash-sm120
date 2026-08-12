#!/usr/bin/env python3
"""Measure steady-state SGLang decode from AIPerf server-metric scrapes.

AIPerf's completed-request throughput includes ramp-up and drain time.  This
tool instead measures a bounded plateau inside the profiling phase and refuses
to call the plateau valid when request occupancy changes, work queues, or new
prefills arrive.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

NANOSECONDS = 1_000_000_000


class AnalysisError(RuntimeError):
    """Raised when the captured interval is not a valid decode plateau."""


@dataclass(frozen=True)
class Point:
    timestamp_ns: int
    value: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--target-concurrency", type=int, required=True)
    parser.add_argument("--settle-seconds", type=float, default=15.0)
    parser.add_argument("--tail-seconds", type=float, default=3.0)
    parser.add_argument("--minimum-window-seconds", type=float, default=30.0)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--minimum-exact-occupancy", type=float, default=0.98)
    parser.add_argument("--average-context-lower", type=float)
    parser.add_argument("--average-context-upper", type=float)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_phase_range(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    try:
        profiling = document["summary"]["phase_time_ranges"]["profiling"]
        return int(profiling["start_ns"]), int(profiling["end_ns"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(
            f"{path} has no valid summary.phase_time_ranges.profiling"
        ) from exc


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                int(record["timestamp_ns"])
                if not isinstance(record["metrics"], dict):
                    raise TypeError("metrics is not an object")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise AnalysisError(f"invalid record at {path}:{line_number}") from exc
            records.append(record)
    records.sort(key=lambda item: int(item["timestamp_ns"]))
    return records


def _metric_samples(
    record: dict[str, Any],
    names: Sequence[str],
    *,
    required_labels: dict[str, str] | None = None,
    allowed_labels: dict[str, set[str]] | None = None,
) -> list[float]:
    return [
        value
        for _, value in _labeled_metric_samples(
            record,
            names,
            required_labels=required_labels,
            allowed_labels=allowed_labels,
        )
    ]


def _labeled_metric_samples(
    record: dict[str, Any],
    names: Sequence[str],
    *,
    required_labels: dict[str, str] | None = None,
    allowed_labels: dict[str, set[str]] | None = None,
) -> list[tuple[dict[str, str], float]]:
    samples: list[dict[str, Any]] = []
    for name in names:
        candidate = record["metrics"].get(name)
        if candidate:
            samples = candidate
            break
    values: list[tuple[dict[str, str], float]] = []
    for sample in samples:
        labels = {
            str(key): str(value) for key, value in sample.get("labels", {}).items()
        }
        if required_labels and any(
            labels.get(key) != value for key, value in required_labels.items()
        ):
            continue
        if allowed_labels and any(
            labels.get(key) not in values for key, values in allowed_labels.items()
        ):
            continue
        try:
            value = float(sample["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append((labels, value))
    return values


def _uses_distributed_request_ownership(
    records: Sequence[dict[str, Any]],
) -> bool:
    dp_ranks: set[str] = set()
    for record in records:
        for labels, _ in _labeled_metric_samples(
            record, ("sglang:num_running_reqs",)
        ):
            if "dp_rank" in labels:
                dp_ranks.add(labels["dp_rank"])
    return len(dp_ranks) > 1


def _rank_metric_values(
    record: dict[str, Any],
    names: Sequence[str],
    *,
    distributed_request_ownership: bool,
    required_labels: dict[str, str] | None = None,
    allowed_labels: dict[str, set[str]] | None = None,
) -> list[float]:
    if not distributed_request_ownership:
        labels = {**(required_labels or {}), "tp_rank": "0"}
        return _metric_samples(
            record,
            names,
            required_labels=labels,
            allowed_labels=allowed_labels,
        )

    samples = _labeled_metric_samples(
        record,
        names,
        required_labels=required_labels,
        allowed_labels=allowed_labels,
    )
    by_series: dict[tuple[tuple[str, str], ...], list[float]] = {}
    for labels, value in samples:
        if "dp_rank" not in labels:
            continue
        series_key = tuple(
            sorted(
                (key, value)
                for key, value in labels.items()
                if key not in {"tp_rank", "pp_rank", "moe_ep_rank"}
            )
        )
        by_series.setdefault(series_key, []).append(value)
    # A logical series can be exported by multiple model-parallel workers for
    # one DP rank. Those samples are replicas. Distinct modes remain distinct
    # series so callers can intentionally sum them.
    return [max(values) for values in by_series.values()]


def _rank_series(
    records: Iterable[dict[str, Any]],
    names: Sequence[str],
    *,
    distributed_request_ownership: bool,
    required_labels: dict[str, str] | None = None,
    allowed_labels: dict[str, set[str]] | None = None,
    reduction: str,
) -> list[Point]:
    points: list[Point] = []
    for record in records:
        values = _rank_metric_values(
            record,
            names,
            distributed_request_ownership=distributed_request_ownership,
            required_labels=required_labels,
            allowed_labels=allowed_labels,
        )
        if not values:
            continue
        if reduction == "sum":
            value = sum(values)
        elif reduction == "max":
            value = max(values)
        else:
            raise ValueError(f"unknown reduction: {reduction}")
        points.append(Point(int(record["timestamp_ns"]), value))
    return points


def _active_context_series(
    records: Iterable[dict[str, Any]],
    *,
    distributed_request_ownership: bool,
) -> list[Point]:
    if not distributed_request_ownership:
        return _rank_series(
            records,
            ("sglang:decode_sum_seq_lens",),
            distributed_request_ownership=False,
            reduction="sum",
        )

    points: list[Point] = []
    for record in records:
        running_samples = _labeled_metric_samples(
            record, ("sglang:num_running_reqs",)
        )
        context_samples = _labeled_metric_samples(
            record, ("sglang:decode_sum_seq_lens",)
        )
        running_by_rank: dict[str, float] = {}
        context_by_rank: dict[str, float] = {}
        for labels, value in running_samples:
            if "dp_rank" in labels:
                running_by_rank[labels["dp_rank"]] = max(
                    value, running_by_rank.get(labels["dp_rank"], value)
                )
        for labels, value in context_samples:
            if "dp_rank" in labels:
                context_by_rank[labels["dp_rank"]] = max(
                    value, context_by_rank.get(labels["dp_rank"], value)
                )
        active_ranks = [rank for rank, value in running_by_rank.items() if value > 0]
        if not active_ranks or any(rank not in context_by_rank for rank in active_ranks):
            continue
        points.append(
            Point(
                int(record["timestamp_ns"]),
                sum(context_by_rank[rank] for rank in active_ranks),
            )
        )
    return points


def _active_rank_count_series(
    records: Iterable[dict[str, Any]],
    *,
    distributed_request_ownership: bool,
) -> list[Point]:
    points: list[Point] = []
    for record in records:
        values = _rank_metric_values(
            record,
            ("sglang:num_running_reqs",),
            distributed_request_ownership=distributed_request_ownership,
        )
        if values:
            points.append(
                Point(int(record["timestamp_ns"]), sum(value > 0 for value in values))
            )
    return points


def _series(
    records: Iterable[dict[str, Any]],
    names: Sequence[str],
    *,
    required_labels: dict[str, str] | None = None,
    allowed_labels: dict[str, set[str]] | None = None,
    reduction: str,
) -> list[Point]:
    points: list[Point] = []
    for record in records:
        values = _metric_samples(
            record,
            names,
            required_labels=required_labels,
            allowed_labels=allowed_labels,
        )
        if not values:
            continue
        if reduction == "sum":
            value = sum(values)
        elif reduction == "max":
            value = max(values)
        else:
            raise ValueError(f"unknown reduction: {reduction}")
        points.append(Point(int(record["timestamp_ns"]), value))
    return points


def _rank_series_or_zero_when_label_never_exists(
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    distributed_request_ownership: bool,
    required_labels: dict[str, str],
) -> tuple[list[Point], bool]:
    points = _rank_series(
        records,
        names,
        distributed_request_ownership=distributed_request_ownership,
        required_labels=required_labels,
        reduction="sum",
    )
    if points:
        return points, True

    # Prometheus does not instantiate every label value at zero. SGLang omits
    # mode=prefill_cache until the process has observed a cache hit. If the
    # metric family is present but the requested label never
    # exists anywhere in the plateau, its counter delta is unambiguously zero.
    family_records = [
        record for record in records if any(name in record["metrics"] for name in names)
    ]
    return (
        [Point(int(record["timestamp_ns"]), 0.0) for record in family_records],
        False,
    )


def _require_monotonic(points: Sequence[Point], name: str) -> None:
    for before, after in pairwise(points):
        if after.value < before.value:
            raise AnalysisError(
                f"{name} reset or decreased at {after.timestamp_ns}: "
                f"{before.value} -> {after.value}"
            )


def _ols_slope(points: Sequence[Point]) -> tuple[float, float]:
    origin = points[0].timestamp_ns
    xs = [(point.timestamp_ns - origin) / NANOSECONDS for point in points]
    ys = [point.value for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0:
        raise AnalysisError("counter samples have no time span")
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(xs, ys, strict=True)
        )
        / denominator
    )
    fitted = [y_mean + slope * (value - x_mean) for value in xs]
    residual = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(ys, fitted, strict=True)
    )
    total = sum((actual - y_mean) ** 2 for actual in ys)
    r_squared = 1.0 if total == 0 else 1.0 - residual / total
    return slope, r_squared


def _delta_rate(points: Sequence[Point]) -> float:
    duration = (points[-1].timestamp_ns - points[0].timestamp_ns) / NANOSECONDS
    if duration <= 0:
        raise AnalysisError("counter samples have no time span")
    return (points[-1].value - points[0].value) / duration


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise AnalysisError("cannot compute a percentile from no samples")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_gauge(points: Sequence[Point]) -> dict[str, float | int]:
    values = [point.value for point in points]
    return {
        "samples": len(values),
        "min": min(values),
        "p05": _percentile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _optional_gauge(
    records: Sequence[dict[str, Any]],
    names: Sequence[str],
    *,
    distributed_request_ownership: bool,
    reduction: str,
) -> dict[str, float | int] | None:
    points = _rank_series(
        records,
        names,
        distributed_request_ownership=distributed_request_ownership,
        reduction=reduction,
    )
    return _summarize_gauge(points) if points else None


def analyze(
    summary_path: Path,
    jsonl_path: Path,
    *,
    target_concurrency: int,
    settle_seconds: float,
    tail_seconds: float,
    minimum_window_seconds: float,
    minimum_samples: int,
    minimum_exact_occupancy: float,
    average_context_lower: float | None = None,
    average_context_upper: float | None = None,
) -> dict[str, Any]:
    if target_concurrency < 1:
        raise AnalysisError("target concurrency must be positive")
    if (average_context_lower is None) != (average_context_upper is None):
        raise AnalysisError("both average-context bounds must be provided together")
    if (
        average_context_lower is not None
        and average_context_upper is not None
        and not 0 < average_context_lower < average_context_upper
    ):
        raise AnalysisError("average-context bounds must be positive and increasing")
    phase_start_ns, phase_end_ns = _load_phase_range(summary_path)
    all_phase_records = [
        record
        for record in _load_records(jsonl_path)
        if phase_start_ns <= int(record["timestamp_ns"]) <= phase_end_ns
    ]
    distributed_request_ownership = _uses_distributed_request_ownership(
        all_phase_records
    )
    context_window = average_context_lower is not None
    if context_window:
        matching_indexes: list[int] = []
        for index, record in enumerate(all_phase_records):
            running_values = _rank_metric_values(
                record,
                ("sglang:num_running_reqs",),
                distributed_request_ownership=distributed_request_ownership,
            )
            context_points = _active_context_series(
                (record,),
                distributed_request_ownership=distributed_request_ownership,
            )
            if not running_values or not context_points:
                continue
            running = sum(running_values)
            average_context = context_points[0].value / target_concurrency
            if (
                running == target_concurrency
                and average_context_lower <= average_context <= average_context_upper
            ):
                matching_indexes.append(index)
        if not matching_indexes:
            raise AnalysisError("no samples matched the average-context window")
        phase_records = all_phase_records[
            matching_indexes[0] : matching_indexes[-1] + 1
        ]
        window_start_ns = int(phase_records[0]["timestamp_ns"])
    else:
        window_start_ns = phase_start_ns + round(settle_seconds * NANOSECONDS)
        phase_records = [
            record
            for record in all_phase_records
            if int(record["timestamp_ns"]) >= window_start_ns
        ]
    running_phase = _rank_series(
        phase_records,
        ("sglang:num_running_reqs",),
        distributed_request_ownership=distributed_request_ownership,
        reduction="sum",
    )
    if not running_phase:
        raise AnalysisError("running-request gauge has no samples after settle period")

    departure_index = next(
        (
            index
            for index, point in enumerate(running_phase)
            if point.value != target_concurrency
        ),
        None,
    )
    terminal_occupancy_drop = False
    first_departure_ns: int | None = None
    if departure_index is not None:
        first_departure = running_phase[departure_index]
        first_departure_ns = first_departure.timestamp_ns
        terminal_occupancy_drop = (
            departure_index > 0
            and first_departure.value < target_concurrency
            and all(
                point.value < target_concurrency
                for point in running_phase[departure_index:]
            )
        )

    if context_window:
        end_reason = "average_context_window"
        window_end_ns = int(phase_records[-1]["timestamp_ns"])
    else:
        end_reason = "phase_tail"
        end_anchor_ns = phase_end_ns
        if terminal_occupancy_drop and first_departure_ns is not None:
            end_reason = "terminal_occupancy_drop"
            end_anchor_ns = first_departure_ns
        window_end_ns = end_anchor_ns - round(tail_seconds * NANOSECONDS)
    window_seconds = (window_end_ns - window_start_ns) / NANOSECONDS
    if window_seconds < minimum_window_seconds:
        raise AnalysisError(
            f"analysis window is {window_seconds:.3f}s; need {minimum_window_seconds:.3f}s"
        )

    records = [
        record
        for record in phase_records
        if int(record["timestamp_ns"]) <= window_end_ns
    ]
    if len(records) < minimum_samples:
        raise AnalysisError(
            f"analysis window has {len(records)} scrapes; need {minimum_samples}"
        )

    decode_realtime = _rank_series(
        records,
        ("sglang:realtime_tokens", "sglang:realtime_tokens_total"),
        distributed_request_ownership=distributed_request_ownership,
        required_labels={"mode": "decode"},
        reduction="sum",
    )
    decode_forward_passes = _rank_series(
        records,
        ("sglang:cuda_graph_passes", "sglang:cuda_graph_passes_total"),
        distributed_request_ownership=distributed_request_ownership,
        allowed_labels={"mode": {"decode_cuda_graph", "decode_none"}},
        reduction="sum",
    )
    prefill_compute, prefill_compute_series_observed = (
        _rank_series_or_zero_when_label_never_exists(
            records,
            ("sglang:realtime_tokens", "sglang:realtime_tokens_total"),
            distributed_request_ownership=distributed_request_ownership,
            required_labels={"mode": "prefill_compute"},
        )
    )
    prefill_cache, prefill_cache_series_observed = (
        _rank_series_or_zero_when_label_never_exists(
            records,
            ("sglang:realtime_tokens", "sglang:realtime_tokens_total"),
            distributed_request_ownership=distributed_request_ownership,
            required_labels={"mode": "prefill_cache"},
        )
    )
    running = _rank_series(
        records,
        ("sglang:num_running_reqs",),
        distributed_request_ownership=distributed_request_ownership,
        reduction="sum",
    )
    queue = _rank_series(
        records,
        ("sglang:num_queue_reqs",),
        distributed_request_ownership=distributed_request_ownership,
        reduction="sum",
    )
    average_context = _active_context_series(
        records,
        distributed_request_ownership=distributed_request_ownership,
    )
    average_context = [
        Point(point.timestamp_ns, point.value / target_concurrency)
        for point in average_context
    ]
    for name, points in (
        ("realtime decode-token counter", decode_realtime),
        ("decode forward-pass counter", decode_forward_passes),
        ("realtime prefill-compute counter", prefill_compute),
        ("realtime prefill-cache counter", prefill_cache),
        ("running-request gauge", running),
        ("queued-request gauge", queue),
        ("average decode-context gauge", average_context),
    ):
        if len(points) < minimum_samples:
            raise AnalysisError(
                f"{name} has {len(points)} samples; need {minimum_samples}"
            )

    _require_monotonic(decode_realtime, "realtime decode-token counter")
    _require_monotonic(decode_forward_passes, "decode forward-pass counter")
    _require_monotonic(prefill_compute, "realtime prefill-compute counter")
    _require_monotonic(prefill_cache, "realtime prefill-cache counter")
    decode_slope, decode_r_squared = _ols_slope(decode_realtime)
    forward_slope, forward_r_squared = _ols_slope(decode_forward_passes)
    prefill_compute_slope, prefill_compute_r_squared = _ols_slope(prefill_compute)
    prefill_cache_slope, prefill_cache_r_squared = _ols_slope(prefill_cache)
    running_summary = _summarize_gauge(running)
    queue_summary = _summarize_gauge(queue)
    average_context_summary = _summarize_gauge(average_context)
    active_rank_summary = _summarize_gauge(
        _active_rank_count_series(
            records,
            distributed_request_ownership=distributed_request_ownership,
        )
    )
    exact_samples = sum(point.value == target_concurrency for point in running)
    exact_fraction = exact_samples / len(running)

    prefill_compute_delta = prefill_compute[-1].value - prefill_compute[0].value
    prefill_cache_delta = prefill_cache[-1].value - prefill_cache[0].value
    decode_counter_delta = decode_realtime[-1].value - decode_realtime[0].value
    forward_counter_delta = (
        decode_forward_passes[-1].value - decode_forward_passes[0].value
    )
    if forward_counter_delta <= 0:
        raise AnalysisError("decode forward-pass counter did not increase")

    validation = {
        "decode_counter_monotonic": True,
        "prefill_compute_counter_unchanged": prefill_compute_delta == 0,
        "prefill_cache_counter_unchanged": prefill_cache_delta == 0,
        "queue_empty": queue_summary["max"] == 0,
        "exact_occupancy_fraction": exact_fraction,
        "exact_occupancy_required": minimum_exact_occupancy,
        "average_context_within_bounds": (
            not context_window
            or (
                average_context_summary["min"] >= average_context_lower
                and average_context_summary["max"] <= average_context_upper
            )
        ),
        "valid": False,
    }
    failures: list[str] = []
    if not validation["prefill_compute_counter_unchanged"]:
        failures.append("prefill-compute counter changed during plateau")
    if not validation["prefill_cache_counter_unchanged"]:
        failures.append("prefill-cache counter changed during plateau")
    if not validation["queue_empty"]:
        failures.append(f"queue reached {queue_summary['max']} requests")
    if exact_fraction < minimum_exact_occupancy:
        failures.append(
            f"exact occupancy held for {exact_fraction:.3%}; "
            f"need {minimum_exact_occupancy:.3%}"
        )
    if not validation["average_context_within_bounds"]:
        failures.append("average decode context left the requested window")
    validation["valid"] = not failures

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "source": {
            "summary": str(summary_path),
            "jsonl": str(jsonl_path),
        },
        "phase": {"start_ns": phase_start_ns, "end_ns": phase_end_ns},
        "plateau": {
            "start_ns": window_start_ns,
            "end_ns": window_end_ns,
            "duration_seconds": window_seconds,
            "selection": end_reason,
            "settle_seconds": 0.0 if context_window else settle_seconds,
            "tail_seconds": 0.0 if context_window else tail_seconds,
            "average_context_lower": average_context_lower,
            "average_context_upper": average_context_upper,
            "scrapes": len(records),
            "end_reason": end_reason,
            "first_occupancy_departure_ns": first_departure_ns,
        },
        "average_context_length": average_context_summary,
        "decode": {
            "target_concurrency": target_concurrency,
            "counter_family": (
                "sum by dp_rank(sglang:realtime_tokens{mode=decode})"
                if distributed_request_ownership
                else "sglang:realtime_tokens{mode=decode,tp_rank=0}"
            ),
            "tokens_per_second_ols": decode_slope,
            "tokens_per_second_delta": _delta_rate(decode_realtime),
            "counter_delta": decode_counter_delta,
            "ols_r_squared": decode_r_squared,
        },
        "engine_work": {
            "counter_family": (
                "sum by dp_rank(sglang:cuda_graph_passes"
                "{mode=decode_cuda_graph|decode_none})"
                if distributed_request_ownership
                else "sglang:cuda_graph_passes"
                "{mode=decode_cuda_graph|decode_none,tp_rank=0}"
            ),
            "forward_passes_per_second_ols": forward_slope,
            "forward_passes_per_second_delta": _delta_rate(decode_forward_passes),
            "counter_delta": forward_counter_delta,
            "ols_r_squared": forward_r_squared,
            "useful_tokens_per_forward_per_request": (
                decode_counter_delta
                / forward_counter_delta
                * active_rank_summary["median"]
                / target_concurrency
            ),
            "active_request_ranks": active_rank_summary,
        },
        "prefill_control": {
            "compute": {
                "series_observed": prefill_compute_series_observed,
                "tokens_per_second_ols": prefill_compute_slope,
                "tokens_per_second_delta": _delta_rate(prefill_compute),
                "counter_delta": prefill_compute_delta,
                "ols_r_squared": prefill_compute_r_squared,
            },
            "cache": {
                "series_observed": prefill_cache_series_observed,
                "tokens_per_second_ols": prefill_cache_slope,
                "tokens_per_second_delta": _delta_rate(prefill_cache),
                "counter_delta": prefill_cache_delta,
                "ols_r_squared": prefill_cache_r_squared,
            },
        },
        "running_requests": running_summary,
        "queued_requests": queue_summary,
        "server_cross_checks": {
            "generation_throughput": _optional_gauge(
                records,
                ("sglang:gen_throughput",),
                distributed_request_ownership=distributed_request_ownership,
                reduction="sum",
            ),
            "spec_accept_rate": _optional_gauge(
                records,
                ("sglang:spec_accept_rate",),
                distributed_request_ownership=distributed_request_ownership,
                reduction="max",
            ),
            "spec_accept_length": _optional_gauge(
                records,
                ("sglang:spec_accept_length",),
                distributed_request_ownership=distributed_request_ownership,
                reduction="max",
            ),
        },
        "validation": validation,
        "failures": failures,
    }
    return result


def main() -> int:
    args = _parse_args()
    try:
        result = analyze(
            args.summary,
            args.jsonl,
            target_concurrency=args.target_concurrency,
            settle_seconds=args.settle_seconds,
            tail_seconds=args.tail_seconds,
            minimum_window_seconds=args.minimum_window_seconds,
            minimum_samples=args.minimum_samples,
            minimum_exact_occupancy=args.minimum_exact_occupancy,
            average_context_lower=args.average_context_lower,
            average_context_upper=args.average_context_upper,
        )
    except (AnalysisError, OSError) as exc:
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

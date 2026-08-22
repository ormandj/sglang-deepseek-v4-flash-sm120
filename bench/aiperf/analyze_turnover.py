#!/usr/bin/env python3
"""Validate one fixed-concurrency, repeated-request turnover cell."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analyze_prefill import (
    NANOSECONDS,
    AnalysisError,
    _concurrency_summary,
    _distribution,
    _load_json,
    _load_records,
    _metric,
    _sglang_counter_stats,
    _summary_metric,
)
from analyze_server_metrics import (
    _load_phase_range,
    _rank_series,
    _summarize_gauge,
    _uses_distributed_request_ownership,
)
from analyze_server_metrics import (
    _load_records as _load_server_records,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--server-summary", type=Path, required=True)
    parser.add_argument("--server-jsonl", type=Path, required=True)
    parser.add_argument("--target-concurrency", type=int, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--target-isl", type=int, required=True)
    parser.add_argument("--target-osl", type=int, required=True)
    parser.add_argument("--isl-tolerance", type=int, default=16)
    parser.add_argument("--minimum-client-occupancy", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _prefill_passes(document: dict[str, Any]) -> dict[str, Any]:
    family = ("sglang:cuda_graph_passes", "sglang:cuda_graph_passes_total")
    modes: dict[str, dict[str, float]] = {}
    for mode in ("prefill_none", "prefill_cuda_graph"):
        metric, total, rate = _sglang_counter_stats(
            document, family, mode=mode, absent_mode_is_zero=True
        )
        modes[mode] = {"total": total, "per_second": rate}
    return {
        "counter_family": metric,
        "modes": modes,
        "total": sum(value["total"] for value in modes.values()),
        "per_second": sum(value["per_second"] for value in modes.values()),
    }


def _counter_total(
    document: dict[str, Any], names: Sequence[str], *, required_labels: dict[str, str]
) -> float:
    """Sum one counter family without double-counting replicated rank series."""
    try:
        metrics = document["metrics"]
    except (KeyError, TypeError) as exc:
        raise AnalysisError("server summary has no metrics object") from exc

    rank_labels = {"tp_rank", "pp_rank", "moe_ep_rank"}
    for name in names:
        metric = metrics.get(name)
        if not isinstance(metric, dict):
            continue
        series = metric.get("series")
        if not isinstance(series, list):
            raise AnalysisError(f"server counter {name} has no series list")

        matches: list[tuple[dict[str, str], float]] = []
        for item in series:
            try:
                labels = {
                    str(key): str(value) for key, value in item["labels"].items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisError(f"server counter {name} has invalid labels") from exc
            if any(labels.get(key) != value for key, value in required_labels.items()):
                continue
            try:
                total = float(item["stats"]["total"])
            except (KeyError, TypeError, ValueError) as exc:
                raise AnalysisError(f"server counter {name} has invalid stats") from exc
            if not math.isfinite(total) or total < 0:
                raise AnalysisError(f"server counter {name} has an invalid total")
            matches.append((labels, total))

        if not matches:
            continue
        if not any(rank_labels & labels.keys() for labels, _ in matches):
            return sum(total for _, total in matches)

        by_owner: dict[tuple[tuple[str, str], ...], float] = {}
        for labels, total in matches:
            owner_key = tuple(
                sorted(
                    (key, value)
                    for key, value in labels.items()
                    if key not in rank_labels
                )
            )
            by_owner[owner_key] = max(total, by_owner.get(owner_key, 0.0))
        return sum(by_owner.values())

    raise AnalysisError(
        f"server summary has no counter {names[0]} with labels {required_labels}"
    )


def _admission_concurrency_summary(
    intervals: list[tuple[int, int]], target_concurrency: int
) -> dict[str, Any]:
    """Summarize occupancy while the client still has requests to admit."""
    last_admission_ns = max(start_ns for start_ns, _ in intervals)
    clipped = [
        (start_ns, min(end_ns, last_admission_ns))
        for start_ns, end_ns in intervals
        if start_ns < last_admission_ns
    ]
    if not clipped:
        raise AnalysisError(
            "turnover requests have no admission window after the initial burst"
        )
    result = _concurrency_summary(clipped, target_concurrency)
    observed_duration_ns = result["window_end_ns"] - result["window_start_ns"]
    admission_duration_ns = last_admission_ns - result["window_start_ns"]
    target_duration_ns = result["fraction_of_window_at_target"] * observed_duration_ns
    active_area = result["time_weighted_mean"] * observed_duration_ns
    result.update(
        {
            "window_end_ns": last_admission_ns,
            "duration_seconds": admission_duration_ns / NANOSECONDS,
            "time_weighted_mean": active_area / admission_duration_ns,
            "fraction_of_window_at_target": target_duration_ns / admission_duration_ns,
        }
    )
    result["last_admission_ns"] = last_admission_ns
    return result


def analyze(
    summary_path: Path,
    records_path: Path,
    server_summary_path: Path,
    server_jsonl_path: Path,
    *,
    target_concurrency: int,
    expected_requests: int,
    target_isl: int,
    target_osl: int,
    isl_tolerance: int,
    minimum_client_occupancy: float,
) -> dict[str, Any]:
    for name, value in (
        ("target concurrency", target_concurrency),
        ("expected requests", expected_requests),
        ("target ISL", target_isl),
        ("target OSL", target_osl),
    ):
        if value < 1:
            raise AnalysisError(f"{name} must be positive")
    if isl_tolerance < 0:
        raise AnalysisError("ISL tolerance cannot be negative")
    if not 0 < minimum_client_occupancy <= 1:
        raise AnalysisError("minimum client occupancy must be within (0, 1]")
    if expected_requests <= target_concurrency:
        raise AnalysisError(
            "expected requests must exceed target concurrency to exercise turnover"
        )

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
            f"profile summary has {summary_count:g} requests; expected {expected_requests}"
        )
    if len(records) != expected_requests:
        failures.append(
            f"record export has {len(records)} requests; expected {expected_requests}"
        )

    intervals: list[tuple[int, int]] = []
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    latency_ms: list[float] = []
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
            isl = _metric(record, "input_sequence_length", "tokens")
            osl = _metric(record, "output_sequence_length", "tokens")
            ttft = _metric(record, "time_to_first_token", "ms")
            itl = _metric(record, "inter_token_latency", "ms")
            latency = _metric(record, "request_latency", "ms")
        except AnalysisError as exc:
            failures.append(f"record {index}: {exc}")
            continue
        if abs(isl - target_isl) > isl_tolerance:
            failures.append(
                f"record {index} ISL is {isl:g}; target {target_isl} +/- {isl_tolerance}"
            )
        if osl != target_osl:
            failures.append(f"record {index} OSL is {osl:g}; expected {target_osl}")
        if min(ttft, itl, latency) <= 0:
            failures.append(f"record {index} has a nonpositive latency metric")
        intervals.append((start_ns, end_ns))
        input_tokens.append(isl)
        output_tokens.append(osl)
        ttft_ms.append(ttft)
        itl_ms.append(itl)
        latency_ms.append(latency)

    if not intervals:
        raise AnalysisError("record export has no successful requests")
    client_concurrency = _concurrency_summary(intervals, target_concurrency)
    admission_concurrency = _admission_concurrency_summary(
        intervals, target_concurrency
    )
    if client_concurrency["max"] != target_concurrency:
        failures.append(
            f"client reached concurrency {client_concurrency['max']}; expected {target_concurrency}"
        )
    if admission_concurrency["fraction_of_window_at_target"] < minimum_client_occupancy:
        failures.append(
            "client held target concurrency for "
            f"{admission_concurrency['fraction_of_window_at_target']:.3f} of the admission window; "
            f"minimum is {minimum_client_occupancy:.3f}"
        )

    phase_start_ns, phase_end_ns = _load_phase_range(server_summary_path)
    server_records = [
        record
        for record in _load_server_records(server_jsonl_path)
        if phase_start_ns <= int(record["timestamp_ns"]) <= phase_end_ns
    ]
    if len(server_records) < 2:
        raise AnalysisError("server metrics have fewer than two profiling samples")
    distributed = _uses_distributed_request_ownership(server_records)
    running = _rank_series(
        server_records,
        ("sglang:num_running_reqs",),
        distributed_request_ownership=distributed,
        reduction="sum",
    )
    waiting = _rank_series(
        server_records,
        ("sglang:num_queue_reqs",),
        distributed_request_ownership=distributed,
        reduction="sum",
    )
    if not running or not waiting:
        raise AnalysisError("server metrics have no running or waiting request gauges")
    running_summary = _summarize_gauge(running)
    waiting_summary = _summarize_gauge(waiting)
    if running_summary["max"] < target_concurrency:
        failures.append(
            f"server running occupancy reached only {running_summary['max']:g}; expected {target_concurrency}"
        )
    if running_summary["max"] > expected_requests:
        failures.append(
            f"server running occupancy reached {running_summary['max']:g}; expected at most {expected_requests}"
        )
    if waiting_summary["max"] > target_concurrency:
        failures.append(
            f"server queue reached {waiting_summary['max']:g}; expected at most {target_concurrency}"
        )

    chat_completion_posts = _counter_total(
        server_summary,
        ("sglang:http_requests", "sglang:http_requests_total"),
        required_labels={"endpoint": "/v1/chat/completions", "method": "POST"},
    )
    if chat_completion_posts != expected_requests:
        failures.append(
            f"server observed {chat_completion_posts:g} chat-completion POSTs; expected {expected_requests}"
        )

    prefill = _prefill_passes(server_summary)
    prefill_passes = float(prefill["total"])
    if not math.isfinite(prefill_passes) or prefill_passes <= 0:
        failures.append("server recorded no positive prefill-pass count")
        requests_per_prefill_pass = None
    else:
        requests_per_prefill_pass = len(intervals) / prefill_passes

    output_throughput = _summary_metric(
        summary, "output_token_throughput", "tokens/sec"
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "engine": "sglang",
        "source": {
            "summary": str(summary_path),
            "records": str(records_path),
            "server_summary": str(server_summary_path),
            "server_jsonl": str(server_jsonl_path),
        },
        "cell": {
            "target_concurrency": target_concurrency,
            "expected_requests": expected_requests,
            "target_isl": target_isl,
            "target_osl": target_osl,
            "isl_tolerance": isl_tolerance,
        },
        "requests": {
            "records_exported": len(records),
            "completed": len(intervals),
            "input_tokens": _distribution(input_tokens),
            "output_tokens": _distribution(output_tokens),
            "output_tokens_per_second": output_throughput,
            "time_to_first_token_ms": _distribution(ttft_ms),
            "inter_token_latency_ms": _distribution(itl_ms),
            "request_latency_ms": _distribution(latency_ms),
        },
        "client_concurrency": {
            "full_request_window": client_concurrency,
            "admission_window": {
                **admission_concurrency,
                "minimum_fraction_at_target": minimum_client_occupancy,
            },
        },
        "server_occupancy": {
            "running": running_summary,
            "waiting": waiting_summary,
        },
        "server_requests": {
            "chat_completion_posts": chat_completion_posts,
            "expected": expected_requests,
        },
        "refill_batching": {
            "prefill_passes": prefill,
            "requests_per_prefill_pass": requests_per_prefill_pass,
            "interpretation": (
                "effective request batching for this one-pass short-prompt shape; "
                "compare only with an identical prompt/chunking method"
            ),
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
            args.server_jsonl,
            target_concurrency=args.target_concurrency,
            expected_requests=args.expected_requests,
            target_isl=args.target_isl,
            target_osl=args.target_osl,
            isl_tolerance=args.isl_tolerance,
            minimum_client_occupancy=args.minimum_client_occupancy,
        )
    except (AnalysisError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            print(f"error: output already exists: {args.output}", file=sys.stderr)
            return 2
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure steady-state vLLM decode from AIPerf server-metric scrapes.

vLLM exposes one observation in ``iteration_tokens_total`` per engine step.
During an exact-occupancy decode plateau, the histogram count is therefore the
cross-engine counterpart to SGLang's decode forward-pass counter.  Live
prompt-token and generation-token deltas derive average sequence progress so
every engine is measured over the same context interval.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analyze_server_metrics import (
    NANOSECONDS,
    AnalysisError,
    Point,
    _delta_rate,
    _load_phase_range,
    _load_records,
    _metric_samples,
    _ols_slope,
    _require_monotonic,
    _series,
    _summarize_gauge,
)

VLLM_LABELS = {"engine": "0"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--target-concurrency", type=int, required=True)
    parser.add_argument("--average-context-lower", type=float, required=True)
    parser.add_argument("--average-context-upper", type=float, required=True)
    parser.add_argument("--minimum-window-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--minimum-exact-occupancy", type=float, default=0.98)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _counter_series(
    records: Sequence[dict[str, Any]], names: Sequence[str]
) -> list[Point]:
    return _series(
        records,
        names,
        required_labels=VLLM_LABELS,
        reduction="sum",
    )


def _context_series(
    records: Sequence[dict[str, Any]],
    *,
    generation_baseline: float,
    prompt_baseline: float,
    target_concurrency: int,
) -> list[Point]:
    generation = _counter_series(
        records,
        ("vllm:generation_tokens", "vllm:generation_tokens_total"),
    )
    prompt = _counter_series(
        records,
        ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
    )
    if len(generation) != len(prompt):
        raise AnalysisError("vLLM prompt and generation counters are not aligned")
    return [
        Point(
            generation_point.timestamp_ns,
            (
                generation_point.value
                - generation_baseline
                + prompt_point.value
                - prompt_baseline
            )
            / target_concurrency,
        )
        for generation_point, prompt_point in zip(generation, prompt, strict=True)
    ]


def _histogram_count_series(
    records: Sequence[dict[str, Any]], names: Sequence[str]
) -> list[Point]:
    points: list[Point] = []
    for record in records:
        samples: list[dict[str, Any]] = []
        for name in names:
            candidate = record["metrics"].get(name)
            if candidate:
                samples = candidate
                break
        values: list[float] = []
        for sample in samples:
            labels = {
                str(key): str(value) for key, value in sample.get("labels", {}).items()
            }
            if any(labels.get(key) != value for key, value in VLLM_LABELS.items()):
                continue
            try:
                value = float(sample["count"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            points.append(Point(int(record["timestamp_ns"]), sum(values)))
    return points


def _required_speculative_counter_series(
    records: Sequence[dict[str, Any]], names: Sequence[str]
) -> list[Point]:
    points = _counter_series(records, names)
    if len(points) != len(records):
        raise AnalysisError(
            f"{names[0]} has {len(points)} samples; need {len(records)}"
        )
    _require_monotonic(points, names[0])
    return points


def _counter_ratio_distribution(
    numerator: Sequence[Point],
    denominator: Sequence[Point],
    *,
    additive: float = 0.0,
) -> dict[str, float | int]:
    if len(numerator) != len(denominator):
        raise AnalysisError("speculative counter series are not aligned")
    values: list[float] = []
    for left_index in range(len(numerator) - 1):
        numerator_left = numerator[left_index]
        numerator_right = numerator[left_index + 1]
        denominator_left = denominator[left_index]
        denominator_right = denominator[left_index + 1]
        if (
            numerator_left.timestamp_ns != denominator_left.timestamp_ns
            or numerator_right.timestamp_ns != denominator_right.timestamp_ns
        ):
            raise AnalysisError("speculative counter timestamps are not aligned")
        numerator_delta = numerator_right.value - numerator_left.value
        denominator_delta = denominator_right.value - denominator_left.value
        if denominator_delta <= 0:
            continue
        value = additive + numerator_delta / denominator_delta
        if not math.isfinite(value) or value < additive:
            raise AnalysisError("invalid speculative counter ratio")
        values.append(value)
    if not values:
        raise AnalysisError("speculative counters have no positive scrape interval")

    numerator_delta = numerator[-1].value - numerator[0].value
    denominator_delta = denominator[-1].value - denominator[0].value
    overall = additive + numerator_delta / denominator_delta
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "overall": overall,
        "intervals": len(values),
    }


def analyze(
    summary_path: Path,
    jsonl_path: Path,
    *,
    target_concurrency: int,
    average_context_lower: float,
    average_context_upper: float,
    minimum_window_seconds: float,
    minimum_samples: int,
    minimum_exact_occupancy: float,
) -> dict[str, Any]:
    if target_concurrency < 1:
        raise AnalysisError("target concurrency must be positive")
    if not 0 < average_context_lower < average_context_upper:
        raise AnalysisError("average-context bounds must be positive and increasing")

    phase_start_ns, phase_end_ns = _load_phase_range(summary_path)
    loaded_records = _load_records(jsonl_path)
    all_phase_records = [
        record
        for record in loaded_records
        if phase_start_ns <= int(record["timestamp_ns"]) <= phase_end_ns
    ]
    baseline_records = [
        record
        for record in loaded_records
        if int(record["timestamp_ns"]) < phase_start_ns
    ]
    if not baseline_records:
        raise AnalysisError("vLLM metrics export has no pre-phase baseline scrape")
    baseline_record = baseline_records[-1:]
    baseline_generation = _counter_series(
        baseline_record,
        ("vllm:generation_tokens", "vllm:generation_tokens_total"),
    )
    if len(baseline_generation) != 1:
        raise AnalysisError("vLLM generation-token counter has no phase baseline")
    generation_baseline = baseline_generation[0].value
    baseline_prompt = _counter_series(
        baseline_record,
        ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
    )
    if len(baseline_prompt) != 1:
        raise AnalysisError("vLLM prompt-token counter has no phase baseline")
    prompt_baseline = baseline_prompt[0].value

    matching_indexes: list[int] = []
    for index, record in enumerate(all_phase_records):
        running_values = _metric_samples(
            record,
            ("vllm:num_requests_running",),
            required_labels=VLLM_LABELS,
        )
        generation_values = _metric_samples(
            record,
            ("vllm:generation_tokens", "vllm:generation_tokens_total"),
            required_labels=VLLM_LABELS,
        )
        prompt_values = _metric_samples(
            record,
            ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
            required_labels=VLLM_LABELS,
        )
        if not running_values or not generation_values or not prompt_values:
            continue
        average_context = (
            sum(generation_values)
            - generation_baseline
            + sum(prompt_values)
            - prompt_baseline
        )
        average_context /= target_concurrency
        if (
            max(running_values) == target_concurrency
            and average_context_lower <= average_context <= average_context_upper
        ):
            matching_indexes.append(index)
    if not matching_indexes:
        raise AnalysisError("no vLLM samples matched the average-context window")

    records = all_phase_records[matching_indexes[0] : matching_indexes[-1] + 1]
    window_start_ns = int(records[0]["timestamp_ns"])
    window_end_ns = int(records[-1]["timestamp_ns"])
    window_seconds = (window_end_ns - window_start_ns) / NANOSECONDS
    if window_seconds < minimum_window_seconds:
        raise AnalysisError(
            f"analysis window is {window_seconds:.3f}s; "
            f"need {minimum_window_seconds:.3f}s"
        )
    if len(records) < minimum_samples:
        raise AnalysisError(
            f"analysis window has {len(records)} scrapes; need {minimum_samples}"
        )

    decode_tokens = _counter_series(
        records,
        ("vllm:generation_tokens", "vllm:generation_tokens_total"),
    )
    engine_steps = _histogram_count_series(
        records,
        ("vllm:iteration_tokens_total", "vllm:iteration_tokens"),
    )
    prompt_compute = _counter_series(
        records,
        ("vllm:prompt_tokens", "vllm:prompt_tokens_total"),
    )
    prompt_cache = _counter_series(
        records,
        ("vllm:prompt_tokens_cached", "vllm:prompt_tokens_cached_total"),
    )
    running = _series(
        records,
        ("vllm:num_requests_running",),
        required_labels=VLLM_LABELS,
        reduction="max",
    )
    queue = _series(
        records,
        ("vllm:num_requests_waiting",),
        required_labels=VLLM_LABELS,
        reduction="max",
    )
    average_context = _context_series(
        records,
        generation_baseline=generation_baseline,
        prompt_baseline=prompt_baseline,
        target_concurrency=target_concurrency,
    )
    for name, points in (
        ("vLLM generation-token counter", decode_tokens),
        ("vLLM engine-step counter", engine_steps),
        ("vLLM prompt-compute counter", prompt_compute),
        ("vLLM prompt-cache counter", prompt_cache),
        ("vLLM running-request gauge", running),
        ("vLLM queued-request gauge", queue),
        ("derived average decode context", average_context),
    ):
        if len(points) < minimum_samples:
            raise AnalysisError(
                f"{name} has {len(points)} samples; need {minimum_samples}"
            )

    for name, points in (
        ("vLLM generation-token counter", decode_tokens),
        ("vLLM engine-step counter", engine_steps),
        ("vLLM prompt-compute counter", prompt_compute),
        ("vLLM prompt-cache counter", prompt_cache),
    ):
        _require_monotonic(points, name)

    decode_slope, decode_r_squared = _ols_slope(decode_tokens)
    step_slope, step_r_squared = _ols_slope(engine_steps)
    prompt_slope, prompt_r_squared = _ols_slope(prompt_compute)
    cache_slope, cache_r_squared = _ols_slope(prompt_cache)
    running_summary = _summarize_gauge(running)
    queue_summary = _summarize_gauge(queue)
    context_summary = _summarize_gauge(average_context)
    input_context_tokens = (
        prompt_compute[0].value - prompt_baseline
    ) / target_concurrency
    if input_context_tokens <= 0:
        raise AnalysisError(
            "vLLM prompt delta did not establish a positive input context"
        )

    decode_delta = decode_tokens[-1].value - decode_tokens[0].value
    step_delta = engine_steps[-1].value - engine_steps[0].value
    prompt_delta = prompt_compute[-1].value - prompt_compute[0].value
    cache_delta = prompt_cache[-1].value - prompt_cache[0].value
    if step_delta <= 0:
        raise AnalysisError("vLLM engine-step counter did not increase")
    exact_fraction = sum(point.value == target_concurrency for point in running) / len(
        running
    )

    validation = {
        "decode_counter_monotonic": True,
        "prefill_compute_counter_unchanged": prompt_delta == 0,
        "prefill_cache_counter_unchanged": cache_delta == 0,
        "queue_empty": queue_summary["max"] == 0,
        "exact_occupancy_fraction": exact_fraction,
        "exact_occupancy_required": minimum_exact_occupancy,
        "average_context_within_bounds": (
            context_summary["min"] >= average_context_lower
            and context_summary["max"] <= average_context_upper
        ),
        "valid": False,
    }
    failures: list[str] = []
    if not validation["prefill_compute_counter_unchanged"]:
        failures.append("vLLM prompt-compute counter changed during decode plateau")
    if not validation["prefill_cache_counter_unchanged"]:
        failures.append("vLLM prompt-cache counter changed during decode plateau")
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

    drafts = _required_speculative_counter_series(
        records,
        ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_drafts_total"),
    )
    accepted = _required_speculative_counter_series(
        records,
        (
            "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_accepted_tokens_total",
        ),
    )
    draft_tokens = _required_speculative_counter_series(
        records,
        (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_draft_tokens_total",
        ),
    )
    drafts_delta = drafts[-1].value - drafts[0].value
    accepted_delta = accepted[-1].value - accepted[0].value
    draft_tokens_delta = draft_tokens[-1].value - draft_tokens[0].value
    acceptance_length = _counter_ratio_distribution(
        accepted,
        drafts,
        additive=1.0,
    )
    acceptance_rate = _counter_ratio_distribution(
        accepted,
        draft_tokens,
    )
    if acceptance_rate["max"] > 1.0:
        raise AnalysisError("vLLM speculative acceptance rate exceeds 1")

    return {
        "schema_version": "1.1",
        "engine": "vllm",
        "source": {"summary": str(summary_path), "jsonl": str(jsonl_path)},
        "phase": {"start_ns": phase_start_ns, "end_ns": phase_end_ns},
        "plateau": {
            "start_ns": window_start_ns,
            "end_ns": window_end_ns,
            "duration_seconds": window_seconds,
            "selection": "derived_average_context_window",
            "settle_seconds": 0.0,
            "tail_seconds": 0.0,
            "average_context_lower": average_context_lower,
            "average_context_upper": average_context_upper,
            "input_context_tokens": input_context_tokens,
            "scrapes": len(records),
            "end_reason": "derived_average_context_window",
            "first_occupancy_departure_ns": None,
        },
        "average_context_length": context_summary,
        "decode": {
            "target_concurrency": target_concurrency,
            "counter_family": "vllm:generation_tokens{engine=0}",
            "tokens_per_second_ols": decode_slope,
            "tokens_per_second_delta": _delta_rate(decode_tokens),
            "counter_delta": decode_delta,
            "ols_r_squared": decode_r_squared,
        },
        "engine_work": {
            "unit": "speculative_decode_steps",
            "counter_family": "vllm:iteration_tokens_total_count{engine=0}",
            "forward_passes_per_second_ols": step_slope,
            "forward_passes_per_second_delta": _delta_rate(engine_steps),
            "counter_delta": step_delta,
            "ols_r_squared": step_r_squared,
            "useful_tokens_per_forward_per_request": (
                decode_delta / step_delta / target_concurrency
            ),
        },
        "prefill_control": {
            "compute": {
                "series_observed": True,
                "tokens_per_second_ols": prompt_slope,
                "tokens_per_second_delta": _delta_rate(prompt_compute),
                "counter_delta": prompt_delta,
                "ols_r_squared": prompt_r_squared,
            },
            "cache": {
                "series_observed": True,
                "tokens_per_second_ols": cache_slope,
                "tokens_per_second_delta": _delta_rate(prompt_cache),
                "counter_delta": cache_delta,
                "ols_r_squared": cache_r_squared,
            },
        },
        "running_requests": running_summary,
        "queued_requests": queue_summary,
        "server_cross_checks": {
            "spec_num_drafts_delta": drafts_delta,
            "spec_accepted_tokens_delta": accepted_delta,
            "spec_draft_tokens_delta": draft_tokens_delta,
            "spec_accept_rate": acceptance_rate,
            "spec_accept_length": acceptance_length,
        },
        "validation": validation,
        "failures": failures,
    }


def main() -> int:
    args = _parse_args()
    try:
        result = analyze(
            args.summary,
            args.jsonl,
            target_concurrency=args.target_concurrency,
            average_context_lower=args.average_context_lower,
            average_context_upper=args.average_context_upper,
            minimum_window_seconds=args.minimum_window_seconds,
            minimum_samples=args.minimum_samples,
            minimum_exact_occupancy=args.minimum_exact_occupancy,
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

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("analyze_server_metrics.py")
SPEC = importlib.util.spec_from_file_location("analyze_server_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_capture(tmp_path: Path, *, contaminated: bool = False) -> tuple[Path, Path]:
    start_ns = 1_000_000_000_000
    end_ns = start_ns + 60_000_000_000
    summary = tmp_path / "server_metrics_export.json"
    summary.write_text(
        json.dumps(
            {
                "summary": {
                    "phase_time_ranges": {
                        "profiling": {"start_ns": start_ns, "end_ns": end_ns}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    jsonl = tmp_path / "server_metrics_export.jsonl"
    lines = []
    for second in range(61):
        prompt_tokens = 256
        running = 4
        if contaminated and second >= 30:
            prompt_tokens += 128
        if contaminated and second == 30:
            running = 5
        metrics = {
            "sglang:realtime_tokens": [
                {
                    "labels": {"mode": "decode", "tp_rank": "0"},
                    "value": 10_000 + 400 * second,
                },
                {
                    "labels": {"mode": "prefill_compute", "tp_rank": "0"},
                    "value": prompt_tokens,
                },
                {
                    "labels": {"mode": "prefill_cache", "tp_rank": "0"},
                    "value": 0,
                },
            ],
            "sglang:generation_tokens": [
                {
                    "labels": {"is_streaming": "true"},
                    "value": 999,
                },
                {
                    "labels": {"is_streaming": "false"},
                    "value": 999_999,
                },
            ],
            "sglang:prompt_tokens": [
                {"labels": {"is_streaming": "true"}, "value": prompt_tokens}
            ],
            "sglang:num_running_reqs": [{"labels": {"tp_rank": "0"}, "value": running}],
            "sglang:num_queue_reqs": [{"labels": {"tp_rank": "0"}, "value": 0}],
            "sglang:decode_sum_seq_lens": [
                {
                    "labels": {"tp_rank": "0"},
                    "value": 4 * (1000 + 100 * second),
                }
            ],
            "sglang:cuda_graph_passes": [
                {
                    "labels": {"mode": "decode_cuda_graph", "tp_rank": "0"},
                    "value": 100 + 20 * second,
                },
                {
                    "labels": {"mode": "decode_none", "tp_rank": "0"},
                    "value": 2,
                },
                {
                    "labels": {"mode": "prefill_none", "tp_rank": "0"},
                    "value": 50 + second,
                },
                {
                    "labels": {"mode": "decode_cuda_graph", "tp_rank": "1"},
                    "value": 1_000 + 30 * second,
                },
            ],
            "sglang:gen_throughput": [{"labels": {"tp_rank": "0"}, "value": 400}],
            "sglang:spec_accept_rate": [{"labels": {"tp_rank": "0"}, "value": 0.75}],
        }
        lines.append(
            json.dumps(
                {"timestamp_ns": start_ns + second * 1_000_000_000, "metrics": metrics}
            )
        )
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, jsonl


def _write_terminal_drop(jsonl: Path) -> None:
    rewritten = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        second = (record["timestamp_ns"] - 1_000_000_000_000) // 1_000_000_000
        if second >= 50:
            record["metrics"]["sglang:num_running_reqs"][0]["value"] = 3
        rewritten.append(json.dumps(record))
    jsonl.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _omit_cache_series(jsonl: Path) -> None:
    rewritten = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["metrics"]["sglang:realtime_tokens"] = [
            sample
            for sample in record["metrics"]["sglang:realtime_tokens"]
            if sample["labels"]["mode"] != "prefill_cache"
        ]
        rewritten.append(json.dumps(record))
    jsonl.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _move_requests_to_second_dp_rank(jsonl: Path) -> None:
    rewritten = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        metrics = record["metrics"]
        metrics["sglang:num_running_reqs"] = [
            {
                "labels": {"dp_rank": "0", "tp_rank": "0"},
                "value": 0,
            },
            {
                "labels": {"dp_rank": "1", "tp_rank": "1"},
                "value": 4,
            },
        ]
        metrics["sglang:num_queue_reqs"] = [
            {
                "labels": {"dp_rank": "0", "tp_rank": "0"},
                "value": 0,
            },
            {
                "labels": {"dp_rank": "1", "tp_rank": "1"},
                "value": 0,
            },
        ]
        active_context = metrics["sglang:decode_sum_seq_lens"][0]["value"]
        metrics["sglang:decode_sum_seq_lens"] = [
            {
                "labels": {"dp_rank": "0", "tp_rank": "0"},
                "value": 123_456,
            },
            {
                "labels": {"dp_rank": "1", "tp_rank": "1"},
                "value": active_context,
            },
        ]
        for family in ("sglang:realtime_tokens", "sglang:cuda_graph_passes"):
            metrics[family] = [
                sample
                for sample in metrics[family]
                if sample["labels"]["tp_rank"] == "0"
            ]
            for sample in metrics[family]:
                sample["labels"]["dp_rank"] = "1"
                sample["labels"]["tp_rank"] = "1"
        metrics["sglang:realtime_tokens"].extend(
            [
                {
                    "labels": {
                        "dp_rank": "0",
                        "tp_rank": "0",
                        "mode": mode,
                    },
                    "value": 7,
                }
                for mode in ("decode", "prefill_compute", "prefill_cache")
            ]
        )
        metrics["sglang:cuda_graph_passes"].append(
            {
                "labels": {
                    "dp_rank": "0",
                    "tp_rank": "0",
                    "mode": "decode_cuda_graph",
                },
                "value": 7,
            }
        )
        rewritten.append(json.dumps(record))
    jsonl.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def _analyze(summary: Path, jsonl: Path):
    return MODULE.analyze(
        summary,
        jsonl,
        target_concurrency=4,
        settle_seconds=10,
        tail_seconds=5,
        minimum_window_seconds=30,
        minimum_samples=30,
        minimum_exact_occupancy=0.98,
    )


def test_valid_plateau_uses_streaming_counter_only(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path)
    result = _analyze(summary, jsonl)
    assert result["validation"]["valid"] is True
    assert result["decode"]["tokens_per_second_ols"] == pytest.approx(400)
    assert result["decode"]["tokens_per_second_delta"] == pytest.approx(400)
    assert result["engine_work"]["forward_passes_per_second_ols"] == pytest.approx(20)
    assert result["engine_work"]["forward_passes_per_second_delta"] == pytest.approx(20)
    assert result["engine_work"][
        "useful_tokens_per_forward_per_request"
    ] == pytest.approx(5)
    assert result["prefill_control"]["compute"]["counter_delta"] == 0
    assert result["running_requests"]["mean"] == 4


def test_prefill_or_extra_request_invalidates_plateau(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path, contaminated=True)
    result = _analyze(summary, jsonl)
    assert result["validation"]["valid"] is False
    assert "prefill-compute counter changed during plateau" in result["failures"]
    assert any("exact occupancy" in failure for failure in result["failures"])


def test_terminal_cohort_completion_defines_plateau_end(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path)
    _write_terminal_drop(jsonl)
    result = _analyze(summary, jsonl)
    assert result["validation"]["valid"] is True
    assert result["plateau"]["end_reason"] == "terminal_occupancy_drop"
    assert result["plateau"]["end_ns"] == 1_045_000_000_000


def test_never_instantiated_zero_cache_label_is_valid(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path)
    _omit_cache_series(jsonl)
    result = _analyze(summary, jsonl)
    assert result["validation"]["valid"] is True
    assert result["prefill_control"]["cache"]["counter_delta"] == 0
    assert result["prefill_control"]["cache"]["series_observed"] is False


def test_context_window_compares_equal_sequence_shapes(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path)
    result = MODULE.analyze(
        summary,
        jsonl,
        target_concurrency=4,
        settle_seconds=10,
        tail_seconds=5,
        minimum_window_seconds=20,
        minimum_samples=20,
        minimum_exact_occupancy=0.98,
        average_context_lower=2000,
        average_context_upper=5000,
    )
    assert result["validation"]["valid"] is True
    assert result["plateau"]["end_reason"] == "average_context_window"
    assert result["average_context_length"]["min"] == 2000
    assert result["average_context_length"]["max"] == 5000


def test_context_window_follows_active_dp_rank(tmp_path: Path) -> None:
    summary, jsonl = _write_capture(tmp_path)
    _move_requests_to_second_dp_rank(jsonl)
    result = MODULE.analyze(
        summary,
        jsonl,
        target_concurrency=4,
        settle_seconds=10,
        tail_seconds=5,
        minimum_window_seconds=20,
        minimum_samples=20,
        minimum_exact_occupancy=0.98,
        average_context_lower=2000,
        average_context_upper=5000,
    )
    assert result["validation"]["valid"] is True
    assert result["decode"]["tokens_per_second_ols"] == pytest.approx(400)
    assert result["engine_work"]["forward_passes_per_second_ols"] == pytest.approx(20)
    assert result["engine_work"][
        "useful_tokens_per_forward_per_request"
    ] == pytest.approx(5)
    assert result["average_context_length"]["min"] == 2000
    assert result["average_context_length"]["max"] == 5000
    assert result["engine_work"]["active_request_ranks"]["median"] == 1

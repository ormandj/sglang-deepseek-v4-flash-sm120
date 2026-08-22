from __future__ import annotations

import json
from pathlib import Path

from analyze_turnover import analyze


def _write_fixture(tmp_path: Path, *, prefill_passes: float = 2.0) -> tuple[Path, ...]:
    summary = tmp_path / "profile_export_aiperf.json"
    records = tmp_path / "profile_export.jsonl"
    server_summary = tmp_path / "server_metrics_export.json"
    server_jsonl = tmp_path / "server_metrics_export.jsonl"
    summary.write_text(
        json.dumps(
            {
                "was_cancelled": False,
                "error_summary": [],
                "request_count": {"unit": "requests", "avg": 4.0},
                "output_token_throughput": {
                    "unit": "tokens/sec",
                    "avg": 200.0,
                },
                "summary": {
                    "phase_time_ranges": {
                        "profiling": {"start_ns": 0, "end_ns": 4_000_000_000}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    request_rows = []
    for index, start_ns in enumerate((0, 0, 2_000_000_000, 2_000_000_000)):
        request_rows.append(
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                    "request_start_ns": start_ns,
                    "request_end_ns": start_ns + 2_000_000_000,
                },
                "metrics": {
                    "input_sequence_length": {
                        "value": 260 + index % 2,
                        "unit": "tokens",
                    },
                    "output_sequence_length": {"value": 256, "unit": "tokens"},
                    "time_to_first_token": {"value": 100, "unit": "ms"},
                    "inter_token_latency": {"value": 7, "unit": "ms"},
                    "request_latency": {"value": 1900, "unit": "ms"},
                },
            }
        )
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
    )
    server_summary.write_text(
        json.dumps(
            {
                "summary": {
                    "phase_time_ranges": {
                        "profiling": {"start_ns": 0, "end_ns": 4_000_000_000}
                    }
                },
                "metrics": {
                    "sglang:cuda_graph_passes": {
                        "series": [
                            {
                                "labels": {"mode": "prefill_none", "tp_rank": "0"},
                                "stats": {"total": prefill_passes, "rate": 0.5},
                            }
                        ]
                    },
                    "sglang:http_requests": {
                        "series": [
                            {
                                "labels": {
                                    "endpoint": "/v1/chat/completions",
                                    "method": "POST",
                                },
                                "stats": {"total": 4.0, "rate": 1.0},
                            },
                            {
                                "labels": {"endpoint": "/health", "method": "GET"},
                                "stats": {"total": 2.0, "rate": 0.5},
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    metric_rows = []
    for second in range(5):
        metric_rows.append(
            {
                "timestamp_ns": second * 1_000_000_000,
                "metrics": {
                    "sglang:num_running_reqs": [
                        {"labels": {"tp_rank": "0"}, "value": 2}
                    ],
                    "sglang:num_queue_reqs": [{"labels": {"tp_rank": "0"}, "value": 0}],
                },
            }
        )
    server_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows), encoding="utf-8"
    )
    return summary, records, server_summary, server_jsonl


def _analyze(tmp_path: Path, *, prefill_passes: float = 2.0) -> dict:
    paths = _write_fixture(tmp_path, prefill_passes=prefill_passes)
    return analyze(
        *paths,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )


def test_valid_turnover_cell_records_effective_refill_batching(tmp_path: Path) -> None:
    result = _analyze(tmp_path)
    assert result["validation"]["valid"] is True
    assert result["requests"]["output_tokens_per_second"] == 200
    assert (
        result["client_concurrency"]["admission_window"]["fraction_of_window_at_target"]
        == 1
    )
    assert result["refill_batching"]["requests_per_prefill_pass"] == 2


def test_single_request_prefills_are_measured_without_shape_specific_rejection(
    tmp_path: Path,
) -> None:
    result = _analyze(tmp_path, prefill_passes=4)
    assert result["validation"]["valid"] is True
    assert result["refill_batching"]["requests_per_prefill_pass"] == 1


def test_wrong_server_occupancy_invalidates_cell(tmp_path: Path) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in server_jsonl.read_text().splitlines()]
    for row in rows:
        row["metrics"]["sglang:num_running_reqs"][0]["value"] = 1
    server_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )
    assert result["validation"]["valid"] is False
    assert "server running occupancy reached only 1; expected 2" in result["failures"]


def test_finished_server_entries_can_temporarily_exceed_client_concurrency(
    tmp_path: Path,
) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in server_jsonl.read_text().splitlines()]
    rows[2]["metrics"]["sglang:num_running_reqs"][0]["value"] = 3
    server_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )
    assert result["validation"]["valid"] is True
    assert result["server_occupancy"]["running"]["max"] == 3
    assert result["server_requests"]["chat_completion_posts"] == 4


def test_untracked_chat_request_invalidates_cell(tmp_path: Path) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    document = json.loads(server_summary.read_text())
    document["metrics"]["sglang:http_requests"]["series"][0]["stats"]["total"] = 5
    server_summary.write_text(json.dumps(document), encoding="utf-8")
    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )
    assert result["validation"]["valid"] is False
    assert "server observed 5 chat-completion POSTs; expected 4" in result["failures"]


def test_http_request_counter_accepts_total_suffix_and_deduplicates_tp_ranks(
    tmp_path: Path,
) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    document = json.loads(server_summary.read_text())
    metric = document["metrics"].pop("sglang:http_requests")
    chat = metric["series"][0]
    chat["labels"]["tp_rank"] = "0"
    duplicate = json.loads(json.dumps(chat))
    duplicate["labels"]["tp_rank"] = "1"
    metric["series"].append(duplicate)
    document["metrics"]["sglang:http_requests_total"] = metric
    server_summary.write_text(json.dumps(document), encoding="utf-8")

    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )

    assert result["validation"]["valid"] is True
    assert result["server_requests"]["chat_completion_posts"] == 4


def test_impossible_server_occupancy_invalidates_cell(tmp_path: Path) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in server_jsonl.read_text().splitlines()]
    rows[2]["metrics"]["sglang:num_running_reqs"][0]["value"] = 5
    server_jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )

    assert result["validation"]["valid"] is False
    assert "server running occupancy reached 5; expected at most 4" in result["failures"]


def test_client_gap_during_admissions_invalidates_cell(tmp_path: Path) -> None:
    summary, records, server_summary, server_jsonl = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    for row in rows[2:]:
        row["metadata"]["request_start_ns"] = 2_500_000_000
        row["metadata"]["request_end_ns"] = 4_500_000_000
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    result = analyze(
        summary,
        records,
        server_summary,
        server_jsonl,
        target_concurrency=2,
        expected_requests=4,
        target_isl=256,
        target_osl=256,
        isl_tolerance=16,
        minimum_client_occupancy=0.90,
    )
    assert result["validation"]["valid"] is False
    assert any("of the admission window" in failure for failure in result["failures"])

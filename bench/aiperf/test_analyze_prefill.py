from __future__ import annotations

import json
from pathlib import Path

from analyze_prefill import analyze


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    summary_path = tmp_path / "profile_export_aiperf.json"
    records_path = tmp_path / "profile_export.jsonl"
    server_path = tmp_path / "server_metrics_export.json"
    summary_path.write_text(
        json.dumps(
            {
                "was_cancelled": False,
                "error_summary": [],
                "request_count": {"unit": "requests", "avg": 4.0},
            }
        ),
        encoding="utf-8",
    )
    records = []
    # Two back-to-back C2 waves. Each prefill is 1 second.
    for index, start_ns in enumerate((0, 0, 1_000_000_000, 1_000_000_000)):
        records.append(
            {
                "metadata": {
                    "benchmark_phase": "profiling",
                    "was_cancelled": False,
                    "request_start_ns": start_ns,
                    "request_end_ns": start_ns + 1_000_000_000,
                },
                "metrics": {
                    "time_to_first_token": {"value": 1000.0, "unit": "ms"},
                    "input_sequence_length": {
                        "value": 8197 + index % 2,
                        "unit": "tokens",
                    },
                    "output_sequence_length": {"value": 1, "unit": "tokens"},
                },
            }
        )
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    server_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "sglang:realtime_tokens": {
                        "series": [
                            {
                                "labels": {"mode": "prefill_compute", "tp_rank": "0"},
                                "stats": {"total": 33792.0, "rate": 16000.0},
                            },
                            {
                                "labels": {"mode": "prefill_cache", "tp_rank": "0"},
                                "stats": {"total": 0.0, "rate": 0.0},
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path, records_path, server_path


def _analyze(tmp_path: Path) -> dict:
    summary, records, server = _write_fixture(tmp_path)
    return analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )


def test_valid_cold_prefill_cell(tmp_path: Path) -> None:
    result = _analyze(tmp_path)
    assert result["validation"]["valid"] is True
    assert result["prefill_concurrency"]["max"] == 2
    assert result["prefill_concurrency"]["fraction_of_window_at_target"] == 1.0
    assert result["requests"]["time_to_first_token_ms"]["median"] == 1000.0
    assert result["server_controls"]["cache"]["total_tokens"] == 0.0


def test_cached_tokens_invalidate_cell(tmp_path: Path) -> None:
    summary, records, server = _write_fixture(tmp_path)
    document = json.loads(server.read_text(encoding="utf-8"))
    document["metrics"]["sglang:realtime_tokens"]["series"][1]["stats"]["total"] = 256.0
    server.write_text(json.dumps(document), encoding="utf-8")
    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )
    assert result["validation"]["valid"] is False
    assert "server attributed 256 tokens to prefill cache" in result["failures"]


def test_absent_sglang_cache_mode_is_zero_when_counter_family_exists(
    tmp_path: Path,
) -> None:
    summary, records, server = _write_fixture(tmp_path)
    document = json.loads(server.read_text(encoding="utf-8"))
    series = document["metrics"]["sglang:realtime_tokens"]["series"]
    document["metrics"]["sglang:realtime_tokens"]["series"] = [
        row for row in series if row["labels"]["mode"] != "prefill_cache"
    ]
    server.write_text(json.dumps(document), encoding="utf-8")

    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )

    assert result["validation"]["valid"] is True
    assert result["server_controls"]["cache"]["total_tokens"] == 0.0


def test_dp_owned_counters_sum_owners_and_deduplicate_tp_replicas(
    tmp_path: Path,
) -> None:
    summary, records, server = _write_fixture(tmp_path)
    document = json.loads(server.read_text(encoding="utf-8"))
    series = []
    for mode, total, rate in (
        ("prefill_compute", 16896.0, 8000.0),
        ("prefill_cache", 0.0, 0.0),
    ):
        for dp_rank in ("0", "1"):
            for tp_rank in ("0", "1"):
                series.append(
                    {
                        "labels": {
                            "mode": mode,
                            "dp_rank": dp_rank,
                            "tp_rank": tp_rank,
                        },
                        "stats": {"total": total, "rate": rate},
                    }
                )
    document["metrics"]["sglang:realtime_tokens"]["series"] = series
    server.write_text(json.dumps(document), encoding="utf-8")

    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )

    assert result["validation"]["valid"] is True
    assert result["server_controls"]["compute"]["total_tokens"] == 33792.0
    assert result["server_controls"]["compute"]["tokens_per_second"] == 16000.0


def test_missing_record_and_concurrency_invalidate_cell(tmp_path: Path) -> None:
    summary, records, server = _write_fixture(tmp_path)
    lines = records.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        document = json.loads(line)
        document["metadata"]["request_start_ns"] = index * 1_000_000_000
        document["metadata"]["request_end_ns"] = (index + 1) * 1_000_000_000
        lines[index] = json.dumps(document)
    records.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )
    assert result["validation"]["valid"] is False
    assert "record export has 3 requests; expected 4" in result["failures"]
    assert "client reached prefill concurrency 1; expected 2" in result["failures"]


def test_error_record_is_reported_without_hiding_other_evidence(tmp_path: Path) -> None:
    summary, records, server = _write_fixture(tmp_path)
    summary_document = json.loads(summary.read_text(encoding="utf-8"))
    summary_document["request_count"]["avg"] = 3.0
    summary_document["error_summary"] = [{"count": 1}]
    summary.write_text(json.dumps(summary_document), encoding="utf-8")
    lines = records.read_text(encoding="utf-8").splitlines()
    error_record = json.loads(lines[0])
    error_record["metrics"] = {"error_isl": {"value": 8197, "unit": "tokens"}}
    error_record["error"] = {"type": "InvalidInferenceResultError"}
    lines[0] = json.dumps(error_record)
    records.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
    )
    assert result["validation"]["valid"] is False
    assert "record 0 failed with InvalidInferenceResultError" in result["failures"]
    assert result["requests"]["completed"] == 3


def test_vllm_prompt_counters_validate_cold_prefill(tmp_path: Path) -> None:
    summary, records, server = _write_fixture(tmp_path)
    document = json.loads(server.read_text(encoding="utf-8"))
    document["metrics"] = {
        "vllm:prompt_tokens": {
            "series": [
                {
                    "labels": {"engine": "0"},
                    "stats": {"total": 33792.0, "rate": 16000.0},
                }
            ]
        },
        "vllm:prompt_tokens_cached": {
            "series": [
                {
                    "labels": {"engine": "0"},
                    "stats": {"total": 0.0, "rate": 0.0},
                }
            ]
        },
    }
    server.write_text(json.dumps(document), encoding="utf-8")
    result = analyze(
        summary,
        records,
        server,
        target_isl=8192,
        target_concurrency=2,
        expected_requests=4,
        isl_tolerance=16,
        maximum_compute_ratio=1.10,
        engine="vllm",
    )
    assert result["validation"]["valid"] is True
    assert result["engine"] == "vllm"
    assert result["server_controls"]["cache"]["total_tokens"] == 0

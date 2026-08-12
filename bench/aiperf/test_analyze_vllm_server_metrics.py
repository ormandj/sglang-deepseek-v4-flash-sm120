from __future__ import annotations

import json
from pathlib import Path

import pytest
from analyze_vllm_server_metrics import analyze


def _write_capture(
    tmp_path: Path, *, prompt_during_decode: bool = False
) -> tuple[Path, Path]:
    summary = tmp_path / "server_metrics_export.json"
    jsonl = tmp_path / "server_metrics_export.jsonl"
    summary.write_text(
        json.dumps(
            {
                "summary": {
                    "phase_time_ranges": {
                        "profiling": {
                            "start_ns": 1_000_000_000,
                            "end_ns": 60_000_000_000,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for second in range(61):
        prompt_tokens = 1000 if second == 0 else 2360
        if prompt_during_decode and second > 0:
            prompt_tokens += second
        metrics = {
            "vllm:num_requests_running": [{"labels": {"engine": "0"}, "value": 4}],
            "vllm:num_requests_waiting": [{"labels": {"engine": "0"}, "value": 0}],
            "vllm:generation_tokens_total": [
                {"labels": {"engine": "0"}, "value": 100 + 400 * second}
            ],
            "vllm:iteration_tokens_total": [
                {
                    "labels": {"engine": "0"},
                    "count": 20 * second,
                    "sum": 400 * second,
                    "buckets": {"+Inf": 20 * second},
                }
            ],
            "vllm:prompt_tokens_total": [
                {"labels": {"engine": "0"}, "value": prompt_tokens}
            ],
            "vllm:prompt_tokens_cached_total": [
                {"labels": {"engine": "0"}, "value": 0}
            ],
            "vllm:spec_decode_num_drafts_total": [
                {"labels": {"engine": "0"}, "value": 80 * second}
            ],
            "vllm:spec_decode_num_draft_tokens_total": [
                {"labels": {"engine": "0"}, "value": 400 * second}
            ],
            "vllm:spec_decode_num_accepted_tokens_total": [
                {"labels": {"engine": "0"}, "value": 320 * second}
            ],
        }
        rows.append(
            json.dumps({"timestamp_ns": second * 1_000_000_000, "metrics": metrics})
        )
    jsonl.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary, jsonl


def _analyze(tmp_path: Path, *, prompt_during_decode: bool = False) -> dict:
    summary, jsonl = _write_capture(tmp_path, prompt_during_decode=prompt_during_decode)
    return analyze(
        summary,
        jsonl,
        target_concurrency=4,
        average_context_lower=1000,
        average_context_upper=3000,
        minimum_window_seconds=10,
        minimum_samples=20,
        minimum_exact_occupancy=0.98,
    )


def test_vllm_engine_step_rate_uses_equal_context_window(tmp_path: Path) -> None:
    result = _analyze(tmp_path)
    assert result["validation"]["valid"] is True
    assert result["engine_work"]["forward_passes_per_second_ols"] == pytest.approx(20)
    assert result["decode"]["tokens_per_second_ols"] == pytest.approx(400)
    assert result["engine_work"][
        "useful_tokens_per_forward_per_request"
    ] == pytest.approx(5)
    acceptance_length = result["server_cross_checks"]["spec_accept_length"]
    acceptance_rate = result["server_cross_checks"]["spec_accept_rate"]
    for statistic in ("mean", "median", "min", "max", "overall"):
        assert acceptance_length[statistic] == pytest.approx(5)
        assert acceptance_rate[statistic] == pytest.approx(0.8)
    assert acceptance_length["intervals"] > 0
    assert acceptance_rate["intervals"] > 0


def test_vllm_prefill_inside_decode_window_is_invalid(tmp_path: Path) -> None:
    result = _analyze(tmp_path, prompt_during_decode=True)
    assert result["validation"]["valid"] is False
    assert "vLLM prompt-compute counter changed" in result["failures"][0]

from __future__ import annotations

import json

import pytest
from summarize_engine_gate import SummaryError, summarize


def _write(path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _speculative(repetition: int) -> dict:
    return {
        "spec_accept_rate": {
            "mean": 0.70 + repetition / 100,
            "median": 0.71 + repetition / 100,
            "min": 0.50,
            "max": 0.90,
        },
        "spec_accept_length": {
            "mean": 4.0 + repetition / 10,
            "median": 4.1 + repetition / 10,
            "min": 2.0,
            "max": 6.0,
        },
    }


def _gate(tmp_path):
    for concurrency, repetitions in {1: 3, 4: 3, 8: 3}.items():
        for repetition in range(1, repetitions + 1):
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "decode-analysis.json",
                {
                    "validation": {"valid": True},
                    "decode": {
                        "target_concurrency": concurrency,
                        "tokens_per_second_ols": 100 * concurrency + repetition,
                    },
                    "engine_work": {
                        "forward_passes_per_second_ols": 50 + repetition,
                        "useful_tokens_per_forward_per_request": 5.5,
                    },
                    "server_cross_checks": _speculative(repetition),
                },
            )
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "profile_export_aiperf.json",
                {
                    metric: {
                        "unit": "ms",
                        "avg": 10 + repetition,
                        "p50": 9 + repetition,
                        "p90": 11 + repetition,
                        "p99": 12 + repetition,
                    }
                    for metric in (
                        "time_to_first_token",
                        "inter_token_latency",
                        "request_latency",
                    )
                },
            )
    for label in ("8k-c1", "32k-c1", "64k-c1", "128k-c1"):
        _write(
            tmp_path / "prefill" / label / "prefill-analysis.json",
            {
                "validation": {"valid": True},
                "requests": {
                    "aggregate_prompt_tokens_per_second": 8000,
                    "time_to_first_token_ms": {"median": 1000},
                    "completed": 3,
                },
            },
        )
    return tmp_path


def test_summarize_quick_gate_retains_every_repetition(tmp_path) -> None:
    result = summarize(_gate(tmp_path), mode="quick", build_id="rc3")
    c1 = result["decode"]["c1"]
    assert result["schema_version"] == "1.2"
    assert c1["engine_forward_passes_per_second"]["count"] == 3
    assert c1["synthetic_decode_tokens_per_second"]["count"] == 3
    assert c1["output_tokens_per_forward_per_request"]["count"] == 3
    assert c1["engine_forward_passes_per_second"]["median"] == 52
    assert [row["id"] for row in c1["repetitions"]] == ["r01", "r02", "r03"]
    assert c1["repetitions"][0]["speculative"]["accept_rate"]["mean"] == 0.71
    assert c1["speculative"]["accept_rate"]["mean_per_run"]["median"] == 0.72
    assert c1["speculative"]["accept_length"]["median_per_run"]["median"] == 4.3
    assert c1["client_latency_ms"]["time_to_first_token"][
        "p50_per_run"
    ]["median"] == 11
    assert result["prefill"]["128k-c1"]["prompt_tokens_per_second"] == 8000


def test_exploratory_decode_requires_priority_cells_without_prefill(tmp_path) -> None:
    root = _gate(tmp_path)
    for repetition in range(1, 4):
        run_id = f"r{repetition:02d}"
        c1 = root / "decode" / "c1" / run_id
        analysis = json.loads(
            (c1 / "decode-analysis.json").read_text(encoding="utf-8")
        )
        analysis["decode"]["target_concurrency"] = 2
        _write(root / "decode" / "c2" / run_id / "decode-analysis.json", analysis)
        profile = json.loads(
            (c1 / "profile_export_aiperf.json").read_text(encoding="utf-8")
        )
        _write(
            root / "decode" / "c2" / run_id / "profile_export_aiperf.json",
            profile,
        )
    for path in root.glob("prefill/*/prefill-analysis.json"):
        path.unlink()

    result = summarize(root, mode="exploratory-decode", build_id="0.8.0-rc5")

    assert set(result["decode"]) == {"c1", "c2", "c4", "c8"}
    assert result["prefill"] == {}


def test_summarize_rejects_missing_repetition(tmp_path) -> None:
    root = _gate(tmp_path)
    path = root / "decode" / "c1" / "r03" / "decode-analysis.json"
    path.unlink()
    with pytest.raises(SummaryError, match="C1 has 2 repetitions"):
        summarize(root, mode="quick", build_id="rc3")


def test_summarize_rejects_invalid_cell(tmp_path) -> None:
    root = _gate(tmp_path)
    path = root / "decode" / "c8" / "r01" / "decode-analysis.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["validation"]["valid"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SummaryError, match="C8/r01 is invalid"):
        summarize(root, mode="quick", build_id="rc3")


def test_vllm_quick_gate_uses_the_same_supported_panel(tmp_path) -> None:
    root = _gate(tmp_path)
    result = summarize(root, mode="quick", build_id="r33", engine="vllm")
    assert result["engine"] == "vllm"
    assert set(result["decode"]) == {"c1", "c4", "c8"}


def test_decode_supplement_requires_only_mid_concurrency_cells(tmp_path) -> None:
    for concurrency, repetitions in {2: 3, 16: 3}.items():
        for repetition in range(1, repetitions + 1):
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "decode-analysis.json",
                {
                    "validation": {"valid": True},
                    "decode": {
                        "target_concurrency": concurrency,
                        "tokens_per_second_ols": 100 * concurrency + repetition,
                    },
                    "engine_work": {
                        "forward_passes_per_second_ols": 50 + repetition,
                        "useful_tokens_per_forward_per_request": 5.5,
                    },
                    "server_cross_checks": _speculative(repetition),
                },
            )
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "profile_export_aiperf.json",
                {
                    metric: {
                        "unit": "ms",
                        "avg": 10 + repetition,
                        "p50": 9 + repetition,
                        "p90": 11 + repetition,
                        "p99": 12 + repetition,
                    }
                    for metric in (
                        "time_to_first_token",
                        "inter_token_latency",
                        "request_latency",
                    )
                },
            )

    result = summarize(
        tmp_path,
        mode="decode-supplement",
        build_id="rc3",
        engine="sglang",
    )

    assert set(result["decode"]) == {"c2", "c16"}
    assert result["prefill"] == {}


def test_repeat_c2_c4_requires_five_independent_repetitions(tmp_path) -> None:
    for concurrency in (2, 4):
        for repetition in range(1, 6):
            run = tmp_path / "decode" / f"c{concurrency}" / f"r{repetition:02d}"
            _write(
                run / "decode-analysis.json",
                {
                    "validation": {"valid": True},
                    "decode": {
                        "target_concurrency": concurrency,
                        "tokens_per_second_ols": 100 * concurrency + repetition,
                    },
                    "engine_work": {
                        "forward_passes_per_second_ols": 50 + repetition,
                        "useful_tokens_per_forward_per_request": 5.5,
                    },
                    "server_cross_checks": _speculative(repetition),
                },
            )
            _write(
                run / "profile_export_aiperf.json",
                {
                    metric: {
                        "unit": "ms",
                        "avg": 10 + repetition,
                        "p50": 9 + repetition,
                        "p90": 11 + repetition,
                        "p99": 12 + repetition,
                    }
                    for metric in (
                        "time_to_first_token",
                        "inter_token_latency",
                        "request_latency",
                    )
                },
            )

    result = summarize(tmp_path, mode="repeat-c2-c4", build_id="rc2-repeat")

    assert set(result["decode"]) == {"c2", "c4"}
    assert result["decode"]["c2"]["engine_forward_passes_per_second"]["count"] == 5
    assert result["prefill"] == {}


def test_repeat_c8_requires_five_independent_repetitions(tmp_path) -> None:
    for repetition in range(1, 6):
        run = tmp_path / "decode" / "c8" / f"r{repetition:02d}"
        _write(
            run / "decode-analysis.json",
            {
                "validation": {"valid": True},
                "decode": {
                    "target_concurrency": 8,
                    "tokens_per_second_ols": 800 + repetition,
                },
                "engine_work": {
                    "forward_passes_per_second_ols": 50 + repetition,
                    "useful_tokens_per_forward_per_request": 5.5,
                },
                "server_cross_checks": _speculative(repetition),
            },
        )
        _write(
            run / "profile_export_aiperf.json",
            {
                metric: {
                    "unit": "ms",
                    "avg": 10 + repetition,
                    "p50": 9 + repetition,
                    "p90": 11 + repetition,
                    "p99": 12 + repetition,
                }
                for metric in (
                    "time_to_first_token",
                    "inter_token_latency",
                    "request_latency",
                )
            },
        )

    result = summarize(tmp_path, mode="repeat-c8", build_id="rc10-repeat")

    assert set(result["decode"]) == {"c8"}
    assert result["decode"]["c8"]["engine_forward_passes_per_second"]["count"] == 5
    assert result["prefill"] == {}


def test_prefill_quick_requires_only_quick_prefill_cells(tmp_path) -> None:
    root = _gate(tmp_path)
    for path in root.glob("decode/c*/r*/decode-analysis.json"):
        path.unlink()

    result = summarize(root, mode="prefill-quick", build_id="rc17")

    assert result["decode"] == {}
    assert set(result["prefill"]) == {
        "8k-c1",
        "32k-c1",
        "64k-c1",
        "128k-c1",
    }


def test_publication_requires_five_repetitions_and_prefill_requests(tmp_path) -> None:
    for concurrency in (1, 2, 4, 8, 16, 32):
        for repetition in range(1, 6):
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "decode-analysis.json",
                {
                    "validation": {"valid": True},
                    "decode": {
                        "target_concurrency": concurrency,
                        "tokens_per_second_ols": 100 * concurrency + repetition,
                    },
                    "engine_work": {
                        "forward_passes_per_second_ols": 50 + repetition,
                        "useful_tokens_per_forward_per_request": 5.5,
                    },
                    "server_cross_checks": _speculative(repetition),
                },
            )
            _write(
                tmp_path
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "profile_export_aiperf.json",
                {
                    metric: {
                        "unit": "ms",
                        "avg": 10 + repetition,
                        "p50": 9 + repetition,
                        "p90": 11 + repetition,
                        "p99": 12 + repetition,
                    }
                    for metric in (
                        "time_to_first_token",
                        "inter_token_latency",
                        "request_latency",
                    )
                },
            )
    for label in ("8k-c1", "32k-c1", "64k-c1", "128k-c1"):
        _write(
            tmp_path / "prefill" / label / "prefill-analysis.json",
            {
                "validation": {"valid": True},
                "requests": {
                    "aggregate_prompt_tokens_per_second": 8000,
                    "time_to_first_token_ms": {"median": 1000},
                    "completed": 5,
                },
            },
        )

    result = summarize(tmp_path, mode="publication", build_id="rc3")

    assert set(result["decode"]) == {"c1", "c2", "c4", "c8", "c16", "c32"}
    assert all(
        cell["engine_forward_passes_per_second"]["count"] == 5
        for cell in result["decode"].values()
    )
    assert all(cell["requests"] == 5 for cell in result["prefill"].values())


def test_publication_rejects_nonuniform_prefill_count(tmp_path) -> None:
    root = _gate(tmp_path)
    for concurrency in (2, 16, 32):
        for repetition in range(1, 6):
            _write(
                root
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "decode-analysis.json",
                {
                    "validation": {"valid": True},
                    "decode": {
                        "target_concurrency": concurrency,
                        "tokens_per_second_ols": 100 * concurrency + repetition,
                    },
                    "engine_work": {
                        "forward_passes_per_second_ols": 50 + repetition,
                        "useful_tokens_per_forward_per_request": 5.5,
                    },
                    "server_cross_checks": _speculative(repetition),
                },
            )
            source = root / "decode" / "c1" / "r01" / "profile_export_aiperf.json"
            target = (
                root
                / "decode"
                / f"c{concurrency}"
                / f"r{repetition:02d}"
                / "profile_export_aiperf.json"
            )
            _write(target, json.loads(source.read_text(encoding="utf-8")))
    for concurrency in (1, 4, 8):
        for repetition in range(4, 6):
            source = root / "decode" / f"c{concurrency}" / "r01" / "decode-analysis.json"
            target = root / "decode" / f"c{concurrency}" / f"r{repetition:02d}" / "decode-analysis.json"
            _write(target, json.loads(source.read_text(encoding="utf-8")))
            client_source = source.with_name("profile_export_aiperf.json")
            client_target = target.with_name("profile_export_aiperf.json")
            _write(client_target, json.loads(client_source.read_text(encoding="utf-8")))
    with pytest.raises(SummaryError, match="completed requests; expected 5"):
        summarize(root, mode="publication", build_id="rc3")

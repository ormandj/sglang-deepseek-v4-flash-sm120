from __future__ import annotations

import json

import pytest
from summarize_engine_gate import SummaryError, summarize


def _write(path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _gate(tmp_path):
    for concurrency, repetitions in {1: 3, 8: 2, 32: 1}.items():
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
                },
            )
    for label in ("8k-c1", "64k-c1", "128k-c1"):
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
    assert c1["engine_forward_passes_per_second"]["count"] == 3
    assert c1["engine_forward_passes_per_second"]["median"] == 52
    assert [row["id"] for row in c1["repetitions"]] == ["r01", "r02", "r03"]
    assert result["prefill"]["128k-c1"]["prompt_tokens_per_second"] == 8000


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


def test_vllm_quick_gate_records_c32_as_outside_panel(tmp_path) -> None:
    root = _gate(tmp_path)
    for path in (root / "decode" / "c32").glob("r*/decode-analysis.json"):
        path.unlink()
    result = summarize(root, mode="quick", build_id="r33", engine="vllm")
    assert result["engine"] == "vllm"
    assert set(result["decode"]) == {"c1", "c8"}


def test_decode_supplement_requires_only_mid_concurrency_cells(tmp_path) -> None:
    for concurrency, repetitions in {2: 6, 4: 4, 16: 2}.items():
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
                },
            )

    result = summarize(
        tmp_path,
        mode="decode-supplement",
        build_id="rc3",
        engine="sglang",
    )

    assert set(result["decode"]) == {"c2", "c4", "c16"}
    assert result["prefill"] == {}

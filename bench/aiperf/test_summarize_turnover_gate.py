from __future__ import annotations

import json
from pathlib import Path

import pytest
from summarize_turnover_gate import SummaryError, summarize


def _write_run(root: Path, concurrency: int, repetition: int, scale: float) -> None:
    path = root / f"c{concurrency}" / f"r{repetition:02d}" / "turnover-analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "cell": {
                    "target_concurrency": concurrency,
                    "expected_requests": 16 if concurrency < 8 else 32,
                    "target_isl": 256,
                    "target_osl": 256,
                },
                "requests": {
                    "output_tokens_per_second": 100 * scale,
                    "time_to_first_token_ms": {"median": 200 / scale},
                    "inter_token_latency_ms": {"median": 20 / scale},
                },
                "refill_batching": {"requests_per_prefill_pass": scale},
                "validation": {"valid": True},
            }
        ),
        encoding="utf-8",
    )


def _write_screen(root: Path) -> None:
    for concurrency in (1, 2, 4, 8):
        for repetition in range(1, 4):
            _write_run(root, concurrency, repetition, 1 + repetition / 10)


def test_screen_summary_requires_all_shapes_and_repetitions(tmp_path: Path) -> None:
    _write_screen(tmp_path)
    result = summarize(tmp_path, mode="screen", build_id="candidate")
    assert set(result["turnover"]) == {"c1", "c2", "c4", "c8"}
    assert result["turnover"]["c8"]["output_tokens_per_second"]["count"] == 3
    assert result["turnover"]["c8"]["requests_per_prefill_pass"]["median"] == 1.2
    assert result["turnover"]["c8"]["shape"]["expected_requests"] == 32


def test_release_screen_requires_only_three_c8_repetitions(tmp_path: Path) -> None:
    for repetition in range(1, 4):
        _write_run(tmp_path, 8, repetition, 1 + repetition / 10)
    result = summarize(tmp_path, mode="release-screen", build_id="candidate")
    assert set(result["turnover"]) == {"c8"}
    assert result["turnover"]["c8"]["output_tokens_per_second"]["count"] == 3


def test_missing_shape_is_rejected(tmp_path: Path) -> None:
    _write_screen(tmp_path)
    for path in (tmp_path / "c8").glob("**/*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(SummaryError, match="turnover cells"):
        summarize(tmp_path, mode="screen", build_id="candidate")


def test_invalid_run_is_rejected(tmp_path: Path) -> None:
    _write_screen(tmp_path)
    path = tmp_path / "c4" / "r02" / "turnover-analysis.json"
    document = json.loads(path.read_text())
    document["validation"]["valid"] = False
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SummaryError, match="C4/r02 is invalid"):
        summarize(tmp_path, mode="screen", build_id="candidate")

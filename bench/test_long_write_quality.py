from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("long_write_quality.py")
SPEC = importlib.util.spec_from_file_location("long_write_quality", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

extract_html = MODULE.extract_html
structural_checks = MODULE.structural_checks
build_parser = MODULE.build_parser


def test_extract_html_closed_fence() -> None:
    assert extract_html("before\n```html\n<html></html>\n```\nafter") == (
        "<html></html>\n"
    )


def test_extract_html_open_fence() -> None:
    assert extract_html("```html\n<html></html>") == "<html></html>"


def test_structural_checks_complete() -> None:
    checks = structural_checks("```html\n<html><script>let x=1;</script></html>\n```")
    assert checks == {
        "fence_closed": True,
        "has_close": True,
        "no_placeholder": True,
    }


def test_structural_checks_reject_placeholder() -> None:
    checks = structural_checks("```html\n<html>TODO: rest of the code</html>\n```")
    assert checks["no_placeholder"] is False


def test_run_defaults_to_five_samples() -> None:
    args = build_parser().parse_args(
        ["run", "--model", "deepseek-v4-flash", "--output", "responses.json"]
    )
    assert args.runs == 5

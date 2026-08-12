"""Run and validate the deterministic long single-file generation check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROMPT = (
    "Write a complete, self-contained single-file HTML5 game: a 3D "
    "Mario-style platformer using vanilla JavaScript and WebGL via raw canvas "
    "(no external libraries, no CDN). It must include: a player that runs and "
    "jumps with gravity and collision, at least 3 distinct level layouts, "
    "collectible coins with a score display, enemies that patrol and can be "
    "defeated by jumping on them, and a win condition. Output the ENTIRE file "
    "in one ```html code block. Do not abbreviate, do not use placeholders, "
    "do not write 'rest of the code here'. Write every line."
)

PLACEHOLDER_RE = re.compile(
    r"(rest of (the )?(code|file)|remainder of|\.\.\. *(rest|more|etc)|"
    r"<!-- *\.\.\. *-->|/\* *\.\.\. *\*/|TODO:)",
    re.IGNORECASE,
)


def extract_html(text: str) -> str:
    match = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```html\s*(.*)$", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


def structural_checks(text: str) -> dict[str, bool]:
    return {
        "fence_closed": text.count("```") >= 2,
        "has_close": "</html>" in text,
        "no_placeholder": PLACEHOLDER_RE.search(text) is None,
    }


def check_javascript(html: str, node_command: str) -> tuple[bool, str]:
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
    scripts = [script for script in scripts if script.strip()]
    if not scripts:
        return False, "no <script> body"
    for index, script in enumerate(scripts):
        with tempfile.NamedTemporaryFile(
            "w", suffix=f"-{index}.js", delete=False
        ) as handle:
            handle.write(script)
            path = handle.name
        try:
            process = subprocess.run(
                [node_command, "--check", path],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            os.unlink(path)
        if process.returncode != 0:
            lines = [line for line in process.stderr.splitlines() if "Error" in line]
            error = lines[0] if lines else process.stderr.strip()
            return False, error[:160]
    return True, ""


def _api_key(env_name: str) -> str:
    return os.environ.get(env_name, "") or os.environ.get("BENCH_API_KEY", "")


def _request(
    *,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - started

    choice = payload["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return {
        "content": content,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "all_budget_in_reasoning": bool(reasoning) and not content.strip(),
        "finish_reason": choice.get("finish_reason"),
        "matched_stop": choice.get("matched_stop"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "elapsed_seconds": round(elapsed, 3),
    }


def run_requests(args: argparse.Namespace) -> int:
    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        raise SystemExit("long-output requests must execute inside the serving pod")
    parsed = urllib.parse.urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise SystemExit("--base-url must be pod-localhost over HTTP")
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    key = _api_key(args.api_key_env)
    for index in range(args.runs):
        try:
            result = _request(
                base_url=args.base_url,
                model=args.model,
                api_key=key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
            )
        except Exception as error:  # noqa: BLE001 - preserve the remaining runs
            result = {"error": repr(error)[:500]}
        result["run"] = index
        runs.append(result)
        print(
            f"[long-output] run{index:02d} finish={result.get('finish_reason')} "
            f"tokens={result.get('completion_tokens')} error={bool(result.get('error'))}",
            flush=True,
        )

    artifact = {
        "schema_version": "1.0",
        "model": args.model,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "runs_requested": args.runs,
        "request_errors": sum(1 for run in runs if run.get("error")),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "provenance": {
            "image_ref": os.environ.get("BENCH_IMAGE_REF", "unknown"),
            "gitops_revision": os.environ.get("BENCH_GITOPS_REVISION", "unknown"),
            "project_revision": os.environ.get("BENCH_PROJECT_REVISION", "unknown"),
            "model_revision": os.environ.get("BENCH_MODEL_REVISION", "unknown"),
        },
        "runs": runs,
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[long-output] requests -> {output}")
    return 1 if artifact["request_errors"] else 0


def validate_requests(args: argparse.Namespace) -> int:
    source = Path(args.input)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    payload = json.loads(source.read_text())
    if shutil.which(args.node_command) is None:
        raise SystemExit(f"JavaScript validator unavailable: {args.node_command}")

    validations: list[dict[str, Any]] = []
    for raw in payload["runs"]:
        if raw.get("error"):
            validations.append(
                {"run": raw["run"], "complete": False, "error": raw["error"]}
            )
            continue
        text = raw.get("content") or ""
        checks = structural_checks(text)
        js_ok, js_error = check_javascript(extract_html(text), args.node_command)
        checks["js_parses"] = js_ok
        validations.append(
            {
                "run": raw["run"],
                "complete": all(checks.values()),
                "checks": checks,
                "js_error": js_error,
                "finish_reason": raw.get("finish_reason"),
                "completion_tokens": raw.get("completion_tokens"),
                "reasoning_chars": raw.get("reasoning_chars"),
                "content_chars": raw.get("content_chars"),
                "elapsed_seconds": raw.get("elapsed_seconds"),
            }
        )

    complete = sum(1 for item in validations if item["complete"])
    completion_tokens = sorted(
        int(item.get("completion_tokens") or 0) for item in validations
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = {
        "schema_version": "1.0",
        "source_sha256": digest,
        "runs": len(validations),
        "complete": complete,
        "complete_rate": complete / len(validations) if validations else 0.0,
        "median_completion_tokens": (
            statistics.median(completion_tokens) if completion_tokens else 0
        ),
        "validations": validations,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[long-output] {complete}/{len(validations)} complete -> {output}")
    return 0 if complete == len(validations) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="issue requests inside the serving pod")
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--runs", type=int, default=5)
    run.add_argument("--max-tokens", type=int, default=131072)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=0.95)
    run.add_argument("--timeout", type=int, default=5400)
    run.add_argument("--reasoning-effort", default="max")
    run.add_argument("--api-key-env", default="LLM_API_KEY")
    run.add_argument("--output", required=True)
    run.set_defaults(func=run_requests)

    validate = subparsers.add_parser(
        "validate", help="validate saved responses with Node.js"
    )
    validate.add_argument("--input", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--node-command", default="node")
    validate.set_defaults(func=validate_requests)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

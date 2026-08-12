#!/usr/bin/env python3
"""Issue one near-maximum-context completion from persisted corpus chunks."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def metrics(base_url: str) -> dict[str, float]:
    names = (
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
    )
    values = {name: 0.0 for name in names}
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace")
            if line.startswith("#"):
                continue
            for name in names:
                if line.startswith((name + "{", name + " ")):
                    values[name] += float(line.rsplit(None, 1)[-1])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--chunks-dir", default="/models/.bench-chunks")
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--cache-salt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.chunks_dir).glob("chunk_*.txt"))[: args.chunks]
    if len(paths) != args.chunks:
        raise RuntimeError(f"requested {args.chunks} chunks, found {len(paths)}")
    prompt = "\n// stable chunk boundary\n".join(path.read_text() for path in paths)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.cache_salt:
        payload["cache_salt"] = args.cache_salt
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(args.api_key_env, "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{args.base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )

    before = metrics(args.base_url)
    started = time.perf_counter()
    first_token = None
    usage: dict[str, object] = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if first_token is None and any(choice.get("text") for choice in choices):
                first_token = time.perf_counter()
    finished = time.perf_counter()
    after = metrics(args.base_url)
    details = usage.get("prompt_tokens_details") or {}
    artifact = {
        "model": args.model,
        "chunk_count": len(paths),
        "cache_salt": args.cache_salt,
        "prompt_bytes": len(prompt.encode()),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "ttft_s": (first_token or finished) - started,
        "wall_s": finished - started,
        "metric_delta": {name: after[name] - before[name] for name in before},
    }
    Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()

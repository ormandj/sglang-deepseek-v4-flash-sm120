#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:-"$repo/release.json"}
lock=${2:-"$repo/stack.lock.json"}

for tool in jq grep; do
  command -v "$tool" >/dev/null || {
    echo "missing required tool: $tool" >&2
    exit 1
  }
done

jq -e . "$manifest" >/dev/null
jq -e . "$lock" >/dev/null

version=$(jq -er '.version' "$manifest")
candidate=$(jq -er '.candidate' "$manifest")
candidate_tag=$(jq -er '.candidate_tag' "$manifest")
stable_tag=$(jq -er '.stable_tag' "$manifest")
cache_schema=$(jq -er '.cache_schema' "$manifest")
product=$(jq -er '.product' "$manifest")

# The target release is stable SemVer. The candidate suffix is derived so the
# exact same image manifest can later be promoted without changing its labels.
semver_re='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
[[ "$version" =~ $semver_re ]] || {
  echo "release version is not strict stable SemVer: $version" >&2
  exit 1
}
[[ "$candidate" =~ ^[1-9][0-9]*$ ]] || {
  echo "candidate must be a positive integer: $candidate" >&2
  exit 1
}
[[ "$candidate_tag" == "v${version}-rc.${candidate}" ]] || {
  echo "candidate_tag must be v${version}-rc.${candidate}: $candidate_tag" >&2
  exit 1
}
[[ "$stable_tag" == "v${version}" ]] || {
  echo "stable_tag must be v${version}: $stable_tag" >&2
  exit 1
}
[[ "$cache_schema" =~ ^v[1-9][0-9]*$ ]] || {
  echo "cache_schema must be vN with N >= 1: $cache_schema" >&2
  exit 1
}

for field in version candidate candidate_tag stable_tag cache_schema; do
  release_value=$(jq -er ".${field}" "$manifest")
  lock_value=$(jq -er ".release.${field}" "$lock")
  [[ "$release_value" == "$lock_value" ]] || {
    echo "release.json and stack.lock.json disagree on $field" >&2
    exit 1
  }
done

[[ "$(jq -er '.image.tag' "$lock")" == "$candidate_tag" ]] || {
  echo "stack.lock.json image tag is not the candidate tag" >&2
  exit 1
}

check_pin() {
  local arg=$1 expected=$2
  actual=$(jq -er --arg arg "$arg" '.pins[] | select(.arg == $arg) | .value' "$lock")
  [[ "$actual" == "$expected" ]] || {
    echo "$arg does not match release.json" >&2
    exit 1
  }
}
check_pin DSV4_0731_RELEASE_VERSION "$version"
check_pin DSV4_0731_RELEASE_CANDIDATE "$candidate"
check_pin DSV4_0731_CACHE_SCHEMA "$cache_schema"

serve_script="$repo/examples/serve-dsv4-0731.sh"
expected_image="ghcr.io/ormandj/${product}:${candidate_tag}"
expected_default="IMAGE=\${IMAGE:-${expected_image}}"
grep -Fx "$expected_default" "$serve_script" >/dev/null || {
  echo "serving script default is not the release candidate: $expected_image" >&2
  exit 1
}

run_guide="$repo/RUN.md"
expected_cache="/srv/cache/sglang-dsv4-0731-${cache_schema}"
grep -F "$expected_cache" "$run_guide" >/dev/null || {
  echo "RUN.md cache directory does not use cache schema $cache_schema" >&2
  exit 1
}

echo "release contract valid: ${candidate_tag} -> ${stable_tag}, cache ${cache_schema}"

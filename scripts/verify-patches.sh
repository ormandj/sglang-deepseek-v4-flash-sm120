#!/usr/bin/env bash
# Verifies that stack.lock.json, the Containerfile, and the copied patch bytes
# describe the same stack, then reproduces the image's source construction:
# clone the pinned SGLang and FlashInfer commits, apply the patches in build
# order, and confirm the final tree hashes.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"

lock=stack.lock.json
containerfile=Containerfile

for tool in jq git; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

if command -v sha256sum >/dev/null; then
  sha256_file() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  sha256_file() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  echo "missing required SHA-256 tool: sha256sum or shasum" >&2
  exit 1
fi

"$repo/scripts/validate-release.sh"

lock_value() {
  jq -r --arg arg "$1" '.pins[] | select(.arg == $arg) | .value' "$lock"
}

echo "== lock and Containerfile agree on every pinned value =="
while IFS=$'\t' read -r arg value; do
  if ! grep -Fxq "ARG ${arg}=${value}" "$containerfile"; then
    echo "Containerfile is missing 'ARG ${arg}=${value}'" >&2
    exit 1
  fi
  printf '  %s\n' "$arg"
done < <(jq -r '.pins[] | [.arg, .value] | @tsv' "$lock")

# Nothing may be pinned in the Containerfile without being recorded in the lock.
while IFS= read -r arg; do
  if [[ "$(lock_value "$arg")" == "" ]]; then
    echo "Containerfile pins ${arg}, which stack.lock.json does not record" >&2
    exit 1
  fi
done < <(sed -n 's/^ARG \([A-Z0-9_]*\)=.*/\1/p' "$containerfile")

echo "== patch bytes match the lock and are the ones the Containerfile copies =="
while IFS=$'\t' read -r path sha; do
  [[ -f "$path" ]] || { echo "missing patch: $path" >&2; exit 1; }
  actual=$(sha256_file "$path")
  if [[ "$actual" != "$sha" ]]; then
    echo "sha256 mismatch for $path: lock $sha, file $actual" >&2
    exit 1
  fi
  if ! grep -Fq "COPY ${path} " "$containerfile"; then
    echo "Containerfile does not copy $path" >&2
    exit 1
  fi
  printf '  %s\n' "$path"
done < <(jq -r '.patches[] | [.path, .sha256] | @tsv' "$lock")

# Any patch file present but unrecorded would silently ship in the build context.
while IFS= read -r path; do
  if [[ "$(jq -r --arg p "$path" '[.patches[] | select(.path == $p)] | length' "$lock")" != "1" ]]; then
    echo "$path is not recorded in stack.lock.json" >&2
    exit 1
  fi
done < <(find patches -type f | sort)

sglang_repo=$(jq -r '.verification.sglang_repository' "$lock")
flashinfer_repo=$(jq -r '.verification.flashinfer_repository' "$lock")
deepgemm_repo=$(jq -r '.verification.deepgemm_repository' "$lock")

sglang_head=$(lock_value DSV4_0731_SGLANG_MAIN_HEAD)
sglang_tree=$(lock_value DSV4_0731_SGLANG_MAIN_TREE)
sglang_effective_tree=$(lock_value DSV4_0731_SGLANG_EFFECTIVE_TREE)
flashinfer_head=$(lock_value DSV4_0731_FLASHINFER_MAIN_HEAD)
flashinfer_tree=$(lock_value DSV4_0731_FLASHINFER_MAIN_TREE)
flashinfer_effective_tree=$(lock_value DSV4_0731_FLASHINFER_EFFECTIVE_TREE)
deepgemm_head=$(lock_value DSV4_0731_DEEPGEMM_MAIN_HEAD)
deepgemm_tree=$(lock_value DSV4_0731_DEEPGEMM_MAIN_TREE)
deepgemm_effective_tree=$(lock_value DSV4_0731_DEEPGEMM_EFFECTIVE_TREE)

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fetch_commit() {
  local dir=$1 url=$2 commit=$3 tree=$4
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth=1 origin "$commit"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  test "$(git -C "$dir" rev-parse HEAD)" = "$commit"
  test "$(git -C "$dir" rev-parse 'HEAD^{tree}')" = "$tree"
}

echo "== SGLang $sglang_head =="
fetch_commit "$work/sglang" "$sglang_repo" "$sglang_head" "$sglang_tree"
while IFS= read -r path; do
  git -C "$work/sglang" apply --index --binary "$repo/$path"
  printf '  applied %s\n' "$path"
done < <(jq -r '[.patches[] | select(.repository == "sglang")] | sort_by(.order)[].path' "$lock")
test "$(git -C "$work/sglang" write-tree)" = "$sglang_effective_tree"
echo "  final tree $sglang_effective_tree"

echo "== FlashInfer $flashinfer_head =="
fetch_commit "$work/flashinfer" "$flashinfer_repo" "$flashinfer_head" "$flashinfer_tree"
while IFS= read -r path; do
  git -C "$work/flashinfer" apply --index --binary "$repo/$path"
  printf '  applied %s\n' "$path"
done < <(jq -r '[.patches[] | select(.repository == "flashinfer")] | sort_by(.order)[].path' "$lock")
test "$(git -C "$work/flashinfer" write-tree)" = "$flashinfer_effective_tree"
echo "  final tree $flashinfer_effective_tree"

echo "== DeepGEMM $deepgemm_head =="
fetch_commit "$work/deepgemm" "$deepgemm_repo" "$deepgemm_head" "$deepgemm_tree"
while IFS= read -r path; do
  git -C "$work/deepgemm" apply --index --binary "$repo/$path"
  printf '  applied %s\n' "$path"
done < <(jq -r '[.patches[] | select(.repository == "deepgemm")] | sort_by(.order)[].path' "$lock")
test "$(git -C "$work/deepgemm" write-tree)" = "$deepgemm_effective_tree"
echo "  final tree $deepgemm_effective_tree"

echo "stack.lock.json, Containerfile, and patches agree; every tree hash reproduces"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEMO_DIR="$ROOT_DIR/examples/wasm-demo"
OUT_DIR="${1:-$ROOT_DIR/dist}"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

GOOS=js GOARCH=wasm go build -C "$DEMO_DIR" -o "$OUT_DIR/cmaes.wasm" .

shopt -s nullglob
assets=("$DEMO_DIR"/*.html "$DEMO_DIR"/*.css "$DEMO_DIR"/*.js "$DEMO_DIR"/*.svg)
shopt -u nullglob

if [ ${#assets[@]} -eq 0 ]; then
  echo "error: no static demo assets found in $DEMO_DIR" >&2
  exit 1
fi

cp "${assets[@]}" "$OUT_DIR/"

wasm_exec=""
for candidate in "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$(go env GOROOT)/misc/wasm/wasm_exec.js"; do
  if [ -f "$candidate" ]; then
    wasm_exec="$candidate"
    break
  fi
done

if [ -z "$wasm_exec" ]; then
  echo "error: wasm_exec.js not found under $(go env GOROOT)" >&2
  exit 1
fi

cp "$wasm_exec" "$OUT_DIR/"

printf "Copied %d static asset(s)\n" "${#assets[@]}"
printf "WASM demo built at %s\n" "$OUT_DIR"

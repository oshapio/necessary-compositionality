#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
  echo "Missing dataset. Use one of: dsprites | mpi3d | pug | clean_dsprites" >&2
  exit 2
fi

REVERSE_ORDER=false
if [[ "${2:-}" == "-r" || "${2:-}" == "--reverse" ]]; then
  REVERSE_ORDER=true
fi

ROOT_OUT="${ROOT_OUT:-${REPO_ROOT}/outputs/clip_models_laion}"
EMB_DIR="${ROOT_OUT}/${DATASET}"

if [[ ! -d "$EMB_DIR" ]]; then
  echo "Dataset directory not found: $EMB_DIR" >&2
  exit 1
fi

if [[ -x ".venv/bin/python" ]]; then
  PY=./.venv/bin/python
else
  PY=python3
fi

shopt -s nullglob
PROBE_DIRS=("$EMB_DIR"/_probes*)
if [[ ${#PROBE_DIRS[@]} -eq 0 ]]; then
  echo "No probe directories found under: $EMB_DIR" >&2
  exit 1
fi

if [[ "$REVERSE_ORDER" == true ]]; then
  REVERSED_PROBE_DIRS=()
  for ((i=${#PROBE_DIRS[@]}-1; i>=0; i--)); do
    REVERSED_PROBE_DIRS+=("${PROBE_DIRS[i]}")
  done
  PROBE_DIRS=("${REVERSED_PROBE_DIRS[@]}")
fi

echo "Dataset: $DATASET"
echo "Embeddings dir: $EMB_DIR"
echo "Reverse probe dir order: $REVERSE_ORDER"
echo "Probe dirs: ${PROBE_DIRS[*]}"

for probes_dir in "${PROBE_DIRS[@]}"; do
  echo "== Probes: $probes_dir =="
  for probes_json in "$probes_dir"/*.json; do
    [[ -f "$probes_json" ]] || continue

    export PROBES_JSON="$probes_json"
    emb_path="$($PY - <<'PY'
import json
import os

path = os.environ.get("PROBES_JSON")
with open(path, "r") as f:
    data = json.load(f)

emb = data.get("emb_path")
if isinstance(emb, str):
    print(emb)
PY
)"

    if [[ -z "$emb_path" ]]; then
      echo "[skip] emb_path missing in probes JSON: $probes_json" >&2
      continue
    fi
    if [[ ! -f "$emb_path" ]]; then
      echo "[skip] embeddings not found for probes: $probes_json (emb_path=$emb_path)" >&2
      continue
    fi

    echo "[ANALYSE] emb=$(basename "$emb_path") probes=$(basename "$probes_json")"
    "$PY" -m complinearity.analyse_embeddings_probes \
      --emb-path "$emb_path" \
      --probes-json "$probes_json"
  done
done

echo "All analyses complete."

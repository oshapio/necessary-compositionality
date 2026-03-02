#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -x ".venv/bin/python" ]]; then
  PY=./.venv/bin/python
else
  PY=python
fi

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/clip_models_laion}"
BATCH=256
BATCH_DINO=32
WORKERS=8

MODEL_SPECS=$(cat <<'MODELS'
clip|ViT-B/32|
clip|ViT-L/14|
openclip|ViT-B-32|laion400m_e32
openclip|ViT-B-16|laion400m_e32
openclip|ViT-L-14|laion2b_s32b_b82k
openclip|hf-hub:google/siglip-large-patch16-256|
openclip|hf-hub:google/siglip2-large-patch16-384|
dino|vit_small_patch16_dinov3|
dino|vit_base_patch16_dinov3|
dino|vit_large_patch16_dinov3|
MODELS
)

run_one() {
  local backend="$1" model="$2" pretrained="$3"
  local batch="$BATCH"
  if [[ "$backend" == "dino" || "$backend" == "timm" ]]; then
    batch="$BATCH_DINO"
  fi
  echo "[RUN] dataset=$DATASET backend=$backend model=$model pretrained=${pretrained:-<default>}"
  local cmd=("$PY" -m complinearity.get_embeddings "${COMMON_ARGS[@]}" --backend "$backend" --model-name "$model" --batch-size "$batch" --num-workers "$WORKERS")
  if [[ -n "$pretrained" ]]; then
    cmd+=(--pretrained "$pretrained")
  fi
  "${cmd[@]}"
}

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(mpi3d)
fi

for DATASET in "${DATASETS[@]}"; do
  case "$DATASET" in
    dsprites|pug|mpi3d|clean_dsprites) ;;
    *)
      echo "[skip] Unknown dataset '$DATASET'" >&2
      continue
      ;;
  esac

  mkdir -p "$OUT_DIR/$DATASET"

  COMMON_ARGS=(--output-path "$OUT_DIR" --dataset "$DATASET")

  echo "=== Dataset: $DATASET ==="
  while IFS='|' read -r backend model pretrained; do
    [[ -z "$backend" || "$backend" =~ ^# ]] && continue
    run_one "$backend" "$model" "${pretrained:-}"
  done <<<"$MODEL_SPECS"
  echo "=== Completed dataset $DATASET. Outputs in: $OUT_DIR/$DATASET ==="
done

echo "All runs completed."

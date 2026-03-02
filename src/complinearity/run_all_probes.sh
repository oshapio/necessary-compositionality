#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${1:-mpi3d}"
ROOT_OUT="${ROOT_OUT:-${REPO_ROOT}/outputs/clip_models_laion}"
OUT_SUFFIX="${OUT_SUFFIX:-""}"
VAL_SPLITS="${VAL_SPLITS:-"0.025 0.05"}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  DATASET="$1"
  shift
fi

ALL_EMB=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      shift
      DATASET="$1"
      ;;
    --out-suffix)
      shift
      OUT_SUFFIX="$1"
      ;;
    --all-embeddings)
      ALL_EMB=1
      ;;
    --single-embedding)
      ALL_EMB=0
      ;;
    -h|--help)
      echo "Usage: $(basename "$0") [DATASET] [--dataset dsprites|clean_dsprites|mpi3d|pug] [--out-suffix SUFFIX] [--all-embeddings|--single-embedding]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
  shift
done

case "$DATASET" in
  dsprites)
    HEADS="shape,size,orientation,posx,posy,color"
    ;;
  clean_dsprites)
    HEADS="color,shape,size,orientation,posx,posy"
    ;;
  mpi3d)
    HEADS="object-color,object-shape,object-size,camera-height,background-color,horizontal-axis,vertical-axis"
    ;;
  *)
    HEADS="character,world,size,texture"
    ;;
esac

if [[ -x ".venv/bin/python" ]]; then
  PY=./.venv/bin/python
else
  PY=python
fi

EPOCHS="${EPOCHS:-500}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-2}"
WD="${WD:-0}"
EMB_DIR="${ROOT_OUT}/${DATASET}"
BASE_OUT_DIR="${ROOT_OUT}/${DATASET}/_probes"

run_probe() {
  local emb_path="$1"
  local val_split="$2"
  local val_tag="${val_split//./p}"
  local out_dir="${BASE_OUT_DIR}${OUT_SUFFIX}_val${val_tag}"
  mkdir -p "$out_dir"

  echo "[PROBE] $emb_path -> $out_dir (val_split=$val_split)"
  "$PY" -m complinearity.train_probes \
    --emb-path "$emb_path" \
    --out-dir "$out_dir" \
    --heads "$HEADS" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr "$LR" \
    --weight-decay "$WD" \
    --val-split "$val_split"
}

run_all_embeddings() {
  local emb_dir="$1"
  local prefix="$2"
  local val_split="$3"

  shopt -s nullglob
  for f in "$emb_dir"/${prefix}_embeddings_*.pkl; do
    run_probe "$f" "$val_split"
  done
  for f in "$emb_dir"/${prefix}_embeddings_*.npz; do
    local base_noext="${f%.*}"
    if [[ -f "${base_noext}.pkl" ]]; then
      continue
    fi
    run_probe "$f" "$val_split"
  done
  shopt -u nullglob
}

for val_split in $VAL_SPLITS; do
  case "$DATASET" in
    dsprites)
      if [[ "$ALL_EMB" -eq 1 ]]; then
        run_all_embeddings "$EMB_DIR" "dsprites" "$val_split"
      else
        run_probe "/mnt/lustre/work/oh/owl661/complin_code/complinearity/outputs/clip_models_laion/dsprites/dsprites_embeddings_clip_ViT-B-32_openai.pkl" "$val_split"
      fi
      ;;
    clean_dsprites)
      if [[ "$ALL_EMB" -eq 1 ]]; then
        run_all_embeddings "$EMB_DIR" "clean_dsprites" "$val_split"
      else
        run_probe "/mnt/lustre/work/oh/owl661/complin_code/complinearity/outputs/clip_models_laion/clean_dsprites/clean_dsprites_embeddings_clip_ViT-B-32_openai.pkl" "$val_split"
      fi
      ;;
    mpi3d)
      run_all_embeddings "$EMB_DIR" "mpi3d" "$val_split"
      ;;
    *)
      if [[ "$ALL_EMB" -eq 1 ]]; then
        run_all_embeddings "$EMB_DIR" "pug" "$val_split"
      else
        run_probe "$EMB_DIR/pug_embeddings_clip_ViT-B-32_openai.pkl" "$val_split"
      fi
      ;;
  esac
done

echo "All probes trained."

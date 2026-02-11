#!/usr/bin/env bash
set -euo pipefail

# Launch single GPU embedding process on GCP
# Usage:
#   ./launch_single_gpu.sh \
#     --in_parquet /path/to/corpus_chunks.parquet \
#     --model google/embeddinggemma-300M \
#     --batch 512 \
#     [--sdpa]

usage(){
  cat <<USAGE
Usage: $0 --in_parquet FILE [--model NAME] [--batch N] [--sdpa]
USAGE
}

IN_PARQUET=""
MODEL="google/embeddinggemma-300M"
BATCH=512
SDPA=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in_parquet) IN_PARQUET="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --sdpa) SDPA=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$IN_PARQUET" ]]; then
  echo "--in_parquet is required" >&2
  usage
  exit 1
fi

DEV="cuda:0"
LOG="embed_single_gpu.log"
echo "Launching single GPU embedding on $DEV..."
if $SDPA; then SDPA_FLAG="--sdpa"; else SDPA_FLAG=""; fi
CUDA_VISIBLE_DEVICES="0" nohup python embed_gcp.py \
  --model "$MODEL" \
  --in_parquet "$IN_PARQUET" \
  --batch "$BATCH" \
  --device "$DEV" \
  $SDPA_FLAG \
  > "$LOG" 2>&1 &

echo "Launched single GPU embedding process. Monitor log: tail -f $LOG"
echo "Process PID: $!"

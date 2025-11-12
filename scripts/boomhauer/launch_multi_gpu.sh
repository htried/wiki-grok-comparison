#!/usr/bin/env bash
set -euo pipefail

# Launch 4 local shard processes across GPUs 0-3
# Usage:
#   ./launch_multi_gpu.sh \
#     --in_parquet /path/to/corpus_chunks.parquet \
#     --model Qwen/Qwen3-Embedding-0.6B \
#     --batch 256 \
#     [--sdpa]

usage(){
  cat <<USAGE
Usage: $0 --in_parquet FILE [--model NAME] [--batch N] [--sdpa]
USAGE
}

IN_PARQUET=""
MODEL="Qwen/Qwen3-Embedding-0.6B"
BATCH=256
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

NUM_SHARDS=4
for i in $(seq 0 $((NUM_SHARDS-1))); do
  # When CUDA_VISIBLE_DEVICES is set, only one GPU is visible and it appears as cuda:0
  # So we always use cuda:0 (or just cuda) when using CUDA_VISIBLE_DEVICES
  DEV="cuda:0"
  LOG="embed_shard_${i}of${NUM_SHARDS}.log"
  echo "Launching shard $i/$NUM_SHARDS on physical GPU $i (visible as $DEV)..."
  if $SDPA; then SDPA_FLAG="--sdpa"; else SDPA_FLAG=""; fi
  CUDA_VISIBLE_DEVICES="$i" nohup python embed.py \
    --model "$MODEL" \
    --in_parquet "$IN_PARQUET" \
    --batch "$BATCH" \
    --device "$DEV" \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
    $SDPA_FLAG \
    > "$LOG" 2>&1 &
done

echo "Launched $NUM_SHARDS shard processes. Monitor logs: tail -f embed_shard_*log"



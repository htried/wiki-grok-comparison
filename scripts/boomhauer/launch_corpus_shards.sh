#!/usr/bin/env bash
set -euo pipefail

# Launch N parallel corpus builders (CPU-bound, shard by article modulo)
# Example:
#   ./launch_corpus_shards.sh \
#     --source wiki \
#     --wiki_fp ~/data/grokipedia_wikipedia_articles.ndjson \
#     --model Qwen/Qwen3-Embedding-0.6B \
#     --window 250 --stride 150 \
#     --num-shards 4

usage(){
  cat <<USAGE
Usage: $0 --source (wiki|grok) [--wiki_fp FILE] [--grok_dir DIR] \
          [--model NAME] [--window N] [--stride N] [--num-shards N]
USAGE
}

SOURCE=""
WIKI_FP=""
GROK_DIR=""
MODEL="Qwen/Qwen3-Embedding-0.6B"
WINDOW=250
STRIDE=150
NUM_SHARDS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2;;
    --wiki_fp) WIKI_FP="$2"; shift 2;;
    --grok_dir) GROK_DIR="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --stride) STRIDE="$2"; shift 2;;
    --num-shards) NUM_SHARDS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SOURCE" ]]; then echo "--source is required" >&2; usage; exit 1; fi
if [[ "$SOURCE" == "wiki" && -z "$WIKI_FP" ]]; then echo "--wiki_fp required for source=wiki" >&2; exit 1; fi
if [[ "$SOURCE" == "grok" && -z "$GROK_DIR" ]]; then echo "--grok_dir required for source=grok" >&2; exit 1; fi

for i in $(seq 0 $((NUM_SHARDS-1))); do
  LOG="build_corpus_${SOURCE}_shard_${i}of${NUM_SHARDS}.log"
  echo "Launching corpus shard $i/$NUM_SHARDS..."
  nohup python build_corpus.py \
    --source "$SOURCE" \
    ${WIKI_FP:+--wiki_fp "$WIKI_FP"} \
    ${GROK_DIR:+--grok_dir "$GROK_DIR"} \
    --model "$MODEL" \
    --window "$WINDOW" \
    --stride "$STRIDE" \
    --shard-id "$i" \
    --num-shards "$NUM_SHARDS" \
    > "$LOG" 2>&1 &
done

echo "Launched $NUM_SHARDS corpus shard processes. Monitor logs: tail -f build_corpus_${SOURCE}_shard_*log"



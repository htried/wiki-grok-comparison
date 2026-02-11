#!/bin/bash
# Nohup command to run embed_gcp.py in the background
# Adjust paths and arguments as needed

nohup python scripts/gcp/embed_gcp.py \
    --in_parquet corpus_chunks.parquet \
    --out_emb corpus_chunks_embeddings_final.npy \
    --out_parquet corpus_chunks_with_ix.parquet \
    --batch 512 \
    --checkpoint-every 50 \
    --sdpa \
    > embed_gcp.log 2>&1 &

echo "Embedding process started in background. PID: $!"
echo "Logs are being written to: embed_gcp.log"
echo "Monitor progress with: tail -f embed_gcp.log"

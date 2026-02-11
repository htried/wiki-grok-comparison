#!/bin/bash
# Commands to verify embeddings alignment with parquet files
# Run these on your GCP VM after combining checkpoints

# Set paths (adjust these to match your actual paths)
EMBEDDINGS="/home/hjt36_cornell_edu/corpus_chunks_embeddings_final.npy"
PARQUET_WITH_IX="/home/hjt36_cornell_edu/corpus_chunks_with_ix.parquet"
PARQUET_ORIGINAL="/home/hjt36_cornell_edu/corpus_chunks.parquet"

echo "=== Quick checks ==="
echo ""

# 1. Check file sizes and row counts
echo "1. File sizes:"
ls -lh "$EMBEDDINGS" "$PARQUET_WITH_IX" "$PARQUET_ORIGINAL" 2>/dev/null | awk '{print $5, $9}'
echo ""

# 2. Quick Python check for row counts
echo "2. Row count check:"
python3 << EOF
import numpy as np
import pandas as pd

emb_path = "$EMBEDDINGS"
parquet_path = "$PARQUET_WITH_IX"
original_path = "$PARQUET_ORIGINAL"

# Load embeddings (memory-mapped)
embs = np.load(emb_path, mmap_mode='r')
print(f"Embeddings: {embs.shape[0]:,} rows, {embs.shape[1]} dimensions")

# Load parquet
df = pd.read_parquet(parquet_path)
print(f"Parquet (with_ix): {len(df):,} rows")

if 'emb_ix' in df.columns:
    print(f"emb_ix range: {df['emb_ix'].min()} to {df['emb_ix'].max()}")
    print(f"emb_ix sequential: {np.array_equal(df['emb_ix'].values, np.arange(len(df)))}")

if original_path:
    df_orig = pd.read_parquet(original_path)
    print(f"Parquet (original): {len(df_orig):,} rows")

# Check alignment
if embs.shape[0] == len(df):
    print("✓ Row counts match!")
else:
    print(f"✗ Row count mismatch: {embs.shape[0]:,} vs {len(df):,}")
EOF

echo ""
echo "=== Full verification ==="
echo ""

# 3. Run full verification script
python3 verify_embeddings.py \
    --embeddings "$EMBEDDINGS" \
    --parquet-with-ix "$PARQUET_WITH_IX" \
    --parquet-original "$PARQUET_ORIGINAL" \
    --sample-check 20

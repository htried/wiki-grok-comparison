# Corpus Building and Embedding on GCP

This guide explains how to run the corpus building and embedding pipeline on an existing GCP VM with GPU support.

## Prerequisites

- GCP VM with:
  - NVIDIA GPU (T4 or similar)
  - CUDA and PyTorch installed
  - Python 3.8+
  - Access to GCS bucket `enwiki-structured-contents-20251028`

## Quick Start

### Option 1: Run the full pipeline (recommended)

The `corpus_runner.py` script runs both corpus building and embedding in sequence:

```bash
cd /path/to/wiki-grok-comparison
python scripts/gcp/corpus_runner.py \
  --gcs_input gs://enwiki-structured-contents-20251028/scraped_data.jsonl \
  --corpus_output corpus_chunks.parquet \
  --emb_output embeddings.npy \
  --batch 256 \
  --sdpa
```

### Option 2: Run steps separately

**Step 1: Build corpus**
```bash
python scripts/gcp/build_corpus_gcp.py \
  --gcs_path gs://enwiki-structured-contents-20251028/scraped_data.jsonl \
  --out corpus_chunks.parquet \
  --model google/embeddinggemma-300M \
  --window 250 \
  --stride 150
```

**Step 2: Embed corpus**
```bash
python scripts/gcp/embed_gcp.py \
  --in_parquet corpus_chunks.parquet \
  --out_emb embeddings.npy \
  --out_parquet corpus_chunks_with_ix.parquet \
  --model google/embeddinggemma-300M \
  --batch 256 \
  --sdpa
```

### Option 3: Use launch script (background)

```bash
cd scripts/gcp
./launch_single_gpu.sh \
  --in_parquet corpus_chunks.parquet \
  --model google/embeddinggemma-300M \
  --batch 256 \
  --sdpa
```

Monitor with: `tail -f embed_single_gpu.log`

## GCS Integration

All scripts support reading from and writing to GCS:

**Read from GCS:**
```bash
python scripts/gcp/build_corpus_gcp.py \
  --gcs_path gs://bucket/path/scraped_data.jsonl \
  --out corpus_chunks.parquet
```

**Write to GCS:**
```bash
python scripts/gcp/build_corpus_gcp.py \
  --gcs_path gs://bucket/path/scraped_data.jsonl \
  --out gs://bucket/output/corpus_chunks.parquet
```

**Read from GCS for embedding:**
```bash
python scripts/gcp/embed_gcp.py \
  --in_parquet gs://bucket/path/corpus_chunks.parquet \
  --out_emb gs://bucket/output/embeddings.npy
```

## Parameters

### Corpus Building (`build_corpus_gcp.py`)
- `--gcs_path`: GCS path to input `scraped_data.jsonl` (default: `gs://enwiki-structured-contents-20251028/scraped_data.jsonl`)
- `--model`: Tokenizer model (default: `google/embeddinggemma-300M`)
- `--window`: Token window size (default: 250)
- `--stride`: Token stride for overlap (default: 150)
- `--out`: Output parquet path (local or `gs://`)
- `--upload`: Upload to GCS if output is local

### Embedding (`embed_gcp.py`)
- `--in_parquet`: Input parquet file (local or `gs://`)
- `--model`: Embedding model (default: `google/embeddinggemma-300M`)
- `--batch`: Batch size (default: 256, adjust based on GPU memory)
- `--sdpa`: Use PyTorch SDPA attention (recommended for T4)
- `--flash-attn`: Use Flash Attention 2 (requires Ampere+ GPUs)
- `--out_emb`: Output embeddings file (`.npy`, local or `gs://`)
- `--out_parquet`: Output parquet with embedding indices (local or `gs://`)

### Pipeline Runner (`corpus_runner.py`)
- `--gcs_input`: GCS path to input data
- `--corpus_output`: Output path for corpus parquet
- `--emb_output`: Output path for embeddings
- `--model`: Model to use (default: `google/embeddinggemma-300M`)
- `--window`, `--stride`, `--batch`: Same as above
- `--sdpa`: Use SDPA attention
- `--skip_corpus`: Skip corpus building step
- `--skip_embedding`: Skip embedding step

## GPU Memory Considerations

For T4 GPUs (16GB VRAM):
- Recommended batch size: 256
- Use `--sdpa` flag for optimized attention
- The scripts include checkpoint/resume support if OOM occurs

## Monitoring

Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

View logs:
```bash
# If using launch_single_gpu.sh
tail -f embed_single_gpu.log

# Check process
ps aux | grep embed_gcp
```

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `--batch` size (try 128 or 64)
- The script will save checkpoints automatically

**GCS Authentication:**
- Ensure the VM has default service account with Storage permissions
- Or set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

**Model Download:**
- First run will download the model (~300MB for embeddinggemma-300M)
- Ensure internet access or pre-download the model

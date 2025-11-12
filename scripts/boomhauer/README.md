# Wikipedia/Grokipedia Embedding Pipeline

This directory contains scripts for building, embedding, and comparing Wikipedia and Grokipedia articles using text embeddings.

## Overview

The pipeline consists of three main stages:

1. **Build Corpus** (`build_corpus.py`) - Chunk articles from Wikipedia and/or Grokipedia into overlapping text segments
2. **Generate Embeddings** (`embed.py`) - Embed text chunks using SentenceTransformers
3. **Compute Similarities** (`similarities.py`) - Calculate pairwise similarity between Wikipedia and Grokipedia chunks for matching articles

## Scripts

### `build_corpus.py`

Builds a chunked corpus from Wikipedia and/or Grokipedia articles. Text is tokenized and split into overlapping windows for embedding.

**Features:**
- Processes Wikipedia and Grokipedia separately or together
- Preserves document structure (headings, sections)
- Creates overlapping chunks for better semantic coverage
- Uses tokenizer from target embedding model to ensure consistency

**Usage:**

```bash
# Process Wikipedia only
python build_corpus.py \
    --wiki_fp ~/data/grokipedia_wikipedia_articles.ndjson \
    --source wiki

# Process Grokipedia only
python build_corpus.py \
    --grok_dir ~/data/scraped_data \
    --source grok

# Process both (original behavior)
python build_corpus.py \
    --wiki_fp ~/data/grokipedia_wikipedia_articles.ndjson \
    --grok_dir ~/data/scraped_data

# Custom parameters
python build_corpus.py \
    --wiki_fp ~/data/wiki.ndjson \
    --grok_dir ~/data/scraped_data \
    --window 250 \
    --stride 150 \
    --out custom_corpus.parquet
```

**Arguments:**
- `--wiki_fp`: Path to Wikipedia JSONL file (optional)
- `--grok_dir`: Path to Grokipedia scraped_data directory with `batch_*.jsonl` files (optional)
- `--source`: Process only 'wiki' or 'grok' (optional, processes both if not specified)
- `--window`: Token window size for chunking (default: 250)
- `--stride`: Token stride for overlapping chunks (default: 150, gives 100 token overlap)
- `--out`: Output parquet file (auto-generated if `--source` is specified)

**Output:**
- Parquet file with columns: `title`, `source`, `chunk_id`, `text`
- Auto-generated filenames:
  - `corpus_chunks_wiki.parquet` when `--source=wiki`
  - `corpus_chunks_grok.parquet` when `--source=grok`
  - `corpus_chunks.parquet` when processing both

### `embed.py`

Generates embeddings for text chunks using SentenceTransformers models.

**Features:**
- Supports GPU acceleration (CUDA)
- Optional Flash Attention 2 for faster inference
- Auto-generates output filenames based on input
- Normalizes embeddings (L2-normalized for cosine similarity)

**Usage:**

```bash
# Basic usage (auto-names outputs)
python embed.py --in_parquet corpus_chunks_wiki.parquet

# With custom outputs
python embed.py \
    --in_parquet corpus_chunks_wiki.parquet \
    --out_emb wiki_embeddings.npy \
    --out_parquet wiki_with_ix.parquet

# With Flash Attention 2 (faster, requires compatible model)
python embed.py \
    --in_parquet corpus_chunks_grok.parquet \
    --flash_attention2

# Custom batch size
python embed.py \
    --in_parquet corpus_chunks.parquet \
    --batch 1024
```

**Arguments:**
- `--in_parquet`: Input parquet file from `build_corpus.py`
- `--out_parquet`: Output parquet with `emb_ix` column (auto-generated if not specified)
- `--out_emb`: Output embeddings `.npy` file (auto-generated if not specified)
- `--batch`: Batch size for embedding (default: 512)
- `--flash_attention2`: Enable Flash Attention 2 if available

**Output:**
- `.npy` file with normalized embeddings (shape: `[n_chunks, embedding_dim]`)
- Parquet file with `emb_ix` column added for indexing embeddings
- Auto-generated filenames:
  - Input: `corpus_chunks_wiki.parquet` → Outputs: `corpus_chunks_wiki_embeddings.npy`, `corpus_chunks_wiki_with_ix.parquet`

### `similarities.py`

Computes pairwise similarity between Wikipedia and Grokipedia chunks for matching articles.

**Features:**
- Supports both combined and separate corpus files
- Auto-detects embedding files if not specified
- Calculates similarity statistics per article (mean, median, max, p90)
- Stores top-1 alignments for each Wikipedia chunk

**Usage:**

```bash
# Using separate files (auto-detects embeddings)
python similarities.py \
    --wiki_parquet corpus_chunks_wiki_with_ix.parquet \
    --grok_parquet corpus_chunks_grok_with_ix.parquet

# Explicit embedding files
python similarities.py \
    --wiki_parquet wiki.parquet \
    --wiki_emb wiki_embeddings.npy \
    --grok_parquet grok.parquet \
    --grok_emb grok_embeddings.npy \
    --out_stats pairwise_stats.parquet \
    --out_top1 pairwise_top1_alignments.parquet

# Original combined mode still works
python similarities.py \
    --in_parquet corpus_chunks_with_ix.parquet \
    --emb embeddings.npy
```

**Arguments:**
- **Combined mode:**
  - `--in_parquet`: Combined corpus parquet file (with both wiki and grok)
  - `--emb`: Combined embeddings file
- **Separate mode:**
  - `--wiki_parquet`: Wikipedia corpus parquet file (required)
  - `--grok_parquet`: Grokipedia corpus parquet file (required)
  - `--wiki_emb`: Wikipedia embeddings file (auto-detected if not specified)
  - `--grok_emb`: Grokipedia embeddings file (auto-detected if not specified)
- `--out_stats`: Output statistics parquet file (default: `pairwise_stats.parquet`)
- `--out_top1`: Output top-1 alignments parquet file (default: `pairwise_top1_alignments.parquet`)

**Output:**
- **Statistics file** (`--out_stats`): Per-article similarity metrics
  - Columns: `title`, `n_w`, `n_g`, `sim_mean`, `sim_median`, `sim_max`, `sim_p90`
- **Top-1 alignments file** (`--out_top1`): Best matching chunk pairs
  - Columns: `title`, `wiki_chunk_id`, `grok_chunk_id`, `similarity`, `wiki_text`, `grok_text`

## Workflow Examples

### Separate Processing Pipeline

```bash
# Step 1: Build corpus for Wikipedia
python build_corpus.py \
    --wiki_fp ~/data/grokipedia_wikipedia_articles.ndjson \
    --source wiki

# Step 2: Build corpus for Grokipedia
python build_corpus.py \
    --grok_dir ~/data/scraped_data \
    --source grok

# Step 3: Embed Wikipedia chunks
python embed.py --in_parquet corpus_chunks_wiki.parquet

# Step 4: Embed Grokipedia chunks
python embed.py --in_parquet corpus_chunks_grok.parquet

# Step 5: Compute similarities
python similarities.py \
    --wiki_parquet corpus_chunks_wiki_with_ix.parquet \
    --grok_parquet corpus_chunks_grok_with_ix.parquet
```

### Combined Processing Pipeline

```bash
# Step 1: Build combined corpus
python build_corpus.py \
    --wiki_fp ~/data/grokipedia_wikipedia_articles.ndjson \
    --grok_dir ~/data/scraped_data

# Step 2: Embed all chunks
python embed.py --in_parquet corpus_chunks.parquet

# Step 3: Compute similarities
python similarities.py \
    --in_parquet corpus_chunks_with_ix.parquet \
    --emb corpus_chunks_embeddings.npy
```

## Dependencies

- `pandas`: DataFrame operations
- `numpy`: Array operations
- `torch`: PyTorch for model inference
- `transformers`: Tokenization (AutoTokenizer)
- `sentence-transformers`: Embedding models (SentenceTransformer)
- `tqdm`: Progress bars
- `pathlib`: Path handling

## Model Configuration

The default model is `google/embeddinggemma-300M`.

You can use other SentenceTransformer models by specifying `--model`, but ensure:
1. The same model is used in `build_corpus.py` (for tokenization) and `embed.py` (for embeddings)
2. The model supports the text format (some models have specific input requirements)

## Tips

1. **Memory Management**: For large datasets, process Wikipedia and Grokipedia separately to reduce memory usage
2. **Batch Size**: Adjust `--batch` in `embed.py` based on available GPU memory
3. **Chunk Overlap**: The default stride of 150 with window 250 gives ~60% overlap. Increase stride for faster processing but less overlap
4. **Flash Attention**: Use `--flash_attention2` if your model supports it for faster embedding generation
5. **GPU Usage**: Embeddings run much faster on GPU. Ensure CUDA is available if using GPU

## Notes

- Title normalization: Spaces in titles are replaced with underscores (`_`) for consistency
- Embeddings are L2-normalized, so cosine similarity = dot product
- Only articles present in both Wikipedia and Grokipedia are included in similarity calculations


#!/usr/bin/env python3
"""
Filter chunks for controversy sections and compute similarities using raw embeddings.
Looks for markdown headers containing "controvers*" in chunk text, then computes
similarities between matching wiki/grok chunks.

Reuses functions from compute_similarities_gcs.py to avoid code duplication.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

# Import shared functions from compute_similarities_gcs.py
# Add the scripts directory to path to import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_similarities_gcs import (TORCH_AVAILABLE, ShardedEmbeddingCache,
                                      list_gcs_files, stream_parquet_gcs,
                                      torch)


def contains_controversy_header(text: str) -> bool:
    """
    Check if text contains a markdown header with "controvers*" in it.
    Only checks headers (lines starting with #), not body text.
    """
    if pd.isna(text) or not text:
        return False
    
    text_str = str(text)
    # Check each line for markdown headers containing "controvers"
    for line in text_str.split('\n'):
        line_stripped = line.strip()
        # Only check lines that start with # (markdown headers)
        if line_stripped.startswith('#'):
            if re.search(r'controvers', line_stripped, re.IGNORECASE):
                return True
    
    return False


def load_controversy_titles(wiki_jsonl_path: str, grok_jsonl_path: str) -> Set[str]:
    """Load normalized titles that appear in BOTH controversy JSONL files."""
    wiki_titles_norm = set()
    grok_titles_norm = set()
    
    # Load Wikipedia titles
    if os.path.exists(wiki_jsonl_path):
        with open(wiki_jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    title = data.get('title', '')
                    if title:
                        title_norm = title.lower().replace(' ', '_')
                        wiki_titles_norm.add(title_norm)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(wiki_titles_norm)} titles from Wikipedia JSONL")
    else:
        print(f"Warning: Wikipedia JSONL not found at {wiki_jsonl_path}")
    
    # Load Grokipedia titles
    if os.path.exists(grok_jsonl_path):
        with open(grok_jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    title = data.get('title', '')
                    if title:
                        title_norm = title.lower().replace(' ', '_')
                        grok_titles_norm.add(title_norm)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(grok_titles_norm)} titles from Grokipedia JSONL")
    else:
        print(f"Warning: Grokipedia JSONL not found at {grok_jsonl_path}")
    
    # Return only titles that appear in BOTH files
    common_titles = wiki_titles_norm & grok_titles_norm
    print(f"Found {len(common_titles)} titles in both files")
    
    return common_titles


def build_controversy_chunk_index(
    gcs_bucket: str,
    gcs_prefix: str,
    parquet_glob: str,
    controversy_titles_norm: Set[str],
    source: str
) -> Tuple[Dict[str, List[Tuple[str, int, int, str]]], Dict[str, str]]:
    """
    Build index of chunks with controversy headers for controversy articles only.
    Returns: {title: [(parquet_file, emb_ix, chunk_id, text), ...]}, {parquet_file: emb_file}
    """
    parquet_files = list_gcs_files(gcs_bucket, gcs_prefix, parquet_glob)
    print(f"Streaming {len(parquet_files)} {source} parquet files to find controversy chunks...")
    
    title_index = defaultdict(list)
    parquet_to_emb = {}
    
    for parquet_file in tqdm(parquet_files, desc=f"Scanning {source}"):
        # Map parquet file to embedding file
        parquet_basename = parquet_file.split('/')[-1]
        parquet_name = parquet_basename.replace('_with_ix.parquet', '').replace('.parquet', '')
        emb_file = f"{parquet_name}_embeddings.npy"
        parquet_to_emb[parquet_file] = emb_file
        
        # Stream batches and filter for controversy articles/chunks
        columns = ['title', 'chunk_id', 'emb_ix', 'text']
        
        for batch_df in stream_parquet_gcs(parquet_file, columns=columns):
            for _, row in batch_df.iterrows():
                title = row['title']
                title_norm = title.lower().replace(' ', '_')
                
                # Only process controversy articles
                if title_norm not in controversy_titles_norm:
                    continue
                
                chunk_id = int(row['chunk_id'])
                emb_ix = int(row['emb_ix'])
                text = str(row.get('text', ''))
                
                # Only keep chunks with controversy headers
                if contains_controversy_header(text):
                    title_index[title].append((parquet_file, emb_ix, chunk_id, text))
    
    print(f"Found {sum(len(chunks) for chunks in title_index.values()):,} {source} chunks with controversy headers")
    print(f"  From {len(title_index):,} unique articles")
    
    return dict(title_index), parquet_to_emb


def compute_controversy_similarities(
    gcs_bucket: str,
    gcs_prefix: str,
    wiki_parquet_glob: str,
    grok_parquet_glob: str,
    controversy_titles_norm: Set[str],
    local_temp_dir: str = "/tmp/controversy_chunks"
):
    """
    Find controversy chunks and compute similarities using raw embeddings.
    """
    os.makedirs(local_temp_dir, exist_ok=True)
    
    # Build indices of controversy chunks
    print("Building Wikipedia controversy chunk index...")
    wiki_index, wiki_parquet_to_emb = build_controversy_chunk_index(
        gcs_bucket, gcs_prefix, wiki_parquet_glob, controversy_titles_norm, 'wiki'
    )
    
    print("Building Grokipedia controversy chunk index...")
    grok_index, grok_parquet_to_emb = build_controversy_chunk_index(
        gcs_bucket, gcs_prefix, grok_parquet_glob, controversy_titles_norm, 'grok'
    )
    
    # Find common titles
    wiki_titles = set(wiki_index.keys())
    grok_titles = set(grok_index.keys())
    common_titles = sorted(wiki_titles & grok_titles)
    
    print(f"\nFound {len(common_titles)} articles with controversy chunks in both sources")
    
    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\n✓ GPU acceleration available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠ GPU not available, using CPU (install PyTorch with CUDA for GPU acceleration)")
    
    # Setup embedding caches
    wiki_emb_dir = f"{local_temp_dir}/wiki_embs"
    grok_emb_dir = f"{local_temp_dir}/grok_embs"
    os.makedirs(wiki_emb_dir, exist_ok=True)
    os.makedirs(grok_emb_dir, exist_ok=True)
    
    wiki_emb_cache = ShardedEmbeddingCache(gcs_bucket, gcs_prefix, wiki_emb_dir)
    grok_emb_cache = ShardedEmbeddingCache(gcs_bucket, gcs_prefix, grok_emb_dir)
    
    # Process article-by-article
    results = []
    
    print(f"\nComputing similarities for {len(common_titles):,} articles...")
    
    for title in tqdm(common_titles, desc="Processing articles"):
        w_chunks = wiki_index[title]  # List of (parquet_file, emb_ix, chunk_id, text)
        g_chunks = grok_index[title]
        
        if len(w_chunks) == 0 or len(g_chunks) == 0:
            continue
        
        # Group chunks by shard file
        w_by_shard = defaultdict(list)
        for parquet_file, emb_ix, chunk_id, text in w_chunks:
            emb_file = wiki_parquet_to_emb[parquet_file]
            w_by_shard[emb_file].append((emb_ix, chunk_id, text))
        
        g_by_shard = defaultdict(list)
        for parquet_file, emb_ix, chunk_id, text in g_chunks:
            emb_file = grok_parquet_to_emb[parquet_file]
            g_by_shard[emb_file].append((emb_ix, chunk_id, text))
        
        # Load embeddings and concatenate
        w_emb_list = []
        w_meta_list = []  # (chunk_id, text)
        for emb_file, chunks in w_by_shard.items():
            embs = wiki_emb_cache.get_embeddings(emb_file)
            for emb_ix, chunk_id, text in chunks:
                w_emb_list.append(embs[emb_ix])
                w_meta_list.append((chunk_id, text))
        
        g_emb_list = []
        g_meta_list = []  # (chunk_id, text)
        for emb_file, chunks in g_by_shard.items():
            embs = grok_emb_cache.get_embeddings(emb_file)
            for emb_ix, chunk_id, text in chunks:
                g_emb_list.append(embs[emb_ix])
                g_meta_list.append((chunk_id, text))
        
        # Stack into arrays
        W = np.vstack(w_emb_list).astype(np.float32)
        G = np.vstack(g_emb_list).astype(np.float32)
        
        # Compute similarity matrix (normalized embeddings -> cosine == dot)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            with torch.no_grad():
                W_gpu = torch.from_numpy(W).cuda()
                G_gpu = torch.from_numpy(G).cuda()
                S_gpu = torch.mm(W_gpu, G_gpu.t())
                S = S_gpu.cpu().numpy()
                del W_gpu, G_gpu, S_gpu
                torch.cuda.empty_cache()
        else:
            S = W @ G.T  # Shape: (n_wiki_chunks, n_grok_chunks)
        
        # Find best matches (top-1 for each wiki chunk)
        best_ix = S.argmax(axis=1)
        best_val = S.max(axis=1)
        
        # Store results
        for j, (gi, sv) in enumerate(zip(best_ix, best_val)):
            wiki_chunk_id, wiki_text = w_meta_list[j]
            grok_chunk_id, grok_text = g_meta_list[gi]
            
            results.append({
                'title': title,
                'wiki_chunk_id': int(wiki_chunk_id),
                'grok_chunk_id': int(grok_chunk_id),
                'similarity': float(sv),
                'wiki_text': wiki_text,
                'grok_text': grok_text,
            })
    
    # Cleanup
    wiki_emb_cache.cleanup()
    grok_emb_cache.cleanup()
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Filter controversy chunks and compute similarities using raw embeddings'
    )
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        required=True,
        help='GCS bucket name'
    )
    parser.add_argument(
        '--gcs-prefix',
        type=str,
        required=True,
        help='GCS prefix (directory) for input/output files'
    )
    parser.add_argument(
        '--wiki-parquet-glob',
        type=str,
        required=True,
        help='Glob pattern for Wiki parquet files (e.g., *wiki*with_ix.parquet)'
    )
    parser.add_argument(
        '--grok-parquet-glob',
        type=str,
        required=True,
        help='Glob pattern for Grok parquet files'
    )
    parser.add_argument(
        '--wiki-jsonl',
        type=str,
        required=True,
        help='Path to results/controversy_sections/wikipedia.jsonl file'
    )
    parser.add_argument(
        '--grok-jsonl',
        type=str,
        required=True,
        help='Path to results/controversy_sections/grokipedia.jsonl file'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='controversy_aligned_chunks',
        help='Output filename prefix (without .parquet extension)'
    )
    parser.add_argument(
        '--local-temp-dir',
        type=str,
        default='/tmp/controversy_chunks',
        help='Local temp directory for processing'
    )
    args = parser.parse_args()
    
    # Load controversy titles
    print("Loading controversy titles from JSONL files...")
    controversy_titles_norm = load_controversy_titles(args.wiki_jsonl, args.grok_jsonl)
    if not controversy_titles_norm:
        raise ValueError("No controversy titles loaded. Check that JSONL files exist and contain data.")
    print(f"Total unique controversy article titles: {len(controversy_titles_norm):,}")
    
    # Compute similarities
    results_df = compute_controversy_similarities(
        args.gcs_bucket,
        args.gcs_prefix,
        args.wiki_parquet_glob,
        args.grok_parquet_glob,
        controversy_titles_norm,
        args.local_temp_dir
    )
    
    print(f"\nComputed similarities for {len(results_df):,} chunk pairs")
    
    # Save results
    output_local = f"{args.local_temp_dir}/{args.output_name}.parquet"
    print(f"Saving results to {output_local}...")
    results_df.to_parquet(output_local)
    print(f"✓ Saved {len(results_df):,} chunk pairs locally")
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(args.gcs_bucket)
    output_gcs = f"{args.gcs_prefix}/{args.output_name}.parquet"
    print(f"Uploading to gs://{args.gcs_bucket}/{output_gcs}...")
    bucket.blob(output_gcs).upload_from_filename(output_local)
    print(f"✓ Uploaded to GCS")
    
    # Cleanup
    if os.path.exists(output_local):
        os.remove(output_local)
    
    print(f"\n✓ Done! Results in gs://{args.gcs_bucket}/{output_gcs}")


if __name__ == '__main__':
    main()

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
import pickle
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


def get_header_level(line: str) -> int:
    """Get the level of a markdown header (number of # characters). Returns 0 if not a header."""
    line_stripped = line.strip()
    if not line_stripped.startswith('#'):
        return 0
    level = 0
    for char in line_stripped:
        if char == '#':
            level += 1
        else:
            break
    return level


def extract_sections_from_chunks(chunks: List[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    """
    Group chunks into sections based on markdown headers.
    A section starts with a header containing "controvers*" and continues until the next header of equal or higher level.
    Only extracts sections that contain "controvers*" in the header.
    
    Returns: List of (start_chunk_idx, end_chunk_idx, section_header_text) tuples.
    """
    sections = []
    current_section_start = None
    current_section_header = None
    current_section_level = None
    
    for chunk_idx, (chunk_id, text) in enumerate(chunks):
        text_str = str(text)
        lines = text_str.split('\n')
        
        # Check for headers in this chunk
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                header_level = get_header_level(line_stripped)
                if header_level > 0:
                    # Check if this is a controversy header
                    is_controversy = re.search(r'controvers', line_stripped, re.IGNORECASE) is not None
                    
                    # If we have a current section and this header is equal or higher level, close current section
                    if current_section_start is not None and header_level <= current_section_level:
                        sections.append((current_section_start, chunk_idx, current_section_header))
                        current_section_start = None
                        current_section_header = None
                        current_section_level = None
                    
                    # Start new section if this is a controversy header
                    if is_controversy and current_section_start is None:
                        current_section_start = chunk_idx
                        current_section_header = line_stripped
                        current_section_level = header_level
                    break
    
    # Close the last section
    if current_section_start is not None:
        sections.append((current_section_start, len(chunks), current_section_header))
    
    return sections


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
    local_temp_dir: str = "/tmp/controversy_chunks",
    batch_articles: int = 100
):
    """
    Find controversy chunks and compute similarities using raw embeddings.
    """
    os.makedirs(local_temp_dir, exist_ok=True)
    
    # Cache directory for indices
    cache_dir = f"{local_temp_dir}/index_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filenames based on parameters
    cache_key = f"{gcs_bucket}_{gcs_prefix}_{wiki_parquet_glob}_{grok_parquet_glob}_{len(controversy_titles_norm)}"
    cache_key = re.sub(r'[^a-zA-Z0-9_]', '_', cache_key)
    wiki_cache_file = os.path.join(cache_dir, f"wiki_index_{cache_key}.pkl")
    grok_cache_file = os.path.join(cache_dir, f"grok_index_{cache_key}.pkl")
    
    # Try to load cached indices
    if os.path.exists(wiki_cache_file) and os.path.exists(grok_cache_file):
        print("Loading cached controversy chunk indices...")
        try:
            with open(wiki_cache_file, 'rb') as f:
                cached_wiki = pickle.load(f)
                wiki_index = cached_wiki['title_index']
                wiki_parquet_to_emb = cached_wiki['parquet_to_emb']
            print(f"  ✓ Loaded Wikipedia index: {len(wiki_index):,} articles, {sum(len(chunks) for chunks in wiki_index.values()):,} chunks")
            
            with open(grok_cache_file, 'rb') as f:
                cached_grok = pickle.load(f)
                grok_index = cached_grok['title_index']
                grok_parquet_to_emb = cached_grok['parquet_to_emb']
            print(f"  ✓ Loaded Grokipedia index: {len(grok_index):,} articles, {sum(len(chunks) for chunks in grok_index.values()):,} chunks")
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), rebuilding...")
            wiki_index = None
            grok_index = None
    else:
        wiki_index = None
        grok_index = None
    
    # Build indices if not cached
    if wiki_index is None:
        print("Building Wikipedia controversy chunk index...")
        wiki_index, wiki_parquet_to_emb = build_controversy_chunk_index(
            gcs_bucket, gcs_prefix, wiki_parquet_glob, controversy_titles_norm, 'wiki'
        )
        # Save to cache
        print(f"Saving Wikipedia index to cache...")
        try:
            with open(wiki_cache_file, 'wb') as f:
                pickle.dump({
                    'title_index': wiki_index,
                    'parquet_to_emb': wiki_parquet_to_emb
                }, f)
            print(f"  ✓ Cached Wikipedia index")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")
    
    if grok_index is None:
        print("Building Grokipedia controversy chunk index...")
        grok_index, grok_parquet_to_emb = build_controversy_chunk_index(
            gcs_bucket, gcs_prefix, grok_parquet_glob, controversy_titles_norm, 'grok'
        )
        # Save to cache
        print(f"Saving Grokipedia index to cache...")
        try:
            with open(grok_cache_file, 'wb') as f:
                pickle.dump({
                    'title_index': grok_index,
                    'parquet_to_emb': grok_parquet_to_emb
                }, f)
            print(f"  ✓ Cached Grokipedia index")
        except Exception as e:
            print(f"  Warning: Failed to save cache: {e}")
    
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
    
    # Pre-identify all unique embedding shard files needed
    print("\nIdentifying required embedding shards...")
    wiki_emb_files = sorted(set(wiki_parquet_to_emb.values()))
    grok_emb_files = sorted(set(grok_parquet_to_emb.values()))
    print(f"Wiki shards: {len(wiki_emb_files)}, Grok shards: {len(grok_emb_files)}")
    
    # Pre-load all embedding shards (reuse from compute_similarities_gcs.py approach)
    wiki_emb_cache = ShardedEmbeddingCache(gcs_bucket, gcs_prefix, wiki_emb_dir)
    grok_emb_cache = ShardedEmbeddingCache(gcs_bucket, gcs_prefix, grok_emb_dir)
    
    # Pre-download all embedding shards to avoid repeated GCS downloads
    print("\nPre-downloading embedding shards (this may take a few minutes)...")
    def filter_existing_emb_files(emb_files, local_dir):
        to_download = []
        for emb_file in emb_files:
            filename = os.path.basename(emb_file)
            local_path = os.path.join(local_dir, filename)
            if not os.path.exists(local_path):
                to_download.append(emb_file)
        return to_download
    
    wiki_emb_files_to_preload = filter_existing_emb_files(wiki_emb_files, wiki_emb_dir)
    grok_emb_files_to_preload = filter_existing_emb_files(grok_emb_files, grok_emb_dir)
    
    if wiki_emb_files_to_preload:
        print(f"Pre-loading {len(wiki_emb_files_to_preload)} wiki embedding shards...")
        for emb_file in tqdm(wiki_emb_files_to_preload, desc="Wiki shards"):
            wiki_emb_cache.get_embeddings(emb_file)
    
    if grok_emb_files_to_preload:
        print(f"Pre-loading {len(grok_emb_files_to_preload)} grok embedding shards...")
        for emb_file in tqdm(grok_emb_files_to_preload, desc="Grok shards"):
            grok_emb_cache.get_embeddings(emb_file)
    
    # Process articles in batches
    results = []
    
    print(f"\nComputing similarities for {len(common_titles):,} articles in batches of {batch_articles}...")
    
    for i in tqdm(range(0, len(common_titles), batch_articles), desc="Processing batches"):
        batch_titles = common_titles[i:i+batch_articles]
        
        for title in batch_titles:
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
            
            # Group chunks into sections
            wiki_sections = extract_sections_from_chunks(w_meta_list)
            grok_sections = extract_sections_from_chunks(g_meta_list)
            
            if len(wiki_sections) == 0 or len(grok_sections) == 0:
                continue
            
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
            
            # For each wiki section, compute average similarity to each grok section
            # Only process sections with "controvers*" in the header
            for w_start, w_end, w_header in wiki_sections:
                # Double-check that this is a controversy section
                if not re.search(r'controvers', w_header, re.IGNORECASE):
                    continue
                
                w_section_chunks = list(range(w_start, w_end))
                w_section_text = ' '.join([w_meta_list[i][1] for i in w_section_chunks])
                w_section_chunk_ids = [w_meta_list[i][0] for i in w_section_chunks]
                
                best_grok_section_idx = None
                best_avg_similarity = -1.0
                
                for g_idx, (g_start, g_end, g_header) in enumerate(grok_sections):
                    # Double-check that this is a controversy section
                    if not re.search(r'controvers', g_header, re.IGNORECASE):
                        continue
                    
                    g_section_chunks = list(range(g_start, g_end))
                    
                    # Compute average similarity between this wiki section and this grok section
                    # S[w_section_chunks, :][:, g_section_chunks] gives the submatrix
                    section_similarities = S[np.ix_(w_section_chunks, g_section_chunks)]
                    avg_sim = float(section_similarities.mean())
                    
                    if avg_sim > best_avg_similarity:
                        best_avg_similarity = avg_sim
                        best_grok_section_idx = g_idx
                
                if best_grok_section_idx is not None:
                    g_start, g_end, g_header = grok_sections[best_grok_section_idx]
                    g_section_chunks = list(range(g_start, g_end))
                    g_section_text = ' '.join([g_meta_list[i][1] for i in g_section_chunks])
                    g_section_chunk_ids = [g_meta_list[i][0] for i in g_section_chunks]
                    
                    results.append({
                        'title': title,
                        'wiki_section_header': w_header,
                        'grok_section_header': g_header,
                        'wiki_chunk_ids': w_section_chunk_ids,
                        'grok_chunk_ids': g_section_chunk_ids,
                        'wiki_section_text': w_section_text,
                        'grok_section_text': g_section_text,
                        'avg_similarity': best_avg_similarity,
                        'n_wiki_chunks': len(w_section_chunks),
                        'n_grok_chunks': len(g_section_chunks),
                    })
        
        # Save incrementally every batch
        if len(results) > 0 and len(results) % (batch_articles * 10) == 0:
            print(f"\n  Saving incremental results ({len(results):,} section pairs so far)...")
            temp_df = pd.DataFrame(results)
            temp_path = f"{local_temp_dir}/temp_results_{len(results)}.parquet"
            temp_df.to_parquet(temp_path)
    
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
    parser.add_argument(
        '--batch-articles',
        type=int,
        default=100,
        help='Number of articles to process before saving (default: 100)'
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
        args.local_temp_dir,
        args.batch_articles
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

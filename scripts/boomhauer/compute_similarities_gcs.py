#!/usr/bin/env python3
"""
Compute similarity between Wikipedia and Grokipedia chunks using embeddings from GCS.
Memory-efficient: processes article-by-article to handle large embeddings.
"""
import argparse
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
from tqdm import tqdm

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def list_gcs_files(bucket_name: str, prefix: str, pattern: str):
    """List files in GCS bucket matching pattern."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    files = []
    pattern_re = re.compile(pattern.replace("*", ".*"))
    for blob in blobs:
        if pattern_re.search(blob.name):
            files.append(f"gs://{bucket_name}/{blob.name}")
    return sorted(files)


def download_gcs_file(gcs_path: str, local_path: str):
    """Download file from GCS."""
    gcs_path = gcs_path.replace("gs://", "")
    parts = gcs_path.split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def stream_parquet_gcs(gcs_path: str, columns: List[str] = None):
    """
    Stream parquet file from GCS using PyArrow.
    Yields batches of rows.
    """
    local_path = tempfile.mktemp(suffix='.parquet')
    try:
        download_gcs_file(gcs_path, local_path)
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=10000):
            yield batch.to_pandas()
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def build_title_index_streaming(
    gcs_bucket: str,
    gcs_prefix: str,
    parquet_glob: str,
    include_text: bool = True,
    cache_dir: str = None
) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Dict[str, str], Dict[Tuple[int, int], str], int]:
    """
    Build title index by streaming parquet files.
    Tracks which shard file each chunk belongs to.
    Caches index to disk to avoid rebuilding on crashes.
    Returns: 
        - {title: [(parquet_file, emb_ix, chunk_id), ...]}
        - {parquet_file: emb_file}
        - {(emb_ix, chunk_id): parquet_file} - fast lookup map
        - total_chunks
    Only loads title, chunk_id, emb_ix, and text columns.
    """
    # Generate cache filename based on parameters
    cache_key = f"{gcs_bucket}_{gcs_prefix}_{parquet_glob}_{include_text}"
    cache_key = re.sub(r'[^a-zA-Z0-9_]', '_', cache_key)
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"index_cache_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"Loading index from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"✓ Loaded cached index: {len(cached_data['title_index']):,} articles, {cached_data['total_chunks']:,} chunks")
                    
                    # Build lookup map if not present (for backward compatibility with old caches)
                    lookup_map = cached_data.get('lookup_map', {})
                    if not lookup_map:
                        print("  Building lookup map from cached index (one-time operation)...")
                        for title, chunks in tqdm(cached_data['title_index'].items(), desc="Building lookup map", total=len(cached_data['title_index'])):
                            for chunk_data in chunks:
                                if len(chunk_data) == 3:
                                    pf, eix, cid = chunk_data
                                else:
                                    pf, eix, cid, _ = chunk_data
                                lookup_map[(eix, cid)] = pf
                        
                        # Save updated cache with lookup_map for future use
                        print("  Saving updated cache with lookup map...")
                        try:
                            cached_data['lookup_map'] = lookup_map
                            with open(cache_file, 'wb') as f:
                                pickle.dump(cached_data, f)
                            print("  ✓ Updated cache saved")
                        except Exception as e:
                            print(f"  Warning: Could not save updated cache: {e}")
                    
                    return cached_data['title_index'], cached_data['parquet_to_emb'], lookup_map, cached_data['total_chunks']
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}), rebuilding...")
                import traceback
                traceback.print_exc()
    
    parquet_files = list_gcs_files(gcs_bucket, gcs_prefix, parquet_glob)
    print(f"Streaming {len(parquet_files)} parquet files to build index...")
    
    title_index = defaultdict(list)
    parquet_to_emb = {}  # Map parquet file to corresponding embedding file
    lookup_map = {}  # Fast lookup: {(emb_ix, chunk_id): parquet_file}
    total_chunks = 0
    
    for parquet_file in tqdm(parquet_files, desc="Building index"):
        # Map parquet file to embedding file
        # e.g., corpus_chunks_grok_shard0of4_shard0of4_with_ix.parquet 
        #   -> corpus_chunks_grok_shard0of4_shard0of4_embeddings.npy
        # Extract filename from GCS path
        parquet_basename = parquet_file.split('/')[-1]  # Get filename
        parquet_name = parquet_basename.replace('_with_ix.parquet', '').replace('.parquet', '')
        emb_file = f"{parquet_name}_embeddings.npy"
        parquet_to_emb[parquet_file] = emb_file
        
        # Stream batches from this parquet file
        columns = ['title', 'chunk_id', 'emb_ix']
        if include_text:
            columns.append('text')
            
        for batch_df in stream_parquet_gcs(parquet_file, columns=columns):
            for _, row in batch_df.iterrows():
                title = row['title']
                chunk_id = int(row['chunk_id'])
                emb_ix = int(row['emb_ix'])
                
                # Store without text to save memory
                title_index[title].append((parquet_file, emb_ix, chunk_id))
                lookup_map[(emb_ix, chunk_id)] = parquet_file
                total_chunks += 1
        
        # Save incremental cache after each file
        if cache_file and total_chunks % 100000 == 0:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'title_index': dict(title_index),
                        'parquet_to_emb': parquet_to_emb,
                        'lookup_map': lookup_map,
                        'total_chunks': total_chunks
                    }, f)
            except Exception as e:
                print(f"Warning: Failed to save incremental cache: {e}")
    
    # Final cache save
    if cache_file:
        print(f"Saving index to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'title_index': dict(title_index),
                    'parquet_to_emb': parquet_to_emb,
                    'lookup_map': lookup_map,
                    'total_chunks': total_chunks
                }, f)
            print(f"✓ Cached index saved")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    return dict(title_index), parquet_to_emb, lookup_map, total_chunks


def load_embeddings_gcs(gcs_path: str, use_mmap: bool = True) -> Tuple[np.ndarray, str]:
    """Load embeddings from GCS, optionally using memory mapping."""
    local_path = tempfile.mktemp(suffix='.npy')
    
    try:
        download_gcs_file(gcs_path, local_path)
        if use_mmap:
            # Memory map for large files
            embs = np.load(local_path, mmap_mode='r')
        else:
            embs = np.load(local_path)
        return embs, local_path
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        raise


class ShardedEmbeddingCache:
    """Cache for sharded embeddings, loads on-demand or pre-loads all."""
    def __init__(self, gcs_bucket: str, gcs_prefix: str, local_temp_dir: str, preload_all: bool = False, emb_files: List[str] = None):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.local_temp_dir = local_temp_dir
        self.cache = {}  # {emb_file: (emb_array, local_path)}
        os.makedirs(local_temp_dir, exist_ok=True)
        
        if preload_all and emb_files:
            self.preload_all(emb_files)
    
    def preload_all(self, emb_files: List[str]):
        """Pre-download and cache all embedding shards."""
        print(f"Pre-downloading {len(emb_files)} embedding shards...")
        for emb_file in tqdm(emb_files, desc="Downloading shards"):
            if emb_file not in self.cache:
                gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{emb_file}"
                try:
                    embs, local_path = load_embeddings_gcs(gcs_path, use_mmap=True)
                    self.cache[emb_file] = (embs, local_path)
                    print(f"  ✓ Loaded {emb_file} (shape: {embs.shape})")
                except Exception as e:
                    print(f"  ✗ Failed to load {emb_file}: {e}")
        print(f"✓ Pre-loaded {len(self.cache)} shards")
    
    def get_embeddings(self, emb_file: str) -> np.ndarray:
        """Get embeddings from cache or load from GCS."""
        if emb_file not in self.cache:
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{emb_file}"
            embs, local_path = load_embeddings_gcs(gcs_path, use_mmap=True)
            self.cache[emb_file] = (embs, local_path)
        return self.cache[emb_file][0]
    
    def cleanup(self):
        """Remove cached files."""
        for emb_file, (embs, local_path) in self.cache.items():
            if os.path.exists(local_path):
                os.remove(local_path)
        self.cache.clear()


def compute_similarities_gcs(
    gcs_bucket: str,
    gcs_prefix: str,
    wiki_emb_gcs: str,
    grok_emb_gcs: str,
    wiki_parquet_glob: str,
    grok_parquet_glob: str,
    output_prefix: str,
    local_temp_dir: str = "/tmp/similarities",
    batch_articles: int = 100,
    include_text: bool = True
):
    """
    Compute similarity between Wikipedia and Grokipedia chunks.
    Streams parquet files to build index, then processes article-by-article.
    """
    os.makedirs(local_temp_dir, exist_ok=True)
    
    # Build title indices by streaming parquet files (tracks which shard each chunk is in)
    # Exclude text from index to save memory - we'll load it on-demand if needed
    cache_dir = f"{local_temp_dir}/index_cache"
    print("Building Wikipedia title index (without text to save memory)...")
    wiki_index, wiki_parquet_to_emb, wiki_lookup_map, wiki_total_chunks = build_title_index_streaming(
        gcs_bucket, gcs_prefix, wiki_parquet_glob, include_text=False, cache_dir=cache_dir
    )
    print(f"Wiki: {len(wiki_index):,} articles, {wiki_total_chunks:,} chunks, {len(wiki_parquet_to_emb)} shards")
    
    print("Building Grokipedia title index (without text to save memory)...")
    grok_index, grok_parquet_to_emb, grok_lookup_map, grok_total_chunks = build_title_index_streaming(
        gcs_bucket, gcs_prefix, grok_parquet_glob, include_text=False, cache_dir=cache_dir
    )
    print(f"Grok: {len(grok_index):,} articles, {grok_total_chunks:,} chunks, {len(grok_parquet_to_emb)} shards")
    
    # Get all unique embedding shard files that will be needed
    print("Identifying required embedding shards...")
    wiki_emb_files = sorted(set(wiki_parquet_to_emb.values()))
    grok_emb_files = sorted(set(grok_parquet_to_emb.values()))
    print(f"Wiki shards: {len(wiki_emb_files)}, Grok shards: {len(grok_emb_files)}")
    
    # Pre-download all embedding shards at startup (avoids repeated GCS downloads)
    print("\nPre-downloading all embedding shards (skipping if already exist, this may take a few minutes)...")
    def filter_existing_emb_files(emb_files, local_dir):
        to_download = []
        for emb_file in emb_files:
            filename = os.path.basename(emb_file)
            local_path = os.path.join(local_dir, filename)
            if not os.path.exists(local_path):
                to_download.append(emb_file)
        return to_download

    wiki_emb_dir = f"{local_temp_dir}/wiki_embs"
    grok_emb_dir = f"{local_temp_dir}/grok_embs"

    os.makedirs(wiki_emb_dir, exist_ok=True)
    os.makedirs(grok_emb_dir, exist_ok=True)

    # Only preload files that aren't already present locally
    wiki_emb_files_to_preload = filter_existing_emb_files(wiki_emb_files, wiki_emb_dir)
    grok_emb_files_to_preload = filter_existing_emb_files(grok_emb_files, grok_emb_dir)

    wiki_emb_cache = ShardedEmbeddingCache(
        gcs_bucket, gcs_prefix, wiki_emb_dir,
        preload_all=True, emb_files=wiki_emb_files_to_preload
    )
    grok_emb_cache = ShardedEmbeddingCache(
        gcs_bucket, gcs_prefix, grok_emb_dir,
        preload_all=True, emb_files=grok_emb_files_to_preload
    )
    
    # Check GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"\n✓ GPU acceleration available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠ GPU not available, using CPU (install PyTorch with CUDA for GPU acceleration)")
    
    # Find common titles
    wiki_titles = set(wiki_index.keys())
    grok_titles = set(grok_index.keys())
    common_titles = sorted(wiki_titles & grok_titles)
    
    print(f"Found {len(common_titles)} articles in both datasets")
    
    # Process article-by-article
    stats_rows = []
    topk_rows = []
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    print(f"\nStarting similarity computation...")
    print(f"Processing {len(common_titles):,} articles in batches of {batch_articles}")
    
    for i in tqdm(range(0, len(common_titles), batch_articles), desc="Processing articles"):
        batch_titles = common_titles[i:i+batch_articles]
        
        for title in batch_titles:
            w_chunks = wiki_index[title]  # List of (parquet_file, emb_ix, chunk_id) - no text
            g_chunks = grok_index[title]
            
            if len(w_chunks) == 0 or len(g_chunks) == 0:
                continue
            
            # Group chunks by shard file
            w_by_shard = defaultdict(list)
            for chunk_data in w_chunks:
                if len(chunk_data) == 4:
                    parquet_file, emb_ix, chunk_id, _ = chunk_data  # Old format with text
                else:
                    parquet_file, emb_ix, chunk_id = chunk_data  # New format without text
                emb_file = wiki_parquet_to_emb[parquet_file]
                w_by_shard[emb_file].append((emb_ix, chunk_id))
            
            g_by_shard = defaultdict(list)
            for chunk_data in g_chunks:
                if len(chunk_data) == 4:
                    parquet_file, emb_ix, chunk_id, _ = chunk_data  # Old format with text
                else:
                    parquet_file, emb_ix, chunk_id = chunk_data  # New format without text
                emb_file = grok_parquet_to_emb[parquet_file]
                g_by_shard[emb_file].append((emb_ix, chunk_id))
            
            # Load embeddings from each shard and concatenate
            w_emb_list = []
            w_meta_list = []  # (chunk_id, parquet_file) for later text lookup if needed
            for emb_file, chunks in w_by_shard.items():
                embs = wiki_emb_cache.get_embeddings(emb_file)
                for emb_ix, chunk_id in chunks:
                    w_emb_list.append(embs[emb_ix])
                    # Fast lookup using lookup_map
                    parquet_file = wiki_lookup_map.get((emb_ix, chunk_id))
                    if not parquet_file:
                        # Fallback for old cached indices
                        for chunk_data in w_chunks:
                            if len(chunk_data) == 3:
                                pf, eix, cid = chunk_data
                            else:
                                pf, eix, cid, _ = chunk_data
                            if eix == emb_ix and cid == chunk_id:
                                parquet_file = pf
                                break
                    w_meta_list.append((chunk_id, parquet_file))
            
            g_emb_list = []
            g_meta_list = []  # (chunk_id, parquet_file) for later text lookup if needed
            for emb_file, chunks in g_by_shard.items():
                embs = grok_emb_cache.get_embeddings(emb_file)
                for emb_ix, chunk_id in chunks:
                    g_emb_list.append(embs[emb_ix])
                    # Fast lookup using lookup_map
                    parquet_file = grok_lookup_map.get((emb_ix, chunk_id))
                    if not parquet_file:
                        # Fallback for old cached indices
                        for chunk_data in g_chunks:
                            if len(chunk_data) == 3:
                                pf, eix, cid = chunk_data
                            else:
                                pf, eix, cid, _ = chunk_data
                            if eix == emb_ix and cid == chunk_id:
                                parquet_file = pf
                                break
                    g_meta_list.append((chunk_id, parquet_file))
            
            # Stack into arrays
            W = np.vstack(w_emb_list).astype(np.float32)
            G = np.vstack(g_emb_list).astype(np.float32)
            
            # Compute similarity matrix (normalized embeddings -> cosine == dot)
            # Use GPU if available, otherwise CPU
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Transfer to GPU and compute
                with torch.no_grad():  # Save memory by not tracking gradients
                    W_gpu = torch.from_numpy(W).cuda()
                    G_gpu = torch.from_numpy(G).cuda()
                    S_gpu = torch.mm(W_gpu, G_gpu.t())  # Matrix multiplication on GPU
                    
                    # Compute stats on GPU
                    sim_mean = float(S_gpu.mean().item())
                    sim_max = float(S_gpu.max().item())
                    
                    # Top-1 alignments (best match for each wiki chunk) on GPU
                    best_val, best_ix = torch.max(S_gpu, dim=1)
                    best_ix = best_ix.cpu().numpy()
                    best_val = best_val.cpu().numpy()
                    
                    # For median and percentile, need CPU numpy
                    S_cpu = S_gpu.cpu().numpy()
                    sim_median = float(np.median(S_cpu))
                    sim_p90 = float(np.percentile(S_cpu, 90))
                    
                    # Clear GPU cache
                    del W_gpu, G_gpu, S_gpu
                    torch.cuda.empty_cache()
                    S = S_cpu
            else:
                # CPU fallback
                S = W @ G.T  # Shape: (n_wiki_chunks, n_grok_chunks)
                sim_mean = float(S.mean())
                sim_median = float(np.median(S))
                sim_max = float(S.max())
                sim_p90 = float(np.percentile(S, 90))
                best_ix = S.argmax(axis=1)
                best_val = S.max(axis=1)
            
            stats_rows.append({
                'title': title,
                'n_w': len(w_chunks),
                'n_g': len(g_chunks),
                'sim_mean': sim_mean,
                'sim_median': sim_median,
                'sim_max': sim_max,
                'sim_p90': sim_p90,
            })
            
            for j, (gi, sv) in enumerate(zip(best_ix, best_val)):
                wiki_chunk_id, wiki_parquet_file = w_meta_list[j]
                grok_chunk_id, grok_parquet_file = g_meta_list[gi]
                
                # Skip text loading for now (too slow) - can add back later if needed
                # Text loading from GCS is extremely slow, so we skip it by default
                wiki_text = ''
                grok_text = ''
                # if include_text:
                #     # This is too slow - would need to download entire parquet files
                #     # Consider pre-loading text into a separate cache if needed
                #     pass
                
                topk_rows.append({
                    'title': title,
                    'wiki_chunk_id': int(wiki_chunk_id),
                    'grok_chunk_id': int(grok_chunk_id),
                    'similarity': float(sv),
                    'wiki_text': wiki_text,
                    'grok_text': grok_text,
                })
        
        # Save incrementally
        if len(stats_rows) > 0 and len(stats_rows) % 1000 == 0:
            pair_stats = pd.DataFrame(stats_rows)
            pair_top1 = pd.DataFrame(topk_rows)
            
            stats_local = f"{local_temp_dir}/pairwise_stats_temp.parquet"
            top1_local = f"{local_temp_dir}/pairwise_top1_temp.parquet"
            pair_stats.to_parquet(stats_local)
            pair_top1.to_parquet(top1_local)
    
    # Final save
    print("Saving final results...")
    pair_stats = pd.DataFrame(stats_rows).sort_values('sim_mean', ascending=True)
    pair_top1 = pd.DataFrame(topk_rows)
    
    stats_local = f"{local_temp_dir}/pairwise_stats.parquet"
    top1_local = f"{local_temp_dir}/pairwise_top1_alignments.parquet"
    pair_stats.to_parquet(stats_local)
    pair_top1.to_parquet(top1_local)
    
    # Upload to GCS
    stats_gcs = f"{gcs_prefix}/{output_prefix}_pairwise_stats.parquet"
    top1_gcs = f"{gcs_prefix}/{output_prefix}_pairwise_top1_alignments.parquet"
    
    print(f"Uploading to gs://{gcs_bucket}/{stats_gcs}...")
    bucket.blob(stats_gcs).upload_from_filename(stats_local)
    print(f"Uploading to gs://{gcs_bucket}/{top1_gcs}...")
    bucket.blob(top1_gcs).upload_from_filename(top1_local)
    
    # Cleanup
    wiki_emb_cache.cleanup()
    grok_emb_cache.cleanup()
    os.remove(stats_local)
    os.remove(top1_local)
    
    print(f"✓ Done! Results in gs://{gcs_bucket}/{gcs_prefix}/{output_prefix}_*.parquet")


def main():
    parser = argparse.ArgumentParser(description='Compute similarities using GCS embeddings')
    parser.add_argument('--gcs-bucket', type=str, required=True,
                       help='GCS bucket name')
    parser.add_argument('--gcs-prefix', type=str, required=True,
                       help='GCS prefix (directory)')
    parser.add_argument('--wiki-emb', type=str, default=None,
                       help='Wiki embeddings filename (if merged). Leave empty to use sharded embeddings.')
    parser.add_argument('--grok-emb', type=str, default=None,
                       help='Grok embeddings filename (if merged). Leave empty to use sharded embeddings.')
    parser.add_argument('--wiki-parquet-glob', type=str, required=True,
                       help='Glob pattern for Wiki parquet files (e.g., *wiki*with_ix.parquet)')
    parser.add_argument('--grok-parquet-glob', type=str, required=True,
                       help='Glob pattern for Grok parquet files')
    parser.add_argument('--output-prefix', type=str, default='similarities',
                       help='Output prefix for result files')
    parser.add_argument('--local-temp-dir', type=str, default='/tmp/similarities',
                       help='Local temp directory')
    parser.add_argument('--batch-articles', type=int, default=100,
                       help='Number of articles to process before saving')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for similarity computation (if available, default: True)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                       help='Disable GPU acceleration')
    args = parser.parse_args()
    
    # Override GPU usage if explicitly disabled
    if not args.use_gpu:
        global TORCH_AVAILABLE
        TORCH_AVAILABLE = False
    
    compute_similarities_gcs(
        args.gcs_bucket,
        args.gcs_prefix,
        args.wiki_emb,
        args.grok_emb,
        args.wiki_parquet_glob,
        args.grok_parquet_glob,
        args.output_prefix,
        args.local_temp_dir,
        args.batch_articles
    )


if __name__ == '__main__':
    main()


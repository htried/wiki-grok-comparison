#!/usr/bin/env python3
"""
Compute similarity between Grokipedia v0.2 corpus and:
(a) Wikipedia page titles
(b) Grokipedia v0.1 page titles

Memory-efficient: processes article-by-article using memory-mapped embeddings.
"""

import argparse
import glob
import json
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


def _norm(s: str) -> str:
    """Normalize title for matching."""
    return " ".join((s or "").strip().lower().split())


def download_gcs_file(gcs_path: str, local_path: str):
    """Download file from GCS."""
    gcs_path_clean = gcs_path.replace("gs://", "")
    parts = gcs_path_clean.split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


def stream_parquet_gcs(gcs_path: str, columns: List[str] = None, cache_dir: str = None):
    """
    Stream parquet file from GCS using PyArrow.
    Yields batches of rows.
    Uses cache if available.
    """
    local_path = None
    
    # Check cache first if cache_dir is provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(gcs_path, cache_dir)
        if os.path.exists(cache_path):
            parquet_file = pq.ParquetFile(cache_path)
            for batch in parquet_file.iter_batches(columns=columns, batch_size=10000):
                yield batch.to_pandas()
            return
        local_path = cache_path
    else:
        local_path = tempfile.mktemp(suffix='.parquet')
    
    try:
        download_gcs_file(gcs_path, local_path)
        parquet_file = pq.ParquetFile(local_path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=10000):
            yield batch.to_pandas()
    finally:
        # Don't remove cached files
        if not cache_dir and os.path.exists(local_path):
            os.remove(local_path)


class ShardedEmbeddingCache:
    """Cache for sharded embeddings, loads on-demand or pre-loads all."""
    def __init__(self, gcs_bucket: str, gcs_prefix: str, local_cache_dir: str, preload_all: bool = False, emb_files: List[str] = None):
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.local_cache_dir = local_cache_dir
        self.cache = {}  # {emb_file: (emb_array, local_path)}
        os.makedirs(local_cache_dir, exist_ok=True)
        
        if preload_all and emb_files:
            self.preload_all(emb_files)
    
    def preload_all(self, emb_files: List[str]):
        """Pre-download and cache all embedding shards."""
        print(f"Pre-downloading {len(emb_files)} embedding shards...")
        for emb_file in tqdm(emb_files, desc="Downloading shards"):
            if emb_file not in self.cache:
                gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{emb_file}"
                try:
                    embs, local_path = load_embeddings_gcs(gcs_path, use_mmap=True, cache_dir=self.local_cache_dir)
                    self.cache[emb_file] = (embs, local_path)
                    print(f"  ✓ Loaded {emb_file} (shape: {embs.shape})")
                except Exception as e:
                    print(f"  ✗ Failed to load {emb_file}: {e}")
        print(f"✓ Pre-loaded {len(self.cache)} shards")
    
    def get_embeddings(self, emb_file: str) -> np.ndarray:
        """Get embeddings from cache or load from GCS."""
        if emb_file not in self.cache:
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}/{emb_file}"
            embs, local_path = load_embeddings_gcs(gcs_path, use_mmap=True, cache_dir=self.local_cache_dir)
            self.cache[emb_file] = (embs, local_path)
        return self.cache[emb_file][0]
    
    def cleanup(self):
        """Remove cached files (only temp files, not persistent cache)."""
        # Don't remove files in cache_dir - they're meant to persist
        self.cache.clear()


def list_shards_gcs(gcs_pattern: str) -> List[str]:
    """List all shard files matching a pattern in GCS."""
    gcs_path_clean = gcs_pattern.replace("gs://", "")
    parts = gcs_path_clean.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    # Extract base pattern (e.g., "corpus_chunks_wiki_shard*of4_embeddings.npy")
    if '*' in prefix:
        base_prefix = prefix.split('*')[0]
        suffix = prefix.split('*')[-1] if '*' in prefix else ""
    else:
        # Single file, not a pattern
        return [f"gs://{gcs_pattern}"]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    shards = []
    for blob in bucket.list_blobs(prefix=base_prefix):
        if blob.name.endswith(suffix) and blob.name.endswith('.npy'):
            shards.append(f"gs://{bucket_name}/{blob.name}")
    
    # Sort shards by extracting numeric indices
    def shard_key(path):
        numbers = re.findall(r'shard(\d+)of(\d+)', path)
        if numbers:
            return tuple(int(n) for n in numbers[-1])
        return (999, 999)
    
    return sorted(shards, key=shard_key)


def list_shards_local(local_pattern: str) -> List[str]:
    """List all shard files matching a pattern locally."""
    shards = glob.glob(local_pattern)
    # Sort by extracting numeric indices
    def shard_key(path):
        numbers = re.findall(r'shard(\d+)of(\d+)', path)
        if numbers:
            return tuple(int(n) for n in numbers[-1])
        return (999, 999)
    return sorted(shards, key=shard_key)


def load_embeddings_sharded(path_or_pattern: str, use_mmap: bool = True, cache_dir: str = None) -> np.ndarray:
    """
    Load embeddings from single file or combine sharded files.
    Supports local paths, GCS paths, or patterns with * wildcard.
    If cache_dir is provided, GCS files will be cached locally.
    """
    # Check if it's a pattern (contains *)
    is_pattern = '*' in path_or_pattern
    
    if is_pattern:
        # List shards
        if path_or_pattern.startswith('gs://'):
            shard_paths = list_shards_gcs(path_or_pattern)
        else:
            shard_paths = list_shards_local(path_or_pattern)
        
        if not shard_paths:
            raise ValueError(f"No shards found matching pattern: {path_or_pattern}")
        
        print(f"  Found {len(shard_paths)} shards, combining...")
        
        # Load and combine shards
        shard_arrays = []
        total_rows = 0
        embedding_dim = None
        
        for i, shard_path in enumerate(shard_paths):
            print(f"    Loading shard {i+1}/{len(shard_paths)}: {shard_path.split('/')[-1]}")
            if shard_path.startswith('gs://'):
                arr, cached_path = load_embeddings_gcs(shard_path, use_mmap=False, cache_dir=cache_dir)
                # Don't remove cached files
                if cache_dir and cached_path.startswith(cache_dir):
                    pass  # Keep cached file
                elif not cache_dir and os.path.exists(cached_path):
                    os.remove(cached_path)
            else:
                arr = np.load(shard_path, mmap_mode='r' if use_mmap else None)
            
            if embedding_dim is None:
                embedding_dim = arr.shape[1] if len(arr.shape) > 1 else arr.shape[0]
            
            shard_arrays.append(arr)
            total_rows += arr.shape[0]
        
        # Combine into single array
        print(f"  Combining {total_rows:,} embeddings...")
        combined = np.vstack(shard_arrays)
        
        # Free memory
        del shard_arrays
        
        return combined
    else:
        # Single file
        if path_or_pattern.startswith('gs://'):
            arr, _ = load_embeddings_gcs(path_or_pattern, use_mmap=use_mmap, cache_dir=cache_dir)
            return arr
        else:
            if use_mmap:
                return np.load(path_or_pattern, mmap_mode='r')
            else:
                return np.load(path_or_pattern)


def load_embeddings_local(local_path: str, use_mmap: bool = True) -> np.ndarray:
    """Load embeddings from local file, optionally using memory mapping."""
    if use_mmap:
        return np.load(local_path, mmap_mode='r')
    else:
        return np.load(local_path)


def get_cache_path(gcs_path: str, cache_dir: str) -> str:
    """Generate a cache file path from a GCS path."""
    gcs_path_clean = gcs_path.replace("gs://", "").replace("/", "_")
    # Sanitize for filesystem
    cache_path = os.path.join(cache_dir, gcs_path_clean)
    return cache_path


def load_embeddings_gcs(gcs_path: str, use_mmap: bool = True, cache_dir: str = None) -> Tuple[np.ndarray, str]:
    """
    Load embeddings from GCS, optionally using memory mapping.
    If cache_dir is provided, will cache files locally for reuse.
    """
    local_path = None
    
    # Check cache first if cache_dir is provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(gcs_path, cache_dir)
        if os.path.exists(cache_path):
            print(f"    Using cached file: {cache_path}")
            if use_mmap:
                embs = np.load(cache_path, mmap_mode='r')
            else:
                embs = np.load(cache_path)
            return embs, cache_path
        local_path = cache_path
    else:
        local_path = tempfile.mktemp(suffix='.npy')
    
    try:
        gcs_path_clean = gcs_path.replace("gs://", "")
        parts = gcs_path_clean.split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""
        
        print(f"    Downloading from GCS: {gcs_path}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        print(f"    Downloaded to: {local_path}")
        
        if use_mmap:
            embs = np.load(local_path, mmap_mode='r')
        else:
            embs = np.load(local_path)
        return embs, local_path
    except Exception:
        if os.path.exists(local_path) and not cache_dir:
            # Only remove temp files, not cached files
            os.remove(local_path)
        raise


def list_parquet_shards_gcs(gcs_pattern: str) -> List[str]:
    """List all parquet shard files matching a pattern in GCS."""
    gcs_path_clean = gcs_pattern.replace("gs://", "")
    parts = gcs_path_clean.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    if '*' in prefix:
        base_prefix = prefix.split('*')[0]
        suffix = prefix.split('*')[-1] if '*' in prefix else ""
    else:
        return [f"gs://{gcs_pattern}"]
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    shards = []
    for blob in bucket.list_blobs(prefix=base_prefix):
        if blob.name.endswith(suffix) and blob.name.endswith('.parquet'):
            shards.append(f"gs://{bucket_name}/{blob.name}")
    
    def shard_key(path):
        import re
        numbers = re.findall(r'shard(\d+)of(\d+)', path)
        if numbers:
            return tuple(int(n) for n in numbers[-1])
        return (999, 999)
    
    return sorted(shards, key=shard_key)


def list_parquet_shards_local(local_pattern: str) -> List[str]:
    """List all parquet shard files matching a pattern locally."""
    shards = glob.glob(local_pattern)
    def shard_key(path):
        numbers = re.findall(r'shard(\d+)of(\d+)', path)
        if numbers:
            return tuple(int(n) for n in numbers[-1])
        return (999, 999)
    return sorted(shards, key=shard_key)


def load_parquet_from_gcs(gcs_path: str, cache_dir: str = None) -> pd.DataFrame:
    """Load parquet file from GCS, with optional caching."""
    local_path = None
    
    # Check cache first if cache_dir is provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = get_cache_path(gcs_path, cache_dir)
        if os.path.exists(cache_path):
            print(f"      Using cached parquet: {cache_path}")
            return pd.read_parquet(cache_path)
        local_path = cache_path
    else:
        local_path = tempfile.mktemp(suffix='.parquet')
    
    # Download from GCS
    gcs_path_clean = gcs_path.replace("gs://", "")
    parts = gcs_path_clean.split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    
    print(f"      Downloading parquet from GCS: {gcs_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"      Downloaded to: {local_path}")
    
    df = pd.read_parquet(local_path)
    
    # Don't remove cached files
    if not cache_dir and os.path.exists(local_path):
        os.remove(local_path)
    
    return df


def build_title_index_streaming_gcs(
    gcs_pattern: str,
    cache_dir: str = None
) -> Tuple[Dict[str, List[Tuple[str, int, int]]], Dict[str, str], Dict[Tuple[int, int], str]]:
    """
    Build title index by streaming parquet files from GCS.
    Tracks which shard file each chunk belongs to.
    Caches index to disk to avoid rebuilding on crashes.
    Returns: 
        - {title: [(parquet_file, emb_ix, chunk_id), ...]}
        - {parquet_file: emb_file}
        - {(emb_ix, chunk_id): parquet_file} - fast lookup map
    """
    # Generate cache filename based on pattern
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = gcs_pattern.replace("gs://", "").replace("/", "_").replace("*", "star")
        cache_key = re.sub(r'[^a-zA-Z0-9_]', '_', cache_key)
        cache_file = os.path.join(cache_dir, f"index_cache_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"    Loading index from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"    ✓ Loaded cached index: {len(cached_data['title_index']):,} articles, {cached_data['total_chunks']:,} chunks")
                    return cached_data['title_index'], cached_data['parquet_to_emb'], cached_data.get('lookup_map', {})
            except Exception as e:
                print(f"    Warning: Failed to load cache ({e}), rebuilding...")
    
    if gcs_pattern.startswith('gs://'):
        shard_paths = list_parquet_shards_gcs(gcs_pattern)
    else:
        shard_paths = list_parquet_shards_local(gcs_pattern)
    
    if not shard_paths:
        raise ValueError(f"No parquet shards found matching pattern: {gcs_pattern}")
    
    print(f"    Streaming {len(shard_paths)} parquet shards to build index...")
    
    title_index = defaultdict(list)
    parquet_to_emb = {}  # Map parquet file to corresponding embedding file
    lookup_map = {}  # Fast lookup: {(emb_ix, chunk_id): parquet_file}
    total_chunks = 0
    
    for parquet_file in tqdm(shard_paths, desc="Building index"):
        # Map parquet file to embedding file
        parquet_basename = parquet_file.split('/')[-1]
        parquet_name = parquet_basename.replace('_with_ix.parquet', '').replace('.parquet', '')
        emb_file = f"{parquet_name}_embeddings.npy"
        parquet_to_emb[parquet_file] = emb_file
        
        # Stream batches from this parquet file
        columns = ['title', 'chunk_id', 'emb_ix']
        for batch_df in stream_parquet_gcs(parquet_file, columns=columns, cache_dir=cache_dir):
            for _, row in batch_df.iterrows():
                title = row.get('title', '')
                if title:
                    chunk_id = int(row.get('chunk_id', -1))
                    emb_ix = int(row.get('emb_ix', -1))
                    
                    if emb_ix >= 0 and chunk_id >= 0:
                        # Titles are stored with underscores, convert to spaces then normalize
                        title_with_spaces = title.replace('_', ' ')
                        norm_title = _norm(title_with_spaces)
                        
                        title_index[norm_title].append((parquet_file, emb_ix, chunk_id))
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
                print(f"    Warning: Failed to save incremental cache: {e}")
    
    # Final cache save
    if cache_file:
        print(f"    Saving index to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'title_index': dict(title_index),
                    'parquet_to_emb': parquet_to_emb,
                    'lookup_map': lookup_map,
                    'total_chunks': total_chunks
                }, f)
            print(f"    ✓ Cached index saved")
        except Exception as e:
            print(f"    Warning: Failed to save cache: {e}")
    
    return dict(title_index), parquet_to_emb, lookup_map


def build_title_index_from_parquet(parquet_path: str, cache_dir: str = None, use_streaming: bool = True) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build title index from parquet file(s).
    For GCS files, uses streaming to avoid loading everything into memory.
    For local files, loads directly.
    Caches index to disk to avoid rebuilding.
    Returns: {normalized_title: [(emb_ix, chunk_id), ...]} for local files
             or {normalized_title: [(parquet_file, emb_ix, chunk_id), ...]} for GCS files
    """
    is_pattern = '*' in parquet_path
    is_gcs = parquet_path.startswith('gs://')
    
    # For GCS files, use streaming approach (returns shard-aware format)
    if is_gcs and use_streaming:
        title_index_sharded, parquet_to_emb, lookup_map = build_title_index_streaming_gcs(parquet_path, cache_dir=cache_dir)
        # Return in shard-aware format - caller will handle differently
        return title_index_sharded, parquet_to_emb, lookup_map
    
    # For local files, check cache first
    cache_file = None
    if cache_dir and not is_gcs:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = parquet_path.replace("/", "_").replace("*", "star")
        cache_key = re.sub(r'[^a-zA-Z0-9_]', '_', cache_key)
        cache_file = os.path.join(cache_dir, f"index_cache_local_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print(f"    Loading local index from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"    ✓ Loaded cached index: {len(cached_data):,} articles")
                    return cached_data
            except Exception as e:
                print(f"    Warning: Failed to load cache ({e}), rebuilding...")
    
    # For local files or non-streaming, use simple approach
    if is_pattern:
        if is_gcs:
            shard_paths = list_parquet_shards_gcs(parquet_path)
        else:
            shard_paths = list_parquet_shards_local(parquet_path)
        
        if not shard_paths:
            raise ValueError(f"No parquet shards found matching pattern: {parquet_path}")
        
        print(f"    Found {len(shard_paths)} parquet shards, combining...")
        
        dfs = []
        for i, shard_path in enumerate(shard_paths):
            print(f"      Loading parquet shard {i+1}/{len(shard_paths)}: {shard_path.split('/')[-1]}")
            if is_gcs:
                df_shard = load_parquet_from_gcs(shard_path, cache_dir=cache_dir)
            else:
                df_shard = pd.read_parquet(shard_path)
            dfs.append(df_shard)
        
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    else:
        # Single parquet file
        if is_gcs:
            df = load_parquet_from_gcs(parquet_path, cache_dir=cache_dir)
        else:
            df = pd.read_parquet(parquet_path)
    
    title_index = defaultdict(list)
    
    for _, row in df.iterrows():
        title = row.get('title', '')
        if title:
            # Titles are stored with underscores, convert to spaces then normalize
            title_with_spaces = title.replace('_', ' ')
            norm_title = _norm(title_with_spaces)
            emb_ix = row.get('emb_ix', -1)
            chunk_id = row.get('chunk_id', -1)
            if emb_ix >= 0 and chunk_id >= 0:
                title_index[norm_title].append((emb_ix, chunk_id))
    
    title_index_dict = dict(title_index)
    
    # Save to cache if cache_file is set
    if cache_file:
        print(f"    Saving local index to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(title_index_dict, f)
            print(f"    ✓ Cached index saved")
        except Exception as e:
            print(f"    Warning: Failed to save cache: {e}")
    
    return title_index_dict


def compute_similarities_v02(
    grok_v02_emb_path: str,
    grok_v02_parquet_path: str,
    wiki_emb_path: str,
    wiki_parquet_path: str,
    grok_v01_emb_path: str = None,
    grok_v01_parquet_path: str = None,
    output_dir: str = "./results/similarities_v02",
    batch_articles: int = 100,
    use_gpu: bool = True,
    cache_dir: str = None,
    local_temp_dir: str = "/tmp/similarities_v02",
    test_mode: bool = False,
    debug: bool = False
):
    """
    Compute similarity between Grokipedia v0.2 and Wikipedia/v0.1.
    
    Args:
        grok_v02_emb_path: Path to v0.2 embeddings (local or gs://)
        grok_v02_parquet_path: Path to v0.2 parquet with metadata
        wiki_emb_path: Path to Wikipedia embeddings (local or gs://)
        wiki_parquet_path: Path to Wikipedia parquet with metadata
        grok_v01_emb_path: Optional path to v0.1 embeddings (local or gs://)
        grok_v01_parquet_path: Optional path to v0.1 parquet with metadata
        output_dir: Output directory for results
        batch_articles: Number of articles to process before saving
        use_gpu: Use GPU for similarity computation
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_temp_dir, exist_ok=True)
    
    if cache_dir is None:
        cache_dir = f"{local_temp_dir}/cache"
    
    # Check GPU availability
    if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"✓ GPU acceleration available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ GPU not available, using CPU")
        use_gpu = False
    
    # Load v0.2 embeddings (local, load directly)
    print("\nLoading Grokipedia v0.2 embeddings...")
    print(f"Loading from {grok_v02_emb_path}...")
    grok_v02_embs = load_embeddings_sharded(grok_v02_emb_path, use_mmap=True, cache_dir=None)
    print(f"  v0.2 embeddings shape: {grok_v02_embs.shape}")
    
    # Check if embeddings are all identical (corruption check)
    if grok_v02_embs.shape[0] > 1:
        first_emb = grok_v02_embs[0]
        # Sample check: compare first 100 embeddings
        sample_size = min(100, grok_v02_embs.shape[0])
        sample_embs = grok_v02_embs[1:sample_size]
        all_same = all(np.array_equal(emb, first_emb) for emb in sample_embs)
        if all_same:
            print(f"  ⚠️  WARNING: First {sample_size} embeddings appear to be IDENTICAL!")
            print(f"  This suggests the embedding file may be corrupted or incorrectly generated.")
            print(f"  Sample embedding (first 10 dims): {first_emb[:10]}")
            # Check a wider range
            if grok_v02_embs.shape[0] > 1000:
                mid_emb = grok_v02_embs[500]
                end_emb = grok_v02_embs[1000]
                print(f"  Checking embeddings at indices 500 and 1000...")
                print(f"    [500] (first 5 dims): {mid_emb[:5]}")
                print(f"    [1000] (first 5 dims): {end_emb[:5]}")
                print(f"    [0] == [500]? {np.array_equal(first_emb, mid_emb)}")
                print(f"    [0] == [1000]? {np.array_equal(first_emb, end_emb)}")
        else:
            # Check how many are unique
            unique_count = len(set(tuple(emb) for emb in grok_v02_embs[:sample_size]))
            print(f"  ✓ Embedding diversity check: {unique_count}/{sample_size} unique in first {sample_size} embeddings")
            if debug:
                print(f"  Sample embeddings (first 3, first 5 dims):")
                for i in range(min(3, grok_v02_embs.shape[0])):
                    print(f"    [{i}]: {grok_v02_embs[i, :5]}")
                # Check some random indices
                if grok_v02_embs.shape[0] > 1000:
                    random_indices = [100, 500, 1000, 5000, min(10000, grok_v02_embs.shape[0]-1)]
                    print(f"  Random embeddings (indices {random_indices}, first 5 dims):")
                    for idx in random_indices:
                        if idx < grok_v02_embs.shape[0]:
                            print(f"    [{idx}]: {grok_v02_embs[idx, :5]}")
                            print(f"    [{idx}] == [0]? {np.array_equal(grok_v02_embs[idx], first_emb)}")
    
    # Build title indices
    print("\nBuilding title indices...")
    
    # v0.2: local file, simple approach
    print(f"Building Grokipedia v0.2 index from {grok_v02_parquet_path}...")
    grok_v02_index = build_title_index_from_parquet(grok_v02_parquet_path, cache_dir=None, use_streaming=False)
    print(f"  v0.2: {len(grok_v02_index):,} articles")
    
    # Wikipedia: GCS, use streaming approach
    print(f"Building Wikipedia index from {wiki_parquet_path}...")
    wiki_index_sharded, wiki_parquet_to_emb, wiki_lookup_map = build_title_index_streaming_gcs(wiki_parquet_path, cache_dir=cache_dir)
    print(f"  Wikipedia: {len(wiki_index_sharded):,} articles")
    
    # Get unique embedding shard files for Wikipedia
    wiki_emb_files = sorted(set(wiki_parquet_to_emb.values()))
    print(f"  Wikipedia shards: {len(wiki_emb_files)}")
    
    # Extract GCS bucket and prefix from wiki_emb_path pattern
    if wiki_emb_path.startswith('gs://'):
        gcs_path_clean = wiki_emb_path.replace("gs://", "")
        parts = gcs_path_clean.split("/", 1)
        wiki_gcs_bucket = parts[0]
        # Extract prefix by removing the filename pattern
        if len(parts) > 1:
            path_parts = parts[1].split("/")
            # Remove the last part (filename pattern) to get prefix
            wiki_gcs_prefix = "/".join(path_parts[:-1]) if len(path_parts) > 1 else ""
        else:
            wiki_gcs_prefix = ""
    else:
        wiki_gcs_bucket = None
        wiki_gcs_prefix = None
    
    # Pre-download Wikipedia embedding shards
    wiki_emb_cache = None
    if wiki_gcs_bucket:
        wiki_emb_dir = f"{local_temp_dir}/wiki_embs"
        os.makedirs(wiki_emb_dir, exist_ok=True)
        
        # Filter existing files
        wiki_emb_files_to_preload = []
        for emb_file in wiki_emb_files:
            filename = os.path.basename(emb_file)
            local_path = os.path.join(wiki_emb_dir, filename)
            if not os.path.exists(local_path):
                wiki_emb_files_to_preload.append(emb_file)
        
        wiki_emb_cache = ShardedEmbeddingCache(
            wiki_gcs_bucket, wiki_gcs_prefix, wiki_emb_dir,
            preload_all=True, emb_files=wiki_emb_files_to_preload
        )
    
    # Grokipedia v0.1 (optional): GCS, use streaming approach
    grok_v01_index_sharded = None
    grok_v01_parquet_to_emb = None
    grok_v01_lookup_map = None
    grok_v01_emb_cache = None
    
    if grok_v01_parquet_path:
        print(f"Building Grokipedia v0.1 index from {grok_v01_parquet_path}...")
        grok_v01_index_sharded, grok_v01_parquet_to_emb, grok_v01_lookup_map = build_title_index_streaming_gcs(grok_v01_parquet_path, cache_dir=cache_dir)
        print(f"  v0.1: {len(grok_v01_index_sharded):,} articles")
        
        grok_v01_emb_files = sorted(set(grok_v01_parquet_to_emb.values()))
        print(f"  v0.1 shards: {len(grok_v01_emb_files)}")
        
        # Extract GCS info for v0.1
        if grok_v01_emb_path and grok_v01_emb_path.startswith('gs://'):
            gcs_path_clean = grok_v01_emb_path.replace("gs://", "")
            parts = gcs_path_clean.split("/", 1)
            grok_v01_gcs_bucket = parts[0]
            # Extract prefix by removing the filename pattern
            if len(parts) > 1:
                path_parts = parts[1].split("/")
                grok_v01_gcs_prefix = "/".join(path_parts[:-1]) if len(path_parts) > 1 else ""
            else:
                grok_v01_gcs_prefix = ""
            
            grok_v01_emb_dir = f"{local_temp_dir}/grok_v01_embs"
            os.makedirs(grok_v01_emb_dir, exist_ok=True)
            
            grok_v01_emb_files_to_preload = []
            for emb_file in grok_v01_emb_files:
                filename = os.path.basename(emb_file)
                local_path = os.path.join(grok_v01_emb_dir, filename)
                if not os.path.exists(local_path):
                    grok_v01_emb_files_to_preload.append(emb_file)
            
            grok_v01_emb_cache = ShardedEmbeddingCache(
                grok_v01_gcs_bucket, grok_v01_gcs_prefix, grok_v01_emb_dir,
                preload_all=True, emb_files=grok_v01_emb_files_to_preload
            )
    
    # Find common titles
    grok_v02_titles = set(grok_v02_index.keys())
    wiki_titles = set(wiki_index_sharded.keys())
    common_v02_wiki = sorted(grok_v02_titles & wiki_titles)
    print(f"\nFound {len(common_v02_wiki):,} articles in both v0.2 and Wikipedia")
    
    if grok_v01_index_sharded:
        grok_v01_titles = set(grok_v01_index_sharded.keys())
        common_v02_v01 = sorted(grok_v02_titles & grok_v01_titles)
        print(f"Found {len(common_v02_v01):,} articles in both v0.2 and v0.1")
    
    # Test mode: limit to first 5 articles
    if test_mode:
        print("\n⚠️  TEST MODE: Processing only first 5 articles")
        common_v02_wiki = common_v02_wiki[:5]
        if grok_v01_index_sharded:
            common_v02_v01 = common_v02_v01[:5]
    
    # Process article-by-article
    stats_rows_v02_wiki = []
    topk_rows_v02_wiki = []
    
    stats_rows_v02_v01 = []
    topk_rows_v02_v01 = []
    
    # Process v0.2 vs Wikipedia
    print(f"\nComputing similarities: Grokipedia v0.2 vs Wikipedia...")
    for idx, title in enumerate(tqdm(common_v02_wiki, desc="v0.2 vs Wikipedia")):
        v02_chunks = grok_v02_index[title]  # List of (emb_ix, chunk_id)
        wiki_chunks = wiki_index_sharded[title]  # List of (parquet_file, emb_ix, chunk_id)
        
        if not v02_chunks or not wiki_chunks:
            if debug:
                print(f"  DEBUG: Skipping {title} - no chunks (v02: {len(v02_chunks)}, wiki: {len(wiki_chunks)})")
            continue
        
        # Get v0.2 embeddings (local, simple)
        v02_emb_list = [grok_v02_embs[emb_ix] for emb_ix, _ in v02_chunks]
        
        if debug and idx < 3:
            emb_indices = [emb_ix for emb_ix, _ in v02_chunks]
            print(f"    V02 chunk indices (first 10): {emb_indices[:10]}")
            print(f"    V02 unique indices: {len(set(emb_indices))}/{len(emb_indices)}")
            print(f"    V02 embedding shapes: {[emb.shape for emb in v02_emb_list[:3]]}")
            # Check if embeddings are all the same
            if len(v02_emb_list) > 1:
                first_emb = v02_emb_list[0]
                # Check if indices are all the same (indexing bug)
                if len(set(emb_indices)) == 1:
                    print(f"    ⚠️  CRITICAL: All chunks use the same embedding index {emb_indices[0]}!")
                # Check if embeddings are all the same (data corruption)
                all_same = all(np.array_equal(emb, first_emb) for emb in v02_emb_list[1:min(5, len(v02_emb_list))])
                print(f"    V02 first 5 embeddings all identical? {all_same}")
                if all_same:
                    print(f"    ⚠️  WARNING: All V02 embeddings appear to be identical!")
                    # Check a few more to be sure
                    if len(v02_emb_list) > 10:
                        all_same_extended = all(np.array_equal(emb, first_emb) for emb in v02_emb_list[5:10])
                        print(f"    V02 embeddings 5-10 also identical? {all_same_extended}")
                else:
                    # Show some differences
                    if len(v02_emb_list) > 1:
                        diff = np.abs(v02_emb_list[0] - v02_emb_list[1])
                        print(f"    V02 embedding difference (first vs second): max={diff.max():.6f}, mean={diff.mean():.6f}")
        
        # Get wiki embeddings (sharded, group by shard first)
        wiki_by_shard = defaultdict(list)
        for chunk_data in wiki_chunks:
            parquet_file, emb_ix, chunk_id = chunk_data
            emb_file = wiki_parquet_to_emb[parquet_file]
            wiki_by_shard[emb_file].append((emb_ix, chunk_id))
        
        wiki_emb_list = []
        for emb_file, chunks in wiki_by_shard.items():
            embs = wiki_emb_cache.get_embeddings(emb_file) if wiki_emb_cache else load_embeddings_sharded(f"gs://{wiki_gcs_bucket}/{wiki_gcs_prefix}/{emb_file}", use_mmap=True, cache_dir=cache_dir)
            for emb_ix, chunk_id in chunks:
                wiki_emb_list.append(embs[emb_ix])
        
        # Stack into arrays
        V02 = np.vstack(v02_emb_list).astype(np.float32)
        W = np.vstack(wiki_emb_list).astype(np.float32)
        
        if debug and idx < 3:
            print(f"\n  DEBUG [{title}]:")
            print(f"    V02 shape: {V02.shape}, W shape: {W.shape}")
            print(f"    V02 sample (first 5 dims): {V02[0, :5]}")
            print(f"    W sample (first 5 dims): {W[0, :5]}")
            print(f"    V02 all zeros? {np.allclose(V02, 0)}")
            print(f"    W all zeros? {np.allclose(W, 0)}")
            print(f"    V02 has NaN? {np.isnan(V02).any()}, has Inf? {np.isinf(V02).any()}")
            print(f"    W has NaN? {np.isnan(W).any()}, has Inf? {np.isinf(W).any()}")
        
        # Normalize both embeddings to ensure cosine similarity
        # Use float64 for norm calculation to avoid overflow with large values
        V02_f64 = V02.astype(np.float64)
        W_f64 = W.astype(np.float64)
        
        # Compute norms in float64 to avoid overflow
        V02_norms = np.linalg.norm(V02_f64, axis=1, keepdims=True).astype(np.float32)
        W_norms = np.linalg.norm(W_f64, axis=1, keepdims=True).astype(np.float32)
        
        # Handle zero/inf norms
        V02_norms = np.where((V02_norms > 1e-8) & np.isfinite(V02_norms), V02_norms, 1.0)
        W_norms = np.where((W_norms > 1e-8) & np.isfinite(W_norms), W_norms, 1.0)
        
        if debug and idx < 3:
            print(f"    V02 norms (float64): min={V02_norms.min():.6f}, max={V02_norms.max():.6f}, mean={V02_norms.mean():.6f}")
            print(f"    W norms: min={W_norms.min():.6f}, max={W_norms.max():.6f}, mean={W_norms.mean():.6f}")
            print(f"    V02 has inf/zero norms: inf={np.isinf(V02_norms).any()}, zero={(V02_norms < 1e-8).any()}")
            print(f"    W has inf/zero norms: inf={np.isinf(W_norms).any()}, zero={(W_norms < 1e-8).any()}")
        
        V02_norm = (V02 / V02_norms).astype(np.float32)
        W_norm = (W / W_norms).astype(np.float32)
        
        if debug and idx < 3:
            V02_norm_check = np.linalg.norm(V02_norm, axis=1)
            W_norm_check = np.linalg.norm(W_norm, axis=1)
            print(f"    V02_norm norms: min={V02_norm_check.min():.6f}, max={V02_norm_check.max():.6f}, mean={V02_norm_check.mean():.6f}")
            print(f"    W_norm norms: min={W_norm_check.min():.6f}, max={W_norm_check.max():.6f}, mean={W_norm_check.mean():.6f}")
        
        # Compute similarity matrix (cosine similarity = dot product of normalized vectors)
        if use_gpu and TORCH_AVAILABLE:
            with torch.no_grad():
                V02_gpu = torch.from_numpy(V02_norm).cuda()
                W_gpu = torch.from_numpy(W_norm).cuda()
                S_gpu = torch.mm(V02_gpu, W_gpu.t())
                
                sim_mean = float(S_gpu.mean().item())
                sim_max = float(S_gpu.max().item())
                best_val, best_ix = torch.max(S_gpu, dim=1)
                best_ix = best_ix.cpu().numpy()
                best_val = best_val.cpu().numpy()
                
                S_cpu = S_gpu.cpu().numpy()
                sim_median = float(np.median(S_cpu))
                sim_p90 = float(np.percentile(S_cpu, 90))
                
                if debug and idx < 3:
                    print(f"    S shape: {S_cpu.shape}")
                    print(f"    S sample values: {S_cpu[0, :5]}")
                    print(f"    S all zeros? {np.allclose(S_cpu, 0)}")
                    print(f"    S has NaN? {np.isnan(S_cpu).any()}, has Inf? {np.isinf(S_cpu).any()}")
                    print(f"    S stats: min={S_cpu.min():.6f}, max={S_cpu.max():.6f}, mean={sim_mean:.6f}, median={sim_median:.6f}")
                
                del V02_gpu, W_gpu, S_gpu
                torch.cuda.empty_cache()
                S = S_cpu
        else:
            S = V02_norm @ W_norm.T
            sim_mean = float(S.mean())
            sim_median = float(np.median(S))
            sim_max = float(S.max())
            sim_p90 = float(np.percentile(S, 90))
            best_ix = S.argmax(axis=1)
            best_val = S.max(axis=1)
            
            if debug and idx < 3:
                print(f"    S shape: {S.shape}")
                print(f"    S sample values: {S[0, :5]}")
                print(f"    S all zeros? {np.allclose(S, 0)}")
                print(f"    S has NaN? {np.isnan(S).any()}, has Inf? {np.isinf(S).any()}")
                print(f"    S stats: min={S.min():.6f}, max={S.max():.6f}, mean={sim_mean:.6f}, median={sim_median:.6f}")
        
        stats_rows_v02_wiki.append({
            'title': title,
            'n_v02': len(v02_chunks),
            'n_wiki': len(wiki_chunks),
            'sim_mean': sim_mean,
            'sim_median': sim_median,
            'sim_max': sim_max,
            'sim_p90': sim_p90,
        })
        
        for j, (gi, sv) in enumerate(zip(best_ix, best_val)):
            v02_chunk_id = v02_chunks[j][1]  # (emb_ix, chunk_id)
            wiki_chunk_id = wiki_chunks[gi][2]  # (parquet_file, emb_ix, chunk_id)
            
            topk_rows_v02_wiki.append({
                'title': title,
                'v02_chunk_id': int(v02_chunk_id),
                'wiki_chunk_id': int(wiki_chunk_id),
                'similarity': float(sv),
            })
        
        # Save incrementally
        if len(stats_rows_v02_wiki) > 0 and len(stats_rows_v02_wiki) % batch_articles == 0:
            stats_df = pd.DataFrame(stats_rows_v02_wiki)
            topk_df = pd.DataFrame(topk_rows_v02_wiki)
            
            stats_path = Path(output_dir) / f'v02_wiki_stats_{len(stats_rows_v02_wiki)}.csv'
            topk_path = Path(output_dir) / f'v02_wiki_top1_{len(topk_rows_v02_wiki)}.csv'
            
            stats_df.to_csv(stats_path, index=False)
            topk_df.to_csv(topk_path, index=False)
    
    # Process v0.2 vs v0.1 (if v0.1 data provided)
    if grok_v01_index_sharded and grok_v01_emb_cache:
        print(f"\nComputing similarities: Grokipedia v0.2 vs v0.1...")
        for idx, title in enumerate(tqdm(common_v02_v01, desc="v0.2 vs v0.1")):
            v02_chunks = grok_v02_index[title]  # List of (emb_ix, chunk_id)
            v01_chunks = grok_v01_index_sharded[title]  # List of (parquet_file, emb_ix, chunk_id)
            
            if not v02_chunks or not v01_chunks:
                continue
            
            # Get v0.2 embeddings (local, simple)
            v02_emb_list = [grok_v02_embs[emb_ix] for emb_ix, _ in v02_chunks]
            
            # Get v0.1 embeddings (sharded, group by shard first)
            v01_by_shard = defaultdict(list)
            for chunk_data in v01_chunks:
                parquet_file, emb_ix, chunk_id = chunk_data
                emb_file = grok_v01_parquet_to_emb[parquet_file]
                v01_by_shard[emb_file].append((emb_ix, chunk_id))
            
            v01_emb_list = []
            for emb_file, chunks in v01_by_shard.items():
                embs = grok_v01_emb_cache.get_embeddings(emb_file)
                for emb_ix, chunk_id in chunks:
                    v01_emb_list.append(embs[emb_ix])
            
            V02 = np.vstack(v02_emb_list).astype(np.float32)
            V01 = np.vstack(v01_emb_list).astype(np.float32)
            
            if debug and idx < 3:
                print(f"\n  DEBUG v0.1 [{title}]:")
                print(f"    V02 shape: {V02.shape}, V01 shape: {V01.shape}")
                print(f"    V02 sample (first 5 dims): {V02[0, :5]}")
                print(f"    V01 sample (first 5 dims): {V01[0, :5]}")
                print(f"    V02 all zeros? {np.allclose(V02, 0)}")
                print(f"    V01 all zeros? {np.allclose(V01, 0)}")
            
            # Normalize both embeddings to ensure cosine similarity
            # Use float64 for norm calculation to avoid overflow with large values
            V02_f64 = V02.astype(np.float64)
            V01_f64 = V01.astype(np.float64)
            
            # Compute norms in float64 to avoid overflow
            V02_norms = np.linalg.norm(V02_f64, axis=1, keepdims=True).astype(np.float32)
            V01_norms = np.linalg.norm(V01_f64, axis=1, keepdims=True).astype(np.float32)
            
            # Handle zero/inf norms
            V02_norms = np.where((V02_norms > 1e-8) & np.isfinite(V02_norms), V02_norms, 1.0)
            V01_norms = np.where((V01_norms > 1e-8) & np.isfinite(V01_norms), V01_norms, 1.0)
            
            if debug and idx < 3:
                print(f"    V02 norms (float64): min={V02_norms.min():.6f}, max={V02_norms.max():.6f}, mean={V02_norms.mean():.6f}")
                print(f"    V01 norms: min={V01_norms.min():.6f}, max={V01_norms.max():.6f}, mean={V01_norms.mean():.6f}")
            
            V02_norm = (V02 / V02_norms).astype(np.float32)
            V01_norm = (V01 / V01_norms).astype(np.float32)
            
            # Compute similarity (cosine similarity = dot product of normalized vectors)
            if use_gpu and TORCH_AVAILABLE:
                with torch.no_grad():
                    V02_gpu = torch.from_numpy(V02_norm).cuda()
                    V01_gpu = torch.from_numpy(V01_norm).cuda()
                    S_gpu = torch.mm(V02_gpu, V01_gpu.t())
                    
                    sim_mean = float(S_gpu.mean().item())
                    sim_max = float(S_gpu.max().item())
                    best_val, best_ix = torch.max(S_gpu, dim=1)
                    best_ix = best_ix.cpu().numpy()
                    best_val = best_val.cpu().numpy()
                    
                    S_cpu = S_gpu.cpu().numpy()
                    sim_median = float(np.median(S_cpu))
                    sim_p90 = float(np.percentile(S_cpu, 90))
                    
                    if debug and idx < 3:
                        print(f"    S shape: {S_cpu.shape}, S stats: min={S_cpu.min():.6f}, max={S_cpu.max():.6f}, mean={sim_mean:.6f}")
                        print(f"    S all zeros? {np.allclose(S_cpu, 0)}")
                    
                    del V02_gpu, V01_gpu, S_gpu
                    torch.cuda.empty_cache()
                    S = S_cpu
            else:
                S = V02_norm @ V01_norm.T
                sim_mean = float(S.mean())
                sim_median = float(np.median(S))
                sim_max = float(S.max())
                sim_p90 = float(np.percentile(S, 90))
                best_ix = S.argmax(axis=1)
                best_val = S.max(axis=1)
                
                if debug and idx < 3:
                    print(f"    S shape: {S.shape}, S stats: min={S.min():.6f}, max={S.max():.6f}, mean={sim_mean:.6f}")
                    print(f"    S all zeros? {np.allclose(S, 0)}")
            
            stats_rows_v02_v01.append({
                'title': title,
                'n_v02': len(v02_chunks),
                'n_v01': len(v01_chunks),
                'sim_mean': sim_mean,
                'sim_median': sim_median,
                'sim_max': sim_max,
                'sim_p90': sim_p90,
            })
            
            for j, (gi, sv) in enumerate(zip(best_ix, best_val)):
                v02_chunk_id = v02_chunks[j][1]  # (emb_ix, chunk_id)
                v01_chunk_id = v01_chunks[gi][2]  # (parquet_file, emb_ix, chunk_id)
                
                topk_rows_v02_v01.append({
                    'title': title,
                    'v02_chunk_id': int(v02_chunk_id),
                    'v01_chunk_id': int(v01_chunk_id),
                    'similarity': float(sv),
                })
            
            # Save incrementally
            if len(stats_rows_v02_v01) > 0 and len(stats_rows_v02_v01) % batch_articles == 0:
                stats_df = pd.DataFrame(stats_rows_v02_v01)
                topk_df = pd.DataFrame(topk_rows_v02_v01)
                
                stats_path = Path(output_dir) / f'v02_v01_stats_{len(stats_rows_v02_v01)}.csv'
                topk_path = Path(output_dir) / f'v02_v01_top1_{len(topk_rows_v02_v01)}.csv'
                
                stats_df.to_csv(stats_path, index=False)
                topk_df.to_csv(topk_path, index=False)
    
    # Save final results
    print("\nSaving final results...")
    
    if stats_rows_v02_wiki:
        stats_df = pd.DataFrame(stats_rows_v02_wiki)
        topk_df = pd.DataFrame(topk_rows_v02_wiki)
        stats_df.to_csv(Path(output_dir) / 'v02_wiki_stats_final.csv', index=False)
        topk_df.to_csv(Path(output_dir) / 'v02_wiki_top1_final.csv', index=False)
        print(f"  Saved v0.2 vs Wikipedia: {len(stats_df):,} articles")
    
    if stats_rows_v02_v01:
        stats_df = pd.DataFrame(stats_rows_v02_v01)
        topk_df = pd.DataFrame(topk_rows_v02_v01)
        stats_df.to_csv(Path(output_dir) / 'v02_v01_stats_final.csv', index=False)
        topk_df.to_csv(Path(output_dir) / 'v02_v01_top1_final.csv', index=False)
        print(f"  Saved v0.2 vs v0.1: {len(stats_df):,} articles")
    
    print("\n✓ Done!")


def main():
    parser = argparse.ArgumentParser(description='Compute similarities between Grokipedia v0.2 and Wikipedia/v0.1')
    parser.add_argument('--grok-v02-emb', type=str, required=True,
                        help='Path to Grokipedia v0.2 embeddings (local or gs://). Supports * wildcard for shards (e.g., "gs://bucket/path/*_embeddings.npy")')
    parser.add_argument('--grok-v02-parquet', type=str, required=True,
                        help='Path to Grokipedia v0.2 parquet with metadata. Supports * wildcard for shards')
    parser.add_argument('--wiki-emb', type=str, required=True,
                        help='Path to Wikipedia embeddings (local or gs://). Supports * wildcard for shards')
    parser.add_argument('--wiki-parquet', type=str, required=True,
                        help='Path to Wikipedia parquet with metadata. Supports * wildcard for shards')
    parser.add_argument('--grok-v01-emb', type=str, default=None,
                        help='Optional: Path to Grokipedia v0.1 embeddings (local or gs://). Supports * wildcard for shards')
    parser.add_argument('--grok-v01-parquet', type=str, default=None,
                        help='Optional: Path to Grokipedia v0.1 parquet with metadata. Supports * wildcard for shards')
    parser.add_argument('--output-dir', type=str, default='./results/similarities_v02',
                        help='Output directory for results')
    parser.add_argument('--batch-articles', type=int, default=100,
                        help='Number of articles to process before saving')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory for caching downloaded files (default: local_temp_dir/cache)')
    parser.add_argument('--local-temp-dir', type=str, default='/tmp/similarities_v02',
                        help='Local temp directory for embeddings and cache')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU for similarity computation (default: True)')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Disable GPU acceleration')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: process only first 5 articles')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: print detailed information for first 3 articles')
    args = parser.parse_args()
    
    compute_similarities_v02(
        args.grok_v02_emb,
        args.grok_v02_parquet,
        args.wiki_emb,
        args.wiki_parquet,
        args.grok_v01_emb,
        args.grok_v01_parquet,
        args.output_dir,
        args.batch_articles,
        args.use_gpu,
        args.cache_dir,
        args.local_temp_dir,
        args.test_mode,
        args.debug
    )


if __name__ == '__main__':
    main()

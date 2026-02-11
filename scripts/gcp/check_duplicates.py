#!/usr/bin/env python3
"""
Check for duplicate rows in embeddings file.
Uses memory-efficient approach to avoid loading everything into RAM.
"""

import argparse
from pathlib import Path

import numpy as np


def check_duplicates(embeddings_path: str, sample_size: int = 100000):
    """
    Check for duplicate rows in embeddings file.
    
    Args:
        embeddings_path: Path to embeddings .npy file
        sample_size: Number of rows to check at a time (for memory efficiency)
    """
    print(f"Loading embeddings from: {embeddings_path}")
    
    try:
        embs = np.load(embeddings_path, mmap_mode='r', allow_pickle=False)
        print(f"✓ Loaded embeddings: shape {embs.shape}")
        total_rows = embs.shape[0]
        embedding_dim = embs.shape[1]
        
        print(f"\nChecking for duplicate rows...")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Sample size per batch: {sample_size:,}")
        
        # Check for exact duplicates by comparing rows in batches
        # Use a more accurate approach: check actual array equality, not just hashes
        print(f"\nChecking for exact duplicate rows...")
        print(f"  (This may take a while for large files)")
        
        # Check consecutive duplicates first (most common issue)
        print(f"\nChecking consecutive duplicates...")
        consecutive_dups = []
        check_every = max(1, total_rows // 100000)  # Sample to avoid taking forever
        for i in range(0, total_rows - 1, check_every):
            if np.array_equal(embs[i], embs[i+1]):
                consecutive_dups.append(i)
                if len(consecutive_dups) <= 10:
                    print(f"  ⚠ Consecutive duplicate at rows {i:,} and {i+1:,}")
        
        if consecutive_dups:
            print(f"\n✗ Found {len(consecutive_dups)} consecutive duplicate pairs (sampled every {check_every} rows)")
            print(f"  Estimated total if pattern continues: ~{len(consecutive_dups) * check_every:,}")
        else:
            print(f"✓ No consecutive duplicates found in sample")
        
        # Check for non-consecutive duplicates (sample-based)
        print(f"\nChecking for non-consecutive duplicates (sampling approach)...")
        sample_indices = np.random.choice(total_rows, size=min(10000, total_rows), replace=False)
        sample_rows = embs[sample_indices]
        
        # Compare sample rows to each other
        unique_in_sample = len(np.unique(sample_rows.view(np.void), axis=0))
        print(f"  Sampled {len(sample_indices):,} rows")
        print(f"  Unique rows in sample: {unique_in_sample:,}")
        if unique_in_sample < len(sample_indices):
            print(f"  ⚠ Found {len(sample_indices) - unique_in_sample} duplicates in sample")
        else:
            print(f"  ✓ No duplicates found in random sample")
        
        # Check a few specific suspicious indices
        if consecutive_dups:
            print(f"\nVerifying first few consecutive duplicates...")
            for i in consecutive_dups[:5]:
                row1 = embs[i]
                row2 = embs[i+1]
                if np.array_equal(row1, row2):
                    diff = np.abs(row1 - row2).max()
                    print(f"  Row {i:,} == Row {i+1:,}: ✓ Confirmed duplicate (max diff: {diff})")
                else:
                    print(f"  Row {i:,} != Row {i+1:,}: ✗ False positive (max diff: {np.abs(row1 - row2).max()})")
        
        del embs
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Check for duplicate rows in embeddings file')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to embeddings .npy file')
    parser.add_argument('--sample-size', type=int, default=100000,
                        help='Number of rows to process at a time (default: 100000)')
    args = parser.parse_args()
    
    check_duplicates(args.embeddings, args.sample_size)


if __name__ == '__main__':
    main()

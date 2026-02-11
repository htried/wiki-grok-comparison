#!/usr/bin/env python3
"""
Verify that embeddings file aligns correctly with parquet metadata files.

Checks:
1. Number of rows match between embeddings and parquet files
2. emb_ix values are sequential and match array indices
3. Embedding dimensions are consistent
4. Sample verification of a few embeddings
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def verify_embeddings(
    embeddings_path: str,
    parquet_with_ix_path: str,
    parquet_original_path: str = None,
    sample_check: int = 10
):
    """
    Verify embeddings alignment with parquet files.
    
    Args:
        embeddings_path: Path to .npy embeddings file
        parquet_with_ix_path: Path to parquet file with emb_ix column
        parquet_original_path: Optional path to original parquet (without emb_ix)
        sample_check: Number of random samples to verify
    """
    print("=" * 70)
    print("Embedding Verification")
    print("=" * 70)
    
    # Load embeddings (memory-mapped to avoid loading into RAM)
    print(f"\n1. Loading embeddings from: {embeddings_path}")
    try:
        embs = np.load(embeddings_path, mmap_mode='r')
        print(f"   ✓ Loaded embeddings: shape {embs.shape}")
        print(f"   ✓ Embedding dimension: {embs.shape[1]}")
        print(f"   ✓ Number of embeddings: {embs.shape[0]:,}")
        print(f"   ✓ Data type: {embs.dtype}")
    except Exception as e:
        print(f"   ✗ Failed to load embeddings: {e}")
        return False
    
    # Load parquet with emb_ix
    print(f"\n2. Loading parquet with emb_ix from: {parquet_with_ix_path}")
    try:
        df_with_ix = pd.read_parquet(parquet_with_ix_path)
        print(f"   ✓ Loaded parquet: {len(df_with_ix):,} rows")
        print(f"   ✓ Columns: {list(df_with_ix.columns)}")
        
        if 'emb_ix' not in df_with_ix.columns:
            print(f"   ✗ ERROR: 'emb_ix' column not found in parquet file!")
            return False
        
        print(f"   ✓ Found 'emb_ix' column")
    except Exception as e:
        print(f"   ✗ Failed to load parquet: {e}")
        return False
    
    # Check row count match
    print(f"\n3. Checking row count alignment...")
    n_embs = embs.shape[0]
    n_parquet = len(df_with_ix)
    
    if n_embs == n_parquet:
        print(f"   ✓ Row counts match: {n_embs:,}")
    else:
        print(f"   ✗ ERROR: Row count mismatch!")
        print(f"      Embeddings: {n_embs:,}")
        print(f"      Parquet: {n_parquet:,}")
        print(f"      Difference: {abs(n_embs - n_parquet):,}")
        return False
    
    # Check emb_ix values
    print(f"\n4. Checking emb_ix values...")
    emb_ix_values = df_with_ix['emb_ix'].values
    
    # Check if sequential starting from 0
    expected_emb_ix = np.arange(n_parquet)
    if np.array_equal(emb_ix_values, expected_emb_ix):
        print(f"   ✓ emb_ix is sequential (0 to {n_parquet-1})")
    else:
        # Check for duplicates or gaps
        unique_emb_ix = np.unique(emb_ix_values)
        if len(unique_emb_ix) != n_parquet:
            print(f"   ✗ ERROR: emb_ix has duplicates!")
            print(f"      Expected {n_parquet} unique values, got {len(unique_emb_ix)}")
            return False
        
        if emb_ix_values.min() != 0:
            print(f"   ✗ ERROR: emb_ix does not start at 0 (min: {emb_ix_values.min()})")
            return False
        
        if emb_ix_values.max() != n_parquet - 1:
            print(f"   ✗ ERROR: emb_ix does not end at {n_parquet-1} (max: {emb_ix_values.max()})")
            return False
        
        # Check if sorted
        if not np.all(emb_ix_values == np.sort(emb_ix_values)):
            print(f"   ⚠ WARNING: emb_ix is not sorted, but values are valid")
        else:
            print(f"   ✓ emb_ix values are valid (0 to {n_parquet-1})")
    
    # Check original parquet if provided
    if parquet_original_path:
        print(f"\n5. Checking original parquet: {parquet_original_path}")
        try:
            df_original = pd.read_parquet(parquet_original_path)
            n_original = len(df_original)
            
            if n_original == n_parquet:
                print(f"   ✓ Row counts match: {n_original:,}")
            else:
                print(f"   ⚠ WARNING: Row count mismatch with original parquet")
                print(f"      Original: {n_original:,}")
                print(f"      With emb_ix: {n_parquet:,}")
                print(f"      Difference: {abs(n_original - n_parquet):,}")
            
            # Check that original parquet has same columns (minus emb_ix)
            original_cols = set(df_original.columns)
            with_ix_cols = set(df_with_ix.columns) - {'emb_ix'}
            
            if original_cols == with_ix_cols:
                print(f"   ✓ Column sets match (excluding emb_ix)")
            else:
                print(f"   ⚠ WARNING: Column sets differ")
                print(f"      Original: {sorted(original_cols)}")
                print(f"      With emb_ix (excluding emb_ix): {sorted(with_ix_cols)}")
        except Exception as e:
            print(f"   ⚠ Could not load original parquet: {e}")
    
    # Sample verification: check a few random indices
    print(f"\n6. Sample verification (checking {sample_check} random indices)...")
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(n_parquet, size=min(sample_check, n_parquet), replace=False)
    
    all_samples_valid = True
    for idx in sorted(sample_indices):
        emb_ix = df_with_ix.iloc[idx]['emb_ix']
        
        # Verify emb_ix matches the row index
        if emb_ix != idx:
            print(f"   ✗ Row {idx}: emb_ix={emb_ix} (expected {idx})")
            all_samples_valid = False
        else:
            # Check that we can access the embedding
            try:
                emb = embs[emb_ix]
                if emb.shape[0] != embs.shape[1]:
                    print(f"   ✗ Row {idx}: embedding shape mismatch")
                    all_samples_valid = False
                elif np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                    print(f"   ✗ Row {idx}: embedding contains NaN or Inf")
                    all_samples_valid = False
                else:
                    print(f"   ✓ Row {idx}: emb_ix={emb_ix}, embedding valid (norm={np.linalg.norm(emb):.4f})")
            except Exception as e:
                print(f"   ✗ Row {idx}: Failed to access embedding: {e}")
                all_samples_valid = False
    
    if all_samples_valid:
        print(f"   ✓ All sample checks passed")
    
    # Summary
    print(f"\n" + "=" * 70)
    if all_samples_valid and n_embs == n_parquet:
        print("✓ VERIFICATION PASSED: Embeddings align correctly with parquet files")
        return True
    else:
        print("✗ VERIFICATION FAILED: Issues found (see above)")
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify embeddings alignment with parquet files')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to embeddings .npy file')
    parser.add_argument('--parquet-with-ix', type=str, required=True,
                        help='Path to parquet file with emb_ix column')
    parser.add_argument('--parquet-original', type=str, default=None,
                        help='Optional: Path to original parquet file (without emb_ix)')
    parser.add_argument('--sample-check', type=int, default=10,
                        help='Number of random samples to verify (default: 10)')
    args = parser.parse_args()
    
    success = verify_embeddings(
        args.embeddings,
        args.parquet_with_ix,
        args.parquet_original,
        args.sample_check
    )
    
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Check for overlaps or duplicates between chunk files.
"""

import argparse
from pathlib import Path

import numpy as np


def check_chunk_overlaps(chunks_dir: str, max_index: int = 5845154):
    """
    Check if chunks have overlaps or duplicates.
    """
    chunks_dir = Path(chunks_dir)
    
    # Get all regular chunks (not final chunks)
    chunk_files = sorted([f for f in chunks_dir.glob('chunk_*.npy') 
                         if not f.name.startswith('chunk_final_')])
    
    print(f"Checking {len(chunk_files)} chunk files...")
    
    # Helper to extract index
    def chunk_index(path: Path) -> int:
        name = path.name
        try:
            if name.startswith('chunk_'):
                return int(name.replace('chunk_', '').replace('.npy', ''))
        except ValueError:
            return float('inf')
        return float('inf')
    
    # Filter chunks with index < max_index
    chunk_files = [f for f in chunk_files if chunk_index(f) < max_index]
    chunk_files = sorted(chunk_files, key=chunk_index)
    
    print(f"Processing {len(chunk_files)} chunks (index < {max_index:,})...\n")
    
    total_rows = 0
    overlaps = []
    chunk_info = []
    
    for i, chunk_file in enumerate(chunk_files):
        arr = np.load(chunk_file, mmap_mode='r', allow_pickle=False)
        rows = arr.shape[0]
        idx = chunk_index(chunk_file)
        
        chunk_info.append({
            'file': chunk_file.name,
            'index': idx,
            'rows': rows,
            'end_index': idx + rows
        })
        
        # Check for consecutive duplicates within this chunk
        if rows > 1:
            consecutive_dups_in_chunk = []
            for j in range(rows - 1):
                if np.array_equal(arr[j], arr[j+1]):
                    consecutive_dups_in_chunk.append(j)
            
            if consecutive_dups_in_chunk:
                print(f"  ⚠ {chunk_file.name}: {len(consecutive_dups_in_chunk)} consecutive duplicates within chunk")
                if len(consecutive_dups_in_chunk) <= 5:
                    print(f"     At positions: {consecutive_dups_in_chunk}")
        
        # Check overlap with previous chunk
        if i > 0:
            prev_info = chunk_info[i-1]
            prev_end = prev_info['end_index']
            current_start = idx
            
            if current_start < prev_end:
                overlap_size = prev_end - current_start
                print(f"  ⚠ OVERLAP: {chunk_file.name} starts at {current_start:,} but previous chunk ends at {prev_end:,}")
                print(f"     Overlap size: {overlap_size:,} rows")
                overlaps.append((prev_info['file'], chunk_file.name, overlap_size))
            elif current_start == prev_end:
                print(f"  ✓ {chunk_file.name}: No gap, no overlap (starts at {current_start:,})")
            else:
                gap = current_start - prev_end
                print(f"  ⚠ GAP: {chunk_file.name} starts at {current_start:,}, gap of {gap:,} rows from previous")
        
        total_rows += rows
        del arr
    
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total chunks: {len(chunk_files)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Overlaps found: {len(overlaps)}")
    
    if overlaps:
        print(f"\n  Overlapping chunks:")
        for prev, curr, size in overlaps:
            print(f"    {prev} overlaps with {curr} by {size:,} rows")
    
    # Check if total matches expected
    expected_rows = max_index
    if total_rows == expected_rows:
        print(f"\n✓ Total rows ({total_rows:,}) matches expected ({expected_rows:,})")
    else:
        diff = total_rows - expected_rows
        print(f"\n⚠ Total rows ({total_rows:,}) differs from expected ({expected_rows:,}) by {diff:,}")


def main():
    parser = argparse.ArgumentParser(description='Check for overlaps between chunk files')
    parser.add_argument('--chunks-dir', type=str,
                        default='/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
                        help='Directory containing chunk files')
    parser.add_argument('--max-index', type=int, default=5845154,
                        help='Maximum index to include (default: 5845154, matches parquet)')
    args = parser.parse_args()
    
    check_chunk_overlaps(args.chunks_dir, args.max_index)


if __name__ == '__main__':
    main()

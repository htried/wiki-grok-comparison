#!/usr/bin/env python3
"""
Reset embedding progress to recalculate from index 0 up to a specific chunk.

This script:
1. Identifies the batch index corresponding to a chunk file
2. Deletes the target chunk and all chunks after it
3. Resets checkpoint index to 0 to recalculate from the beginning
"""

import argparse
import re
from pathlib import Path


def extract_index_from_chunk_name(chunk_name: str) -> int:
    """Extract batch index from chunk filename like 'chunk_0002226688.npy'."""
    match = re.search(r'chunk_(\d+)\.npy', chunk_name)
    if match:
        return int(match.group(1))
    match = re.search(r'chunk_final_(\d+)\.npy', chunk_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract index from chunk name: {chunk_name}")


def reset_to_chunk(
    target_chunk: str,
    checkpoint_dir: str = '/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
    checkpoint_idx_file: str = '/home/hjt36_cornell_edu/corpus_chunks_embeddings_checkpoint_idx.txt',
    dry_run: bool = False
):
    """
    Reset embedding progress to recalculate from index 0 up to a specific chunk.
    
    This will:
    - Delete the target chunk and all chunks after it
    - Reset checkpoint index to 0
    - When you re-run embed_gcp.py, it will recalculate from the beginning up to the target chunk
    
    Args:
        target_chunk: Chunk filename (e.g., 'chunk_0002226688.npy') or just the index number (e.g., '2226688')
        checkpoint_dir: Directory containing chunk files
        checkpoint_idx_file: Path to checkpoint index file
        dry_run: If True, only show what would be deleted without actually deleting
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_idx_file = Path(checkpoint_idx_file)
    
    # Extract target index
    if target_chunk.isdigit():
        target_index = int(target_chunk)
        target_chunk_name = f'chunk_{target_index:010d}.npy'
    else:
        target_index = extract_index_from_chunk_name(target_chunk)
        target_chunk_name = Path(target_chunk).name
    
    print(f"Target chunk: {target_chunk_name}")
    print(f"Target index: {target_index:,}")
    
    # Get all chunk files
    all_chunks = sorted(checkpoint_dir.glob('chunk_*.npy'))
    
    if not all_chunks:
        print("No chunk files found!")
        return
    
    print(f"\nFound {len(all_chunks)} chunk files")
    
    # Find chunks to delete (target chunk and all after it)
    chunks_to_delete = []
    target_chunk_found = False
    
    for chunk_file in all_chunks:
        chunk_index = extract_index_from_chunk_name(chunk_file.name)
        
        if chunk_index == target_index:
            target_chunk_found = True
            chunks_to_delete.append((chunk_file, chunk_index))
            print(f"  ✓ Found target chunk: {chunk_file.name} (will be deleted and recalculated)")
        elif chunk_index > target_index:
            chunks_to_delete.append((chunk_file, chunk_index))
    
    if not target_chunk_found:
        print(f"  ⚠ Warning: Target chunk {target_chunk_name} not found!")
        print(f"  Available chunks range from {extract_index_from_chunk_name(all_chunks[0].name):,} to {extract_index_from_chunk_name(all_chunks[-1].name):,}")
        return
    
    if not chunks_to_delete:
        print("\n✓ No chunks to delete - target chunk is the last one")
        return
    
    print(f"\nChunks to delete ({len(chunks_to_delete)}):")
    for chunk_file, idx in chunks_to_delete[:10]:  # Show first 10
        print(f"  - {chunk_file.name} (index {idx:,})")
    if len(chunks_to_delete) > 10:
        print(f"  ... and {len(chunks_to_delete) - 10} more")
    
    # Delete chunks
    if not dry_run:
        print(f"\nDeleting {len(chunks_to_delete)} chunks...")
        for chunk_file, idx in chunks_to_delete:
            chunk_file.unlink()
            print(f"  ✓ Deleted {chunk_file.name}")
        print(f"\n✓ Deleted {len(chunks_to_delete)} chunks")
    else:
        print(f"\n[DRY RUN] Would delete {len(chunks_to_delete)} chunks")
    
    # Update checkpoint index file
    # User wants to recalculate from 0 UP TO the target chunk
    # So we set checkpoint_idx to 0 to start from the beginning
    new_checkpoint_idx = 0
    
    if checkpoint_idx_file.exists():
        old_idx = checkpoint_idx_file.read_text().strip()
        print(f"\nCurrent checkpoint index: {old_idx}")
        
        if not dry_run:
            checkpoint_idx_file.write_text(str(new_checkpoint_idx))
            print(f"✓ Updated checkpoint index to: {new_checkpoint_idx:,}")
            print(f"\nYou can now re-run embed_gcp.py and it will resume from index {new_checkpoint_idx:,}")
        else:
            print(f"[DRY RUN] Would update checkpoint index to: {new_checkpoint_idx:,}")
    else:
        print(f"\nCheckpoint index file not found: {checkpoint_idx_file}")
        if not dry_run:
            checkpoint_idx_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_idx_file.write_text(str(new_checkpoint_idx))
            print(f"✓ Created checkpoint index file with value: {new_checkpoint_idx:,}")
        else:
            print(f"[DRY RUN] Would create checkpoint index file with value: {new_checkpoint_idx:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Reset embedding progress to recalculate from index 0 up to a specific chunk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  python reset_to_chunk.py chunk_0002226688.npy
  python reset_to_chunk.py 2226688
  python reset_to_chunk.py chunk_0002226688.npy --dry-run  # Preview changes
        '''
    )
    parser.add_argument('target_chunk', type=str,
                        help='Target chunk filename (e.g., chunk_0002226688.npy) or just the index number (e.g., 2226688). This chunk and all after it will be deleted.')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
                        help='Directory containing chunk files')
    parser.add_argument('--checkpoint-idx-file', type=str,
                        default='/home/hjt36_cornell_edu/corpus_chunks_embeddings_checkpoint_idx.txt',
                        help='Path to checkpoint index file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually doing it')
    args = parser.parse_args()
    
    reset_to_chunk(
        args.target_chunk,
        args.checkpoint_dir,
        args.checkpoint_idx_file,
        args.dry_run
    )


if __name__ == '__main__':
    main()

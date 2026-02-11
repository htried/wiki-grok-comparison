#!/usr/bin/env python3
"""
Reset checkpoint to 0 to process from the beginning up to the first existing chunk.

This script:
1. Finds the first (lowest index) chunk in the checkpoint directory
2. Sets checkpoint index to 0 (without deleting any chunks)
3. Provides the command to run embed_gcp.py with --final-index set to the first chunk index
   This ensures existing chunks are preserved and not overwritten.
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


def reset_to_first_chunk(
    checkpoint_dir: str = '/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
    checkpoint_idx_file: str = '/home/hjt36_cornell_edu/corpus_chunks_embeddings_checkpoint_idx.txt',
    dry_run: bool = False
):
    """
    Find the first chunk and reset checkpoint to 0 (keeping all existing chunks).
    
    Args:
        checkpoint_dir: Directory containing chunk files
        checkpoint_idx_file: Path to checkpoint index file
        dry_run: If True, only show what would be done without actually doing it
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_idx_file = Path(checkpoint_idx_file)
    
    # Get all chunk files
    all_chunks = sorted(checkpoint_dir.glob('chunk_*.npy'))
    
    if not all_chunks:
        print("No chunk files found in checkpoint directory!")
        print(f"  Directory: {checkpoint_dir}")
        print("\nSetting checkpoint to 0 anyway...")
        first_chunk_index = None
    else:
        # Find the first (lowest index) chunk
        first_chunk = min(all_chunks, key=lambda f: extract_index_from_chunk_name(f.name))
        first_chunk_index = extract_index_from_chunk_name(first_chunk.name)
        
        print(f"Found {len(all_chunks)} chunk files in {checkpoint_dir}")
        print(f"First chunk: {first_chunk.name}")
        print(f"First chunk index: {first_chunk_index:,}")
        
        # Show a few chunks for context
        print(f"\nChunk index range:")
        chunk_indices = [extract_index_from_chunk_name(c.name) for c in all_chunks]
        print(f"  Lowest: {min(chunk_indices):,}")
        print(f"  Highest: {max(chunk_indices):,}")
        if len(all_chunks) <= 10:
            print(f"\nAll chunks:")
            for chunk_file in sorted(all_chunks, key=lambda f: extract_index_from_chunk_name(f.name)):
                idx = extract_index_from_chunk_name(chunk_file.name)
                marker = " <-- FIRST" if idx == first_chunk_index else ""
                print(f"  - {chunk_file.name} (index {idx:,}){marker}")
    
    # Set checkpoint index to 0
    new_checkpoint_idx = 0
    
    if checkpoint_idx_file.exists():
        old_idx = checkpoint_idx_file.read_text().strip()
        print(f"\nCurrent checkpoint index: {old_idx}")
        
        if not dry_run:
            checkpoint_idx_file.write_text(str(new_checkpoint_idx))
            print(f"✓ Updated checkpoint index to: {new_checkpoint_idx:,}")
            if first_chunk_index is not None:
                print(f"\n✓ Ready to process from index 0 up to (but not including) index {first_chunk_index:,}")
                print(f"\nTo run embed_gcp.py with final-index protection:")
                print(f"  python3 scripts/gcp/embed_gcp.py --in_parquet corpus_chunks.parquet --final-index {first_chunk_index:,} [other args]")
                print(f"\nThis will:")
                print(f"  - Process from index 0")
                print(f"  - Stop before index {first_chunk_index:,} (preserving existing chunks)")
            else:
                print(f"\nYou can now re-run embed_gcp.py and it will start from index 0")
                print(f"  (No existing chunks found, so no --final-index needed)")
        else:
            print(f"[DRY RUN] Would update checkpoint index to: {new_checkpoint_idx:,}")
            if first_chunk_index is not None:
                print(f"  First existing chunk is at index {first_chunk_index:,}")
                print(f"  Would recommend running with: --final-index {first_chunk_index:,}")
    else:
        print(f"\nCheckpoint index file not found: {checkpoint_idx_file}")
        if not dry_run:
            checkpoint_idx_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_idx_file.write_text(str(new_checkpoint_idx))
            print(f"✓ Created checkpoint index file with value: {new_checkpoint_idx:,}")
            if first_chunk_index is not None:
                print(f"\nTo run embed_gcp.py with final-index protection:")
                print(f"  python3 scripts/gcp/embed_gcp.py --in_parquet corpus_chunks.parquet --final-index {first_chunk_index:,} [other args]")
        else:
            print(f"[DRY RUN] Would create checkpoint index file with value: {new_checkpoint_idx:,}")
            if first_chunk_index is not None:
                print(f"  Would recommend running with: --final-index {first_chunk_index:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Reset checkpoint to 0 to process from beginning (keeps all existing chunks)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script finds the first chunk in the checkpoint directory and sets the checkpoint
index to 0. All existing chunks are kept.

When you run embed_gcp.py, use the --final-index argument (shown in the output)
to stop processing before the first existing chunk, preserving all existing chunks.

Example:
  python reset_to_first_chunk.py
  # Then run the command shown in the output, which includes --final-index
  python reset_to_first_chunk.py --dry-run  # Preview changes
        '''
    )
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
                        help='Directory containing chunk files')
    parser.add_argument('--checkpoint-idx-file', type=str,
                        default='/home/hjt36_cornell_edu/corpus_chunks_embeddings_checkpoint_idx.txt',
                        help='Path to checkpoint index file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually doing it')
    args = parser.parse_args()
    
    reset_to_first_chunk(
        args.checkpoint_dir,
        args.checkpoint_idx_file,
        args.dry_run
    )


if __name__ == '__main__':
    main()

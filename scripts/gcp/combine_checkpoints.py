#!/usr/bin/env python3
"""
Simple script to combine a checkpoint file and chunk_*.npy files
into one final embeddings .npy file.

This version favors correctness and uses an on-disk memmap
(`open_memmap`) so it does NOT need to hold all embeddings in RAM.
"""

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def main():
    parser = argparse.ArgumentParser(description='Combine checkpoint and chunk files into final embeddings')
    parser.add_argument(
        '--checkpoint_file',
        type=str,
        default='/home/hjt36_cornell_edu/corpus_chunks_embeddings_checkpoint.npy',
        help='Path to main checkpoint file (if exists)',
    )
    parser.add_argument(
        '--chunks_dir',
        type=str,
        default='/home/hjt36_cornell_edu/corpus_chunks_checkpoint_chunks',
        help='Directory containing chunk_*.npy files',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='/home/hjt36_cornell_edu/corpus_chunks_embeddings_final.npy',
        help='Output file path (will be overwritten)',
    )
    args = parser.parse_args()

    checkpoint_file = Path(args.checkpoint_file)
    chunks_dir = Path(args.chunks_dir)
    output_file = Path(args.output_file)

    # 1) All chunk_*.npy (and chunk_final_*.npy) files, sorted by numeric index
    def chunk_index(p: Path) -> int:
        name = p.name
        try:
            if name.startswith('chunk_final_'):
                return int(name.replace('chunk_final_', '').replace('.npy', ''))
            if name.startswith('chunk_'):
                return int(name.replace('chunk_', '').replace('.npy', ''))
        except ValueError:
            pass
        return 10**15  # put unparseable names at the end

    chunk_files = sorted(chunks_dir.glob('chunk*.npy'), key=chunk_index)
    print(f"Found {len(chunk_files)} chunk files in {chunks_dir}")

    if not chunk_files and not checkpoint_file.exists():
        print("No checkpoint or chunk files found; nothing to combine.")
        return

    # 2) First pass: determine total_rows and embedding_dim
    total_rows = 0
    embedding_dim = None

    def _load_shape(path: Path, label: str):
        nonlocal total_rows, embedding_dim
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"{label} {path} has unexpected shape {arr.shape}")
        rows, dim = arr.shape
        if embedding_dim is None:
            embedding_dim = dim
        elif embedding_dim != dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {embedding_dim}, got {dim} in {path}"
            )
        total_rows += rows
        print(f"{label} {path.name}: shape={arr.shape} (total rows so far: {total_rows:,})")
        del arr

    # Optional checkpoint first
    file_order: list[tuple[str, Path]] = []
    if checkpoint_file.exists():
        print(f"Will include checkpoint file: {checkpoint_file}")
        _load_shape(checkpoint_file, "checkpoint")
        file_order.append(("checkpoint", checkpoint_file))
    else:
        print(f"No checkpoint file found at {checkpoint_file} (this is fine).")

    # Then all chunks
    for f in chunk_files:
        file_order.append(("chunk", f))
        _load_shape(f, "chunk")

    print(f"\nTotal rows: {total_rows:,}, embedding_dim: {embedding_dim}")

    # 3) Create on-disk .npy via open_memmap and fill incrementally
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating output memmap at {output_file} ...")
    combined = open_memmap(
        output_file,
        mode='w+',
        dtype='float32',
        shape=(total_rows, embedding_dim),
    )

    current_row = 0
    for label, path in file_order:
        print(f"Writing {label} from {path.name} ...")
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        rows = arr.shape[0]
        combined[current_row:current_row + rows] = arr.astype('float32')
        current_row += rows
        print(f"  wrote {rows:,} rows (progress: {current_row:,}/{total_rows:,})")
        del arr

    # Flush to disk
    del combined
    print(f"\n✓ Saved final embeddings to {output_file} (rows={total_rows:,}, dim={embedding_dim})")


if __name__ == '__main__':
    main()

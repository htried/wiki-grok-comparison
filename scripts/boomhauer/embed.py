import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MODEL = 'Qwen/Qwen3-Embedding-0.6B'

def main():
    parser = argparse.ArgumentParser(description='Embed chunked corpus (supports sharded multi-GPU)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--in_parquet', type=str, default='corpus_chunks.parquet')
    parser.add_argument('--out_parquet', type=str, default=None,
                        help='Output parquet with emb_ix. Auto-generates if not specified.')
    parser.add_argument('--out_emb', type=str, default=None,
                        help='Output embeddings file. Auto-generates if not specified.')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--sdpa', action='store_true', help='Use PyTorch SDPA attention (optimized, works on all GPUs)')
    parser.add_argument('--flash-attn', action='store_true', help='Use Flash Attention 2 (requires Ampere+ GPUs, e.g., A100, H100, RTX 30/40 series)')
    parser.add_argument('--device', type=str, default=None, help='Device to use, e.g., cuda:0, cuda:1, or cpu. Defaults to auto.')
    parser.add_argument('--shard-id', type=int, default=0, help='Shard id for distributed embedding (0-indexed)')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_kwargs = {}
    tokenizer_kwargs = {}
    
    # Attention mechanism selection (mutually exclusive)
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("✓ Using Flash Attention 2 (requires Ampere+ GPUs: A100, H100, RTX 30/40 series)")
        # Flash Attention works best with left padding for batch processing
        tokenizer_kwargs["padding_side"] = "left"
    elif args.sdpa:
        model_kwargs["attn_implementation"] = "sdpa"
        print("✓ Using PyTorch SDPA (optimized attention, works on all GPUs)")
    else:
        # Default to SDPA if available, otherwise native
        try:
            model_kwargs["attn_implementation"] = "sdpa"
            print("✓ Using PyTorch SDPA (default)")
        except:
            print("Using default attention implementation")

    # Read parquet FIRST (before loading model) to avoid simultaneous model loading
    # This also helps stagger memory usage across processes
    print("Loading parquet file...")
    corpus_df = pd.read_parquet(args.in_parquet)
    shard_id = int(args.shard_id)
    num_shards = int(args.num_shards)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f'Invalid shard-id {shard_id} for num-shards {num_shards}')
    # Even split; last shard takes the remainder
    total_rows = len(corpus_df)
    base = total_rows // num_shards
    rem = total_rows % num_shards
    start_row = shard_id * base + min(shard_id, rem)
    end_row = start_row + base + (1 if shard_id < rem else 0)
    shard_df = corpus_df.iloc[start_row:end_row].reset_index(drop=True)
    texts = shard_df['text'].tolist()
    
    # Free the full dataframe from memory (only keep shard_df)
    del corpus_df
    import gc
    gc.collect()
    
    # NOW load the model (after parquet is loaded and freed)
    # This helps stagger model loading across processes
    print("Loading model...")
    # Force model to load directly to GPU if available, avoiding CPU RAM copy
    if device.startswith('cuda'):
        # Ensure we're using the right GPU
        torch.cuda.set_device(int(device.split(':')[1]) if ':' in device else 0)
        # Clear any existing cache
        torch.cuda.empty_cache()
    
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs=model_kwargs if model_kwargs else None,
        tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs else None,
    )
    
    # Clear GPU cache after model load
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print(f"Embedding {len(texts)} chunks...")
    if device == 'cuda':
        print(f"Using GPU with batch size: {args.batch}")
    
    # Checkpoint support: per-shard checkpoint files
    name_prefix = f"{Path(args.in_parquet).stem}_shard{shard_id}of{num_shards}"
    checkpoint_file = Path(args.in_parquet).parent / f'{name_prefix}_embeddings_checkpoint.npy'
    checkpoint_idx_file = Path(args.in_parquet).parent / f'{name_prefix}_embeddings_checkpoint_idx.txt'
    start_idx = 0
    existing_embs = None
    
    if checkpoint_file.exists() and checkpoint_idx_file.exists():
        print(f"Found checkpoint: {checkpoint_file}")
        try:
            existing_embs = np.load(checkpoint_file)
            start_idx = int(checkpoint_idx_file.read_text().strip())
            print(f"Resuming from index {start_idx} (already processed {start_idx}/{len(texts)} chunks)")
            print(f"Existing checkpoint has shape: {existing_embs.shape}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting from scratch.")
            start_idx = 0
            existing_embs = None
    
    embs = []
    
    # Save progress every N batches
    save_every = 100  # Save every 100 batches
    
    # Calculate total number of batches for accurate ETA
    total_batches = (len(texts) - start_idx + args.batch - 1) // args.batch  # Ceiling division
    batch_indices = list(range(start_idx, len(texts), args.batch))
    
    for batch_num, i in enumerate(tqdm(batch_indices, desc=f'Embedding shard {shard_id}/{num_shards}', 
                                        total=total_batches, unit='batch')):
        batch = texts[i:i+args.batch]
        if i < 1:
            print(f"Batch type: {type(batch)}")
            print(f"Batch length (number of text chunks): {len(batch)}")
            print(f"First chunk length (characters): {len(batch[0]) if batch else 0}")
            print(f"First chunk preview: {batch[0][:200] if batch else 'N/A'}...")
            print(f"Last chunk length (characters): {len(batch[-1]) if batch else 0}")
            print(f"Last chunk preview: {batch[-1][:200] if batch else 'N/A'}...")
        try:
            with torch.no_grad():
                e = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            if i < 1:
                print(f"Embedding output shape: {e.shape}")
                print(f"Expected: ({len(batch)}, embedding_dim)")
                print(f"Actual: {e.shape}")
                if e.shape[0] != len(batch):
                    print(f"⚠️  WARNING: Number of embeddings ({e.shape[0]}) doesn't match number of input texts ({len(batch)})!")
            embs.append(e)
            
            # Save checkpoint periodically (every save_every batches)
            if (batch_num + 1) % save_every == 0:
                # Combine existing checkpoint with new embeddings
                current_embs = np.vstack(embs).astype('float32')
                if existing_embs is not None:
                    all_embs = np.vstack([existing_embs, current_embs]).astype('float32')
                else:
                    all_embs = current_embs
                np.save(checkpoint_file, all_embs)
                next_idx = min(i + args.batch, len(texts))
                checkpoint_idx_file.write_text(str(next_idx))
                print(f"\n[Checkpoint saved at index {next_idx}]")
                # Clear embs list since we've saved to checkpoint - prevents duplication on next checkpoint
                embs = []
                # Update existing_embs to include what we just saved
                existing_embs = all_embs
            
            # Clear GPU cache periodically to avoid fragmentation
            if device == 'cuda' and (batch_num + 1) % 10 == 0:
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n⚠ CUDA OOM at batch {i}! Saving checkpoint and exiting...")
            if len(embs) > 0:
                # Combine existing checkpoint with new embeddings
                current_embs = np.vstack(embs).astype('float32')
                if existing_embs is not None:
                    all_embs = np.vstack([existing_embs, current_embs]).astype('float32')
                else:
                    all_embs = current_embs
                np.save(checkpoint_file, all_embs)
                checkpoint_idx_file.write_text(str(i))
                print(f"Saved checkpoint at index {i}. Resume by running the same command.")
            raise
    
    # Combine existing checkpoint with new embeddings
    if len(embs) > 0:
        current_embs = np.vstack(embs).astype('float32')
        if existing_embs is not None:
            embs = np.vstack([existing_embs, current_embs]).astype('float32')
        else:
            embs = current_embs
    
    # Clean up checkpoint files on successful completion
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    if checkpoint_idx_file.exists():
        checkpoint_idx_file.unlink()
    print("Checkpoint files cleaned up.")
    
    # Final cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()

    shard_df['emb_ix'] = np.arange(len(shard_df))
    
    # Auto-generate output filenames if not provided
    in_path = Path(args.in_parquet)
    base_name = f"{in_path.stem}_shard{shard_id}of{num_shards}"
    
    if args.out_emb is None:
        out_emb = in_path.parent / f'{base_name}_embeddings.npy'
    else:
        out_emb = Path(args.out_emb)
    
    if args.out_parquet is None:
        out_parquet = in_path.parent / f'{base_name}_with_ix.parquet'
    else:
        out_parquet = Path(args.out_parquet)
    
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(out_emb, embs)
    shard_df.to_parquet(out_parquet)
    print(f"Wrote embeddings: {out_emb} (shape: {embs.shape})")
    print(f"Wrote metadata: {out_parquet}")
    if 'source' in corpus_df.columns:
        print(f"  Source distribution: {corpus_df['source'].value_counts().to_dict()}")

if __name__ == '__main__':
    main()
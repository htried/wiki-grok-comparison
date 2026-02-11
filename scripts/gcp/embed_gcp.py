import argparse
import shutil
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Global flag for clean shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print("\n⚠️  Interrupt received. Saving checkpoint before exit...")
    shutdown_requested = True
    # The checkpoint saving will happen in the main loop

DEFAULT_MODEL = 'google/embeddinggemma-300M'

def parse_gcs_path(gcs_path: str):
    """Parse a GCS path into bucket and blob name."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with 'gs://'")
    path = gcs_path.replace('gs://', '')
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name

def download_from_gcs(gcs_path: str, local_path: Path):
    """Download a file from GCS to local path."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob does not exist: {gcs_path}")
    
    print(f"Downloading {gcs_path} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    print(f"Downloaded to {local_path}")

def upload_to_gcs(local_path: Path, gcs_path: str):
    """Upload a local file to GCS."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    print(f"Uploading {local_path} to {gcs_path}...")
    blob.upload_from_filename(str(local_path))
    print(f"Uploaded to {gcs_path}")

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description='Embed chunked corpus (GCP single GPU version)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--in_parquet', type=str, default='corpus_chunks.parquet',
                        help='Input parquet file (local or gs:// path)')
    parser.add_argument('--out_parquet', type=str, default=None,
                        help='Output parquet with emb_ix. Auto-generates if not specified.')
    parser.add_argument('--out_emb', type=str, default=None,
                        help='Output embeddings file (local or gs://). Auto-generates if not specified.')
    parser.add_argument('--batch', type=int, default=512, 
                        help='Batch size for embedding (default: 512 for T4. Will auto-reduce if OOM occurs)')
    parser.add_argument('--sdpa', action='store_true', help='Use PyTorch SDPA attention (optimized, works on all GPUs)')
    parser.add_argument('--flash-attn', action='store_true', help='Use Flash Attention 2 (requires Ampere+ GPUs, e.g., A100, H100, RTX 30/40 series)')
    parser.add_argument('--device', type=str, default=None, help='Device to use, e.g., cuda:0, cuda:1, or cpu. Defaults to auto.')
    parser.add_argument('--checkpoint-every', type=int, default=50, help='Save checkpoint every N batches (default: 50). Reduce if running out of memory during checkpoint saves.')
    parser.add_argument('--final-index', type=int, default=None, help='Stop processing at this index (exclusive). Useful for processing up to a specific point without overwriting existing chunks.')
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

    # Handle input parquet (local or GCS)
    in_parquet_is_gcs = args.in_parquet.startswith('gs://')
    if in_parquet_is_gcs:
        import tempfile
        local_in_parquet = Path(tempfile.mktemp(suffix='.parquet'))
        download_from_gcs(args.in_parquet, local_in_parquet)
        in_parquet_path = local_in_parquet
    else:
        in_parquet_path = Path(args.in_parquet)
    
    # Read parquet FIRST (before loading model) to avoid simultaneous model loading
    print("Loading parquet file...")
    corpus_df = pd.read_parquet(in_parquet_path)
    texts = corpus_df['text'].tolist()
    
    # Clean up downloaded file if it was from GCS
    if in_parquet_is_gcs:
        local_in_parquet.unlink()
    
    # NOW load the model (after parquet is loaded and freed)
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
    
    # Checkpoint support
    in_path = Path(args.in_parquet)
    if in_parquet_is_gcs:
        # Use just the filename part for checkpoint naming
        base_name = Path(in_path.name).stem + "_rerun"
    else:
        base_name = in_path.stem + "_rerun"
    checkpoint_file = Path(f'{base_name}_embeddings_checkpoint.npy')
    checkpoint_idx_file = Path(f'{base_name}_embeddings_checkpoint_idx.txt')
    checkpoint_dir = checkpoint_file.parent / f'{base_name}_checkpoint_chunks'
    print(f"Checkpoint files will be saved as:")
    print(f"  Main checkpoint: {checkpoint_file.absolute()}")
    print(f"  Index file: {checkpoint_idx_file.absolute()}")
    print(f"  Incremental chunks: {checkpoint_dir.absolute()}")
    start_idx = 0
    existing_embs = None
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for checkpoint (either main file or incremental chunks)
    if checkpoint_idx_file.exists():
        try:
            start_idx = int(checkpoint_idx_file.read_text().strip())
            if checkpoint_file.exists():
                checkpoint_size = checkpoint_file.stat().st_size
                print(f"Found main checkpoint: {checkpoint_file}")
                print(f"Resuming from index {start_idx} (already processed {start_idx}/{len(texts)} chunks)")
                print(f"Checkpoint file size: {checkpoint_size / (1024**3):.2f} GB")
                print("  (Using incremental chunk saves to avoid memory issues)")
            elif checkpoint_dir.exists() and len(list(checkpoint_dir.glob('chunk_*.npy'))) > 0:
                chunk_count = len(list(checkpoint_dir.glob('chunk_*.npy')))
                print(f"Found incremental checkpoint chunks: {chunk_count} chunks")
                print(f"Resuming from index {start_idx} (already processed {start_idx}/{len(texts)} chunks)")
                print("  (Chunks will be combined at the end)")
            else:
                print(f"Found checkpoint index but no checkpoint files. Starting from scratch.")
                start_idx = 0
            existing_embs = None  # Don't load into memory - we'll use incremental saves
        except Exception as e:
            print(f"Warning: Could not read checkpoint: {e}. Starting from scratch.")
            start_idx = 0
            existing_embs = None
    else:
        start_idx = 0
        existing_embs = None
    
    embs = []
    
    # Save progress every N batches
    # Each checkpoint save appends ~(save_every * batch_size) embeddings
    # For large checkpoints, reducing save_every helps minimize memory usage during concatenation
    save_every = args.checkpoint_every
    
    # Dynamic batch size - will reduce if OOM occurs
    current_batch_size = args.batch
    min_batch_size = 32  # Don't go below this
    
    # Recalculate batch indices when batch size changes
    def recalculate_batches(batch_size):
        end_idx = len(texts)
        if args.final_index is not None:
            end_idx = min(end_idx, args.final_index)
        return list(range(start_idx, end_idx, batch_size))
    
    # Calculate total number of batches for accurate ETA
    batch_indices = recalculate_batches(current_batch_size)
    total_batches = len(batch_indices)
    
    if args.final_index is not None:
        print(f"⚠️  Final index set to {args.final_index:,}. Processing will stop before index {args.final_index:,}.")
        print(f"   Will process indices {start_idx:,} to {min(len(texts), args.final_index):,} (exclusive)")
    
    for batch_num, i in enumerate(tqdm(batch_indices, desc='Embedding', 
                                        total=total_batches, unit='batch')):
        # Check if we've reached the final index
        if args.final_index is not None and i >= args.final_index:
            print(f"\n✓ Reached final index {args.final_index:,}. Stopping processing.")
            break
        
        # Check for shutdown request
        if shutdown_requested:
            print("\n⚠️  Shutdown requested. Saving checkpoint...")
            if len(embs) > 0:
                try:
                    current_embs = np.vstack(embs).astype('float32')
                    if checkpoint_file.exists():
                        existing_data = np.load(checkpoint_file, mmap_mode='r')
                        all_embs = np.concatenate([existing_data, current_embs], axis=0).astype('float32')
                        del existing_data
                    else:
                        all_embs = current_embs
                    np.save(checkpoint_file, all_embs)
                    checkpoint_idx_file.write_text(str(i))
                    print(f"Checkpoint saved at index {i}")
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
            sys.exit(0)
        
        # Adjust batch size if we're near the final index
        batch_end = i + current_batch_size
        if args.final_index is not None and batch_end > args.final_index:
            batch_end = args.final_index
            if batch_end <= i:
                # We've reached the final index, skip this batch
                print(f"\n✓ Reached final index {args.final_index:,}. Stopping processing.")
                break
        
        batch = texts[i:batch_end]
        
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
                try:
                    print(f"\n[Preparing checkpoint at batch {batch_num + 1}, index {i}]")
                    
                    # Combine new embeddings first
                    current_embs = np.vstack(embs).astype('float32')
                    print(f"  Current batch embeddings shape: {current_embs.shape}")
                                        
                    # Use incremental checkpoint strategy: save new embeddings as separate chunk files
                    # This avoids loading the entire checkpoint into memory
                    chunk_file = checkpoint_dir / f'chunk_{i:010d}.npy'
                    print(f"  Saving incremental chunk: {chunk_file.name} ({current_embs.shape[0]:,} embeddings)")
                    np.save(chunk_file, current_embs)
                    
                    # Update index file to track progress
                    # The main checkpoint will be rebuilt at the end if needed
                    
                    next_idx = min(i + current_batch_size, len(texts))
                    temp_idx_file = checkpoint_idx_file.with_suffix('.txt.tmp')
                    temp_idx_file.write_text(str(next_idx))
                    shutil.move(str(temp_idx_file), str(checkpoint_idx_file))
                    
                    print(f"[Incremental checkpoint saved at index {next_idx}]")
                    # Clear embs list since we've saved to checkpoint - prevents duplication on next checkpoint
                    embs = []
                    # Don't keep existing_embs in memory - we'll load from file when needed
                    existing_embs = None
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                except MemoryError as mem_error:
                    print(f"\n⚠️  MEMORY ERROR saving checkpoint at batch {batch_num + 1}: {mem_error}")
                    print("  This might be due to large checkpoint size. Consider reducing save_every interval.")
                    import traceback
                    traceback.print_exc()
                    # Don't clear embs - keep them for next checkpoint attempt
                    # Continue processing but warn user
                    print("Continuing processing, but checkpoint not saved. Will retry at next checkpoint interval.")
                except Exception as checkpoint_error:
                    print(f"\n⚠️  ERROR saving checkpoint at batch {batch_num + 1}: {checkpoint_error}")
                    import traceback
                    traceback.print_exc()
                    # Don't clear embs - keep them for next checkpoint attempt
                    # Continue processing but warn user
                    print("Continuing processing, but checkpoint not saved. Will retry at next checkpoint interval.")
            
            # Clear GPU cache periodically to avoid fragmentation
            if device == 'cuda' and (batch_num + 1) % 10 == 0:
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n⚠ CUDA OOM at batch {i} with batch size {current_batch_size}!")
            
            # Try reducing batch size and retry
            if current_batch_size > min_batch_size:
                new_batch_size = max(min_batch_size, current_batch_size // 2)
                print(f"  Reducing batch size from {current_batch_size} to {new_batch_size} and retrying...")
                current_batch_size = new_batch_size
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Recalculate batch indices with new batch size
                batch_indices = recalculate_batches(current_batch_size)
                total_batches = len(batch_indices)
                
                # Retry with smaller batch
                try:
                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        e = model.encode(
                            batch,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )
                        embs.append(e)
                        continue
                except torch.cuda.OutOfMemoryError:
                    print(f"  Still OOM even with batch size {current_batch_size}. Saving checkpoint and exiting...")
            
            # If we can't reduce further or reduction didn't help, save checkpoint and exit
            if len(embs) > 0:
                try:
                    # Combine existing checkpoint with new embeddings
                    current_embs = np.vstack(embs).astype('float32')
                    if existing_embs is not None:
                        all_embs = np.vstack([existing_embs, current_embs]).astype('float32')
                    else:
                        all_embs = current_embs
                    np.save(checkpoint_file, all_embs)
                    checkpoint_idx_file.write_text(str(i))
                    print(f"Saved checkpoint at index {i}. Resume by running the same command.")
                except Exception as save_error:
                    print(f"⚠️  Failed to save checkpoint during OOM: {save_error}")
                    import traceback
                    traceback.print_exc()
            raise
        except Exception as e:
            # Catch any other exceptions (memory errors, I/O errors, etc.)
            print(f"\n⚠️  Unexpected error at batch {i} (batch_num {batch_num}): {e}")
            import traceback
            traceback.print_exc()
            # Try to save checkpoint before exiting
            if len(embs) > 0:
                try:
                    current_embs = np.vstack(embs).astype('float32')
                    if existing_embs is not None:
                        all_embs = np.vstack([existing_embs, current_embs]).astype('float32')
                    else:
                        all_embs = current_embs
                    np.save(checkpoint_file, all_embs)
                    checkpoint_idx_file.write_text(str(i))
                    print(f"Saved checkpoint at index {i} before exit.")
                except Exception as save_error:
                    print(f"⚠️  Failed to save checkpoint: {save_error}")
            raise
    
    # Determine the last processed index
    if args.final_index is not None:
        last_processed_index = args.final_index
    else:
        last_processed_index = len(texts)
    
    # Combine existing checkpoint with new embeddings
    # If we used incremental checkpoints, combine all chunks
    if len(embs) > 0:
        current_embs = np.vstack(embs).astype('float32')
        # Save final chunk
        if checkpoint_dir.exists() and len(list(checkpoint_dir.glob('chunk_*.npy'))) > 0:
            final_chunk_file = checkpoint_dir / f'chunk_final_{last_processed_index:010d}.npy'
            np.save(final_chunk_file, current_embs)
            print(f"Saved final chunk: {final_chunk_file.name} (index up to {last_processed_index:,})")
        
        if existing_embs is not None:
            embs = np.vstack([existing_embs, current_embs]).astype('float32')
        else:
            embs = current_embs
    
    # Update checkpoint index file to reflect the last processed index
    if checkpoint_idx_file.exists() or len(embs) > 0 or (checkpoint_dir.exists() and len(list(checkpoint_dir.glob('chunk_*.npy'))) > 0):
        checkpoint_idx_file.write_text(str(last_processed_index))
        print(f"✓ Updated checkpoint index to {last_processed_index:,}")
    
    # If we have incremental chunks, optionally combine them into main checkpoint
    # (Skip this if memory is limited - chunks can be combined later)
    chunk_files = sorted(checkpoint_dir.glob('chunk_*.npy')) if checkpoint_dir.exists() else []
    if chunk_files and not checkpoint_file.exists():
        print(f"\nCombining {len(chunk_files)} incremental checkpoint chunks...")
        print("  (This may use significant memory. If it fails, chunks can be combined later.)")
        try:
            chunk_arrays = [np.load(f) for f in chunk_files]
            embs = np.concatenate(chunk_arrays, axis=0).astype('float32')
            del chunk_arrays
            np.save(checkpoint_file, embs)
            print(f"  Combined checkpoint saved: {checkpoint_file}")
        except MemoryError:
            print(f"  ⚠️  Not enough memory to combine chunks. Keeping incremental chunks.")
            print(f"  Chunks can be combined later or loaded individually.")
    
    # Clean up checkpoint files on successful completion
    # But keep them if we're using --final-index (incomplete run)
    if args.final_index is None:
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        if checkpoint_idx_file.exists():
            checkpoint_idx_file.unlink()
        print("Checkpoint files cleaned up.")
    else:
        print(f"✓ Checkpoint files kept (incomplete run up to index {last_processed_index:,})")
    
    # Final cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()

    corpus_df['emb_ix'] = np.arange(len(corpus_df))
    
    # Auto-generate output filenames if not provided
    if args.out_emb is None:
        out_emb = Path(f'{base_name}_embeddings.npy')
    else:
        out_emb = Path(args.out_emb) if not args.out_emb.startswith('gs://') else args.out_emb
    
    if args.out_parquet is None:
        out_parquet = Path(f'{base_name}_with_ix.parquet')
    else:
        out_parquet = Path(args.out_parquet) if not args.out_parquet.startswith('gs://') else args.out_parquet
    
    # Save embeddings locally first
    if isinstance(out_emb, Path):
        out_emb.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_emb, embs)
        print(f"Wrote embeddings: {out_emb} (shape: {embs.shape})")
    else:
        # GCS path - save to temp file then upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        np.save(tmp_path, embs)
        upload_to_gcs(Path(tmp_path), out_emb)
        Path(tmp_path).unlink()
        print(f"Uploaded embeddings to {out_emb} (shape: {embs.shape})")
    
    # Save parquet
    if isinstance(out_parquet, Path):
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        corpus_df.to_parquet(out_parquet)
        print(f"Wrote metadata: {out_parquet}")
    else:
        # GCS path - save to temp file then upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        corpus_df.to_parquet(tmp_path)
        upload_to_gcs(Path(tmp_path), out_parquet)
        Path(tmp_path).unlink()
        print(f"Uploaded metadata to {out_parquet}")
    
    if 'source' in corpus_df.columns:
        print(f"  Source distribution: {corpus_df['source'].value_counts().to_dict()}")

if __name__ == '__main__':
    main()

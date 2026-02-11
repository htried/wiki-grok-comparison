# to be run on GCP with data in gs://enwiki-structured-contents-20251028/scraped_data.jsonl

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from google.cloud import storage
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_GCS_PATH = 'gs://enwiki-structured-contents-20251028/scraped_data.jsonl'
DEFAULT_MODEL = 'google/embeddinggemma-300M'
DEFAULT_WINDOW = 250
DEFAULT_STRIDE = 150  # => 100 overlap
tokenizer = None

def tokenize_count(text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)

def sliding_chunks(text: str, window=250, stride=150) -> List[str]:
    ids = tokenize_count(text)
    out = []
    for start in range(0, max(1, len(ids)), stride):
        piece = ids[start:start+window]
        if not piece: break
        out.append(detokenize(piece))
        if start+window >= len(ids): break
    return out

def parse_gcs_path(gcs_path: str):
    """Parse a GCS path into bucket and blob name."""
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with 'gs://'")
    path = gcs_path.replace('gs://', '')
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    return bucket_name, blob_name

def grok_iter_articles_gcs(gcs_path: str):
    """Iterate over Grokipedia articles from a GCS JSONL file."""
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob does not exist: {gcs_path}")
    
    # Stream the file content
    with blob.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            data = (obj.get('data') or {})
            title = data.get('title') or obj.get('title')
            sections = data.get('sections') or []
            parts = []
            for sec in sections:
                lvl = (sec.get('level') or '').lower()
                name = (sec.get('title') or '').strip()
                if not name: continue
                if lvl == 'h1':
                    parts.append(f"# {name}")
                elif lvl == 'h2':
                    parts.append(f"## {name}")
                else:
                    parts.append(f"### {name}")
                for item in (sec.get('content') or []):
                    if isinstance(item, dict):
                        txt = item.get('text') or ''
                        if txt.strip():
                            parts.append(txt)
            full = '\n\n'.join(parts).strip()
            yield title, full

def build_corpus(gcs_path: str, window_tokens: int, stride_tokens: int) -> pd.DataFrame:
    """
    Build corpus from Grokipedia GCS source.
    
    Args:
        gcs_path: GCS path to scraped_data.jsonl file
        window_tokens: Token window size for chunking
        stride_tokens: Token stride for overlapping chunks
    
    Returns:
        DataFrame with columns: title, source, chunk_id, text
    """
    rows = []
    
    # Grokipedia from GCS
    for title, text in tqdm(grok_iter_articles_gcs(gcs_path), desc='Grok'):
        if not title or not text: continue
        for i, chunk in enumerate(sliding_chunks(text, window_tokens, stride_tokens)):
            rows.append({'title': title.replace(' ', '_'),
                        'source': 'grok',
                        'chunk_id': i,
                        'text': chunk})
    return pd.DataFrame(rows)

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
    parser = argparse.ArgumentParser(description='Build grok chunked corpus from GCS (GCP version)')
    parser.add_argument('--gcs_path', type=str, default=DEFAULT_GCS_PATH,
                        help='GCS path to scraped_data.jsonl file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE)
    parser.add_argument('--out', type=str, default=None, 
                        help='Output parquet file (local or gs:// path). If not specified, auto-generates name.')
    parser.add_argument('--upload', action='store_true',
                        help='Upload output to GCS if --out is a local path')
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Auto-generate output filename if not provided
    if args.out is None:
        out_path = Path('corpus_chunks.parquet').resolve()
    else:
        out_path = Path(args.out) if not args.out.startswith('gs://') else None
        out_gcs_path = args.out if args.out.startswith('gs://') else None
    
    print(f"Output will be written to {out_path.absolute() if out_path else out_gcs_path}")
    
    try:
        corpus_df = build_corpus(args.gcs_path, args.window, args.stride)
        
        print(f"Built corpus with {len(corpus_df)} rows")
        
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Writing parquet to {out_path.absolute()}...")
            corpus_df.to_parquet(out_path)
            
            # Verify file was written
            if out_path.exists():
                file_size = out_path.stat().st_size
                print(f"Wrote {len(corpus_df)} rows to {out_path.absolute()} (file size: {file_size} bytes)")
                if len(corpus_df) > 0:
                    print(f"  Source distribution: {corpus_df['source'].value_counts().to_dict()}")
                else:
                    print(f"Warning: Processed 0 articles. Empty parquet written.")
                
                # Upload to GCS if requested
                if args.upload and out_gcs_path:
                    upload_to_gcs(out_path, out_gcs_path)
            else:
                raise RuntimeError(f"File {out_path.absolute()} was not created after to_parquet() call!")
        else:
            # Write directly to GCS (download, write, upload)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            corpus_df.to_parquet(tmp_path)
            upload_to_gcs(Path(tmp_path), out_gcs_path)
            Path(tmp_path).unlink()  # Clean up temp file
            
    except Exception as e:
        import sys
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == '__main__':
    main()

# to be run on boomhauer once data is done rsyncing over

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_WIKI_FP = Path('~/data/grokipedia_wikipedia_articles.ndjson').expanduser()
DEFAULT_GROK_DIR = Path('~/data/scraped_data').expanduser()
# Default to Qwen3 Embedding model so tokenization matches downstream embeddings
DEFAULT_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
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

def wiki_iter_articles(fp: Path):
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            title = obj.get('name')
            # Build a structural text that preserves headings + abstracts
            sections = obj.get('sections') or []
            parts = []
            # Map Abstract -> first h1-equivalent
            for i, sec in enumerate(sections):
                name = (sec.get('name') or '').strip()
                if not name: continue
                heading = f"# {name}" if name.lower() == 'abstract' else f"## {name}"
                parts.append(heading)
                for p in (sec.get('has_parts') or []):
                    val = p.get('value') or p.get('text') or ''
                    if isinstance(val, str) and val.strip():
                        parts.append(val)
            full = '\n\n'.join(parts).strip()
            yield title, full

def grok_iter_articles(scraped_dir: Path):
    for fp in sorted(scraped_dir.glob('*.jsonl')):
        with fp.open('r', encoding='utf-8') as f:
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

def build_corpus(wiki_fp: Path, grok_dir: Path, window_tokens: int, stride_tokens: int, source: str = None,
                 shard_id: int = 0, num_shards: int = 1, sub_shard_id: int = 0, sub_num_shards: int = 1) -> pd.DataFrame:
    """
    Build corpus from wiki and/or grok sources.
    
    Args:
        wiki_fp: Path to Wikipedia JSONL file
        grok_dir: Path to Grokipedia scraped_data directory
        window_tokens: Token window size for chunking
        stride_tokens: Token stride for overlapping chunks
        source: If specified ('wiki' or 'grok'), only process that source
    
    Returns:
        DataFrame with columns: title, source, chunk_id, text
    """
    rows = []
    
    # Wikipedia
    if wiki_fp is not None and (source is None or source == 'wiki'):
        idx = 0
        shard_count = 0  # Count of articles in this shard
        for title, text in tqdm(wiki_iter_articles(wiki_fp), desc='Wiki'):
            # First check if article belongs to this shard
            in_shard = (idx % num_shards) == shard_id
            idx += 1
            if not in_shard:
                continue
            
            # If sub-sharding is enabled, further filter based on position within this shard
            if sub_num_shards > 1:
                in_sub_shard = (shard_count % sub_num_shards) == sub_shard_id
                shard_count += 1
                if not in_sub_shard:
                    continue
            else:
                shard_count += 1
            
            if not title or not text: continue
            for i, chunk in enumerate(sliding_chunks(text, window_tokens, stride_tokens)):
                rows.append({'title': title.replace(' ', '_'),
                            'source': 'wiki',
                            'chunk_id': i,
                            'text': chunk})
    
    # Grokipedia
    if grok_dir is not None and (source is None or source == 'grok'):
        idx = 0
        shard_count = 0  # Count of articles in this shard
        for title, text in tqdm(grok_iter_articles(grok_dir), desc='Grok'):
            # First check if article belongs to this shard
            in_shard = (idx % num_shards) == shard_id
            idx += 1
            if not in_shard:
                continue
            
            # If sub-sharding is enabled, further filter based on position within this shard
            if sub_num_shards > 1:
                in_sub_shard = (shard_count % sub_num_shards) == sub_shard_id
                shard_count += 1
                if not in_sub_shard:
                    continue
            else:
                shard_count += 1
            
            if not title or not text: continue
            for i, chunk in enumerate(sliding_chunks(text, window_tokens, stride_tokens)):
                rows.append({'title': title.replace(' ', '_'),
                            'source': 'grok',
                            'chunk_id': i,
                            'text': chunk})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description='Build wiki/grok chunked corpus (shardable)')
    parser.add_argument('--wiki_fp', type=str, default=None, help='Path to Wikipedia JSONL file')
    parser.add_argument('--grok_dir', type=str, default=None, help='Path to Grokipedia scraped_data directory')
    parser.add_argument('--source', type=str, choices=['wiki', 'grok'], default=None, 
                        help='Process only this source (wiki or grok). If not specified, processes both.')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    parser.add_argument('--stride', type=int, default=DEFAULT_STRIDE)
    parser.add_argument('--out', type=str, default=None, 
                        help='Output parquet file. If not specified and --source is set, auto-generates name.')
    parser.add_argument('--shard-id', type=int, default=0, help='Shard id (0-indexed)')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards')
    parser.add_argument('--sub-shard-id', type=int, default=0, help='Sub-shard id within the shard (0-indexed)')
    parser.add_argument('--sub-num-shards', type=int, default=1, help='Total number of sub-shards within this shard')
    args = parser.parse_args()

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    wiki_fp = Path(args.wiki_fp).expanduser() if args.wiki_fp is not None else None
    grok_dir = Path(args.grok_dir).expanduser() if args.grok_dir is not None else None
    
    # Validate inputs
    if args.source == 'wiki' and wiki_fp is None:
        raise ValueError("--wiki_fp is required when --source=wiki")
    if args.source == 'grok' and grok_dir is None:
        raise ValueError("--grok_dir is required when --source=grok")
    
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"Invalid --shard-id {args.shard_id} for --num-shards {args.num_shards}")
    if args.sub_shard_id < 0 or args.sub_shard_id >= args.sub_num_shards:
        raise ValueError(f"Invalid --sub-shard-id {args.sub_shard_id} for --sub-num-shards {args.sub_num_shards}")

    # Auto-generate output filename if not provided (before processing, so we can write even on error)
    shard_suffix = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    if args.sub_num_shards > 1:
        shard_suffix += f"_sub{args.sub_shard_id}of{args.sub_num_shards}"
    if args.out is None:
        if args.source:
            out_path = Path(f'corpus_chunks_{args.source}{shard_suffix}.parquet').resolve()
        else:
            out_path = Path(f'corpus_chunks{shard_suffix}.parquet').resolve()
    else:
        out_path = Path(args.out).resolve()
    
    print(f"Shard {args.shard_id}: Output will be written to {out_path.absolute()}")
    
    try:
        corpus_df = build_corpus(wiki_fp, grok_dir, args.window, args.stride, args.source,
                                 shard_id=args.shard_id, num_shards=args.num_shards,
                                 sub_shard_id=args.sub_shard_id, sub_num_shards=args.sub_num_shards)
        
        print(f"Shard {args.shard_id}: Built corpus with {len(corpus_df)} rows")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Shard {args.shard_id}: Writing parquet to {out_path.absolute()}...")
        corpus_df.to_parquet(out_path)
        
        # Verify file was written
        if out_path.exists():
            file_size = out_path.stat().st_size
            print(f"Wrote {len(corpus_df)} rows to {out_path.absolute()} (file size: {file_size} bytes)")
            if len(corpus_df) > 0:
                print(f"  Source distribution: {corpus_df['source'].value_counts().to_dict()}")
            else:
                print(f"Warning: Shard {args.shard_id} processed 0 articles. Empty parquet written.")
        else:
            raise RuntimeError(f"File {out_path.absolute()} was not created after to_parquet() call!")
            
    except Exception as e:
        # Ensure we write an empty parquet even on error, so downstream knows the shard "completed"
        import sys
        import traceback
        print(f"ERROR in shard {args.shard_id}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(out_path)  # Write empty file
            if out_path.exists():
                print(f"Wrote empty parquet to {out_path.absolute()} due to error")
            else:
                print(f"CRITICAL: Failed to write even empty parquet to {out_path.absolute()}", file=sys.stderr)
        except Exception as e2:
            print(f"CRITICAL: Could not write error marker file: {e2}", file=sys.stderr)
        raise

if __name__ == '__main__':
    main()
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Compute wiki/grok pairwise similarities')
    # Support both combined and separate files
    parser.add_argument('--in_parquet', type=str, default=None,
                        help='Combined corpus parquet (with both wiki and grok). Use --wiki_parquet/--grok_parquet for separate files.')
    parser.add_argument('--wiki_parquet', type=str, default=None,
                        help='Wikipedia corpus parquet file')
    parser.add_argument('--grok_parquet', type=str, default=None,
                        help='Grokipedia corpus parquet file')
    # Sharded inputs: provide globs that expand to multiple files
    parser.add_argument('--wiki_parquet_glob', type=str, default=None,
                        help='Glob for multiple wiki parquet shards (e.g., \'/path/corpus_chunks_wiki_shard*of*.parquet\')')
    parser.add_argument('--grok_parquet_glob', type=str, default=None,
                        help='Glob for multiple grok parquet shards')
    parser.add_argument('--emb', type=str, default=None,
                        help='Combined embeddings file. Use --wiki_emb/--grok_emb for separate files.')
    parser.add_argument('--wiki_emb', type=str, default=None,
                        help='Wikipedia embeddings file')
    parser.add_argument('--grok_emb', type=str, default=None,
                        help='Grokipedia embeddings file')
    parser.add_argument('--wiki_emb_glob', type=str, default=None,
                        help='Glob for multiple wiki embedding shards (e.g., \'*_shard*of*_embeddings.npy\')')
    parser.add_argument('--grok_emb_glob', type=str, default=None,
                        help='Glob for multiple grok embedding shards')
    parser.add_argument('--out_stats', type=str, default='pairwise_stats.parquet')
    parser.add_argument('--out_top1', type=str, default='pairwise_top1_alignments.parquet')
    args = parser.parse_args()

    # Determine if using combined or separate files
    if args.in_parquet:
        # Combined mode
        corpus_df = pd.read_parquet(args.in_parquet)
        embs = np.load(args.emb)  # L2-normalized -> cosine == dot
        wiki_df = corpus_df[corpus_df['source'] == 'wiki'].copy()
        grok_df = corpus_df[corpus_df['source'] == 'grok'].copy()
        wiki_embs = embs[wiki_df['emb_ix'].values]
        grok_embs = embs[grok_df['emb_ix'].values]
        
        # Reset emb_ix for indexing into separate arrays
        wiki_df = wiki_df.reset_index(drop=True)
        grok_df = grok_df.reset_index(drop=True)
        wiki_df['emb_ix'] = np.arange(len(wiki_df))
        grok_df['emb_ix'] = np.arange(len(grok_df))
        
    elif (args.wiki_parquet and args.grok_parquet) or (args.wiki_parquet_glob and args.grok_parquet_glob):
        # Separate (possibly sharded) mode
        import glob
        import re
        def sort_by_shard(files):
            # Expect names with ..._shard{K}of{N}...
            def shard_key(p):
                m = re.search(r"_shard(\d+)of(\d+)", Path(p).stem)
                if m:
                    return (int(m.group(2)), int(m.group(1)))
                return (1, 0)
            return sorted(files, key=shard_key)

        if args.wiki_parquet_glob:
            wiki_parquet_files = sort_by_shard(glob.glob(args.wiki_parquet_glob))
            if not wiki_parquet_files:
                raise FileNotFoundError(f"No files matched --wiki_parquet_glob: {args.wiki_parquet_glob}")
            wiki_df = pd.concat([pd.read_parquet(p) for p in wiki_parquet_files], ignore_index=True)
        else:
            wiki_df = pd.read_parquet(args.wiki_parquet)

        if args.grok_parquet_glob:
            grok_parquet_files = sort_by_shard(glob.glob(args.grok_parquet_glob))
            if not grok_parquet_files:
                raise FileNotFoundError(f"No files matched --grok_parquet_glob: {args.grok_parquet_glob}")
            grok_df = pd.concat([pd.read_parquet(p) for p in grok_parquet_files], ignore_index=True)
        else:
            grok_df = pd.read_parquet(args.grok_parquet)
        
        # Auto-detect embedding files if not provided
        if args.wiki_emb_glob:
            wiki_emb_files = sort_by_shard(glob.glob(args.wiki_emb_glob))
            if not wiki_emb_files:
                raise FileNotFoundError(f"No files matched --wiki_emb_glob: {args.wiki_emb_glob}")
            wiki_embs = np.vstack([np.load(p) for p in wiki_emb_files])
        elif args.wiki_emb:
            wiki_emb_path = args.wiki_emb
            wiki_embs = np.load(wiki_emb_path)
        else:
            wiki_path = Path(args.wiki_parquet)
            # Try common naming patterns
            emb_path = wiki_path.parent / f'{wiki_path.stem.replace("_with_ix", "")}_embeddings.npy'
            if not emb_path.exists():
                emb_path = wiki_path.parent / f'{wiki_path.stem}_embeddings.npy'
            if not emb_path.exists():
                raise FileNotFoundError(f"Could not auto-detect Wikipedia embeddings. Please specify --wiki_emb")
            wiki_embs = np.load(emb_path)
        
        if args.grok_emb_glob:
            grok_emb_files = sort_by_shard(glob.glob(args.grok_emb_glob))
            if not grok_emb_files:
                raise FileNotFoundError(f"No files matched --grok_emb_glob: {args.grok_emb_glob}")
            grok_embs = np.vstack([np.load(p) for p in grok_emb_files])
        elif args.grok_emb:
            grok_emb_path = args.grok_emb
            grok_embs = np.load(grok_emb_path)
        else:
            grok_path = Path(args.grok_parquet)
            # Try common naming patterns
            emb_path = grok_path.parent / f'{grok_path.stem.replace("_with_ix", "")}_embeddings.npy'
            if not emb_path.exists():
                emb_path = grok_path.parent / f'{grok_path.stem}_embeddings.npy'
            if not emb_path.exists():
                raise FileNotFoundError(f"Could not auto-detect Grokipedia embeddings. Please specify --grok_emb")
            grok_embs = np.load(emb_path)
    else:
        raise ValueError("Must provide either --in_parquet (combined) or both --wiki_parquet and --grok_parquet (separate)")

    # Ensure emb_ix is properly set for separate arrays
    # For sharded inputs, ignore any preexisting emb_ix and reindex to match concatenation order
    wiki_df = wiki_df.reset_index(drop=True)
    grok_df = grok_df.reset_index(drop=True)
    wiki_df['emb_ix'] = np.arange(len(wiki_df))
    grok_df['emb_ix'] = np.arange(len(grok_df))

    stats_rows = []
    topk_rows = []

    # Get all unique titles that appear in both
    wiki_titles = set(wiki_df['title'].unique())
    grok_titles = set(grok_df['title'].unique())
    common_titles = wiki_titles & grok_titles

    print(f"Found {len(common_titles)} titles common to both datasets")

    for title in tqdm(sorted(common_titles), desc='Pairwise'):
        w = wiki_df[wiki_df['title'] == title]
        g = grok_df[grok_df['title'] == title]
        if len(w) == 0 or len(g) == 0:
            continue
        
        W = wiki_embs[w['emb_ix'].values]
        G = grok_embs[g['emb_ix'].values]
        S = W @ G.T  # L2-normalized -> cosine == dot

        stats_rows.append({
            'title': title,
            'n_w': len(w), 'n_g': len(g),
            'sim_mean': float(S.mean()),
            'sim_median': float(np.median(S)),
            'sim_max': float(S.max()),
            'sim_p90': float(np.percentile(S, 90)),
        })

        best_ix = S.argmax(axis=1)
        best_val = S.max(axis=1)
        for i, (gi, sv) in enumerate(zip(best_ix, best_val)):
            topk_rows.append({
                'title': title,
                'wiki_chunk_id': int(w.iloc[i]['chunk_id']),
                'grok_chunk_id': int(g.iloc[gi]['chunk_id']),
                'similarity': float(sv),
                'wiki_text': w.iloc[i]['text'],
                'grok_text': g.iloc[gi]['text'],
            })

    pair_stats = pd.DataFrame(stats_rows).sort_values('sim_mean', ascending=True)
    pair_top1  = pd.DataFrame(topk_rows)

    pair_stats.to_parquet(args.out_stats)
    pair_top1.to_parquet(args.out_top1)
    print(f"Wrote stats: {args.out_stats} and top1: {args.out_top1}")

if __name__ == '__main__':
    main()
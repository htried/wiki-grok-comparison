#!/usr/bin/env python3
"""
GCP Shard Runner for Edit Requests Fetcher
Runs a subset of slugs based on start_idx and end_idx
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts/gcp directory to path
sys.path.insert(0, str(Path(__file__).parent))

import edit_requests_scraper as scraper


async def main():
    parser = argparse.ArgumentParser(description='Run edit requests fetcher on a slug range')
    parser.add_argument('--start_idx', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--shard_id', type=int, default=None, help='Optional shard ID for naming')
    parser.add_argument('--slugs_file', type=str, default=None, help='Path to slugs file (local or gs://), or "hf"/"huggingface" to load from HuggingFace dataset')
    args = parser.parse_args()
    
    if args.start_idx >= args.end_idx:
        print("Error: start_idx must be less than end_idx")
        sys.exit(1)
    
    shard_id_str = f" (Shard {args.shard_id})" if args.shard_id is not None else ""
    print(f"Running edit requests fetcher{shard_id_str}")
    print(f"Slug range: {args.start_idx:,} to {args.end_idx:,} ({args.end_idx - args.start_idx:,} slugs)")
    
    # Load slugs
    try:
        if args.slugs_file:
            all_slugs = scraper.load_slugs_from_file(args.slugs_file)
            source_desc = "HuggingFace dataset" if args.slugs_file.lower() in ['hf', 'huggingface', 'hf_dataset'] else f"file: {args.slugs_file}"
            print(f"Loaded {len(all_slugs):,} total slugs from {source_desc}")
        else:
            # Default to the standard location, fallback to HuggingFace
            default_file = "results/overall/grokipedia_w_license.txt"
            if os.path.exists(default_file):
                all_slugs = scraper.load_slugs_from_file(default_file)
                print(f"Loaded {len(all_slugs):,} total slugs from default file: {default_file}")
            else:
                print(f"Default file not found: {default_file}")
                print("Falling back to HuggingFace dataset...")
                all_slugs = scraper.load_slugs_from_file("hf")
                print(f"Loaded {len(all_slugs):,} total slugs from HuggingFace dataset")
    except Exception as e:
        print(f"Error loading slugs: {e}")
        sys.exit(1)
    
    # Validate indices
    if args.end_idx > len(all_slugs):
        print(f"Warning: end_idx ({args.end_idx:,}) exceeds total slugs ({len(all_slugs):,}). Adjusting to {len(all_slugs):,}")
        args.end_idx = len(all_slugs)
    
    if args.start_idx < 0:
        print(f"Error: start_idx ({args.start_idx}) must be >= 0")
        sys.exit(1)
    
    # Extract shard slugs
    shard_slugs = all_slugs[args.start_idx:args.end_idx]
    print(f"Processing {len(shard_slugs):,} slugs (indices {args.start_idx:,} to {args.end_idx:,})")
    
    # Configuration - optimized for distributed fetching
    config = {
        'max_concurrent': 50,  # Concurrent requests
        'rate_limit': 100,  # Requests per minute per instance
        'api_timeout': 30,  # Timeout for API requests
        'batch_size': 1000,  # Save batch every N successful items
        'skip_on_error': True,
        'output_dir': 'edit_requests_data',
        'gcs_bucket': 'enwiki-structured-contents-20251028',
        'gcs_project': 'vitaly-gcp',
        'proxy_url': None  # Can be set via environment variable if needed
    }
    
    # Check for proxy credentials
    brightdata_username = os.getenv('BRIGHTDATA_USERNAME')
    brightdata_password = os.getenv('BRIGHTDATA_PASSWORD')
    if brightdata_username and brightdata_password:
        config['proxy_url'] = f'http://{brightdata_username}:{brightdata_password}@brd.superproxy.io:33335'
        print("BrightData proxy configured")
    
    # Run fetching phase
    print(f"\nStarting edit requests fetching phase...")
    success_count = await scraper.fetch_edit_requests_phase(
        shard_slugs,
        config,
        start_index=args.start_idx,  # Use absolute start index for blob naming
        shard_id=args.shard_id
    )
    
    print(f"\nShard Summary:")
    print(f"Processed {len(shard_slugs):,} slugs")
    print(f"Successfully fetched {success_count:,} pages")
    print(f"Success rate: {success_count/len(shard_slugs)*100:.2f}%" if len(shard_slugs) > 0 else "0%")

if __name__ == '__main__':
    asyncio.run(main())


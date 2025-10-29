#!/usr/bin/env python3
"""
GCP Shard Runner for Grokipedia Scraper
Runs a subset of URLs based on start_idx and end_idx
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import grokipedia_scraper as scraper


async def main():
    parser = argparse.ArgumentParser(description='Run grokipedia scraper on a URL range')
    parser.add_argument('--start_idx', type=int, required=True, help='Start index (inclusive)')
    parser.add_argument('--end_idx', type=int, required=True, help='End index (exclusive)')
    parser.add_argument('--shard_id', type=int, default=None, help='Optional shard ID for naming')
    args = parser.parse_args()
    
    if args.start_idx >= args.end_idx:
        print("Error: start_idx must be less than end_idx")
        sys.exit(1)
    
    shard_id_str = f" (Shard {args.shard_id})" if args.shard_id is not None else ""
    print(f"Running scraper{shard_id_str}")
    print(f"URL range: {args.start_idx:,} to {args.end_idx:,} ({args.end_idx - args.start_idx:,} URLs)")
    
    # Load URLs from HuggingFace dataset
    try:
        all_urls = scraper.load_urls_from_hf()
        print(f"Loaded {len(all_urls):,} total URLs from HuggingFace")
    except Exception as e:
        print(f"Error loading URLs: {e}")
        sys.exit(1)
    
    # Validate indices
    if args.end_idx > len(all_urls):
        print(f"Warning: end_idx ({args.end_idx:,}) exceeds total URLs ({len(all_urls):,}). Adjusting to {len(all_urls):,}")
        args.end_idx = len(all_urls)
    
    if args.start_idx < 0:
        print(f"Error: start_idx ({args.start_idx}) must be >= 0")
        sys.exit(1)
    
    # Extract shard URLs
    shard_urls = all_urls[args.start_idx:args.end_idx]
    print(f"Processing {len(shard_urls):,} URLs (indices {args.start_idx:,} to {args.end_idx:,})")
    
    # Configuration - optimized for distributed scraping (lower per-instance limits)
    config = {
        'max_concurrent': 50,  # Lower for distributed scraping
        'rate_limit': 100,  # Lower rate limit per instance (more instances = more total throughput)
        'scraping_timeout': 60,
        'scraping_batch_size': 300,
        'batch_delay': 0.5,
        'skip_on_error': True,
        'output_dir': 'scraped_data',
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
    
    # Run scraping phase
    print(f"\nStarting scraping phase...")
    success_count = await scraper.scraping_phase(
        shard_urls,
        config,
        start_index=args.start_idx,  # Use absolute start index for blob naming
        shard_id=args.shard_id
    )
    
    print(f"\nShard Summary:")
    print(f"Processed {len(shard_urls):,} URLs")
    print(f"Successfully scraped {success_count:,} pages")
    print(f"Success rate: {success_count/len(shard_urls)*100:.2f}%" if len(shard_urls) > 0 else "0%")

if __name__ == '__main__':
    asyncio.run(main())

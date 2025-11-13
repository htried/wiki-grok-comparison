#!/usr/bin/env python3
"""
Local test script for fixed issues scraper (no proxy, no GCS)
Tests the scraper on a small set of slugs locally
"""

import asyncio
import sys
from pathlib import Path

# Add scripts/gcp directory to path
sys.path.insert(0, str(Path(__file__).parent))

import fixed_issues_scraper as scraper


async def main():
    # Test with a small set of titles (will be converted to slugs automatically)
    # You can modify this list or load from a file
    test_titles = [
        "Indiana Jones and the Dial of Destiny",
        "The Beatles",
        "Python (programming language)",
        "Barack Obama",
        "Elon Musk",
        "Fullback (rugby league)",
        "Palindrome",
        "Cold Winter",
        "Russell Hicks",
    ]
    
    # Convert titles to slugs
    test_slugs = [scraper.title_to_slug(title) for title in test_titles]
    
    # Or load from file (uncomment to use)
    # slugs_file = "results/overall/grokipedia_w_license.txt"
    # test_slugs = scraper.load_slugs_from_file(slugs_file)[:10]  # First 10 slugs
    
    print(f"Testing fixed issues scraper locally")
    print(f"Number of slugs to test: {len(test_slugs)}")
    print(f"Slugs: {test_slugs}\n")
    
    # Configuration for local testing (no proxy, no GCS)
    config = {
        'max_concurrent': 5,  # Lower for local testing
        'rate_limit': 30,  # Lower rate limit for local testing
        'api_timeout': 30,
        'batch_size': 10,  # Small batch size for testing
        'skip_on_error': True,
        'output_dir': 'fixed_issues_test_output',  # Local output directory
        'gcs_bucket': None,  # No GCS upload
        'gcs_project': None,
        'proxy_url': None  # No proxy
    }
    
    print("Configuration:")
    print(f"  Max concurrent: {config['max_concurrent']}")
    print(f"  Rate limit: {config['rate_limit']} requests/minute")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Proxy: None (direct connection)\n")
    
    # Run fetching phase
    print("Starting fixed issues fetching...")
    success_count = await scraper.fetch_fixed_issues_phase(
        test_slugs,
        config,
        start_index=0,
        shard_id=None
    )
    
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"{'='*60}")
    print(f"Processed {len(test_slugs)} slugs")
    print(f"Successfully fetched {success_count} pages")
    print(f"Success rate: {success_count/len(test_slugs)*100:.2f}%" if len(test_slugs) > 0 else "0%")
    print(f"\nResults saved to: {config['output_dir']}/")
    print(f"Check the output files for fixedIssues data")


if __name__ == '__main__':
    asyncio.run(main())


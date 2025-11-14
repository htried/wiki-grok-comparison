#!/usr/bin/env python3
"""
Download all fixed_issues files from GCS, consolidate successful entries,
and identify pages that need to be retried.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from google.cloud import storage
from tqdm import tqdm

# Configuration
GCS_BUCKET = "enwiki-structured-contents-20251028"
GCS_PREFIX = "fixed_issues"
OUTPUT_DIR = Path("results/fixed_issues")
CONSOLIDATED_FILE = OUTPUT_DIR / "all_fixed_issues.jsonl"
FAILED_SLUGS_FILE = OUTPUT_DIR / "failed_slugs.txt"
STATS_FILE = OUTPUT_DIR / "consolidation_stats.json"

def list_gcs_files(bucket_name, prefix):
    """List all files in GCS with the given prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.jsonl')]

def download_file(bucket_name, blob_name, local_path):
    """Download a file from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def load_original_slugs(slugs_file):
    """Load the original list of slugs/titles."""
    slugs = []
    with open(slugs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                slugs.append(line)
    return slugs

def title_to_slug(title):
    """Convert title to slug format (same as in fixed_issues_scraper.py)."""
    return title.replace(" ", "_")

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Listing files in gs://{GCS_BUCKET}/{GCS_PREFIX}/...")
    gcs_files = list_gcs_files(GCS_BUCKET, f"{GCS_PREFIX}/")
    print(f"Found {len(gcs_files)} JSONL files")
    
    # Download all files
    print("\nDownloading files...")
    local_files = []
    for blob_name in tqdm(gcs_files, desc="Downloading"):
        local_path = OUTPUT_DIR / blob_name.replace("/", "_")
        download_file(GCS_BUCKET, blob_name, local_path)
        local_files.append(local_path)
    
    # Process all files and consolidate
    print("\nProcessing files and consolidating...")
    successful_slugs = set()
    all_entries = []
    failed_slugs = set()
    shard_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})
    
    for local_file in tqdm(local_files, desc="Processing"):
        # Extract shard info from filename
        parts = local_file.stem.split("_")
        shard_id = None
        for part in parts:
            if part.startswith("shard"):
                try:
                    shard_id = part.replace("shard", "")
                    break
                except:
                    pass
        
        with open(local_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    slug = entry.get('slug')
                    
                    if slug:
                        successful_slugs.add(slug)
                        all_entries.append(entry)
                        if shard_id:
                            shard_stats[shard_id]["success"] += 1
                            shard_stats[shard_id]["total"] += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {local_file}: {e}")
                    continue
    
    # Write consolidated file
    print(f"\nWriting consolidated file: {CONSOLIDATED_FILE}")
    with open(CONSOLIDATED_FILE, 'w', encoding='utf-8') as f:
        for entry in tqdm(all_entries, desc="Writing"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Load original slugs file to find failed ones
    slugs_file = Path("results/overall/grokipedia_w_license.txt")
    if not slugs_file.exists():
        print(f"\nWarning: Could not find {slugs_file}")
        print("Skipping failed slugs identification. You may need to specify the path manually.")
    else:
        print(f"\nLoading original slugs from {slugs_file}...")
        original_titles = load_original_slugs(slugs_file)
        original_slugs = {title_to_slug(title) for title in original_titles}
        
        # Find failed slugs (in original but not in successful)
        failed_slugs = original_slugs - successful_slugs
        
        print(f"\nWriting failed slugs to {FAILED_SLUGS_FILE}")
        with open(FAILED_SLUGS_FILE, 'w', encoding='utf-8') as f:
            for slug in sorted(failed_slugs):
                f.write(slug + '\n')
    
    # Write statistics
    stats = {
        "total_files": len(gcs_files),
        "total_entries": len(all_entries),
        "successful_slugs": len(successful_slugs),
        "failed_slugs": len(failed_slugs) if slugs_file.exists() else None,
        "shard_stats": dict(shard_stats)
    }
    
    print(f"\nWriting statistics to {STATS_FILE}")
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CONSOLIDATION SUMMARY")
    print("="*60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total entries: {stats['total_entries']:,}")
    print(f"Successful slugs: {stats['successful_slugs']:,}")
    if stats['failed_slugs'] is not None:
        print(f"Failed slugs: {stats['failed_slugs']:,}")
    print(f"\nConsolidated file: {CONSOLIDATED_FILE}")
    if failed_slugs:
        print(f"Failed slugs file: {FAILED_SLUGS_FILE}")
    print(f"Statistics file: {STATS_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()



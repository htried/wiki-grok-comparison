#!/usr/bin/env python3
"""
Download all edit_requests files from GCS, consolidate successful entries,
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
GCS_PREFIX = "edit_requests"
OUTPUT_DIR = Path("results/edit_requests")
CONSOLIDATED_FILE = OUTPUT_DIR / "all_edit_requests.jsonl"
FAILED_SLUGS_FILE = OUTPUT_DIR / "failed_slugs.txt"
STATS_FILE = OUTPUT_DIR / "consolidation_stats.json"

def list_gcs_files(bucket_name, prefix):
    """List all files in GCS with the given prefix."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [blob.name for blob in blobs if blob.name.endswith('.jsonl') and 'failed' not in blob.name]

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
    """Convert title to slug format (same as in edit_requests_scraper.py)."""
    import re
    # Greek letter to English name mapping
    GREEK_LETTERS = {
        'Α': 'Alpha', 'α': 'alpha',
        'Β': 'Beta', 'β': 'beta',
        'Γ': 'Gamma', 'γ': 'gamma',
        'Δ': 'Delta', 'δ': 'delta',
        'Ε': 'Epsilon', 'ε': 'epsilon',
        'Ζ': 'Zeta', 'ζ': 'zeta',
        'Η': 'Eta', 'η': 'eta',
        'Θ': 'Theta', 'θ': 'theta',
        'Ι': 'Iota', 'ι': 'iota',
        'Κ': 'Kappa', 'κ': 'kappa',
        'Λ': 'Lambda', 'λ': 'lambda',
        'Μ': 'Mu', 'μ': 'mu',
        'Ν': 'Nu', 'ν': 'nu',
        'Ξ': 'Xi', 'ξ': 'xi',
        'Ο': 'Omicron', 'ο': 'omicron',
        'Π': 'Pi', 'π': 'pi',
        'Ρ': 'Rho', 'ρ': 'rho',
        'Σ': 'Sigma', 'σ': 'sigma', 'ς': 'sigma',
        'Τ': 'Tau', 'τ': 'tau',
        'Υ': 'Upsilon', 'υ': 'upsilon',
        'Φ': 'Phi', 'φ': 'phi', 'ϕ': 'phi',
        'Χ': 'Chi', 'χ': 'chi',
        'Ψ': 'Psi', 'ψ': 'psi',
        'Ω': 'Omega', 'ω': 'omega',
    }
    
    slug = title.replace(' ', '_')
    for greek_char, english_name in GREEK_LETTERS.items():
        slug = slug.replace(greek_char, english_name)
    slug = re.sub(r'([^_])\(', r'\1_(', slug)
    return slug

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
    total_edit_requests = 0
    
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
                        # Count total edit requests
                        edit_requests = entry.get('editRequests', [])
                        total_edit_requests += len(edit_requests)
                        if shard_id:
                            shard_stats[shard_id]["success"] += 1
                            shard_stats[shard_id]["total"] += 1
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {local_file}: {e}")
                    continue
    
    # Also check for failed slugs files
    print("\nChecking for failed slugs files...")
    failed_blobs = [blob.name for blob in storage.Client().bucket(GCS_BUCKET).list_blobs(prefix=f"{GCS_PREFIX}/failed_") if blob.name.endswith('.jsonl')]
    if failed_blobs:
        print(f"Found {len(failed_blobs)} failed slugs files")
        for blob_name in failed_blobs:
            local_path = OUTPUT_DIR / blob_name.replace("/", "_")
            download_file(GCS_BUCKET, blob_name, local_path)
            with open(local_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        slug = entry.get('slug')
                        if slug:
                            failed_slugs.add(slug)
                    except json.JSONDecodeError:
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
        "total_edit_requests": total_edit_requests,
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
    print(f"Total edit requests: {stats['total_edit_requests']:,}")
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


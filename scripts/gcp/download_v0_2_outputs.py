#!/usr/bin/env python3
"""
Download all outputs from grokipedia_scraper_v0_2.py from GCS and consolidate into single JSONL files.

Downloads:
- All batch files from shard_*/ directories -> results/v0.2_scrape/scraped_data.jsonl
- All failed pages -> results/v0.2_scrape/failed_pages.jsonl
"""

import json
import logging
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GCS_BUCKET = "enwiki-structured-contents-20251028"
OUTPUT_DIR = Path("results/v0.2_scrape")
SCRAPED_OUTPUT = OUTPUT_DIR / "scraped_data.jsonl"
FAILED_OUTPUT = OUTPUT_DIR / "failed_pages.jsonl"


def list_all_blobs(bucket_name, prefix=None):
    """List all blobs in a GCS bucket with optional prefix"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist")
            return []
        
        blobs = list(bucket.list_blobs(prefix=prefix))
        logger.info(f"Found {len(blobs)} blobs with prefix '{prefix}'" if prefix else f"Found {len(blobs)} total blobs")
        return blobs
    except Exception as e:
        logger.error(f"Failed to list blobs: {e}")
        raise


def download_blob_content(bucket_name, blob_name):
    """Download content from a GCS blob"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            logger.warning(f"Blob {blob_name} does not exist")
            return None
        
        content = blob.download_as_text()
        return content
    except Exception as e:
        logger.error(f"Failed to download blob {blob_name}: {e}")
        return None


def consolidate_scraped_data():
    """Download and consolidate all scraped data batch files"""
    logger.info("=" * 60)
    logger.info("Downloading and consolidating scraped data...")
    logger.info("=" * 60)
    
    # Find all batch files (shard_*/batch_*.jsonl or scraped_data/batch_*.jsonl)
    all_blobs = list_all_blobs(GCS_BUCKET)
    
    # Filter for batch files
    batch_blobs = []
    for blob in all_blobs:
        name = blob.name
        if ('shard_' in name and '/batch_' in name and name.endswith('.jsonl')) or \
           (name.startswith('scraped_data/batch_') and name.endswith('.jsonl')):
            batch_blobs.append(name)
    
    logger.info(f"Found {len(batch_blobs)} batch files to download")
    
    if not batch_blobs:
        logger.warning("No batch files found!")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download and consolidate
    total_lines = 0
    with open(SCRAPED_OUTPUT, 'w', encoding='utf-8') as outfile:
        for blob_name in tqdm(batch_blobs, desc="Downloading batches"):
            content = download_blob_content(GCS_BUCKET, blob_name)
            if content:
                lines = content.strip().split('\n')
                valid_lines = [line for line in lines if line.strip()]
                total_lines += len(valid_lines)
                
                # Write each line (already JSON)
                for line in valid_lines:
                    if line.strip():
                        outfile.write(line.strip() + '\n')
                
                logger.debug(f"Downloaded {blob_name}: {len(valid_lines)} items")
    
    logger.info(f"✓ Consolidated {total_lines} items into {SCRAPED_OUTPUT}")
    logger.info(f"  File size: {SCRAPED_OUTPUT.stat().st_size / 1024 / 1024:.2f} MB")


def consolidate_failed_pages():
    """Download and consolidate all failed page files"""
    logger.info("=" * 60)
    logger.info("Downloading and consolidating failed pages...")
    logger.info("=" * 60)
    
    # Find all failed page files
    all_blobs = list_all_blobs(GCS_BUCKET, prefix="failed_pages/")
    
    # Filter for JSONL files
    failed_blobs = [blob.name for blob in all_blobs if blob.name.endswith('.jsonl')]
    
    logger.info(f"Found {len(failed_blobs)} failed page files to download")
    
    if not failed_blobs:
        logger.warning("No failed page files found!")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download and consolidate
    total_lines = 0
    seen_urls = set()  # Deduplicate by URL
    
    with open(FAILED_OUTPUT, 'w', encoding='utf-8') as outfile:
        for blob_name in tqdm(failed_blobs, desc="Downloading failed pages"):
            content = download_blob_content(GCS_BUCKET, blob_name)
            if content:
                lines = content.strip().split('\n')
                valid_lines = [line for line in lines if line.strip()]
                
                for line in valid_lines:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            url = item.get('url', '')
                            
                            # Deduplicate by URL
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                outfile.write(line.strip() + '\n')
                                total_lines += 1
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {blob_name}: {line[:100]}")
                            continue
                
                logger.debug(f"Downloaded {blob_name}: {len(valid_lines)} items")
    
    logger.info(f"✓ Consolidated {total_lines} unique failed pages into {FAILED_OUTPUT}")
    logger.info(f"  File size: {FAILED_OUTPUT.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main function"""
    logger.info("Starting download and consolidation of v0.2 scraper outputs...")
    logger.info(f"GCS Bucket: {GCS_BUCKET}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Consolidate scraped data
        consolidate_scraped_data()
        
        # Consolidate failed pages
        consolidate_failed_pages()
        
        logger.info("=" * 60)
        logger.info("✓ Download and consolidation complete!")
        logger.info(f"  Scraped data: {SCRAPED_OUTPUT}")
        logger.info(f"  Failed pages: {FAILED_OUTPUT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during download: {e}")
        raise


if __name__ == "__main__":
    main()


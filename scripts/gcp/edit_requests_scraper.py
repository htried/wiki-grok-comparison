"""
Grokipedia Edit Requests API Scraper
Fetches edit requests data from the Grokipedia API endpoint
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import aiohttp
from aiolimiter import AsyncLimiter
from google.cloud import storage
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

def title_to_slug(title: str) -> str:
    """Convert a title to a slug format (spaces -> underscores, handle special chars).
    
    Examples:
    - "Fullback (rugby league)" -> "Fullback_(rugby_league)"
    - "Sofía Gómez Villafañe" -> "Sofía_Gómez_Villafañe"
    - "The Silent Sea(TV series)" -> "The_Silent_Sea_(TV_series)"
    - "ΦX174" -> "Phi_X_174"
    """
    # Replace spaces with underscores
    slug = title.replace(' ', '_')
    
    # Replace Greek letters with their English names
    for greek_char, english_name in GREEK_LETTERS.items():
        slug = slug.replace(greek_char, english_name)
    
    # Add underscore before opening parenthesis if missing
    # Fix patterns like "Title(description)" -> "Title_(description)"
    slug = re.sub(r'([^_])\(', r'\1_(', slug)
    
    # The API handles URL encoding, so we don't need to do much else
    return slug


def load_slugs_from_hf():
    """Load slugs from HuggingFace dataset by extracting them from URLs.
    
    The dataset contains URLs like https://grokipedia.com/page/Slug_Name
    We extract the slug from the URL path.
    """
    try:
        from datasets import load_dataset
        
        hf_token = os.getenv('HF_API_TOKEN')
        
        # Load dataset (with or without token)
        if hf_token:
            dataset = load_dataset("stefan-it/grokipedia-urls", token=hf_token)
        else:
            dataset = load_dataset("stefan-it/grokipedia-urls")
        
        # Convert to pandas DataFrame
        if isinstance(dataset, dict):
            # If dataset has splits, use 'train' or first available
            split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
            df_urls = dataset[split_name].to_pandas()
        else:
            # Single split
            df_urls = dataset.to_pandas()
        
        urls = df_urls['url'].tolist()
        
        # Extract slugs from URLs
        # URLs are like: https://grokipedia.com/page/Slug_Name
        slugs = []
        for url in urls:
            try:
                # Extract the slug from the URL path
                # Pattern: https://grokipedia.com/page/Slug_Name
                if '/page/' in url:
                    slug_part = url.split('/page/')[-1].split('?')[0].split('#')[0]
                    # URL decode if needed
                    from urllib.parse import unquote
                    slug = unquote(slug_part)
                    slugs.append(slug)
                else:
                    logger.warning(f"Unexpected URL format: {url}")
            except Exception as e:
                logger.warning(f"Failed to extract slug from URL {url}: {e}")
                continue
        
        logger.info(f"Loaded {len(slugs):,} slugs from HuggingFace dataset")
        return slugs
    except Exception as e:
        logger.error(f"Failed to load slugs from HuggingFace: {e}")
        raise


def load_slugs_from_file(file_path: str):
    """Load titles from a text file (one title per line) and convert to slugs.
    
    Supports:
    - Local file path
    - GCS path (gs://bucket/path)
    - Text file with one title per line (titles may have spaces)
    - Special value "hf" or "huggingface" to load from HuggingFace dataset
    
    Titles are automatically converted to slug format (spaces -> underscores).
    """
    # Check if user wants to load from HuggingFace
    if file_path.lower() in ['hf', 'huggingface', 'hf_dataset']:
        return load_slugs_from_hf()
    
    try:
        # Handle GCS paths
        if file_path.startswith('gs://'):
            # Download from GCS to temp file
            from google.cloud import storage
            gcs_path = file_path.replace('gs://', '')
            parts = gcs_path.split('/', 1)
            bucket_name = parts[0]
            blob_path = parts[1] if len(parts) > 1 else ""
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Read from temp file
            with open(tmp_path, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
            
            # Clean up temp file
            os.unlink(tmp_path)
        else:
            # Local file
            with open(file_path, 'r', encoding='utf-8') as f:
                titles = [line.strip() for line in f if line.strip()]
        
        # Convert titles to slugs
        slugs = [title_to_slug(title) for title in titles]
        logger.info(f"Loaded {len(slugs):,} slugs from {file_path}")
        return slugs
    
    except Exception as e:
        logger.error(f"Error loading slugs from {file_path}: {e}")
        raise


async def fetch_edit_requests(session, limiter, slug, config, skip_on_error=True):
    """Fetch edit requests for a single slug from the Grokipedia API.
    
    Args:
        session: aiohttp ClientSession
        limiter: AsyncLimiter for rate limiting
        slug: Page slug to fetch
        config: Configuration dict
        skip_on_error: If True, return error dict instead of raising
    
    Returns:
        dict with 'success', 'slug', 'editRequests', 'totalCount', etc.
    """
    try:
        # URL encode the slug
        encoded_slug = quote(slug, safe='')
        url = f"https://grokipedia.com/api/list-edit-requests-by-slug?slug={encoded_slug}&limit=500"
        
        proxy_url = config.get('proxy_url')
        
        async with limiter:
            async with session.get(
                url,
                timeout=config.get('api_timeout', 30),
                headers={'Accept': 'application/json'},
                proxy=proxy_url,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if we got valid data
                    if not isinstance(data, dict):
                        return {
                            'success': False,
                            'error': 'invalid_response_format',
                            'slug': slug,
                            'status_code': 200
                        }
                    
                    # Extract edit requests
                    edit_requests = data.get('editRequests', [])
                    total_count = data.get('totalCount', 0)
                    has_more = data.get('hasMore', False)
                    
                    # Only store what we need - the edit requests array
                    return {
                        'success': True,
                        'slug': slug,
                        'editRequests': edit_requests,
                        'totalCount': total_count,
                        'hasMore': has_more
                    }
                elif response.status == 404:
                    return {
                        'success': False,
                        'error': 'page_not_found',
                        'slug': slug,
                        'status_code': 404
                    }
                else:
                    return {
                        'success': False,
                        'error': f'http_error_{response.status}',
                        'slug': slug,
                        'status_code': response.status
                    }
    except asyncio.TimeoutError:
        if skip_on_error:
            # Retry with exponential backoff
            for delay in [2, 4, 8]:
                await asyncio.sleep(delay)
                try:
                    async with limiter:
                        async with session.get(
                            url,
                            timeout=config.get('api_timeout', 30),
                            headers={'Accept': 'application/json'},
                            proxy=proxy_url,
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                edit_requests = data.get('editRequests', [])
                                total_count = data.get('totalCount', 0)
                                has_more = data.get('hasMore', False)
                                return {
                                    'success': True,
                                    'slug': slug,
                                    'editRequests': edit_requests,
                                    'totalCount': total_count,
                                    'hasMore': has_more
                                }
                except:
                    continue
            return {
                'success': False,
                'error': 'timeout_retries_exhausted',
                'slug': slug
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'slug': slug
        }


def upload_to_gcs(bucket_name, blob_name, content, project_id=None):
    """Upload content to GCS bucket"""
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist")
            return False
        
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type='application/jsonl')
        logger.info(f"✓ Successfully uploaded {blob_name} to {bucket_name} ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def fetch_edit_requests_phase(slugs, config, start_index=0, shard_id=None):
    """Run edit requests fetching phase"""
    limiter = AsyncLimiter(max_rate=config['rate_limit'], time_period=60)
    connector = aiohttp.TCPConnector(
        limit=config['max_concurrent'],
        force_close=True,
        enable_cleanup_closed=True
    )
    
    batch_size = config.get('batch_size', 1000)
    skip_on_error = config.get('skip_on_error', True)
    gcs_bucket = config.get('gcs_bucket')
    gcs_project = config.get('gcs_project')
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=config.get('api_timeout', 60))
    ) as session:
        success_count = 0
        fail_count = 0
        results_data = []
        failed_slugs = []
        batch_count = 0
        last_save_index = 0
        items_processed_count = 0
        
        # Create progress bar
        desc = f"Fetching Edit Requests (Shard {shard_id})" if shard_id is not None else "Fetching Edit Requests"
        pbar = tqdm(total=len(slugs), desc=desc, initial=0)
        
        # Process in batches
        for i in range(0, len(slugs), config['max_concurrent']):
            batch = slugs[i:i + config['max_concurrent']]
            
            # Create tasks for concurrent requests
            tasks = [fetch_edit_requests(session, limiter, slug, config, skip_on_error) for slug in batch]
            
            # Use gather with return_exceptions to handle individual failures
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Batch gather failed: {e}")
                batch_results = [{'success': False, 'error': str(e), 'slug': slug} for slug in batch]
            
            # Process results
            for j, result in enumerate(batch_results):
                # Handle exceptions
                if isinstance(result, Exception):
                    fail_count += 1
                    slug = batch[j] if j < len(batch) else 'unknown'
                    failed_slugs.append({
                        'slug': slug,
                        'error': str(result),
                        'failed_at': datetime.now().isoformat()
                    })
                    continue
                
                if result['success']:
                    success_count += 1
                    items_processed_count += 1
                    results_data.append({
                        'slug': result['slug'],
                        'editRequests': result['editRequests'],
                        'totalCount': result['totalCount'],
                        'hasMore': result.get('hasMore', False),
                        'fetched_at': datetime.now().isoformat()
                    })
                else:
                    fail_count += 1
                    failed_slugs.append({
                        'slug': result.get('slug', 'unknown'),
                        'error': result.get('error', 'unknown'),
                        'status_code': result.get('status_code'),
                        'failed_at': datetime.now().isoformat()
                    })
            
            pbar.update(len(batch))
            pbar.set_postfix({
                'success': success_count,
                'failed': fail_count,
                'fail_rate': f'{fail_count/(success_count+fail_count)*100:.1f}%' if (success_count+fail_count) > 0 else '0%'
            })
            
            # Save batch when we hit the batch_size limit
            should_save = False
            save_reason = ""
            
            if len(results_data) >= batch_size:
                should_save = True
                save_reason = "batch_size"
            elif len(results_data) >= 100 and (i - last_save_index) >= 5000:
                should_save = True
                save_reason = "progress"
            
            if should_save and gcs_bucket:
                # Prepare batch data
                batch_to_save = results_data[:batch_size]
                remaining_data = results_data[batch_size:]
                
                # Convert to JSONL
                jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in batch_to_save]
                jsonl_content = '\n'.join(jsonl_lines) + '\n'
                
                # Generate blob name
                shard_suffix = f"_shard{shard_id}" if shard_id is not None else ""
                blob_name = f"edit_requests/edit_requests_{start_index + items_processed_count - len(batch_to_save)}_{start_index + items_processed_count - 1}{shard_suffix}.jsonl"
                
                # Upload to GCS
                if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                    batch_count += 1
                    logger.info(f"Saved batch {batch_count} ({save_reason}): {len(batch_to_save)} items to {blob_name}")
                    last_save_index = i
                    # Clear saved data from memory
                    results_data = remaining_data
                else:
                    logger.warning(f"Failed to upload batch {batch_count + 1} to GCS")
        
        pbar.close()
        
        # Save remaining data
        if results_data and gcs_bucket:
            jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in results_data]
            jsonl_content = '\n'.join(jsonl_lines) + '\n'
            
            shard_suffix = f"_shard{shard_id}" if shard_id is not None else ""
            final_start = start_index + items_processed_count - len(results_data)
            final_end = start_index + items_processed_count - 1
            blob_name = f"edit_requests/edit_requests_{final_start}_{final_end}{shard_suffix}.jsonl"
            
            if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                batch_count += 1
                logger.info(f"Saved final batch {batch_count}: {len(results_data)} items to {blob_name}")
        
        # Save failed slugs if any
        if failed_slugs and gcs_bucket:
            failed_jsonl_lines = [json.dumps(item, ensure_ascii=False) for item in failed_slugs]
            failed_jsonl_content = '\n'.join(failed_jsonl_lines) + '\n'
            
            shard_suffix = f"_shard{shard_id}" if shard_id is not None else ""
            failed_blob_name = f"edit_requests/failed_edit_requests{shard_suffix}.jsonl"
            
            if upload_to_gcs(gcs_bucket, failed_blob_name, failed_jsonl_content, gcs_project):
                logger.info(f"Saved {len(failed_slugs)} failed slugs to {failed_blob_name}")
        
        logger.info(f"Phase complete: {success_count:,} successful, {fail_count:,} failed")
        return success_count


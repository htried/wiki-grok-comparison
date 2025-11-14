"""
Grokipedia Fixed Issues API Scraper
Fetches fixedIssues data from the Grokipedia API endpoint
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


def load_slugs_from_file(file_path: str):
    """Load titles from a text file (one title per line) and convert to slugs.
    
    Supports:
    - Local file path
    - GCS path (gs://bucket/path)
    - Text file with one title per line (titles may have spaces)
    
    Titles are automatically converted to slug format (spaces -> underscores).
    """
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
            
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
            temp_path = temp_file.name
            temp_file.close()
            
            blob.download_to_filename(temp_path)
            file_path = temp_path
            logger.info(f"Downloaded slugs file from GCS to {temp_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Slugs file not found: {file_path}")
        
        titles = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                title = line.strip()
                if title:
                    titles.append(title)
        
        # Convert titles to slugs (spaces -> underscores)
        slugs = [title_to_slug(title) for title in titles]
        
        logger.info(f"Loaded {len(slugs)} titles from file: {file_path}")
        logger.info(f"Sample slugs: {slugs[:3] if len(slugs) >= 3 else slugs}")
        return slugs
    except Exception as e:
        logger.error(f"Failed to load slugs from file '{file_path}': {e}")
        raise


async def fetch_fixed_issues(session, limiter, slug, config, skip_on_error=True):
    """Fetch fixedIssues for a single slug from the Grokipedia API
    
    Uses minimal parameters to reduce data transfer:
    - includeContent=false: Excludes full content (saves ~90KB)
    - No validateLinks: Saves ~8KB per request
    Note: API doesn't support fetching ONLY fixedIssues, so we still get
    citations, images, slug, title, description, metadata, stats, linkedPages
    """
    proxy_url = config.get('proxy_url')
    base_url = "https://grokipedia.com/api/page"
    
    # URL encode the slug
    encoded_slug = quote(slug, safe='')
    url = f"{base_url}?slug={encoded_slug}&includeContent=false"
    
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
                    # Check if page was found
                    if not data.get('found', True) or data.get('page') is None:
                        return {
                            'success': False,
                            'error': 'page_not_found',
                            'slug': slug,
                            'status_code': 200
                        }
                    # Extract fixedIssues from response
                    fixed_issues = data.get('page', {}).get('fixedIssues', [])
                    # Only store what we need - don't store full response to minimize data transfer
                    return {
                        'success': True,
                        'slug': slug,
                        'fixedIssues': fixed_issues,
                        'fixedIssuesCount': len(fixed_issues)
                    }
                elif response.status == 404:
                    return {
                        'success': False,
                        'error': 'not_found',
                        'slug': slug,
                        'status_code': 404
                    }
                else:
                    return {
                        'success': False,
                        'error': f'status_{response.status}',
                        'slug': slug,
                        'status_code': response.status
                    }
    except asyncio.TimeoutError:
        if skip_on_error:
            return {
                'success': False,
                'error': 'timeout',
                'slug': slug
            }
        else:
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
                                fixed_issues = data.get('page', {}).get('fixedIssues', [])
                                return {
                                    'success': True,
                                    'slug': slug,
                                    'fixedIssues': fixed_issues,
                                    'fixedIssuesCount': len(fixed_issues),
                                    'data': data
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


async def fetch_fixed_issues_phase(slugs, config, start_index=0, shard_id=None):
    """Run fixed issues fetching phase"""
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
        desc = f"Fetching Fixed Issues (Shard {shard_id})" if shard_id is not None else "Fetching Fixed Issues"
        pbar = tqdm(total=len(slugs), desc=desc, initial=0)
        
        # Process in batches
        for i in range(0, len(slugs), config['max_concurrent']):
            batch = slugs[i:i + config['max_concurrent']]
            
            # Create tasks for concurrent requests
            tasks = [fetch_fixed_issues(session, limiter, slug, config, skip_on_error) for slug in batch]
            
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
                        'fixedIssues': result['fixedIssues'],
                        'fixedIssuesCount': result['fixedIssuesCount'],
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
            
            if should_save and results_data:
                # Calculate absolute indices for blob naming
                batch_start = start_index + (items_processed_count - len(results_data))
                batch_end = start_index + items_processed_count - 1
                
                # Prepare JSONL content
                jsonl_content = '\n'.join(json.dumps(item) for item in results_data)
                
                # Upload to GCS or save locally
                if gcs_bucket:
                    # Construct blob name
                    if shard_id is not None:
                        blob_name = f'fixed_issues/shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                    else:
                        blob_name = f'fixed_issues/batch_{batch_start}_{batch_end}.jsonl'
                    
                    logger.info(f"Preparing to upload batch: blob_name='{blob_name}', batch_start={batch_start}, batch_end={batch_end}, items={len(results_data)}")
                    
                    if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                        logger.info(f"✓ Uploaded batch to GCS: {blob_name} ({len(results_data)} items)")
                    else:
                        # Fallback to local save if GCS fails
                        output_dir = Path(config.get('output_dir', 'fixed_issues_data'))
                        if shard_id is not None:
                            output_dir = output_dir / f'shard_{shard_id}'
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                        with open(batch_file, 'w') as f:
                            f.write(jsonl_content)
                        logger.info(f"Saved batch locally (GCS failed): {batch_file} ({len(results_data)} items)")
                else:
                    # Save locally if no GCS bucket configured
                    output_dir = Path(config.get('output_dir', 'fixed_issues_data'))
                    if shard_id is not None:
                        output_dir = output_dir / f'shard_{shard_id}'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                    with open(batch_file, 'w') as f:
                        f.write(jsonl_content)
                    logger.info(f"Saved batch: {batch_file} ({len(results_data)} items) - Reason: {save_reason}")
                
                # Save checkpoint
                checkpoint = {
                    'last_processed_index': i,
                    'success_count': success_count,
                    'fail_count': fail_count,
                    'total_processed': i + len(batch),
                    'last_save_index': batch_end,
                    'batch_count': batch_count,
                    'save_reason': save_reason,
                    'shard_id': shard_id
                }
                
                checkpoint_file = f'fixed_issues_checkpoint_shard_{shard_id}.json' if shard_id is not None else 'fixed_issues_checkpoint.json'
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                
                last_save_index = i
                batch_count += 1
                results_data = []
            
            # Save failed slugs periodically
            if len(failed_slugs) >= 1000:
                failed_content = '\n'.join(json.dumps(item) for item in failed_slugs)
                
                if gcs_bucket:
                    failed_blob = f'fixed_issues/failed/failed_shard_{shard_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl' if shard_id is not None else f'fixed_issues/failed/failed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
                    if upload_to_gcs(gcs_bucket, failed_blob, failed_content, gcs_project):
                        logger.info(f"Uploaded {len(failed_slugs)} failed slugs to GCS")
                        failed_slugs = []
                else:
                    failed_file = Path(f'fixed_issues_failed_partial_shard_{shard_id}.jsonl' if shard_id is not None else 'fixed_issues_failed_partial.jsonl')
                    with open(failed_file, 'a') as f:
                        f.write(failed_content + '\n')
                    logger.info(f"Saved {len(failed_slugs)} failed slugs to partial file")
                    failed_slugs = []
            
            # Adaptive delay - increase if failure rate is high
            if (success_count + fail_count) > 0:
                failure_rate = fail_count / (success_count + fail_count)
                if failure_rate > 0.1:  # More than 10% failure rate
                    delay = min(2.0, failure_rate * 10)
                    await asyncio.sleep(delay)
        
        pbar.close()
        
        # Save any remaining results
        if results_data:
            batch_start = start_index + (items_processed_count - len(results_data))
            batch_end = start_index + items_processed_count - 1
            
            jsonl_content = '\n'.join(json.dumps(item) for item in results_data)
            
            if gcs_bucket:
                if shard_id is not None:
                    blob_name = f'fixed_issues/shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                else:
                    blob_name = f'fixed_issues/batch_{batch_start}_{batch_end}.jsonl'
                
                if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                    logger.info(f"✓ Uploaded final batch to GCS: {blob_name} ({len(results_data)} items)")
                else:
                    output_dir = Path(config.get('output_dir', 'fixed_issues_data'))
                    if shard_id is not None:
                        output_dir = output_dir / f'shard_{shard_id}'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                    with open(batch_file, 'w') as f:
                        f.write(jsonl_content)
                    logger.info(f"Saved final batch locally: {batch_file}")
            else:
                output_dir = Path(config.get('output_dir', 'fixed_issues_data'))
                if shard_id is not None:
                    output_dir = output_dir / f'shard_{shard_id}'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                with open(batch_file, 'w') as f:
                    f.write(jsonl_content)
                logger.info(f"Saved final batch: {batch_file}")
        
        # Save final stats
        stats = {
            'total_processed': len(slugs),
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': f'{success_count/len(slugs)*100:.2f}%' if len(slugs) > 0 else '0%',
            'completed_at': datetime.now().isoformat(),
            'shard_id': shard_id
        }
        
        stats_file = f'fixed_issues_stats_shard_{shard_id}.json' if shard_id is not None else 'fixed_issues_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Fixed issues fetching complete: {success_count} successful, {fail_count} failed out of {len(slugs)} slugs")
        return success_count


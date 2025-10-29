"""
Grokipedia Scraper Module
Extract shared logic from notebook for use in sharded deployment
"""

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup
from google.cloud import storage
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_grokipedia_html(html_content, url, title=None):
    """Parse grokipedia HTML and extract structured data"""
    if title is None:
        title = url.split('/page/')[-1]
    
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {
        'title': title,
        'url': url,
        'main_title': None,
        'sections': [],
        'paragraphs': [],
        'tables': [],
        'references': [],
        'metadata': {
            'has_edits': False,  # Flag to indicate if edits exist
            'fact_check_timestamp': None
        }
    }
    
    # Find article container
    article = soup.find('div', class_='mx-auto max-w-[850px]')
    if not article:
        return data
    
    # Extract main title (h1)
    h1 = article.find('h1')
    if h1:
        data['main_title'] = h1.get_text(strip=True)
    
    # Check if edits exist (look for "See Edits" button anywhere on page)
    edits_btn = soup.find('button', string=lambda x: x and 'See Edits' in str(x))
    if edits_btn:
        data['metadata']['has_edits'] = True
        
        # Extract edit count if visible in button text
        btn_text = edits_btn.get_text(strip=True)
        if '(' in btn_text and ')' in btn_text:
            import re
            count_match = re.search(r'\((\d+)\)', btn_text)
            if count_match:
                data['metadata']['edits_count'] = int(count_match.group(1))
    
    # Extract sections with proper content
    for heading in article.find_all(['h1', 'h2', 'h3'], id=True):
        section_data = {
            'level': heading.name,
            'id': heading.get('id'),
            'title': heading.get_text(strip=True),
            'content': []
        }
        
        # Walk through siblings after heading
        current = heading.next_sibling
        while current:
            if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3']:
                if current.name <= heading.name:
                    break
            
            if hasattr(current, 'name'):
                if current.name == 'span' and 'mb-4' in (current.get('class') or []):
                    text = current.get_text(strip=True)
                    if text:
                        # Join sentences with proper spacing
                        section_data['content'].append({'type': 'paragraph', 'text': ' '.join(text.split())})
                elif current.name == 'ul':
                    items = [li.get_text(strip=True) for li in current.find_all('li')]
                    if items:
                        section_data['content'].append({'type': 'list', 'items': items})
                elif current.name == 'ol':
                    items = [li.get_text(strip=True) for li in current.find_all('li')]
                    if items:
                        section_data['content'].append({'type': 'ordered_list', 'items': items})
            
            current = current.next_sibling
        
        data['sections'].append(section_data)
    
    # Extract paragraphs with proper spacing
    for span in article.find_all('span', class_='mb-4'):
        text = span.get_text(strip=True)
        # Normalize whitespace
        text = ' '.join(text.split())
        if text and text not in data['paragraphs']:
            data['paragraphs'].append(text)
    
    # Extract tables
    for table in article.find_all('table'):
        table_data = []
        headers = []
        
        if table.find('thead'):
            for th in table.find('thead').find_all('th'):
                headers.append(th.get_text(strip=True))
        
        if table.find('tbody'):
            for tr in table.find('tbody').find_all('tr'):
                row = []
                for td in tr.find_all('td'):
                    row.append(td.get_text(strip=True))
                if row:
                    table_data.append(row)
        
        if headers or table_data:
            data['tables'].append({'headers': headers, 'rows': table_data})
    
    # Extract references WITH links
    references_section = soup.find('div', id='references')
    if references_section:
        for li in references_section.find_all('li'):
            ref_text = li.get_text(strip=True)
            ref_link = None
            
            link = li.find('a')
            if link and link.get('href'):
                ref_link = {'href': link.get('href'), 'text': link.get_text(strip=True)}
            
            if ref_text:
                data['references'].append({'text': ref_text, 'link': ref_link})
    
    # Remove references from paragraphs
    data['paragraphs'] = [p for p in data['paragraphs'] 
                          if not any(ref['text'].split()[0:3] == p.split()[0:3] for ref in data['references'])]
    
    return data


async def discover_page_exists(session, limiter, title, config):
    """Check if a grokipedia page exists using HEAD request"""
    url = f"https://grokipedia.com/page/{quote(title)}"
    
    try:
        async with limiter:
            async with session.head(url, timeout=config['discovery_timeout']) as response:
                status = response.status
                if status == 200:
                    return {'title': title, 'url': url, 'status': 'exists', 'checked_at': datetime.now().isoformat()}
                elif status == 404:
                    return {'title': title, 'url': url, 'status': 'not_found', 'checked_at': datetime.now().isoformat()}
                elif status == 429:  # Rate limited
                    await asyncio.sleep(5)
                    return {'title': title, 'url': url, 'status': 'rate_limited', 'checked_at': datetime.now().isoformat()}
                else:
                    return {'title': title, 'url': url, 'status': f'error_{status}', 'checked_at': datetime.now().isoformat()}
    except Exception as e:
        return {'title': title, 'url': url, 'status': 'error', 'error': str(e), 'checked_at': datetime.now().isoformat()}


async def discovery_phase(titles, config, shard_id=None):
    """Run discovery phase to find which pages exist"""
    limiter = AsyncLimiter(max_rate=config['rate_limit'], time_period=60)
    connector = aiohttp.TCPConnector(limit=config['max_concurrent'])
    
    # Create shard-specific output directory
    output_dir = Path(config.get('output_dir', 'discovered_titles'))
    if shard_id is not None:
        output_dir = output_dir / f'shard_{shard_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        discovered_count = 0
        not_found_count = 0
        error_count = 0
        results = []
        
        # Create progress bar
        desc = f"Discovery (Shard {shard_id})" if shard_id is not None else "Discovery"
        pbar = tqdm(total=len(titles), desc=desc, initial=0)
        
        # Process in batches
        for i in range(0, len(titles), config['max_concurrent']):
            batch = titles[i:i + config['max_concurrent']]
            
            # Create tasks for concurrent requests
            tasks = [discover_page_exists(session, limiter, title, config) for title in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Process results
            for result in batch_results:
                if result['status'] == 'exists':
                    discovered_count += 1
                elif result['status'] == 'not_found':
                    not_found_count += 1
                else:
                    error_count += 1
                
                results.append(result)
            
            pbar.update(len(batch))
            pbar.set_postfix({
                'found': discovered_count,
                'not_found': not_found_count,
                'errors': error_count
            })
            
            # Add delay between batches to avoid rate limiting
            if i % config['max_concurrent'] < config['max_concurrent'] - 1:
                await asyncio.sleep(config.get('batch_delay', 1))
            
            # Save batch if we've accumulated enough results
            if len(results) >= config['discovery_batch_size']:
                # Save to JSONL
                batch_num = (i // config['discovery_batch_size']) + 1
                batch_start = (batch_num - 1) * config['discovery_batch_size']
                batch_end = batch_start + len(results)
                
                batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                with open(batch_file, 'w') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')
                
                # Save checkpoint
                if shard_id is not None:
                    checkpoint_file = f'discovery_checkpoint_shard_{shard_id}.json'
                else:
                    checkpoint_file = 'discovery_checkpoint.json'
                
                checkpoint = {
                    'last_processed_index': i,
                    'discovered_count': discovered_count,
                    'not_found_count': not_found_count,
                    'error_count': error_count,
                    'total_processed': i + len(batch),
                    'shard_id': shard_id
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                
                results = []
        
        # Save any remaining results
        if results:
            batch_num = (len(titles) // config['discovery_batch_size'])
            batch_start = batch_num * config['discovery_batch_size']
            batch_end = batch_start + len(results)
            
            batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
            with open(batch_file, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
        
        pbar.close()
        
        # Save final stats
        if shard_id is not None:
            stats_file = f'discovery_stats_shard_{shard_id}.json'
        else:
            stats_file = 'discovery_stats.json'
        
        stats = {
            'total_checked': len(titles),
            'discovered': discovered_count,
            'not_found': not_found_count,
            'errors': error_count,
            'completed_at': datetime.now().isoformat(),
            'shard_id': shard_id
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Discovery complete: Found {discovered_count} existing pages out of {len(titles)} checked")
        return discovered_count


async def scrape_page(session, limiter, url, config, skip_on_error=True):
    """Scrape a single grokipedia page"""
    proxy_url = config.get('proxy_url')
    try:
        title = url.split('/page/')[-1]
        async with limiter:
            async with session.get(
                url,
                timeout=config.get('scraping_timeout', 60),
                headers={'Accept-Encoding': 'gzip, deflate'},
                proxy=proxy_url,
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    data = parse_grokipedia_html(html, url, title)
                    return {'success': True, 'data': data}
                elif response.status == 404:
                    return {'success': False, 'error': 'not_found', 'title': title, 'url': url}
                else:
                    return {'success': False, 'error': f'status_{response.status}', 'title': title, 'url': url}
    except asyncio.TimeoutError:
        if skip_on_error:
            return {'success': False, 'error': 'timeout', 'title': title, 'url': url}
        else:
            for delay in [2, 4, 8]:
                await asyncio.sleep(delay)
                try:
                    async with limiter:
                        async with session.get(
                            url,
                            timeout=config.get('scraping_timeout', 60),
                            headers={'Accept-Encoding': 'gzip, deflate'},
                            proxy=proxy_url,
                        ) as response:
                            if response.status == 200:
                                html = await response.text()
                                data = parse_grokipedia_html(html, url, title)
                                return {'success': True, 'data': data}
                except:
                    continue
            return {'success': False, 'error': 'timeout_retries_exhausted', 'title': title, 'url': url}
    except Exception as e:
        return {'success': False, 'error': str(e), 'title': title, 'url': url}


def upload_to_gcs(bucket_name, blob_name, content, project_id=None):
    """Upload content to Google Cloud Storage"""
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type='application/jsonl')
        return True
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        return False


async def scraping_phase(urls, config, start_index=0, shard_id=None):
    """Run scraping phase to extract data from discovered pages"""
    limiter = AsyncLimiter(max_rate=config['rate_limit'], time_period=60)
    connector = aiohttp.TCPConnector(
        limit=config['max_concurrent'], 
        force_close=True, 
        enable_cleanup_closed=True
    )
    
    batch_size = config.get('scraping_batch_size', 300)
    skip_on_error = config.get('skip_on_error', True)
    gcs_bucket = config.get('gcs_bucket')
    gcs_project = config.get('gcs_project')
    
    async with aiohttp.ClientSession(
        connector=connector, 
        timeout=aiohttp.ClientTimeout(total=config.get('scraping_timeout', 90))
    ) as session:
        success_count = 0
        fail_count = 0
        scraped_data = []
        failed_pages = []
        batch_count = 0
        last_save_index = 0
        
        # Create progress bar
        desc = f"Scraping (Shard {shard_id})" if shard_id is not None else "Scraping"
        pbar = tqdm(total=len(urls), desc=desc, initial=start_index)
        
        # Process in batches
        for i in range(start_index, len(urls), config['max_concurrent']):
            batch = urls[i:i + config['max_concurrent']]
            
            # Create tasks for concurrent requests
            tasks = [scrape_page(session, limiter, url, config, skip_on_error) for url in batch]
            
            # Use gather with return_exceptions to handle individual failures
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Batch gather failed: {e}")
                batch_results = [{'success': False, 'error': str(e), 'title': 'batch_error', 'url': url} for url in batch]
            
            # Process results
            for j, result in enumerate(batch_results):
                # Handle exceptions
                if isinstance(result, Exception):
                    fail_count += 1
                    url = batch[j] if j < len(batch) else 'unknown'
                    failed_pages.append({
                        'url': url,
                        'error': str(result),
                        'failed_at': datetime.now().isoformat()
                    })
                    continue
                
                if result['success']:
                    success_count += 1
                    scraped_data.append({
                        'title': result['data']['title'],
                        'url': result['data']['url'],
                        'data': result['data'],
                        'scraped_at': datetime.now().isoformat()
                    })
                else:
                    fail_count += 1
                    failed_pages.append({
                        'url': result.get('url', 'unknown'),
                        'title': result.get('title', 'unknown'),
                        'error': result.get('error', 'unknown'),
                        'failed_at': datetime.now().isoformat()
                    })
            
            pbar.update(len(batch))
            pbar.set_postfix({
                'success': success_count,
                'failed': fail_count,
                'fail_rate': f'{fail_count/(success_count+fail_count)*100:.1f}%' if (success_count+fail_count) > 0 else '0%'
            })
            
            # Save batch when we hit the batch_size limit OR every 100 successful items
            should_save = False
            save_reason = ""
            
            if len(scraped_data) >= batch_size:
                should_save = True
                save_reason = "batch_size"
            elif len(scraped_data) >= 100 and (i - last_save_index) >= 5000:
                should_save = True
                save_reason = "progress"
            
            if should_save and scraped_data:
                batch_start = start_index + (batch_count * batch_size)
                batch_end = batch_start + len(scraped_data)
                
                # Prepare JSONL content
                jsonl_content = '\n'.join(json.dumps(item) for item in scraped_data)
                
                # Upload to GCS or save locally
                if gcs_bucket:
                    blob_name = f'scraped_data/batch_{batch_start}_{batch_end}.jsonl'
                    if shard_id is not None:
                        blob_name = f'shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                    
                    if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                        logger.info(f"Uploaded batch to GCS: {blob_name} ({len(scraped_data)} items)")
                    else:
                        # Fallback to local save if GCS fails
                        output_dir = Path(config.get('output_dir', 'scraped_data'))
                        if shard_id is not None:
                            output_dir = output_dir / f'shard_{shard_id}'
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                        with open(batch_file, 'w') as f:
                            f.write(jsonl_content)
                        logger.info(f"Saved batch locally (GCS failed): {batch_file} ({len(scraped_data)} items)")
                else:
                    # Save locally if no GCS bucket configured
                    output_dir = Path(config.get('output_dir', 'scraped_data'))
                    if shard_id is not None:
                        output_dir = output_dir / f'shard_{shard_id}'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                    with open(batch_file, 'w') as f:
                        f.write(jsonl_content)
                    logger.info(f"Saved batch: {batch_file} ({len(scraped_data)} items) - Reason: {save_reason}")
                
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
                
                checkpoint_file = f'scraping_checkpoint_shard_{shard_id}.json' if shard_id is not None else 'scraping_checkpoint.json'
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                
                last_save_index = i
                batch_count += 1
                scraped_data = []
            
            # Save failed pages periodically
            if len(failed_pages) >= 1000:
                failed_content = '\n'.join(json.dumps(item) for item in failed_pages)
                
                if gcs_bucket:
                    failed_blob = f'failed_pages/failed_shard_{shard_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl' if shard_id is not None else f'failed_pages/failed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
                    if upload_to_gcs(gcs_bucket, failed_blob, failed_content, gcs_project):
                        logger.info(f"Uploaded {len(failed_pages)} failed pages to GCS")
                        failed_pages = []
                else:
                    failed_file = Path(f'scraping_failed_partial_shard_{shard_id}.jsonl' if shard_id is not None else 'scraping_failed_partial.jsonl')
                    with open(failed_file, 'a') as f:
                        f.write(failed_content + '\n')
                    logger.info(f"Saved {len(failed_pages)} failed pages to partial file")
                    failed_pages = []
            
            # Adaptive delay - increase if failure rate is high
            failure_rate = fail_count / (success_count + fail_count) if (success_count + fail_count) > 0 else 0
            if failure_rate > 0.3:
                delay = 1.0
                logger.warning(f"High failure rate ({failure_rate*100:.1f}%), increasing delay to {delay}s")
            else:
                delay = config.get('batch_delay', 0.1)
            
            await asyncio.sleep(delay)
        
        # Save any remaining results
        if scraped_data:
            batch_start = start_index + (batch_count * batch_size)
            batch_end = batch_start + len(scraped_data)
            
            jsonl_content = '\n'.join(json.dumps(item) for item in scraped_data)
            
            if gcs_bucket:
                blob_name = f'scraped_data/batch_{batch_start}_{batch_end}.jsonl'
                if shard_id is not None:
                    blob_name = f'shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                
                if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                    logger.info(f"Uploaded final batch to GCS: {blob_name} ({len(scraped_data)} items)")
            else:
                output_dir = Path(config.get('output_dir', 'scraped_data'))
                if shard_id is not None:
                    output_dir = output_dir / f'shard_{shard_id}'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                with open(batch_file, 'w') as f:
                    f.write(jsonl_content)
                logger.info(f"Saved final batch: {batch_file} ({len(scraped_data)} items)")
        
        # Save all failed pages
        if failed_pages:
            failed_content = '\n'.join(json.dumps(item) for item in failed_pages)
            
            if gcs_bucket:
                failed_blob = f'failed_pages/failed_shard_{shard_id}_final.jsonl' if shard_id is not None else 'failed_pages/failed_final.jsonl'
                upload_to_gcs(gcs_bucket, failed_blob, failed_content, gcs_project)
                logger.info(f"Uploaded {len(failed_pages)} failed pages to GCS")
            else:
                failed_file = Path(f'scraping_failed_shard_{shard_id}.jsonl' if shard_id is not None else 'scraping_failed.jsonl')
                with open(failed_file, 'w') as f:
                    f.write(failed_content)
                logger.info(f"Saved {len(failed_pages)} failed pages")
        
        pbar.close()
        
        # Save final stats
        stats = {
            'total_scraped': len(urls),
            'success': success_count,
            'failed': fail_count,
            'completed_at': datetime.now().isoformat(),
            'shard_id': shard_id
        }
        
        stats_file = f'scraping_stats_shard_{shard_id}.json' if shard_id is not None else 'scraping_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Scraping complete: Successfully scraped {success_count} pages out of {len(urls)} attempted")
        return success_count


def load_titles(pkl_file='enwiki_titles_20251027.pkl'):
    """Load and sort enwiki titles from pandas DataFrame"""
    df = pd.read_pickle(pkl_file)
    titles = df['page_title'].tolist()
    titles_sorted = sorted(titles)
    logger.info(f"Loaded {len(titles_sorted)} titles from DataFrame")
    return titles_sorted


def load_urls_from_hf():
    """Load URLs from HuggingFace dataset"""
    try:
        import os

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
        logger.info(f"Loaded {len(urls)} URLs from HuggingFace dataset")
        return urls
    except Exception as e:
        logger.error(f"Failed to load URLs from HuggingFace: {e}")
        raise


if __name__ == '__main__':
    # Configuration
    config = {
        'max_concurrent': 20,
        'rate_limit': 300,
        'discovery_timeout': 10,
        'discovery_batch_size': 5000,
        'batch_delay': 1,
        'output_dir': 'discovered_titles'
    }
    
    titles = load_titles()
    discovered_count = asyncio.run(discovery_phase(titles, config))
    print(f"Found {discovered_count} existing pages")


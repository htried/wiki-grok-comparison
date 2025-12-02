"""
Grokipedia Scraper Module v0.2
Improved version with better parsing for pages like https://grokipedia.com/page/%E2%89%A1
Extracts links, references, and content more comprehensively
"""

import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

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


def ensure_spaces_around_links(element):
    """
    Ensure that all links in an element have spaces around them.
    This prevents links from being concatenated with surrounding text.
    
    Args:
        element: BeautifulSoup element to process
        
    Returns:
        BeautifulSoup element with spaces around links
    """
    if not element or not hasattr(element, 'find_all'):
        return element
    
    # Find all links in the element (in reverse order to preserve indices)
    links = list(element.find_all('a'))
    for link in reversed(links):
        # Check if we need a space before the link
        prev = link.previous_sibling
        needs_space_before = True
        if prev is not None:
            if isinstance(prev, str):
                # Check if the string ends with whitespace
                if prev.rstrip() != prev:
                    needs_space_before = False
            else:
                # For element siblings, check their text
                prev_text = prev.get_text() if hasattr(prev, 'get_text') else ''
                if prev_text and prev_text.rstrip() != prev_text:
                    needs_space_before = False
        
        if needs_space_before:
            link.insert_before(' ')
        
        # Check if we need a space after the link
        next_sib = link.next_sibling
        needs_space_after = True
        if next_sib is not None:
            if isinstance(next_sib, str):
                # Check if the string starts with whitespace
                if next_sib.lstrip() != next_sib:
                    needs_space_after = False
            else:
                # For element siblings, check their text
                next_text = next_sib.get_text() if hasattr(next_sib, 'get_text') else ''
                if next_text and next_text.lstrip() != next_text:
                    needs_space_after = False
        
        if needs_space_after:
            link.insert_after(' ')
    
    return element


def extract_links_from_element(element, base_url=None):
    """
    Extract all links from an element, preserving their context.
    
    Args:
        element: BeautifulSoup element to extract links from
        base_url: Base URL for resolving relative links
        
    Returns:
        List of link dictionaries with href, text, and context
    """
    links = []
    if not element or not hasattr(element, 'find_all'):
        return links
    
    for link in element.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Resolve relative URLs
        if base_url and href:
            href = urljoin(base_url, href)
        
        if href:
            links.append({
                'href': href,
                'text': text,
                'anchor_text': text
            })
    
    return links


def dedupe_sections(data):
    """
    Remove duplicate content from h1 sections that also appears in other sections.
    
    For each h1 section, removes content items (identified by (type, text)) that
    also appear in any non-h1 section. This fixes mis-parsing issues where h1
    sections accidentally include content from other sections.
    
    Args:
        data: Dictionary with 'sections' key containing list of section dicts
        
    Returns:
        Modified data dictionary with deduplicated sections
    """
    sections = data.get('sections') or []
    if not sections:
        return data
    
    # Precompute content signatures for each section: set of (type, text)
    # Only include items that have both 'type' and 'text' fields
    section_signatures = []
    for s in sections:
        content = s.get('content') or []
        sig = set()
        for item in content:
            if isinstance(item, dict) and 'type' in item and 'text' in item:
                sig.add((item.get('type'), item.get('text')))
        section_signatures.append(sig)
    
    # For each h1 section, build the "other sections" signature and dedupe its content
    for idx, s in enumerate(sections):
        if s.get('level') != 'h1':
            continue
        
        # Union of all content in other sections (exclude this h1's own section)
        other_sig = set()
        for j, sig in enumerate(section_signatures):
            if j != idx:
                other_sig |= sig
        
        content = s.get('content') or []
        deduped = [
            item for item in content
            if not (isinstance(item, dict) and 'type' in item and 'text' in item and 
                   (item.get('type'), item.get('text')) in other_sig)
        ]
        s['content'] = deduped
    
    return data


def parse_grokipedia_html(html_content, url, title=None):
    """
    Parse grokipedia HTML and extract structured data (v0.2 - improved parsing)
    
    Improvements:
    - Better link extraction from all content
    - More robust section parsing
    - Extracts links from paragraphs and sections
    - Better handling of different HTML structures
    """
    if title is None:
        title = url.split('/page/')[-1]
        # Decode URL-encoded title
        try:
            from urllib.parse import unquote
            title = unquote(title)
        except:
            pass
    
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {
        'title': title,
        'url': url,
        'main_title': None,
        'sections': [],
        'paragraphs': [],
        'tables': [],
        'references': [],
        'links': [],  # All links found in the page
        'metadata': {
            'has_edits': False,
            'fact_check_timestamp': None,
            'edits_count': None
        }
    }
    
    # Find article container - try multiple possible containers
    article = soup.find('div', class_='mx-auto max-w-[850px]')
    if not article:
        # Try alternative containers
        article = soup.find('article')
        if not article:
            article = soup.find('main')
            if not article:
                article = soup.find('div', class_=re.compile(r'max-w'))
    
    if not article:
        logger.warning(f"No article container found for {url}")
        return data
    
    # Extract main title (h1)
    h1 = article.find('h1')
    if h1:
        data['main_title'] = h1.get_text(strip=True)
    
    # Check if edits exist (look for "See Edits" button anywhere on page)
    edits_btn = soup.find('button', string=lambda x: x and 'See Edits' in str(x))
    if not edits_btn:
        # Try alternative ways to find edits button
        edits_btn = soup.find('button', class_=re.compile(r'edit', re.I))
    
    if edits_btn:
        data['metadata']['has_edits'] = True
        
        # Extract edit count if visible in button text
        btn_text = edits_btn.get_text(strip=True)
        if '(' in btn_text and ')' in btn_text:
            count_match = re.search(r'\((\d+)\)', btn_text)
            if count_match:
                data['metadata']['edits_count'] = int(count_match.group(1))
    
    # Extract fact-check timestamp if available
    fact_check_elem = soup.find(string=re.compile(r'Fact-checked by Grok', re.I))
    if fact_check_elem:
        parent = fact_check_elem.find_parent()
        if parent:
            fact_text = parent.get_text()
            # Try to extract timestamp like "5 days ago" or a date
            time_match = re.search(r'(\d+\s+(?:day|hour|minute|second)s?\s+ago|Fact-checked by Grok\s+([^\.]+))', fact_text, re.I)
            if time_match:
                data['metadata']['fact_check_timestamp'] = time_match.group(0)
    
    # Extract sections with proper content
    headings = article.find_all(['h1', 'h2', 'h3'], id=True)
    if not headings:
        # Try without id requirement
        headings = article.find_all(['h1', 'h2', 'h3'])
    
    for heading in headings:
        section_data = {
            'level': heading.name,
            'id': heading.get('id'),
            'title': heading.get_text(strip=True),
            'content': [],
            'links': []  # Links found in this section
        }
        
        # Extract links from heading itself
        heading_links = extract_links_from_element(heading, url)
        section_data['links'].extend(heading_links)
        
        # Walk through siblings after heading
        current = heading.next_sibling
        while current:
            if hasattr(current, 'name') and current.name in ['h1', 'h2', 'h3']:
                if current.name <= heading.name:
                    break
            
            if hasattr(current, 'name'):
                if current.name == 'span' and 'mb-4' in (current.get('class') or []):
                    # Ensure spaces around links before extracting text
                    ensure_spaces_around_links(current)
                    
                    # Extract links from this paragraph
                    para_links = extract_links_from_element(current, url)
                    section_data['links'].extend(para_links)
                    
                    text = current.get_text(separator=' ')
                    if text:
                        # Normalize whitespace (multiple spaces to single space)
                        text = re.sub(r'\s+', ' ', text).strip()
                        para_data = {'type': 'paragraph', 'text': text}
                        if para_links:
                            para_data['links'] = para_links
                        section_data['content'].append(para_data)
                elif current.name == 'ul':
                    items = []
                    for li in current.find_all('li'):
                        ensure_spaces_around_links(li)
                        
                        # Extract links from list item
                        li_links = extract_links_from_element(li, url)
                        section_data['links'].extend(li_links)
                        
                        item_text = li.get_text(separator=' ')
                        item_text = re.sub(r'\s+', ' ', item_text).strip()
                        items.append(item_text)
                    if items:
                        list_data = {'type': 'list', 'items': items}
                        section_data['content'].append(list_data)
                elif current.name == 'ol':
                    items = []
                    for li in current.find_all('li'):
                        ensure_spaces_around_links(li)
                        
                        # Extract links from list item
                        li_links = extract_links_from_element(li, url)
                        section_data['links'].extend(li_links)
                        
                        item_text = li.get_text(separator=' ')
                        item_text = re.sub(r'\s+', ' ', item_text).strip()
                        items.append(item_text)
                    if items:
                        list_data = {'type': 'ordered_list', 'items': items}
                        section_data['content'].append(list_data)
                elif current.name == 'p':
                    # Handle paragraph tags directly
                    ensure_spaces_around_links(current)
                    para_links = extract_links_from_element(current, url)
                    section_data['links'].extend(para_links)
                    
                    text = current.get_text(separator=' ')
                    if text:
                        text = re.sub(r'\s+', ' ', text).strip()
                        para_data = {'type': 'paragraph', 'text': text}
                        if para_links:
                            para_data['links'] = para_links
                        section_data['content'].append(para_data)
                elif current.name == 'div':
                    # Check if div contains text content
                    # Check if div contains text content (but extract after ensuring spaces)
                    ensure_spaces_around_links(current)
                    div_text = current.get_text(separator=' ')
                    if div_text and len(div_text.strip()) > 10:  # Only if substantial content
                        div_links = extract_links_from_element(current, url)
                        section_data['links'].extend(div_links)
                        
                        div_text = re.sub(r'\s+', ' ', div_text).strip()
                        para_data = {'type': 'paragraph', 'text': div_text}
                        if div_links:
                            para_data['links'] = div_links
                        section_data['content'].append(para_data)
            
            current = current.next_sibling
        
        data['sections'].append(section_data)
        # Add section links to main links list
        data['links'].extend(section_data['links'])
    
    # Extract paragraphs with proper spacing (fallback for paragraphs not in sections)
    for span in article.find_all('span', class_='mb-4'):
        # Ensure spaces around links before extracting text
        ensure_spaces_around_links(span)
        
        # Extract links
        para_links = extract_links_from_element(span, url)
        data['links'].extend(para_links)
        
        text = span.get_text(separator=' ')
        # Normalize whitespace (multiple spaces to single space)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text not in data['paragraphs']:
            data['paragraphs'].append(text)
    
    # Also check for paragraph tags
    for p in article.find_all('p'):
        ensure_spaces_around_links(p)
        para_links = extract_links_from_element(p, url)
        data['links'].extend(para_links)
        
        text = p.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text not in data['paragraphs']:
            data['paragraphs'].append(text)
    
    # Extract tables
    for table in article.find_all('table'):
        table_data = []
        headers = []
        table_links = []
        
        if table.find('thead'):
            for th in table.find('thead').find_all('th'):
                header_text = th.get_text(strip=True)
                headers.append(header_text)
                # Extract links from headers
                th_links = extract_links_from_element(th, url)
                table_links.extend(th_links)
        
        if table.find('tbody'):
            for tr in table.find('tbody').find_all('tr'):
                row = []
                for td in tr.find_all('td'):
                    cell_text = td.get_text(strip=True)
                    row.append(cell_text)
                    # Extract links from cells
                    td_links = extract_links_from_element(td, url)
                    table_links.extend(td_links)
                if row:
                    table_data.append(row)
        
        if headers or table_data:
            table_obj = {'headers': headers, 'rows': table_data}
            if table_links:
                table_obj['links'] = table_links
            data['tables'].append(table_obj)
            data['links'].extend(table_links)
    
    # Extract references WITH links
    references_section = soup.find('div', id='references')
    if not references_section:
        # Try alternative ways to find references
        references_section = soup.find('section', id='references')
        if not references_section:
            references_section = soup.find('div', class_=re.compile(r'reference', re.I))
    
    if references_section:
        for li in references_section.find_all('li'):
            ensure_spaces_around_links(li)
            ref_text = li.get_text(separator=' ')
            ref_text = re.sub(r'\s+', ' ', ref_text).strip()
            ref_link = None
            
            link = li.find('a')
            if link and link.get('href'):
                href = link.get('href')
                # Resolve relative URLs
                if url:
                    href = urljoin(url, href)
                ref_link = {
                    'href': href,
                    'text': link.get_text(strip=True),
                    'anchor_text': link.get_text(strip=True)
                }
                data['links'].append(ref_link)
            
            if ref_text:
                ref_obj = {'text': ref_text}
                if ref_link:
                    ref_obj['link'] = ref_link
                data['references'].append(ref_obj)
    
    # Extract all other links from the article (comprehensive link extraction)
    all_links_in_article = article.find_all('a', href=True)
    seen_hrefs = set()
    for link in all_links_in_article:
        href = link.get('href', '')
        if not href:
            continue
        
        # Resolve relative URLs
        if url:
            href = urljoin(url, href)
        
        # Skip if we've already seen this link
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)
        
        # Skip if it's already in our links list
        if any(l.get('href') == href for l in data['links']):
            continue
        
        link_text = link.get_text(strip=True)
        link_obj = {
            'href': href,
            'text': link_text,
            'anchor_text': link_text
        }
        data['links'].append(link_obj)
    
    # Remove references from paragraphs
    data['paragraphs'] = [p for p in data['paragraphs'] 
                          if not any(ref['text'].split()[0:3] == p.split()[0:3] for ref in data['references'])]
    
    # Deduplicate h1 section content that appears in other sections
    data = dedupe_sections(data)
    
    # Deduplicate links (keep unique hrefs)
    seen_links = set()
    unique_links = []
    for link in data['links']:
        href = link.get('href')
        if href and href not in seen_links:
            seen_links.add(href)
            unique_links.append(link)
    data['links'] = unique_links
    
    return data


# Import the rest of the functions from the original scraper
# (discovery, scraping, GCS functions, etc. remain the same)
# For brevity, I'll include the key async functions

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


def load_urls_from_file(file_path: str):
    """Load URLs from a local file.

    Supports:
    - .jsonl: one JSON object per line, with a 'url' field
    - .json: either a list of URLs, or an object with key 'urls'
    - any other text file: one URL per non-empty line
    """
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"URLs file not found: {file_path}")

        urls: list[str] = []
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            with path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "url" in obj:
                            urls.append(obj["url"])
                    except json.JSONDecodeError:
                        continue
        elif suffix == ".json":
            with path.open("r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    # assume list of URL strings
                    urls = [str(u) for u in data]
                elif isinstance(data, dict):
                    if "urls" in data and isinstance(data["urls"], list):
                        urls = [str(u) for u in data["urls"]]
        else:
            with path.open("r") as f:
                for line in f:
                    url = line.strip()
                    if url:
                        urls.append(url)

        urls_count = len(urls)
        logger.info(f"Loaded {urls_count} URLs from file: {file_path}")
        return urls
    except Exception as e:
        logger.error(f"Failed to load URLs from file '{file_path}': {e}")
        raise


def list_gcs_blobs(bucket_name, prefix=None, max_results=10, project_id=None):
    """List existing blobs in a GCS bucket for debugging"""
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist")
            return []
        
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=max_results))
        return [blob.name for blob in blobs]
    except Exception as e:
        logger.error(f"Failed to list blobs: {e}")
        return []


def upload_to_gcs(bucket_name, blob_name, content, project_id=None):
    """Upload content to Google Cloud Storage"""
    try:
        # Validate inputs
        if not bucket_name:
            logger.error("Bucket name is empty or None")
            return False
        if not blob_name:
            logger.error("Blob name is empty or None")
            return False
        if content is None:
            logger.error("Content is None")
            return False
        if isinstance(content, str) and len(content) == 0:
            logger.warning("Content is empty string, but uploading anyway")
        
        logger.info(f"Attempting to upload to GCS: bucket={bucket_name}, blob={blob_name}, content_length={len(content) if content else 0}")
        
        # Use default service account credentials (should be available on GCE instances)
        # If project_id is not provided, try to get it from environment
        try:
            if project_id is None:
                import os
                project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT')
            
            client = storage.Client(project=project_id)
            bucket = client.bucket(bucket_name)
        except Exception as auth_error:
            logger.error(f"Failed to initialize GCS client: {auth_error}")
            logger.error("Make sure the instance has the 'Storage Object Admin' role or sufficient permissions")
            raise
        
        # Check if bucket exists
        if not bucket.exists():
            logger.error(f"Bucket {bucket_name} does not exist")
            # Try to list what buckets are available (for debugging)
            try:
                buckets = list(client.list_buckets())
                logger.error(f"Available buckets: {[b.name for b in buckets]}")
            except:
                pass
            return False
        
        # For debugging: list a few existing blobs with similar prefix
        blob_prefix = '/'.join(blob_name.split('/')[:-1]) if '/' in blob_name else ''
        if blob_prefix:
            existing_blobs = list_gcs_blobs(bucket_name, prefix=blob_prefix, max_results=5, project_id=project_id)
            if existing_blobs:
                logger.info(f"Found {len(existing_blobs)} existing blobs with prefix '{blob_prefix}': {existing_blobs[:3]}...")
            else:
                logger.info(f"No existing blobs found with prefix '{blob_prefix}' (this might be the first upload)")
        
        blob = bucket.blob(blob_name)
        blob.upload_from_string(content, content_type='application/jsonl')
        logger.info(f"✓ Successfully uploaded {blob_name} to {bucket_name} ({len(content)} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        logger.error(f"  Bucket: {bucket_name}, Blob: '{blob_name}', Content length: {len(content) if content else 0}")
        logger.error(f"  Blob name type: {type(blob_name)}, Blob name repr: {repr(blob_name)}")
        import traceback
        logger.error(traceback.format_exc())
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
        items_processed_count = 0  # Track total items processed for blob naming
        
        # Create progress bar
        desc = f"Scraping (Shard {shard_id})" if shard_id is not None else "Scraping"
        # Note: start_index is for blob naming, not loop indexing (urls is already a slice)
        pbar = tqdm(total=len(urls), desc=desc, initial=0)
        
        # Process in batches - always start from 0 since urls is already the shard slice
        for i in range(0, len(urls), config['max_concurrent']):
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
                    items_processed_count += 1
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
                # Calculate absolute indices for blob naming
                # scraped_data contains items that were successfully scraped from start_index onwards
                # Since items_processed_count tracks total successful items, we can calculate ranges
                batch_start = start_index + (items_processed_count - len(scraped_data))
                batch_end = start_index + items_processed_count - 1
                
                # Prepare JSONL content
                jsonl_content = '\n'.join(json.dumps(item) for item in scraped_data)
                
                # Upload to GCS or save locally
                if gcs_bucket:
                    # Construct blob name - match existing pattern if possible
                    if shard_id is not None:
                        blob_name = f'shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                    else:
                        blob_name = f'scraped_data/batch_{batch_start}_{batch_end}.jsonl'
                    
                    logger.info(f"Preparing to upload batch: blob_name='{blob_name}', batch_start={batch_start}, batch_end={batch_end}, items={len(scraped_data)}")
                    
                    if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                        logger.info(f"✓ Uploaded batch to GCS: {blob_name} ({len(scraped_data)} items)")
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
            # Calculate absolute indices for final batch
            batch_start = start_index + (items_processed_count - len(scraped_data))
            batch_end = start_index + items_processed_count - 1
            
            jsonl_content = '\n'.join(json.dumps(item) for item in scraped_data)
            
            if gcs_bucket:
                if shard_id is not None:
                    blob_name = f'shard_{shard_id}/batch_{batch_start}_{batch_end}.jsonl'
                else:
                    blob_name = f'scraped_data/batch_{batch_start}_{batch_end}.jsonl'
                
                if upload_to_gcs(gcs_bucket, blob_name, jsonl_content, gcs_project):
                    logger.info(f"Uploaded final batch to GCS: {blob_name} ({len(scraped_data)} items)")
                else:
                    # Fallback to local save
                    output_dir = Path(config.get('output_dir', 'scraped_data'))
                    if shard_id is not None:
                        output_dir = output_dir / f'shard_{shard_id}'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    batch_file = output_dir / f'batch_{batch_start}_{batch_end}.jsonl'
                    with open(batch_file, 'w') as f:
                        f.write(jsonl_content)
                    logger.warning(f"Failed to upload final batch to GCS, saved locally: {batch_file} ({len(scraped_data)} items)")
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
                if upload_to_gcs(gcs_bucket, failed_blob, failed_content, gcs_project):
                    logger.info(f"Uploaded {len(failed_pages)} failed pages to GCS")
                else:
                    logger.warning(f"Failed to upload failed pages to GCS, saving locally")
                    failed_file = Path(f'scraping_failed_shard_{shard_id}.jsonl' if shard_id is not None else 'scraping_failed.jsonl')
                    with open(failed_file, 'w') as f:
                        f.write(failed_content)
                    logger.info(f"Saved {len(failed_pages)} failed pages locally")
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


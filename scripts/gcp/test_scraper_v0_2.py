#!/usr/bin/env python3
"""
Test script for grokipedia_scraper_v0_2.py
Tests the scraper on a sample URL to verify it's working correctly
Supports testing with and without BrightData proxy
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

import grokipedia_scraper_v0_2 as scraper


def test_parse_html_from_file(html_file_path, url=None, title=None):
    """Test parse_grokipedia_html with HTML from a file"""
    print(f"\n{'='*80}")
    print(f"Testing parse_grokipedia_html with HTML from file")
    print(f"File: {html_file_path}")
    print(f"{'='*80}\n")
    
    from pathlib import Path
    html_file = Path(html_file_path)
    if not html_file.exists():
        print(f"✗ File not found: {html_file_path}")
        return None
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    
    print(f"✓ Loaded HTML ({len(html):,} bytes)")
    
    if url is None:
        url = f"https://grokipedia.com/page/{html_file.stem}"
    if title is None:
        title = html_file.stem
    
    # Test parser directly
    data = scraper.parse_grokipedia_html(html, url, title)
    
    print(f"\n📊 Parser Results:")
    print(f"  Title: {data.get('title')}")
    print(f"  Main title: {data.get('main_title')}")
    print(f"  Sections: {len(data.get('sections', []))}")
    print(f"  Paragraphs: {len(data.get('paragraphs', []))}")
    print(f"  Tables: {len(data.get('tables', []))}")
    print(f"  References: {len(data.get('references', []))}")
    print(f"  Links: {len(data.get('links', []))}")
    
    total_content = (
        len(data.get('sections', [])) +
        len(data.get('paragraphs', [])) +
        len(data.get('tables', [])) +
        len(data.get('references', []))
    )
    
    print(f"\n  Total content items: {total_content}")
    
    if total_content == 0 and not data.get('main_title'):
        print(f"\n⚠️  WARNING: No content extracted!")
        print(f"  HTML length: {len(html):,} bytes")
        print(f"  Has <article> tag: {'<article' in html}")
        print(f"  Has 'mx-auto' class: {'mx-auto' in html}")
        print(f"  Has 'max-w-[850px]' class: {'max-w-[850px]' in html}")
        
        # Show first 500 chars of HTML for debugging
        print(f"\n  First 500 chars of HTML:")
        print(f"  {html[:500]}...")
    
    return data


async def test_parse_html_directly(url, config, use_proxy=False):
    """Test parse_grokipedia_html function directly with fetched HTML"""
    proxy_status = "WITH PROXY" if use_proxy else "WITHOUT PROXY"
    print(f"\n{'='*80}")
    print(f"Testing parse_grokipedia_html directly on: {url}")
    print(f"Mode: {proxy_status}")
    print(f"{'='*80}\n")
    
    proxy_url = config.get('proxy_url') if use_proxy else None
    
    async with scraper.aiohttp.ClientSession() as session:
        async with session.get(
            url,
            timeout=scraper.aiohttp.ClientTimeout(total=60),
            headers={'Accept-Encoding': 'gzip, deflate'},
            proxy=proxy_url,
        ) as response:
            if response.status == 200:
                html = await response.text()
                print(f"✓ Fetched HTML ({len(html):,} bytes)")
                
                # Test parser directly
                title = url.split('/page/')[-1]
                data = scraper.parse_grokipedia_html(html, url, title)
                
                print(f"\n📊 Parser Results:")
                print(f"  Main title: {data.get('main_title')}")
                print(f"  Sections: {len(data.get('sections', []))}")
                print(f"  Paragraphs: {len(data.get('paragraphs', []))}")
                print(f"  Tables: {len(data.get('tables', []))}")
                print(f"  References: {len(data.get('references', []))}")
                print(f"  Links: {len(data.get('links', []))}")
                
                total_content = (
                    len(data.get('sections', [])) +
                    len(data.get('paragraphs', [])) +
                    len(data.get('tables', [])) +
                    len(data.get('references', []))
                )
                
                if total_content == 0 and not data.get('main_title'):
                    print(f"\n⚠️  WARNING: No content extracted!")
                    print(f"  HTML length: {len(html):,} bytes")
                    print(f"  Has <article> tag: {'<article' in html}")
                    print(f"  Has 'mx-auto' class: {'mx-auto' in html}")
                    print(f"  Has 'max-w-[850px]' class: {'max-w-[850px]' in html}")
                    
                    # Save HTML sample for debugging
                    from pathlib import Path
                    debug_dir = Path('debug_html_samples')
                    debug_dir.mkdir(exist_ok=True)
                    safe_title = title.replace('/', '_')[:50]
                    debug_file = debug_dir / f"{safe_title}_{'proxy' if use_proxy else 'direct'}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(html)
                    print(f"  Saved HTML sample to: {debug_file}")
                
                return {
                    'success': True,
                    'data': data,
                    'html_length': len(html),
                    'total_content': total_content
                }
            else:
                print(f"✗ Failed to fetch HTML: status {response.status}")
                return {
                    'success': False,
                    'error': f'status_{response.status}',
                    'html_length': 0,
                    'total_content': 0
                }


async def test_scrape_url(url, config, use_proxy=False):
    """Test scraping a single URL with given config"""
    proxy_status = "WITH PROXY" if use_proxy else "WITHOUT PROXY"
    print(f"\n{'='*80}")
    print(f"Testing scraper on: {url}")
    print(f"Mode: {proxy_status}")
    print(f"{'='*80}\n")
    
    limiter = scraper.AsyncLimiter(max_rate=config['rate_limit'], time_period=60)
    proxy_url = config.get('proxy_url') if use_proxy else None
    
    async with scraper.aiohttp.ClientSession() as session:
        # First, fetch HTML and test parser directly
        print("Step 1: Fetching HTML and testing parser directly...")
        async with session.get(
            url,
            timeout=scraper.aiohttp.ClientTimeout(total=60),
            headers={'Accept-Encoding': 'gzip, deflate'},
            proxy=proxy_url,
        ) as response:
            if response.status == 200:
                html = await response.text()
                print(f"  ✓ Fetched HTML ({len(html):,} bytes)")
                
                # Test parser directly
                title = url.split('/page/')[-1]
                parser_data = scraper.parse_grokipedia_html(html, url, title)
                
                parser_content = (
                    len(parser_data.get('sections', [])) +
                    len(parser_data.get('paragraphs', [])) +
                    len(parser_data.get('tables', [])) +
                    len(parser_data.get('references', []))
                )
                
                print(f"  Parser results: {parser_content} content items, main_title: {parser_data.get('main_title')}")
                
                if parser_content == 0 and not parser_data.get('main_title'):
                    print(f"  ⚠️  WARNING: Parser extracted no content!")
            else:
                print(f"  ✗ Failed to fetch HTML: status {response.status}")
                html = None
        
        # Now test the full scrape_page flow
        print("\nStep 2: Testing full scrape_page flow...")
        result = await scraper.scrape_page(session, limiter, url, config, skip_on_error=True)
        
        if result['success']:
            data = result['data']
            
            print(f"✓ Successfully scraped page")
            print(f"\nTitle: {data.get('title')}")
            print(f"Main Title: {data.get('main_title')}")
            print(f"URL: {data.get('url')}")
            
            # Check for parse warnings
            if data.get('_parse_warning'):
                print(f"\n⚠️  WARNING: {data['_parse_warning']}")
            
            print(f"\nMetadata:")
            metadata = data.get('metadata', {})
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            
            print(f"\nSections: {len(data.get('sections', []))}")
            for i, section in enumerate(data.get('sections', [])[:3]):  # Show first 3
                print(f"  Section {i+1}: {section.get('title')} (level: {section.get('level')})")
                print(f"    Content items: {len(section.get('content', []))}")
                print(f"    Links: {len(section.get('links', []))}")
                if section.get('content'):
                    first_content = section['content'][0]
                    if isinstance(first_content, dict):
                        text_preview = first_content.get('text', '')[:100]
                        print(f"    First content preview: {text_preview}...")
            
            print(f"\nParagraphs: {len(data.get('paragraphs', []))}")
            if data.get('paragraphs'):
                print(f"  First paragraph preview: {data['paragraphs'][0][:100]}...")
            
            print(f"\nTables: {len(data.get('tables', []))}")
            
            print(f"\nReferences: {len(data.get('references', []))}")
            for i, ref in enumerate(data.get('references', [])[:3]):  # Show first 3
                print(f"  Reference {i+1}: {ref.get('text', '')[:80]}...")
                if ref.get('link'):
                    print(f"    Link: {ref['link'].get('href', '')[:80]}")
            
            print(f"\nTotal Links Found: {len(data.get('links', []))}")
            for i, link in enumerate(data.get('links', [])[:5]):  # Show first 5
                print(f"  Link {i+1}: {link.get('href', '')[:80]}")
                print(f"    Text: {link.get('text', '')[:60]}")
            
            # Calculate content summary
            total_content = (
                len(data.get('sections', [])) +
                len(data.get('paragraphs', [])) +
                len(data.get('tables', [])) +
                len(data.get('references', []))
            )
            
            print(f"\n📊 Content Summary:")
            print(f"  Total content items: {total_content}")
            print(f"  Has main title: {data.get('main_title') is not None}")
            print(f"  Has sections: {len(data.get('sections', [])) > 0}")
            print(f"  Has paragraphs: {len(data.get('paragraphs', [])) > 0}")
            
            # Compare with direct parser results if we have them
            if html is not None:
                parser_content = (
                    len(parser_data.get('sections', [])) +
                    len(parser_data.get('paragraphs', [])) +
                    len(parser_data.get('tables', [])) +
                    len(parser_data.get('references', []))
                )
                if parser_content != total_content:
                    print(f"\n  ⚠️  Content mismatch detected!")
                    print(f"    Direct parser: {parser_content} items")
                    print(f"    scrape_page: {total_content} items")
                else:
                    print(f"\n  ✓ Parser and scrape_page results match ({total_content} items)")
            
            # Save full output to a file for inspection
            proxy_suffix = "_proxy" if use_proxy else "_direct"
            output_file = Path(f'test_scraper_output{proxy_suffix}.json')
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Full output saved to: {output_file}")
            
            return {
                'success': True,
                'data': data,
                'total_content': total_content,
                'has_main_title': data.get('main_title') is not None
            }
        else:
            print(f"✗ Failed to scrape page")
            print(f"  Error: {result.get('error')}")
            print(f"  Title: {result.get('title')}")
            return {
                'success': False,
                'error': result.get('error'),
                'total_content': 0,
                'has_main_title': False
            }


async def fetch_raw_html(url, proxy_url=None):
    """Fetch raw HTML to check length and basic structure"""
    try:
        async with scraper.aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=scraper.aiohttp.ClientTimeout(total=60),
                headers={'Accept-Encoding': 'gzip, deflate'},
                proxy=proxy_url,
            ) as response:
                if response.status == 200:
                    html = await response.text()
                    return {
                        'success': True,
                        'html_length': len(html),
                        'status_code': response.status,
                        'has_article_tag': '<article' in html,
                        'has_mx_auto': 'mx-auto' in html,
                        'has_max_w': 'max-w-[850px]' in html,
                    }
                else:
                    return {
                        'success': False,
                        'status_code': response.status,
                        'html_length': 0
                    }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'html_length': 0
        }


async def compare_proxy_vs_direct(url, config_with_proxy, config_without_proxy):
    """Compare scraping results with and without proxy"""
    print(f"\n{'#'*80}")
    print(f"COMPARISON TEST: {url}")
    print(f"{'#'*80}\n")
    
    # First, check raw HTML
    print("Fetching raw HTML for comparison...")
    html_direct = await fetch_raw_html(url, proxy_url=None)
    html_proxy = await fetch_raw_html(url, proxy_url=config_with_proxy.get('proxy_url'))
    
    print(f"\n📄 Raw HTML Comparison:")
    print(f"  Direct access:")
    print(f"    Success: {html_direct.get('success')}")
    print(f"    HTML length: {html_direct.get('html_length', 0):,} bytes")
    print(f"    Status code: {html_direct.get('status_code', 'N/A')}")
    if html_direct.get('success'):
        print(f"    Has <article> tag: {html_direct.get('has_article_tag')}")
        print(f"    Has 'mx-auto' class: {html_direct.get('has_mx_auto')}")
        print(f"    Has 'max-w-[850px]' class: {html_direct.get('has_max_w')}")
    
    print(f"  Proxy access:")
    print(f"    Success: {html_proxy.get('success')}")
    print(f"    HTML length: {html_proxy.get('html_length', 0):,} bytes")
    print(f"    Status code: {html_proxy.get('status_code', 'N/A')}")
    if html_proxy.get('success'):
        print(f"    Has <article> tag: {html_proxy.get('has_article_tag')}")
        print(f"    Has 'mx-auto' class: {html_proxy.get('has_mx_auto')}")
        print(f"    Has 'max-w-[850px]' class: {html_proxy.get('has_max_w')}")
    
    if html_direct.get('success') and html_proxy.get('success'):
        length_diff = html_direct.get('html_length', 0) - html_proxy.get('html_length', 0)
        print(f"\n  ⚠️  HTML length difference: {length_diff:,} bytes")
        if abs(length_diff) > 10000:
            print(f"     WARNING: Significant difference in HTML length!")
    
    # Now test scraping
    print(f"\n{'─'*80}")
    result_direct = await test_scrape_url(url, config_without_proxy, use_proxy=False)
    await asyncio.sleep(2)  # Small delay between tests
    
    print(f"\n{'─'*80}")
    result_proxy = await test_scrape_url(url, config_with_proxy, use_proxy=True)
    
    # Compare results
    print(f"\n{'#'*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'#'*80}\n")
    
    print(f"Direct Access:")
    print(f"  Success: {result_direct.get('success')}")
    print(f"  Total content items: {result_direct.get('total_content', 0)}")
    print(f"  Has main title: {result_direct.get('has_main_title', False)}")
    
    print(f"\nProxy Access:")
    print(f"  Success: {result_proxy.get('success')}")
    print(f"  Total content items: {result_proxy.get('total_content', 0)}")
    print(f"  Has main title: {result_proxy.get('has_main_title', False)}")
    
    if result_direct.get('success') and result_proxy.get('success'):
        content_diff = result_direct.get('total_content', 0) - result_proxy.get('total_content', 0)
        print(f"\n  Content difference: {content_diff} items")
        if content_diff != 0:
            print(f"     ⚠️  WARNING: Content extraction differs between direct and proxy!")
    
    return result_direct, result_proxy


async def main():
    """Main test function"""
    # Test URLs - including the triple bar symbol page
    test_urls = [
        "https://grokipedia.com/page/%E2%89%A1",  # Triple bar symbol (URL encoded)
        "https://grokipedia.com/page/Formula_D",  # Regular page
        "https://grokipedia.com/page/The_Church_of_Jesus_Christ_of_Latter-day_Saints",
        "https://grokipedia.com/page/Codeblack_Films"  # From the batch file
    ]
    
    print("Grokipedia Scraper v0.2 Test")
    print("=" * 80)
    
    # Check for proxy credentials
    brightdata_username = os.getenv('BRIGHTDATA_USERNAME')
    brightdata_password = os.getenv('BRIGHTDATA_PASSWORD')
    proxy_url = None
    if brightdata_username and brightdata_password:
        proxy_url = f'http://{brightdata_username}:{brightdata_password}@brd.superproxy.io:33335'
        print("✓ BrightData proxy credentials found")
    else:
        print("⚠️  No BrightData proxy credentials found (set BRIGHTDATA_USERNAME and BRIGHTDATA_PASSWORD)")
        print("   Testing will proceed without proxy comparison")
    
    # Base config
    base_config = {
        'max_concurrent': 1,
        'rate_limit': 60,
        'scraping_timeout': 60,
        'skip_on_error': True
    }
    
    config_without_proxy = base_config.copy()
    config_with_proxy = base_config.copy()
    config_with_proxy['proxy_url'] = proxy_url
    
    # Parse command line arguments
    test_mode = 'normal'
    urls_to_test = []
    html_file_to_test = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            urls_to_test = test_urls
        elif sys.argv[1] == "--compare" or sys.argv[1] == "--proxy":
            test_mode = 'compare'
            if len(sys.argv) > 2:
                if sys.argv[2] == "--all":
                    urls_to_test = test_urls
                else:
                    urls_to_test = [sys.argv[2]]
            else:
                urls_to_test = [test_urls[0]]  # Default to first URL
        elif sys.argv[1] == "--html" or sys.argv[1].endswith('.html'):
            # Test with HTML file
            test_mode = 'html_file'
            html_file_to_test = sys.argv[1] if sys.argv[1].endswith('.html') else sys.argv[2]
        else:
            urls_to_test = [sys.argv[1]]
    else:
        urls_to_test = [test_urls[0]]  # Default to triple bar symbol
    
    if not proxy_url and test_mode == 'compare':
        print("\n⚠️  Cannot run comparison test without proxy credentials")
        print("   Falling back to normal test mode")
        test_mode = 'normal'
    
    results = []
    
    if test_mode == 'html_file':
        # Test parser with HTML from file
        data = test_parse_html_from_file(html_file_to_test)
        if data:
            print("\n✓ Parser test completed")
            return 0
        else:
            print("\n✗ Parser test failed")
            return 1
    elif test_mode == 'compare':
        # Comparison mode: test with and without proxy
        print(f"\n🔬 Running comparison tests (with and without proxy)")
        for url in urls_to_test:
            result_direct, result_proxy = await compare_proxy_vs_direct(
                url, config_with_proxy, config_without_proxy
            )
            results.append((url, result_direct.get('success'), result_proxy.get('success')))
            await asyncio.sleep(2)  # Delay between URLs
    else:
        # Normal mode: test with or without proxy based on config
        use_proxy = proxy_url is not None
        config = config_with_proxy if use_proxy else config_without_proxy
        
        for url in urls_to_test:
            result = await test_scrape_url(url, config, use_proxy=use_proxy)
            results.append((url, result.get('success')))
            await asyncio.sleep(1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*80}")
    print("Test Summary:")
    print(f"{'='*80}")
    
    if test_mode == 'compare':
        for url, direct_success, proxy_success in results:
            direct_status = "✓" if direct_success else "✗"
            proxy_status = "✓" if proxy_success else "✗"
            print(f"{direct_status} Direct / {proxy_status} Proxy: {url}")
    else:
        for url, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {url}")
    
    if test_mode == 'compare':
        all_passed = all(direct and proxy for _, direct, proxy in results)
    else:
        all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


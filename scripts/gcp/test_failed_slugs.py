#!/usr/bin/env python3
"""
Test failed slugs to find the correct Grokipedia slug format
Uses curl/requests to test API responses
"""

import json
import re
import subprocess
import sys
import unicodedata
from urllib.parse import quote

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

def normalize_slug(slug):
    """Try different normalization approaches"""
    variants = []
    
    # Original
    variants.append(slug)
    
    # Apply title_to_slug normalization (Greek letters + underscore before parentheses)
    normalized = slug.replace(' ', '_')
    # Replace Greek letters
    for greek_char, english_name in GREEK_LETTERS.items():
        normalized = normalized.replace(greek_char, english_name)
    # Add underscore before opening parenthesis if missing
    normalized = re.sub(r'([^_])\(', r'\1_(', normalized)
    if normalized != slug:
        variants.append(normalized)
    
    # Remove special Unicode characters (like ʻ, ʿ, etc.)
    no_marks = ''.join(c for c in slug if unicodedata.category(c)[0] != 'M')
    if no_marks != slug:
        variants.append(no_marks)
    
    # ASCII-only (remove non-ASCII)
    ascii_only = slug.encode('ascii', 'ignore').decode('ascii')
    if ascii_only != slug:
        variants.append(ascii_only)
    
    # Remove special quote characters
    no_quotes = slug.replace("'", "").replace("ʻ", "").replace("ʿ", "").replace("ʾ", "").replace("'", "")
    if no_quotes != slug:
        variants.append(no_quotes)
    
    # NFKD normalization (decompose characters)
    nfkd = unicodedata.normalize('NFKD', slug)
    ascii_nfkd = nfkd.encode('ascii', 'ignore').decode('ascii')
    if ascii_nfkd != slug:
        variants.append(ascii_nfkd)
    
    # Remove all non-ASCII and special punctuation
    simple = ''.join(c if c.isalnum() or c in '_-' else '_' for c in slug)
    if simple != slug:
        variants.append(simple)
    
    return list(set(variants))  # Remove duplicates

def test_slug(slug):
    """Test a slug variant against the API"""
    base_url = "https://grokipedia.com/api/page"
    encoded = quote(slug, safe='')
    url = f"{base_url}?slug={encoded}&includeContent=false"
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-w', '\n%{http_code}', url],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            output = result.stdout
            parts = output.rsplit('\n', 1)
            if len(parts) == 2:
                body = parts[0]
                status_code = parts[1]
                
                if status_code == '200':
                    try:
                        data = json.loads(body)
                        if data.get('found') and data.get('page'):
                            return True, data
                        else:
                            return False, data
                    except:
                        return False, {'error': 'invalid_json'}
                else:
                    return False, {'status': status_code}
        
        return False, {'error': 'curl_failed'}
    except Exception as e:
        return False, {'error': str(e)}

def main():
    if len(sys.argv) > 1:
        slugs_file = sys.argv[1]
    else:
        slugs_file = '/tmp/test_slugs_sample.txt'
    
    print(f"Testing slugs from: {slugs_file}")
    print("="*80)
    
    with open(slugs_file, 'r', encoding='utf-8') as f:
        slugs = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for slug in slugs:
        print(f"\nTesting: {slug}")
        variants = normalize_slug(slug)
        print(f"  Generated {len(variants)} variants")
        
        found = False
        for i, variant in enumerate(variants[:10], 1):  # Limit to 10 variants
            if variant == slug:
                print(f"  Variant {i}: {variant} (original)")
            else:
                print(f"  Variant {i}: {variant}")
            
            success, data = test_slug(variant)
            if success:
                print(f"    ✓ FOUND! Using variant: {variant}")
                results.append({
                    'original': slug,
                    'working_slug': variant,
                    'found': True
                })
                found = True
                break
            else:
                if data.get('found') == False:
                    print(f"    ✗ Not found (found=false)")
                else:
                    print(f"    ✗ Failed: {data}")
        
        if not found:
            print(f"  ✗ No working variant found for: {slug}")
            results.append({
                'original': slug,
                'working_slug': None,
                'found': False
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    found_count = sum(1 for r in results if r['found'])
    print(f"Found: {found_count}/{len(results)}")
    
    print("\nWorking slugs:")
    for r in results:
        if r['found']:
            print(f"  {r['original']} -> {r['working_slug']}")
    
    print("\nStill not found:")
    for r in results:
        if not r['found']:
            print(f"  {r['original']}")

if __name__ == "__main__":
    main()


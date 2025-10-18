"""
Functions for downloading and parsing the Wikimedia structured Wikipedia dataset.
"""

from datasets import load_dataset
from typing import List, Dict, Tuple


def download_wikipedia_dataset(language: str = "en", split: str = "train") -> 'Dataset':
    """
    Download the most recent Wikimedia structured Wikipedia dataset for a given language.
    
    Args:
        language: Language code (default: "en" for English)
        split: Dataset split to load (default: "train")
    
    Returns:
        Dataset object containing the Wikipedia data
    
    Example:
        >>> dataset = download_wikipedia_dataset("en")
        >>> print(f"Downloaded {len(dataset)} articles")
    """
    dataset = load_dataset(
        "wikimedia/wikipedia",
        f"20231101.{language}",
        split=split,
        trust_remote_code=True
    )
    return dataset


def parse_page_content(page: Dict) -> Tuple[str, str]:
    """
    Parse out the page title and lead content from a Wikipedia page entry.
    
    Args:
        page: A single Wikipedia page dictionary from the dataset
    
    Returns:
        Tuple of (title, lead_content)
    
    Example:
        >>> title, lead = parse_page_content(dataset[0])
        >>> print(f"Title: {title}")
    """
    title = page.get("title", "")
    text = page.get("text", "")
    
    # Extract lead section (content before the first section heading)
    # Lead content is typically everything before the first "==" heading
    if "\n==" in text:
        lead_content = text.split("\n==")[0].strip()
    else:
        # If no sections, take first few paragraphs (up to 500 chars as fallback)
        lead_content = text[:500].strip()
    
    return title, lead_content


def parse_multiple_pages(dataset: 'Dataset', num_pages: int = None) -> List[Tuple[str, str]]:
    """
    Parse multiple Wikipedia pages to extract titles and lead content.
    
    Args:
        dataset: Dataset object from download_wikipedia_dataset()
        num_pages: Number of pages to parse (default: all pages)
    
    Returns:
        List of tuples containing (title, lead_content) for each page
    
    Example:
        >>> dataset = download_wikipedia_dataset("en")
        >>> pages = parse_multiple_pages(dataset, num_pages=100)
        >>> print(f"Parsed {len(pages)} pages")
    """
    if num_pages is None:
        num_pages = len(dataset)
    
    parsed_pages = []
    for i in range(min(num_pages, len(dataset))):
        title, lead = parse_page_content(dataset[i])
        parsed_pages.append((title, lead))
    
    return parsed_pages


def get_page_by_title(dataset: 'Dataset', target_title: str) -> Tuple[str, str]:
    """
    Search for a specific page by title and return its title and lead content.
    
    Args:
        dataset: Dataset object from download_wikipedia_dataset()
        target_title: The title to search for (case-sensitive)
    
    Returns:
        Tuple of (title, lead_content) if found, otherwise (None, None)
    
    Example:
        >>> dataset = download_wikipedia_dataset("en")
        >>> title, lead = get_page_by_title(dataset, "Python (programming language)")
    """
    for page in dataset:
        title = page.get("title", "")
        if title == target_title:
            return parse_page_content(page)
    
    return None, None

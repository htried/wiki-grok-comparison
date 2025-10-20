"""
Functions for downloading and parsing the Wikimedia structured Wikipedia dataset.
"""

from typing import Any, Dict, List, Tuple, Union

from datasets import Dataset


def parse_page_content(page: Dict) -> Tuple[str, str]:
    """
    Parse out the page title and abstract from a Wikipedia page entry.
    
    Args:
        page: A single Wikipedia page dictionary from the dataset
    
    Returns:
        Tuple of (title, abstract)
    
    Example:
        >>> title, abstract = parse_page_content(dataset[0])
        >>> print(f"Title: {title}")
    """
    title = page.get("name", "")
    abstract = page.get("abstract", "")
    
    return title, abstract


def parse_multiple_pages(dataset: Dataset, num_pages: int = None) -> List[Tuple[str, str]]:
    """
    Parse multiple Wikipedia pages to extract titles and abstracts.
    
    Args:
        dataset: Hugging Face Dataset object
        num_pages: Number of pages to parse (default: all pages)
    
    Returns:
        List of tuples containing (title, abstract) for each page
    
    Example:
        >>> dataset = load_dataset("wikipedia", "20220301.en")["train"]
        >>> pages = parse_multiple_pages(dataset, num_pages=100)
        >>> print(f"Parsed {len(pages)} pages")
    """
    if num_pages is None:
        num_pages = len(dataset)
    
    parsed_pages = []
    for i in range(min(num_pages, len(dataset))):
        title, abstract = parse_page_content(dataset[i])
        parsed_pages.append((title, abstract))
    
    return parsed_pages


def get_page_by_title(dataset: Dataset, target_title: str) -> Tuple[str, str]:
    """
    Search for a specific page by title and return its title and abstract.
    
    Args:
        dataset: Hugging Face Dataset object
        target_title: The title to search for (case-sensitive)
    
    Returns:
        Tuple of (title, abstract) if found, otherwise (None, None)
    
    Example:
        >>> dataset = load_dataset("wikipedia", "20220301.en")["train"]
        >>> title, abstract = get_page_by_title(dataset, "Python (programming language)")
    """
    for page in dataset:
        title = page.get("name", "")
        if title == target_title:
            return parse_page_content(page)
    
    return None, None


def extract_section_text(section: Dict) -> str:
    """
    Recursively extract text content from a section and its nested parts.
    
    Args:
        section: A section dictionary from the sections field
    
    Returns:
        Combined text content from the section
    """
    text_parts = []
    
    # Extract text from has_parts if it exists
    if "has_parts" in section:
        for part in section["has_parts"]:
            if "value" in part and part["value"]:
                text_parts.append(part["value"])
            
            # Recursively extract from nested parts
            if "has_parts" in part:
                nested_text = extract_section_text(part)
                if nested_text:
                    text_parts.append(nested_text)
    
    return " ".join(text_parts)


def extract_all_sections_text(page: Dict) -> str:
    """
    Extract all text content from the sections field of a page.
    
    Args:
        page: A single Wikipedia page dictionary from the dataset
    
    Returns:
        Combined text content from all sections
    """
    sections = page.get("sections", [])
    all_text = []
    
    for section in sections:
        section_text = extract_section_text(section)
        if section_text:
            all_text.append(section_text)
    
    return " ".join(all_text)


def extract_infobox_text(page: Dict) -> str:
    """
    Extract text content from infoboxes.
    
    Args:
        page: A single Wikipedia page dictionary from the dataset
    
    Returns:
        Combined text content from all infoboxes
    """
    infoboxes = page.get("infoboxes", [])
    infobox_text = []
    
    for infobox in infoboxes:
        if "has_parts" in infobox:
            for part in infobox["has_parts"]:
                if "has_parts" in part:
                    for subpart in part["has_parts"]:
                        if "value" in subpart and subpart["value"]:
                            infobox_text.append(subpart["value"])
                        if "name" in subpart and subpart["name"]:
                            infobox_text.append(subpart["name"])
    
    return " ".join(infobox_text)


def get_comprehensive_content(page: Dict, include_sections: bool = True, 
                            include_infoboxes: bool = True) -> str:
    """
    Get comprehensive content from a page including abstract, sections, and infoboxes.
    
    Args:
        page: A single Wikipedia page dictionary from the dataset
        include_sections: Whether to include content from sections
        include_infoboxes: Whether to include content from infoboxes
    
    Returns:
        Combined text content from the specified sources
    """
    content_parts = []
    
    # Always include abstract
    abstract = page.get("abstract", "")
    if abstract:
        content_parts.append(abstract)
    
    # Include sections if requested
    if include_sections:
        sections_text = extract_all_sections_text(page)
        if sections_text:
            content_parts.append(sections_text)
    
    # Include infoboxes if requested
    if include_infoboxes:
        infobox_text = extract_infobox_text(page)
        if infobox_text:
            content_parts.append(infobox_text)
    
    return " ".join(content_parts)

"""
Example usage of the updated Wikipedia dataset and embeddings functionality.
This demonstrates how to use the new schema-aware functions.
"""

from embeddings import (embed_dataset_pages, embed_page_from_dict,
                        find_similar_pages, load_embedding_model)
from wikipedia_dataset import (extract_all_sections_text, extract_infobox_text,
                               get_comprehensive_content, get_page_by_title,
                               parse_multiple_pages, parse_page_content)


def example_usage():
    """
    Example of how to use the updated functions with Hugging Face datasets.
    """
    # Note: This is a conceptual example. In practice, you would load a real dataset:
    # from datasets import load_dataset
    # dataset = load_dataset("wikipedia", "20220301.en")["train"]
    
    print("Example usage of updated Wikipedia dataset functions:")
    print("=" * 60)
    
    # Example page dictionary (based on the new schema)
    example_page = {
        "name": "Python (programming language)",
        "abstract": "Python is a high-level, general-purpose programming language.",
        "sections": [
            {
                "type": "section",
                "name": "History",
                "has_parts": [
                    {
                        "type": "paragraph",
                        "value": "Python was created by Guido van Rossum and first released in 1991."
                    }
                ]
            }
        ],
        "infoboxes": [
            {
                "type": "infobox",
                "name": "programming language",
                "has_parts": [
                    {
                        "has_parts": [
                            {
                                "name": "Paradigm",
                                "value": "Multi-paradigm"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # 1. Parse basic page content
    title, abstract = parse_page_content(example_page)
    print(f"Title: {title}")
    print(f"Abstract: {abstract}")
    print()
    
    # 2. Extract content from sections
    sections_text = extract_all_sections_text(example_page)
    print(f"Sections text: {sections_text}")
    print()
    
    # 3. Extract content from infoboxes
    infobox_text = extract_infobox_text(example_page)
    print(f"Infobox text: {infobox_text}")
    print()
    
    # 4. Get comprehensive content
    full_content = get_comprehensive_content(example_page)
    print(f"Comprehensive content: {full_content}")
    print()
    
    # 5. Embedding examples (conceptual - would need actual model)
    print("Embedding examples:")
    print("- embed_page_from_dict(page, content_type='abstract')")
    print("- embed_page_from_dict(page, content_type='sections')")
    print("- embed_page_from_dict(page, content_type='comprehensive')")
    print()
    
    # 6. Dataset-level operations (conceptual)
    print("Dataset operations:")
    print("- embed_dataset_pages(dataset, content_type='comprehensive')")
    print("- find_similar_pages('Python programming', dataset, content_type='comprehensive')")
    print()
    
    print("Key improvements:")
    print("- Works with Hugging Face Dataset objects")
    print("- Supports new schema with 'name' and 'abstract' fields")
    print("- Can extract content from structured sections and infoboxes")
    print("- Multiple content types for embeddings: abstract, sections, comprehensive")
    print("- Enhanced similarity search with page titles in results")

if __name__ == "__main__":
    example_usage()

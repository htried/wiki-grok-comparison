"""
Example usage of the Wikipedia dataset and embedding utilities.

This file demonstrates how to use the functions in Jupyter notebooks.
"""

from wikipedia_dataset import (
    download_wikipedia_dataset,
    parse_page_content,
    parse_multiple_pages
)
from embeddings import (
    load_embedding_model,
    embed_text,
    embed_page_content,
    compare_embeddings,
    compare_multiple_embeddings,
    find_most_similar
)


def example_download_and_parse():
    """Example: Download and parse Wikipedia pages."""
    print("=" * 60)
    print("Example 1: Download and Parse Wikipedia Pages")
    print("=" * 60)
    
    # Download the dataset (this may take a while on first run)
    print("Downloading Wikipedia dataset...")
    dataset = download_wikipedia_dataset("en")
    print(f"Downloaded dataset with {len(dataset)} articles\n")
    
    # Parse a single page
    print("Parsing first page:")
    title, lead = parse_page_content(dataset[0])
    print(f"Title: {title}")
    print(f"Lead content (first 200 chars): {lead[:200]}...\n")
    
    # Parse multiple pages
    print("Parsing first 5 pages:")
    pages = parse_multiple_pages(dataset, num_pages=5)
    for i, (title, lead) in enumerate(pages):
        print(f"{i+1}. {title} - {len(lead)} characters")
    print()


def example_embedding_single_text():
    """Example: Create embeddings for text."""
    print("=" * 60)
    print("Example 2: Create Embeddings for Text")
    print("=" * 60)
    
    # Load the embedding model
    print("Loading embedding model...")
    model = load_embedding_model()
    print("Model loaded!\n")
    
    # Embed a single text
    text = "Python is a high-level programming language."
    embedding = embed_text(text, model)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}\n")
    
    # Embed multiple texts
    texts = [
        "Python is a programming language",
        "Java is a programming language",
        "The cat sat on the mat"
    ]
    embeddings = embed_text(texts, model)
    print(f"Embedded {len(texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}\n")


def example_compare_embeddings():
    """Example: Compare embeddings."""
    print("=" * 60)
    print("Example 3: Compare Embeddings")
    print("=" * 60)
    
    # Load model and create embeddings
    model = load_embedding_model()
    
    text1 = "Python programming language"
    text2 = "Programming in Python"
    text3 = "Cooking pasta recipes"
    
    emb1 = embed_text(text1, model)
    emb2 = embed_text(text2, model)
    emb3 = embed_text(text3, model)
    
    # Compare pairs
    sim_1_2 = compare_embeddings(emb1, emb2)
    sim_1_3 = compare_embeddings(emb1, emb3)
    sim_2_3 = compare_embeddings(emb2, emb3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}\n")
    
    print(f"Similarity (1 vs 2): {sim_1_2:.4f}")
    print(f"Similarity (1 vs 3): {sim_1_3:.4f}")
    print(f"Similarity (2 vs 3): {sim_2_3:.4f}\n")


def example_wikipedia_embeddings():
    """Example: Embed Wikipedia pages and compare."""
    print("=" * 60)
    print("Example 4: Embed Wikipedia Pages and Compare")
    print("=" * 60)
    
    # Download dataset
    print("Downloading Wikipedia dataset...")
    dataset = download_wikipedia_dataset("en")
    
    # Load model
    print("Loading embedding model...")
    model = load_embedding_model()
    
    # Parse and embed a few pages
    print("\nEmbedding first 10 pages...")
    pages = parse_multiple_pages(dataset, num_pages=10)
    embeddings = []
    titles = []
    
    for title, lead in pages:
        if lead:  # Only embed if there's content
            embedding = embed_page_content(title, lead, model)
            embeddings.append(embedding)
            titles.append(title)
    
    print(f"Created embeddings for {len(embeddings)} pages\n")
    
    # Compare all pairs
    import numpy as np
    embeddings_array = np.array(embeddings)
    similarities = compare_multiple_embeddings(embeddings_array)
    
    print("Pairwise similarities (first 5x5):")
    for i in range(min(5, len(titles))):
        print(f"{titles[i][:30]:30s}", end="")
        for j in range(min(5, len(titles))):
            print(f" {similarities[i,j]:5.3f}", end="")
        print()
    print()


def example_find_similar_pages():
    """Example: Find most similar pages to a query."""
    print("=" * 60)
    print("Example 5: Find Most Similar Pages to a Query")
    print("=" * 60)
    
    # Download dataset
    print("Downloading Wikipedia dataset...")
    dataset = download_wikipedia_dataset("en")
    
    # Load model
    print("Loading embedding model...")
    model = load_embedding_model()
    
    # Parse and embed some pages
    print("\nEmbedding first 50 pages...")
    pages = parse_multiple_pages(dataset, num_pages=50)
    embeddings = []
    titles = []
    
    for title, lead in pages:
        if lead:
            embedding = embed_page_content(title, lead, model)
            embeddings.append(embedding)
            titles.append(title)
    
    import numpy as np
    embeddings_array = np.array(embeddings)
    
    # Create a query
    query_text = "Computer science and programming"
    query_embedding = embed_text(query_text, model)
    
    # Find most similar pages
    top_matches = find_most_similar(query_embedding, embeddings_array, top_k=5)
    
    print(f"\nQuery: {query_text}")
    print("\nMost similar pages:")
    for idx, score in top_matches:
        print(f"{score:.4f} - {titles[idx]}")
    print()


if __name__ == "__main__":
    """
    Run examples. 
    
    Note: The first example will download the Wikipedia dataset which can take
    significant time and storage space. Comment out examples as needed.
    """
    
    # Uncomment the examples you want to run:
    
    # example_download_and_parse()
    # example_embedding_single_text()
    example_compare_embeddings()
    # example_wikipedia_embeddings()
    # example_find_similar_pages()

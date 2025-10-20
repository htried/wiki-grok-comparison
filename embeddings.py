"""
Functions for embedding text and comparing embeddings.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Default model for embeddings
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load a sentence transformer model for creating embeddings.
    
    Args:
        model_name: Name of the sentence-transformers model to use
                   (default: "all-MiniLM-L6-v2", a fast and efficient model)
    
    Returns:
        Loaded SentenceTransformer model
    
    Example:
        >>> model = load_embedding_model()
        >>> # or use a different model
        >>> model = load_embedding_model("all-mpnet-base-v2")
    """
    model = SentenceTransformer(model_name)
    return model


def embed_text(text: Union[str, List[str]], model: SentenceTransformer = None) -> np.ndarray:
    """
    Create embeddings for text or a list of texts.
    
    Args:
        text: A single text string or list of text strings to embed
        model: Pre-loaded SentenceTransformer model (will load default if None)
    
    Returns:
        Numpy array of embeddings (shape: [num_texts, embedding_dim])
    
    Example:
        >>> model = load_embedding_model()
        >>> embedding = embed_text("Hello world", model)
        >>> print(embedding.shape)
    """
    if model is None:
        model = load_embedding_model()
    
    embeddings = model.encode(text, convert_to_numpy=True)
    return embeddings


def embed_page_content(title: str, abstract: str, model: SentenceTransformer = None) -> np.ndarray:
    """
    Create an embedding for a Wikipedia page by combining title and abstract.
    
    Args:
        title: Page title
        abstract: Page abstract/summary
        model: Pre-loaded SentenceTransformer model (will load default if None)
    
    Returns:
        Numpy array embedding for the combined text
    
    Example:
        >>> model = load_embedding_model()
        >>> embedding = embed_page_content("Python", "Python is a programming language...", model)
    """
    # Combine title and abstract with a separator
    combined_text = f"{title}. {abstract}"
    return embed_text(combined_text, model)


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compare two embeddings using cosine similarity.
    
    Args:
        embedding1: First embedding (numpy array)
        embedding2: Second embedding (numpy array)
    
    Returns:
        Cosine similarity score between -1 and 1 (higher means more similar)
    
    Example:
        >>> model = load_embedding_model()
        >>> emb1 = embed_text("Python programming", model)
        >>> emb2 = embed_text("Programming in Python", model)
        >>> similarity = compare_embeddings(emb1, emb2)
        >>> print(f"Similarity: {similarity:.4f}")
    """
    # Ensure embeddings are 2D for sklearn
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(embedding1, embedding2)[0, 0]
    return float(similarity)


def compare_multiple_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities for multiple embeddings.
    
    Args:
        embeddings: 2D numpy array where each row is an embedding
    
    Returns:
        2D numpy array of pairwise cosine similarities (shape: [n, n])
    
    Example:
        >>> model = load_embedding_model()
        >>> texts = ["Python", "Java", "JavaScript"]
        >>> embeddings = embed_text(texts, model)
        >>> similarities = compare_multiple_embeddings(embeddings)
        >>> print(similarities)
    """
    similarities = cosine_similarity(embeddings)
    return similarities


def find_most_similar(query_embedding: np.ndarray, 
                      candidate_embeddings: np.ndarray, 
                      top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Find the most similar embeddings to a query embedding.
    
    Args:
        query_embedding: Query embedding (1D or 2D numpy array)
        candidate_embeddings: 2D numpy array of candidate embeddings to compare against
        top_k: Number of top similar candidates to return
    
    Returns:
        List of tuples (index, similarity_score) sorted by similarity (highest first)
    
    Example:
        >>> model = load_embedding_model()
        >>> query = embed_text("Python programming", model)
        >>> candidates = embed_text(["Java", "C++", "Python", "Ruby"], model)
        >>> top_matches = find_most_similar(query, candidates, top_k=2)
        >>> for idx, score in top_matches:
        ...     print(f"Index {idx}: {score:.4f}")
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return as list of (index, score) tuples
    results = [(int(idx), float(similarities[idx])) for idx in top_indices]
    return results


def embed_page_from_dict(page: Dict, model: SentenceTransformer = None, 
                        content_type: str = "abstract") -> np.ndarray:
    """
    Create an embedding for a Wikipedia page from a page dictionary.
    
    Args:
        page: Wikipedia page dictionary from the dataset
        model: Pre-loaded SentenceTransformer model (will load default if None)
        content_type: Type of content to embed ("abstract", "sections", "comprehensive")
    
    Returns:
        Numpy array embedding for the specified content
    
    Example:
        >>> model = load_embedding_model()
        >>> embedding = embed_page_from_dict(page, model, "comprehensive")
    """
    if model is None:
        model = load_embedding_model()
    
    title = page.get("name", "")
    
    if content_type == "abstract":
        abstract = page.get("abstract", "")
        return embed_page_content(title, abstract, model)
    
    elif content_type == "sections":
        # Import here to avoid circular imports
        from wikipedia_dataset import extract_all_sections_text
        sections_text = extract_all_sections_text(page)
        combined_text = f"{title}. {sections_text}"
        return embed_text(combined_text, model)
    
    elif content_type == "comprehensive":
        # Import here to avoid circular imports
        from wikipedia_dataset import get_comprehensive_content
        full_content = get_comprehensive_content(page)
        combined_text = f"{title}. {full_content}"
        return embed_text(combined_text, model)
    
    else:
        raise ValueError("content_type must be 'abstract', 'sections', or 'comprehensive'")


def embed_dataset_pages(dataset: Dataset, model: SentenceTransformer = None, 
                       content_type: str = "abstract", num_pages: int = None) -> np.ndarray:
    """
    Create embeddings for multiple pages in a dataset.
    
    Args:
        dataset: Hugging Face Dataset object
        model: Pre-loaded SentenceTransformer model (will load default if None)
        content_type: Type of content to embed ("abstract", "sections", "comprehensive")
        num_pages: Number of pages to embed (default: all pages)
    
    Returns:
        2D numpy array of embeddings (shape: [num_pages, embedding_dim])
    
    Example:
        >>> model = load_embedding_model()
        >>> embeddings = embed_dataset_pages(dataset, model, "comprehensive", 100)
    """
    if model is None:
        model = load_embedding_model()
    
    if num_pages is None:
        num_pages = len(dataset)
    
    embeddings = []
    for i in range(min(num_pages, len(dataset))):
        embedding = embed_page_from_dict(dataset[i], model, content_type)
        embeddings.append(embedding)
    
    return np.vstack(embeddings)


def find_similar_pages(query_text: str, dataset: Dataset, model: SentenceTransformer = None,
                      content_type: str = "abstract", top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Find the most similar pages in a dataset to a query text.
    
    Args:
        query_text: Text to search for
        dataset: Hugging Face Dataset object
        model: Pre-loaded SentenceTransformer model (will load default if None)
        content_type: Type of content to compare ("abstract", "sections", "comprehensive")
        top_k: Number of top similar pages to return
    
    Returns:
        List of tuples (index, title, similarity_score) sorted by similarity (highest first)
    
    Example:
        >>> model = load_embedding_model()
        >>> results = find_similar_pages("Python programming", dataset, model, "comprehensive")
        >>> for idx, title, score in results:
        ...     print(f"{title}: {score:.4f}")
    """
    if model is None:
        model = load_embedding_model()
    
    # Create query embedding
    query_embedding = embed_text(query_text, model)
    
    # Create embeddings for all pages
    page_embeddings = embed_dataset_pages(dataset, model, content_type)
    
    # Find most similar
    similar_indices = find_most_similar(query_embedding, page_embeddings, top_k)
    
    # Add page titles to results
    results = []
    for idx, score in similar_indices:
        title = dataset[idx].get("name", f"Page {idx}")
        results.append((idx, title, score))
    
    return results

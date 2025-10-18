# wiki-grok-comparison
Comparison between Wikipedia and Grokipedia

## Overview

This repository provides Python utilities for working with the Wikimedia structured Wikipedia dataset in Jupyter notebooks. The tools allow you to:

1. Download the latest Wikimedia/structured-wikipedia English Wikipedia dataset
2. Parse out page titles and lead content
3. Embed page titles and lead content using sentence transformers
4. Compare embeddings using cosine similarity

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download and Parse Wikipedia Data

```python
from wikipedia_dataset import download_wikipedia_dataset, parse_page_content, parse_multiple_pages

# Download the English Wikipedia dataset
dataset = download_wikipedia_dataset("en")

# Parse a single page
title, lead = parse_page_content(dataset[0])
print(f"Title: {title}")
print(f"Lead: {lead[:200]}...")

# Parse multiple pages
pages = parse_multiple_pages(dataset, num_pages=100)
```

### 2. Create Embeddings

```python
from embeddings import load_embedding_model, embed_text, embed_page_content

# Load the embedding model
model = load_embedding_model()

# Embed a single text
embedding = embed_text("Python is a programming language", model)

# Embed a Wikipedia page (title + lead content)
embedding = embed_page_content(title, lead, model)

# Embed multiple texts at once
texts = ["Python", "Java", "C++"]
embeddings = embed_text(texts, model)
```

### 3. Compare Embeddings

```python
from embeddings import compare_embeddings, compare_multiple_embeddings, find_most_similar

# Compare two embeddings
emb1 = embed_text("Python programming", model)
emb2 = embed_text("Programming in Python", model)
similarity = compare_embeddings(emb1, emb2)
print(f"Similarity: {similarity:.4f}")

# Compare multiple embeddings (pairwise)
embeddings = embed_text(["Python", "Java", "Cooking"], model)
similarities = compare_multiple_embeddings(embeddings)

# Find most similar embeddings to a query
query = embed_text("machine learning", model)
candidates = embed_text(["AI", "cooking", "neural networks", "recipes"], model)
top_matches = find_most_similar(query, candidates, top_k=2)
```

### 4. Complete Example

See `example_usage.py` for complete working examples that demonstrate all functionality.

## Files

- `requirements.txt` - Python dependencies
- `wikipedia_dataset.py` - Functions for downloading and parsing Wikipedia data
- `embeddings.py` - Functions for creating and comparing embeddings
- `example_usage.py` - Example usage demonstrating all features
- `example_notebook.ipynb` - Jupyter notebook with interactive examples

## Notes

- The Wikipedia dataset is large and may take significant time to download on first use
- The default embedding model is `all-MiniLM-L6-v2` (fast and efficient)
- You can use different models by passing the model name to `load_embedding_model()`

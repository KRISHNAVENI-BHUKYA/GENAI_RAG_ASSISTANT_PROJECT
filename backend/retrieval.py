import re
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Retrieval")


def chunk_text(text: str, max_words: int = 100, overlap_words: int = 20) -> list:
    """
    Splits long text blocks into smaller, overlapping chunks to preserve local context.
    Default target is 100 words (~150 tokens) per chunk, which is optimal for small documents.
    """
    words = re.findall(r'\S+', text)
    if len(words) <= max_words:
        return [text]

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        # Move forward by max_words - overlap_words to create overlapping chunks
        i += (max_words - overlap_words)
        
    return chunks


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Computes standard cosine similarity between two numeric lists/vectors using numpy.
    Returns a score between -1.0 and 1.0 (normally 0.0 to 1.0 for positive text embeddings).
    """
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
        
    return float(dot_product / (norm_a * norm_b))


def search_similar_documents(query_vector: list, document_chunks: list, threshold: float = 0.70, top_k: int = 3) -> list:
    """
    Compares the query vector against all document chunk vectors.
    Ranks them by cosine similarity and returns the top_k results above the threshold.
    
    document_chunks format:
    [
        {
            "id": "doc_id_chunk_idx",
            "title": "Document Title",
            "content": "Chunked text contents...",
            "embedding": [0.23, -0.45, ...]
        }
    ]
    """
    results = []
    
    for idx, chunk in enumerate(document_chunks):
        chunk_vector = chunk.get("embedding")
        if not chunk_vector:
            logger.warning(f"Chunk at index {idx} in document '{chunk.get('title')}' is missing an embedding. Skipping.")
            continue
            
        score = cosine_similarity(query_vector, chunk_vector)
        
        results.append({
            "chunk": chunk,
            "score": score
        })
        
    # Sort by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Filter by threshold
    filtered_results = [r for r in results if r["score"] >= threshold]
    
    logger.info(f"Similarity search finished. Found {len(filtered_results)} / {len(results)} chunks above threshold {threshold}.")
    
    return filtered_results[:top_k]

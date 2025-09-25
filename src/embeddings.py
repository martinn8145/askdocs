import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from loader import load_and_chunk, get_chunk_stats

# Initialize the sentence transformer model (will download on first use)
@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_local_embeddings(
    texts: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings using local sentence transformer model.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for processing (default: 32)

    Returns:
        List of embedding vectors
    """
    try:
        # Load the model
        model = load_embedding_model()

        # Generate embeddings
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

        # Convert to list format
        return embeddings.tolist()

    except Exception as e:
        raise Exception(f"Error generating local embeddings: {str(e)}")


def embed_document_chunks(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Tuple[List[List[float]], List[str], Dict[str, Any]]:
    """
    Load a document, chunk it, and generate embeddings for each chunk.

    Args:
        file_path: Path to the document file
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        Tuple of (embeddings, chunk_texts, metadata)
    """
    # Load and chunk the document
    chunks = load_and_chunk(file_path, chunk_size, chunk_overlap)

    if not chunks:
        raise ValueError("No chunks were generated from the document")

    # Generate embeddings
    embeddings = get_local_embeddings(chunks)

    # Create metadata
    chunk_stats = get_chunk_stats(chunks)
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model": "all-MiniLM-L6-v2",
        "stats": chunk_stats
    }

    return embeddings, chunks, metadata


def embed_multiple_documents(
    file_paths: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Tuple[np.ndarray, List[str], List[str], List[int], List[Dict[str, Any]]]:
    """
    Process multiple documents and generate embeddings for visualization.

    Args:
        file_paths: List of paths to document files
        chunk_size: Size of each chunk in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        Tuple of (embeddings_array, chunk_texts, document_names, chunk_indices, metadata_list)
    """
    all_embeddings = []
    all_chunk_texts = []
    all_document_names = []
    all_chunk_indices = []
    all_metadata = []

    for file_path in file_paths:
        try:
            # Process each document
            embeddings, chunks, metadata = embed_document_chunks(
                file_path, chunk_size, chunk_overlap
            )

            # Add to combined lists
            all_embeddings.extend(embeddings)
            all_chunk_texts.extend(chunks)

            # Create document name and chunk indices
            doc_name = os.path.splitext(os.path.basename(file_path))[0]
            all_document_names.extend([doc_name] * len(chunks))
            all_chunk_indices.extend(list(range(len(chunks))))
            all_metadata.append(metadata)

        except Exception as e:
            st.error(f"Error processing {file_path}: {str(e)}")
            continue

    if not all_embeddings:
        raise ValueError("No embeddings were generated from any documents")

    return (
        np.array(all_embeddings),
        all_chunk_texts,
        all_document_names,
        all_chunk_indices,
        all_metadata
    )


def create_sample_embeddings_from_file(
    file_path: str,
    n_samples: int = 10,
    chunk_size: int = 200,
    embedding_dim: int = 384,  # Updated for MiniLM
    random_state: int = 42
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    """
    Create sample embeddings from a real document file for testing.

    Args:
        file_path: Path to the document file
        n_samples: Maximum number of chunks to sample
        chunk_size: Size of each chunk in tokens
        embedding_dim: Embedding dimensionality (384 for MiniLM)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (embeddings, chunk_texts, document_names, chunk_indices)
    """
    try:
        # Load and chunk the document
        chunks = load_and_chunk(file_path, chunk_size=chunk_size, chunk_overlap=25)

        if not chunks:
            raise ValueError("No chunks were generated from the document")

        # Limit to n_samples
        if len(chunks) > n_samples:
            np.random.seed(random_state)
            indices = np.random.choice(len(chunks), n_samples, replace=False)
            chunks = [chunks[i] for i in sorted(indices)]

        # Generate real embeddings using local model
        embeddings = get_local_embeddings(chunks)

        # Create metadata
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        document_names = [doc_name] * len(chunks)
        chunk_indices = list(range(len(chunks)))

        return np.array(embeddings), chunks, document_names, chunk_indices

    except Exception as e:
        st.error(f"Error creating sample embeddings: {str(e)}")
        # Fallback to completely synthetic data
        from visualization import generate_sample_embeddings
        return generate_sample_embeddings(n_docs=1, chunks_per_doc=n_samples, embedding_dim=embedding_dim, random_state=random_state)


def get_embedding_stats(embeddings: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics about the embeddings.

    Args:
        embeddings: Array of embeddings

    Returns:
        Dictionary with embedding statistics
    """
    return {
        "num_embeddings": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "mean_norm": np.mean(np.linalg.norm(embeddings, axis=1)),
        "std_norm": np.std(np.linalg.norm(embeddings, axis=1)),
        "memory_usage_mb": embeddings.nbytes / (1024 * 1024)
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing embeddings module...")

    # Test with sample document
    test_file = "../test/example.txt"

    try:
        embeddings, chunks, doc_names, chunk_indices = create_sample_embeddings_from_file(
            test_file, n_samples=5
        )

        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {embeddings.shape}")
        print(f"First chunk preview: {chunks[0][:100]}...")

        stats = get_embedding_stats(embeddings)
        print(f"Embedding stats: {stats}")

    except Exception as e:
        print(f"Test failed: {e}")


class DocumentEmbeddingStore:
    """Store and retrieve document embeddings using FAISS."""

    def __init__(self, embedding_dim: int = 384):  # Updated for MiniLM
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
        self.chunks = []
        self.metadata = []

    def add_chunks(self, embeddings: np.ndarray, chunks: List[str], doc_metadata: Dict[str, Any]):
        """Add document chunks and their embeddings to the store."""
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to FAISS index
        self.index.add(embeddings_normalized.astype('float32'))

        # Store chunks and metadata
        self.chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            self.metadata.append({
                **doc_metadata,
                'chunk_index': len(self.chunks) - len(chunks) + i,
                'chunk_text': chunk
            })

    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[Dict]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return [], [], []

        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.index.search(query_normalized, min(k, self.index.ntotal))

        # Return results
        results_chunks = [self.chunks[idx] for idx in indices[0]]
        results_scores = scores[0].tolist()
        results_metadata = [self.metadata[idx] for idx in indices[0]]

        return results_chunks, results_scores, results_metadata

    def clear(self):
        """Clear all stored embeddings and chunks."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        self.metadata = []


# Global embedding store (updated for MiniLM dimensions)
embedding_store = DocumentEmbeddingStore()


def embed_chunks(chunks: List[str], doc_metadata: Dict[str, Any]) -> bool:
    """
    Generate embeddings for document chunks and store them.

    Args:
        chunks: List of text chunks
        doc_metadata: Document metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate embeddings using local model
        embeddings_list = get_local_embeddings(chunks)
        embeddings = np.array(embeddings_list)

        # Add to store
        embedding_store.add_chunks(embeddings, chunks, doc_metadata)

        return True

    except Exception as e:
        st.error(f"Error embedding chunks: {str(e)}")
        return False


def retrieve_chunks(query: str, k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Retrieve relevant chunks for a query.

    Args:
        query: User query
        k: Number of chunks to retrieve

    Returns:
        Tuple of (chunks, scores)
    """
    try:
        # Generate query embedding using local model
        query_embeddings = get_local_embeddings([query])
        query_embedding = np.array(query_embeddings[0])

        # Search for similar chunks
        chunks, scores, metadata = embedding_store.search(query_embedding, k)

        return chunks, scores

    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return [], []


def get_embedding_store() -> DocumentEmbeddingStore:
    """Get the global embedding store."""
    return embedding_store


def clear_embedding_store():
    """Clear the global embedding store."""
    global embedding_store
    embedding_store.clear()
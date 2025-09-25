import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st


def reduce_embeddings_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D using t-SNE algorithm.

    Args:
        embeddings: Array of shape (n_samples, n_features) containing embeddings
        perplexity: t-SNE perplexity parameter (5-50, default=30)
        learning_rate: t-SNE learning rate (10-1000, default=200)
        n_iter: Number of iterations for optimization (default=1000)
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 2) with 2D coordinates
    """
    if embeddings.shape[0] < 4:
        raise ValueError("t-SNE requires at least 4 samples")

    # Adjust perplexity if we have fewer samples
    max_perplexity = min(perplexity, (embeddings.shape[0] - 1) // 3)
    if max_perplexity < perplexity:
        st.warning(f"Adjusted perplexity from {perplexity} to {max_perplexity} due to small sample size")
        perplexity = max_perplexity

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=random_state,
        init='random'
    )

    return tsne.fit_transform(embeddings)


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D using UMAP algorithm.

    Args:
        embeddings: Array of shape (n_samples, n_features) containing embeddings
        n_neighbors: UMAP n_neighbors parameter (2-100, default=15)
        min_dist: UMAP min_dist parameter (0.0-0.99, default=0.1)
        n_components: Number of dimensions to reduce to (default=2)
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, n_components) with reduced coordinates
    """
    if embeddings.shape[0] < 4:
        raise ValueError("UMAP requires at least 4 samples")

    # Adjust n_neighbors if we have fewer samples
    max_neighbors = min(n_neighbors, embeddings.shape[0] - 1)
    if max_neighbors < n_neighbors:
        st.warning(f"Adjusted n_neighbors from {n_neighbors} to {max_neighbors} due to small sample size")
        n_neighbors = max_neighbors

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )

    return reducer.fit_transform(embeddings)


def create_embedding_scatter_plot(
    embeddings_2d: np.ndarray,
    labels: List[str],
    colors: List[str],
    texts: Optional[List[str]] = None,
    title: str = "Document Embeddings Visualization",
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create an interactive scatter plot of 2D embeddings using Plotly.

    Args:
        embeddings_2d: Array of shape (n_samples, 2) with 2D coordinates
        labels: List of labels for each point (e.g., "Doc1_Chunk3")
        colors: List of color categories for each point
        texts: Optional list of text snippets to show on hover
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        Plotly figure object
    """
    if texts is None:
        texts = [f"Chunk {i+1}" for i in range(len(embeddings_2d))]

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
        'color': colors,
        'text': texts
    })

    # Create scatter plot with color coding
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='color',
        hover_name='label',
        hover_data={
            'x': ':.3f',
            'y': ':.3f',
            'color': False
        },
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2'},
        width=width,
        height=height
    )

    # Add custom hover text
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'X: %{x:.3f}<br>' +
                     'Y: %{y:.3f}<br>' +
                     'Text: %{customdata}<br>' +
                     '<extra></extra>',
        customdata=texts,
        hovertext=labels
    )

    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            title="Document/Chunk",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=50, r=150, t=50, b=50)
    )

    return fig


def prepare_visualization_data(
    embeddings: np.ndarray,
    chunk_texts: List[str],
    document_names: List[str],
    chunk_indices: List[int]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare data for visualization by creating labels, colors, and hover texts.

    Args:
        embeddings: Array of embeddings
        chunk_texts: List of text chunks
        document_names: List of document names for each chunk
        chunk_indices: List of chunk indices within each document

    Returns:
        Tuple of (labels, colors, hover_texts)
    """
    labels = []
    colors = []
    hover_texts = []

    for i, (doc_name, chunk_idx, text) in enumerate(zip(document_names, chunk_indices, chunk_texts)):
        # Create label
        label = f"{doc_name}_Chunk{chunk_idx + 1}"
        labels.append(label)

        # Use document name as color category
        colors.append(doc_name)

        # Create hover text (truncate long texts)
        hover_text = text[:200] + "..." if len(text) > 200 else text
        hover_texts.append(hover_text)

    return labels, colors, hover_texts


def create_embedding_visualization(
    embeddings: np.ndarray,
    chunk_texts: List[str],
    document_names: List[str],
    chunk_indices: List[int],
    method: str = "tsne",
    **kwargs
) -> go.Figure:
    """
    Complete pipeline to create embedding visualization.

    Args:
        embeddings: Array of high-dimensional embeddings
        chunk_texts: List of text chunks
        document_names: List of document names for each chunk
        chunk_indices: List of chunk indices within each document
        method: Dimensionality reduction method ("tsne" or "umap")
        **kwargs: Additional parameters for the reduction algorithm

    Returns:
        Plotly figure object
    """
    # Validate inputs
    if len(embeddings) != len(chunk_texts) != len(document_names) != len(chunk_indices):
        raise ValueError("All input lists must have the same length")

    if len(embeddings) < 4:
        raise ValueError("Need at least 4 embeddings for visualization")

    # Reduce dimensionality
    if method.lower() == "tsne":
        embeddings_2d = reduce_embeddings_tsne(embeddings, **kwargs)
        title = "Document Embeddings (t-SNE)"
    elif method.lower() == "umap":
        embeddings_2d = reduce_embeddings_umap(embeddings, **kwargs)
        title = "Document Embeddings (UMAP)"
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    # Prepare visualization data
    labels, colors, hover_texts = prepare_visualization_data(
        embeddings, chunk_texts, document_names, chunk_indices
    )

    # Create scatter plot
    fig = create_embedding_scatter_plot(
        embeddings_2d, labels, colors, hover_texts, title
    )

    return fig


def generate_sample_embeddings(
    n_docs: int = 3,
    chunks_per_doc: int = 5,
    embedding_dim: int = 384,  # Updated for MiniLM
    random_state: int = 42
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    """
    Generate sample embeddings for testing visualization.

    Args:
        n_docs: Number of documents
        chunks_per_doc: Number of chunks per document
        embedding_dim: Embedding dimensionality (384 for MiniLM)
        random_state: Random seed

    Returns:
        Tuple of (embeddings, chunk_texts, document_names, chunk_indices)
    """
    np.random.seed(random_state)

    embeddings = []
    chunk_texts = []
    document_names = []
    chunk_indices = []

    sample_texts = [
        "This is a sample chunk about machine learning and artificial intelligence.",
        "The document discusses various aspects of natural language processing.",
        "Technical specifications and implementation details are covered here.",
        "Research findings and experimental results are presented in this section.",
        "Conclusions and future work directions are outlined in the final part."
    ]

    for doc_idx in range(n_docs):
        doc_name = f"Document_{doc_idx + 1}"

        # Generate document-specific embeddings (with some clustering)
        doc_center = np.random.randn(embedding_dim) * 0.5

        for chunk_idx in range(chunks_per_doc):
            # Add some noise around document center
            embedding = doc_center + np.random.randn(embedding_dim) * 0.3
            embedding = embedding / np.linalg.norm(embedding)  # Normalize

            embeddings.append(embedding)
            chunk_texts.append(f"{sample_texts[chunk_idx % len(sample_texts)]} (from {doc_name})")
            document_names.append(doc_name)
            chunk_indices.append(chunk_idx)

    return np.array(embeddings), chunk_texts, document_names, chunk_indices
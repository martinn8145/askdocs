import streamlit as st
import os
from typing import List, Dict, Any
import time
import numpy as np
from visualization import create_embedding_visualization, generate_sample_embeddings
from embeddings import create_sample_embeddings_from_file, get_embedding_stats, embed_chunks, clear_embedding_store, get_embedding_store
from loader import load_and_chunk
from qa_chain import get_answer

st.set_page_config(
    page_title="AskDocs - AI-Powered Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def decode_bytes_with_fallback(content_bytes: bytes) -> str:
    """
    Decode bytes to string with multiple encoding fallbacks.

    Args:
        content_bytes: Bytes content to decode

    Returns:
        str: Decoded string content

    Raises:
        Exception: If all encoding attempts fail
    """
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    # If all encodings fail, raise an error
    raise Exception(f"Could not decode content with any of the attempted encodings: {encodings}")

def initialize_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    if "valid_files" not in st.session_state:
        st.session_state.valid_files = []
    if "embeddings_data" not in st.session_state:
        st.session_state.embeddings_data = None
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processing_messages" not in st.session_state:
        st.session_state.processing_messages = {"success": [], "error": []}

def validate_uploaded_files(uploaded_files):
    """Validate file size and count limits"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes (increased for local processing)
    MAX_FILES = 10  # Increased from 3 to 10

    valid_files = []
    errors = []

    if len(uploaded_files) > MAX_FILES:
        errors.append(f"Maximum {MAX_FILES} files allowed. You uploaded {len(uploaded_files)} files.")
        uploaded_files = uploaded_files[:MAX_FILES]  # Take only first 10 files

    for file in uploaded_files:
        if file.size > MAX_FILE_SIZE:
            errors.append(f"File '{file.name}' is {file.size / (1024*1024):.1f}MB. Maximum size is 50MB.")
        else:
            valid_files.append(file)

    return valid_files, errors

def format_file_size(bytes_size):
    """Convert bytes to human readable format"""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

def process_uploaded_files(uploaded_files):
    """Process uploaded files and create embeddings using local models."""
    if not uploaded_files:
        return False

    # Clear previous messages
    st.session_state.processing_messages = {"success": [], "error": []}

    try:
        # Clear previous embeddings
        clear_embedding_store()
        st.session_state.documents_processed = False

        # Create progress bar for multiple files
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_files = []
        failed_files = []

        for i, file_obj in enumerate(uploaded_files):
            # Update progress
            progress_bar.progress((i) / len(uploaded_files))
            status_text.text(f"Processing {file_obj.name} ({i+1}/{len(uploaded_files)})...")

            # Save file temporarily
            temp_path = f"temp_{file_obj.name}"
            with open(temp_path, "wb") as f:
                f.write(file_obj.getvalue())

            try:
                # Load and chunk document
                chunks = load_and_chunk(temp_path)

                if chunks:
                    # Create metadata
                    doc_metadata = {
                        "file_name": file_obj.name,
                        "file_size": file_obj.size,
                        "file_type": file_obj.type
                    }

                    # Generate local embeddings
                    success = embed_chunks(chunks, doc_metadata)

                    if success:
                        processed_files.append((file_obj.name, len(chunks)))
                    else:
                        failed_files.append((file_obj.name, "Failed to generate embeddings"))

                else:
                    failed_files.append((file_obj.name, "No chunks generated"))

            except Exception as e:
                failed_files.append((file_obj.name, str(e)))

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Complete progress and clean up UI elements
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        # Store results in session state for persistent display
        store = get_embedding_store()
        if store.index.ntotal > 0:
            st.session_state.documents_processed = True
            success_msg = f"Successfully processed {len(processed_files)} document(s) with {store.index.ntotal} total chunks!"
            for filename, chunk_count in processed_files:
                success_msg += f"\n• {filename}: {chunk_count} chunks"
            st.session_state.processing_messages["success"].append(success_msg)

        if failed_files:
            error_msg = "Some files failed to process:"
            for filename, error in failed_files:
                error_msg += f"\n• {filename}: {error}"
            st.session_state.processing_messages["error"].append(error_msg)

        return len(processed_files) > 0

    except Exception as e:
        error_msg = f"Error during document processing: {str(e)}"
        st.session_state.processing_messages["error"].append(error_msg)
        return False


def render_visualization_tab():
    """Render the embedding visualization tab."""
    st.header("Document Embeddings Visualization")
    st.markdown("Explore document relationships through 2D embeddings using t-SNE or UMAP algorithms.")

    # Algorithm selection
    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "Dimensionality Reduction Algorithm",
            ["t-SNE", "UMAP"],
            help="Choose between t-SNE (better for local structure) or UMAP (better for global structure)"
        )

    with col2:
        data_source = st.selectbox(
            "Data Source",
            ["Sample Data", "Uploaded Documents"],
            help="Use sample data for testing or your uploaded documents"
        )

    # Algorithm-specific parameters
    if algorithm == "t-SNE":
        col3, col4 = st.columns(2)
        with col3:
            perplexity = st.slider(
                "Perplexity",
                min_value=5,
                max_value=50,
                value=30,
                help="Balance between local and global aspects (5-50)"
            )
        with col4:
            learning_rate = st.slider(
                "Learning Rate",
                min_value=10,
                max_value=1000,
                value=200,
                help="Step size for optimization (10-1000)"
            )

        algo_params = {"perplexity": perplexity, "learning_rate": learning_rate}

    else:  # UMAP
        col3, col4 = st.columns(2)
        with col3:
            n_neighbors = st.slider(
                "N Neighbors",
                min_value=2,
                max_value=100,
                value=15,
                help="Number of neighboring points used for manifold approximation"
            )
        with col4:
            min_dist = st.slider(
                "Min Distance",
                min_value=0.0,
                max_value=0.99,
                value=0.1,
                step=0.05,
                help="Minimum distance between points in low-dimensional representation"
            )

        algo_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}

    # Generate visualization button
    col_vis, col_refresh = st.columns([2, 1])

    with col_vis:
        generate_viz = st.button("Generate Visualization", type="primary", use_container_width=True)

    with col_refresh:
        if st.button("Clear Cache", use_container_width=True):
            st.session_state.embeddings_data = None
            st.rerun()

    # Generate and display visualization
    if generate_viz:
        try:
            with st.spinner(f"Generating {algorithm} visualization..."):
                if data_source == "Sample Data":
                    # Use sample data
                    embeddings, chunk_texts, document_names, chunk_indices = generate_sample_embeddings(
                        n_docs=3, chunks_per_doc=8, random_state=42
                    )
                    st.success("Sample embeddings generated!")

                else:
                    # Use uploaded documents
                    if not st.session_state.valid_files:
                        st.error("No uploaded documents found. Please upload documents first.")
                        return

                    # For demo, use the first uploaded file
                    if len(st.session_state.valid_files) > 0:
                        # Create a temporary file to work with
                        temp_file = "temp_doc.txt"
                        try:
                            with open(temp_file, "w", encoding="utf-8") as f:
                                content = st.session_state.valid_files[0].read()
                                if isinstance(content, bytes):
                                    content = decode_bytes_with_fallback(content)
                                f.write(content)
                        except Exception as e:
                            st.error(f"Error processing uploaded document: {str(e)}")
                            st.error("The document may contain unsupported characters or encoding. Please try saving the file as UTF-8 text.")
                            return

                        embeddings, chunk_texts, document_names, chunk_indices = create_sample_embeddings_from_file(
                            temp_file, n_samples=15, chunk_size=300
                        )

                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)

                        st.success(f"Embeddings generated from {len(st.session_state.valid_files)} document(s)!")
                    else:
                        st.error("Could not process uploaded documents.")
                        return

                # Create visualization
                method_mapping = {"t-SNE": "tsne", "UMAP": "umap"}
                fig = create_embedding_visualization(
                    embeddings, chunk_texts, document_names, chunk_indices,
                    method=method_mapping[algorithm], **algo_params
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                # Store embeddings for analysis
                st.session_state.embeddings_data = {
                    "embeddings": embeddings,
                    "chunk_texts": chunk_texts,
                    "document_names": document_names,
                    "chunk_indices": chunk_indices
                }

                # Display statistics
                st.subheader("Embedding Statistics")
                col1, col2, col3, col4 = st.columns(4)

                stats = get_embedding_stats(embeddings)

                with col1:
                    st.metric("Total Chunks", stats["num_embeddings"])
                with col2:
                    st.metric("Embedding Dim", stats["embedding_dimension"])
                with col3:
                    st.metric("Mean Norm", f"{stats['mean_norm']:.3f}")
                with col4:
                    st.metric("Memory Usage", f"{stats['memory_usage_mb']:.1f} MB")

                # Display document breakdown
                if st.session_state.embeddings_data:
                    st.subheader("Document Breakdown")
                    doc_counts = {}
                    for doc_name in document_names:
                        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1

                    for doc_name, count in doc_counts.items():
                        st.write(f"**{doc_name}:** {count} chunks")

        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            st.exception(e)

    # Display cached visualization if available
    elif st.session_state.embeddings_data is not None:
        st.info("Previous visualization cached. Click 'Generate Visualization' to create a new one with different parameters.")

def main():
    initialize_session_state()

    # Header
    st.title("AskDocs")
    st.subheader("AI-Powered Document Q&A with RAG")

    # Create tabs
    tab1, tab2 = st.tabs(["Document Q&A", "Embedding Visualization"])

    with tab1:
        render_main_qa_tab()

    with tab2:
        render_visualization_tab()

def render_main_qa_tab():
    """Render the main Q&A tab."""

    # Sidebar for file management
    with st.sidebar:
        st.header("Document Management")

        # File uploader with validation
        st.markdown("**File Upload Limits (Local Processing):**")
        st.markdown("• Maximum 10 files per session")
        st.markdown("• Maximum 50MB per file")
        st.markdown("• Supported formats: PDF, TXT")
        st.markdown("")

        uploaded_files = st.file_uploader(
            "Choose your documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="The drag-and-drop area shows Streamlit's default 200MB limit, but this app enforces a 50MB limit per file."
        )

        # Validate uploaded files
        if uploaded_files:
            valid_files, errors = validate_uploaded_files(uploaded_files)
            st.session_state.valid_files = valid_files

            # Show validation errors
            for error in errors:
                st.error(error)
        else:
            st.session_state.valid_files = []

        # Display valid uploaded files
        if st.session_state.valid_files:
            st.subheader(f"Uploaded Files ({len(st.session_state.valid_files)}/3)")
            for idx, file in enumerate(st.session_state.valid_files):
                with st.expander(f"File: {file.name}"):
                    st.write(f"**File type:** {file.type}")
                    st.write(f"**File size:** {format_file_size(file.size)}")

                    if st.button(f"Remove {file.name}", key=f"remove_{idx}"):
                        st.session_state.valid_files.pop(idx)
                        st.session_state.documents_processed = False
                        clear_embedding_store()
                        st.rerun()

        st.markdown("---")

        # Processing status
        st.subheader("Processing Status")
        if st.session_state.valid_files:
            if not st.session_state.documents_processed:
                st.warning(f"{len(st.session_state.valid_files)} file(s) ready for processing")

                if st.button("Process Documents", type="primary", use_container_width=True):
                    process_uploaded_files(st.session_state.valid_files)
                    st.rerun()
            else:
                store = get_embedding_store()
                st.success(f"{len(st.session_state.valid_files)} file(s) processed ({store.index.ntotal} chunks indexed)")
                if st.button("Reprocess Documents", use_container_width=True):
                    process_uploaded_files(st.session_state.valid_files)
                    st.rerun()

            remaining_slots = 10 - len(st.session_state.valid_files)
            if remaining_slots > 0:
                st.info(f"{remaining_slots} more file slot(s) available")
        else:
            st.info("No files uploaded yet (0/10 files)")

        # Display persistent processing messages
        if st.session_state.processing_messages["success"]:
            for msg in st.session_state.processing_messages["success"]:
                st.success(msg)

        if st.session_state.processing_messages["error"]:
            for msg in st.session_state.processing_messages["error"]:
                st.error(msg)

            # Clear messages button
            if st.button("Clear Messages", key="clear_messages"):
                st.session_state.processing_messages = {"success": [], "error": []}
                st.rerun()

        # Visualization status
        st.markdown("---")
        st.subheader("Local AI Models")
        st.info("• Embeddings: all-MiniLM-L6-v2")
        st.info("• Q&A Model: google/flan-t5-base")
        if st.session_state.embeddings_data:
            st.success("Cached embeddings ready for visualization")
        st.checkbox("Enable Vector Search Debug", disabled=True, help="Coming soon: FAISS search insights")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query section
        st.header("Ask Your Questions")

        query = st.text_area(
            "Enter your question about the uploaded documents:",
            placeholder="e.g., What are the main findings in the research paper? What does the contract say about payment terms?",
            height=100,
            help="Ask any question about your uploaded documents. The AI will search through the content to provide accurate answers."
        )

        col_search, col_clear = st.columns([1, 1])
        with col_search:
            search_button = st.button("Search Documents", type="primary", use_container_width=True)
        with col_clear:
            if st.button("Clear History", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()

        # Answer section
        st.header("AI Responses")

        # Handle different states
        if not st.session_state.valid_files:
            st.info("Please upload some documents in the sidebar to get started!")
        elif not st.session_state.documents_processed:
            st.info("Please process your uploaded documents first using the 'Process Documents' button in the sidebar!")
        elif not query and not st.session_state.conversation_history:
            st.info("Your documents are ready! Ask a question to see AI-powered answers.")
        else:
            # Handle search button click
            if search_button and query.strip():
                if st.session_state.documents_processed:
                    with st.spinner("Analyzing documents and generating response..."):
                        # Use RAG pipeline
                        answer, source_chunks = get_answer(query)

                    # Add to conversation history
                    response = {
                        "query": query,
                        "answer": answer,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "sources": [f"Chunk {i+1}: {chunk[:100]}..." for i, chunk in enumerate(source_chunks)]
                    }
                    st.session_state.conversation_history.append(response)
                else:
                    st.error("Please process your documents first!")

            # Display conversation history
            for i, item in enumerate(reversed(st.session_state.conversation_history)):
                with st.container():
                    st.markdown(f"**Query #{len(st.session_state.conversation_history) - i}** *({item['timestamp']})*")
                    st.markdown(f"*{item['query']}*")

                    st.markdown("**Answer:**")
                    st.markdown(item['answer'])

                    # Source citations
                    with st.expander("Sources"):
                        if item['sources']:
                            for source in item['sources']:
                                st.markdown(f"• {source}")
                        else:
                            st.markdown("No sources available")

                    st.markdown("---")

    with col2:
        # Stats and info panel
        st.header("Document Stats")

        if st.session_state.valid_files:
            total_files = len(st.session_state.valid_files)
            total_size = sum(file.size for file in st.session_state.valid_files)

            # Display stats
            st.metric("Total Documents", total_files)
            st.metric("Total Size", format_file_size(total_size))

            # File type breakdown
            file_types = {}
            for file in st.session_state.valid_files:
                ext = file.name.split('.')[-1].upper()
                file_types[ext] = file_types.get(ext, 0) + 1

            st.subheader("File Types")
            for file_type, count in file_types.items():
                st.write(f"**{file_type}:** {count} file(s)")
        else:
            st.info("Upload documents to see statistics")

        st.markdown("---")

        # Performance and embedding info
        st.header("Performance")
        if st.session_state.documents_processed:
            store = get_embedding_store()
            st.metric("Documents Indexed", len(st.session_state.valid_files))
            st.metric("Total Chunks", store.index.ntotal)
            st.metric("Embedding Dimensions", "384", help="all-MiniLM-L6-v2 local embeddings")
        else:
            st.metric("Documents Indexed", "0", help="Process documents to see stats")
            st.metric("Total Chunks", "0", help="Process documents to see stats")
            st.metric("Embedding Dimensions", "384", help="all-MiniLM-L6-v2 local embeddings")

if __name__ == "__main__":
    main()
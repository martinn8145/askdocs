import os
from typing import List, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


def load_and_chunk(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Load a PDF or TXT file and split it into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        file_path (str): Path to the PDF or TXT file
        chunk_size (int): Target size of each chunk in tokens (default: 500)
        chunk_overlap (int): Number of overlapping tokens between chunks (default: 50)

    Returns:
        List[str]: List of text chunks

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is not supported
        Exception: For other file processing errors
    """

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file extension
    _, ext = os.path.splitext(file_path.lower())

    # Load text based on file type
    if ext == '.pdf':
        text = _load_pdf(file_path)
    elif ext == '.txt':
        text = _load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and TXT files are supported.")

    # Handle empty files
    if not text.strip():
        return []

    # Initialize text splitter
    # Convert token count to approximate character count (1 token H 4 characters for English)
    char_chunk_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=char_chunk_size,
        chunk_overlap=char_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    # Split text into chunks
    chunks = text_splitter.split_text(text)

    # Validate chunks are within token limits using tiktoken
    validated_chunks = []
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

    for chunk in chunks:
        token_count = len(encoding.encode(chunk))
        if token_count > chunk_size * 1.2:  # Allow 20% buffer for token estimation
            # If chunk is still too large, split it further
            smaller_splitter = RecursiveCharacterTextSplitter(
                chunk_size=char_chunk_size // 2,
                chunk_overlap=char_overlap // 2,
                length_function=len
            )
            smaller_chunks = smaller_splitter.split_text(chunk)
            validated_chunks.extend(smaller_chunks)
        else:
            validated_chunks.append(chunk)

    return validated_chunks


def _load_pdf(file_path: str) -> str:
    """
    Load text from a PDF file.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue

            return text.strip()

    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")


def _load_txt(file_path: str) -> str:
    """
    Load text from a TXT file.

    Args:
        file_path (str): Path to the TXT file

    Returns:
        str: Content of the text file
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, raise an error
        raise Exception(f"Could not decode file with any of the attempted encodings: {encodings}")

    except Exception as e:
        raise Exception(f"Error reading text file: {e}")


def get_chunk_stats(chunks: List[str]) -> dict:
    """
    Get statistics about the chunks.

    Args:
        chunks (List[str]): List of text chunks

    Returns:
        dict: Statistics including count, avg tokens, min/max tokens
    """
    if not chunks:
        return {"count": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0}

    encoding = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(encoding.encode(chunk)) for chunk in chunks]

    return {
        "count": len(chunks),
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "total_tokens": sum(token_counts)
    }


# Test function
if __name__ == "__main__":
    # Example usage and testing
    print("Testing load_and_chunk function...")

    # Test with a simple text file
    test_file = "test_example.txt"

    # Create a sample text file for testing
    sample_text = """
    This is a sample document for testing the load_and_chunk function.

    The function should be able to load this text file and split it into appropriate chunks
    based on the specified token limits. Each chunk should be roughly 500 tokens with
    50 tokens of overlap between consecutive chunks.

    This helps maintain context when processing documents for retrieval augmented generation
    (RAG) applications. The chunks will be used to create embeddings and store them in a
    vector database for efficient similarity search.

    LangChain's RecursiveCharacterTextSplitter is designed to split text in a way that
    preserves semantic meaning by trying different separators in order of preference.

    This is important for maintaining the quality of the document chunks and ensuring
    that related information stays together when possible.
    """ * 10  # Repeat to create a longer document

    try:
        # Create test file
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)

        # Test the function
        chunks = load_and_chunk(test_file)
        stats = get_chunk_stats(chunks)

        print(f"Successfully loaded and chunked file: {test_file}")
        print(f"Number of chunks: {stats['count']}")
        print(f"Average tokens per chunk: {stats['avg_tokens']:.1f}")
        print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"\nFirst chunk preview (first 200 chars):")
        print(f"'{chunks[0][:200]}...'")

        if len(chunks) > 1:
            print(f"\nSecond chunk preview (first 200 chars):")
            print(f"'{chunks[1][:200]}...'")

        # Clean up test file
        os.remove(test_file)
        print(f"\nTest completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        # Clean up test file if it exists
        if os.path.exists(test_file):
            os.remove(test_file)
import os
from typing import Dict, List, Tuple, Optional, Any
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from embeddings import retrieve_chunks
from templates.qa_prompt import prompt_template

# Load and cache the local Q&A model
@st.cache_resource
def load_qa_model():
    """Load and cache the local Q&A model."""
    try:
        # Use FLAN-T5-base for question answering (good balance of quality and speed)
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create a text2text generation pipeline
        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.1
        )

        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading Q&A model: {str(e)}")
        return None


def get_answer(query: str, k: int = 5) -> Tuple[str, List[str]]:
    """
    Get an answer to a user query using RAG with retrieved chunks and local models.

    Args:
        query: User question
        k: Number of chunks to retrieve

    Returns:
        Tuple of (answer, source_chunks)
    """
    try:
        # Load the local Q&A model
        qa_pipeline = load_qa_model()
        if qa_pipeline is None:
            return "Error: Could not load the local Q&A model.", []

        # Retrieve relevant chunks using local embeddings
        chunks, scores = retrieve_chunks(query, k=k)

        if not chunks:
            return "I don't have any relevant information to answer your question. Please upload and process some documents first.", []

        # Prepare context from retrieved chunks (truncate if too long)
        context_parts = []
        max_context_length = 800  # Keep context manageable for local model

        for i, chunk in enumerate(chunks):
            chunk_text = f"Source {i+1}: {chunk[:200]}..." if len(chunk) > 200 else f"Source {i+1}: {chunk}"
            if len("\n".join(context_parts + [chunk_text])) > max_context_length:
                break
            context_parts.append(chunk_text)

        context = "\n".join(context_parts)

        # Create a simple prompt suitable for T5
        simple_prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Truncate prompt if too long for the model
        if len(simple_prompt) > 1000:
            simple_prompt = simple_prompt[:1000] + "..."

        # Generate answer using local model
        try:
            response = qa_pipeline(simple_prompt, max_new_tokens=200, num_return_sequences=1)
            answer = response[0]['generated_text'].strip()

            # Clean up the answer if it repeats the prompt
            if answer.startswith(simple_prompt):
                answer = answer[len(simple_prompt):].strip()

            if not answer or len(answer) < 10:
                answer = "Based on the provided context, I can see relevant information but cannot generate a clear answer. Please try rephrasing your question."

        except Exception as model_error:
            st.error(f"Model generation error: {str(model_error)}")
            # Fallback to simple extraction from chunks
            answer = f"Based on the retrieved information: {chunks[0][:300]}..." if chunks else "No relevant information found."

        return answer, chunks

    except Exception as e:
        return f"Error generating answer: {str(e)}", []
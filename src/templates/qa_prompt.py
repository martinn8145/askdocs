# Create custom prompt template for document-based Q&A
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.
Always cite which document or section your answer comes from when possible.

Context:
{context}

Question: {question}

Answer: """
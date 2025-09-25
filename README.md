# AskDocs - AI-Powered Document Q&A with RAG

AskDocs is a powerful Streamlit application that enables intelligent document analysis and question-answering using Retrieval-Augmented Generation (RAG). Upload your documents and ask natural language questions to get precise, context-aware answers backed by your document content.

## =€ Features

### =Ú Document Management
- **Multi-format support**: Upload PDF and TXT files
- **Robust encoding handling**: Automatically detects and handles various text encodings (UTF-8, Windows-1252, Latin-1, etc.)
- **Support for special characters**: Handles emojis, accented characters, and international text
- **Batch processing**: Upload up to 3 documents simultaneously
- **Smart chunking**: Automatically splits documents into optimal chunks for processing

### > AI-Powered Q&A
- **Local AI models**: Uses sentence-transformers for embeddings (no external API needed)
- **Semantic search**: Finds relevant document sections using vector similarity
- **Context-aware answers**: Provides answers based on actual document content
- **Conversation history**: Maintains chat history for better context understanding
- **Source attribution**: Shows which document sections were used for each answer

### =Ê Advanced Analytics & Visualization
- **Document Statistics**: View detailed stats about your uploaded documents
- **Embedding Visualization**: Interactive 2D visualization of document embeddings using:
  - **t-SNE**: Better for exploring local structure and clusters
  - **UMAP**: Better for preserving global structure and relationships
- **Performance Monitoring**: Track query response times and system performance
- **Chunk Analysis**: See how documents are split and processed

### =' Technical Features
- **FAISS Integration**: Efficient vector storage and similarity search
- **Configurable Parameters**: Adjust chunk sizes, overlap, and embedding parameters
- **Memory Efficient**: Optimized for local processing without cloud dependencies
- **Real-time Processing**: Live updates as documents are processed

## =à Installation & Setup

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended for larger documents

### 1. Clone the Repository
```bash
git clone <repository-url>
cd askdocs
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```env
# Optional: Configure any API keys or custom settings
# The app works completely offline by default
```

### 5. Run the Application
```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

## =Ö How to Use

### Step 1: Upload Documents
1. Navigate to the **Document Management** section in the sidebar
2. Click "Choose files" and select your PDF or TXT documents
3. Wait for the documents to be processed (you'll see a progress indicator)

### Step 2: Ask Questions
1. Go to the main **Ask Your Questions** section
2. Type your question in natural language
3. Press Enter or click Submit
4. View the AI-generated answer with source references

### Step 3: Explore Visualizations
1. Navigate to the **Document Embeddings Visualization** tab
2. Choose between t-SNE or UMAP algorithms
3. Adjust parameters like perplexity (t-SNE) or neighbors (UMAP)
4. Explore the interactive 2D visualization of your document relationships

### Step 4: Monitor Performance
1. Check the **Document Stats** section for file information
2. View the **Performance** section for response times and system metrics

## =¡ Use Cases

- **Research**: Analyze academic papers, research documents, and literature
- **Legal**: Review contracts, legal documents, and case files
- **Business**: Process reports, manuals, and business documents
- **Education**: Study materials, textbooks, and course content
- **Personal**: Organize and search through personal document collections

## =' Configuration

### Chunk Settings
- **Chunk Size**: Controls how documents are split (default: 500 tokens)
- **Chunk Overlap**: Overlap between chunks for better context (default: 50 tokens)

### Embedding Model
- Uses `all-MiniLM-L6-v2` by default for balanced performance and accuracy
- Model runs locally - no internet required after initial download

### Visualization Parameters
- **t-SNE**: Adjust perplexity (5-50) and learning rate (10-1000)
- **UMAP**: Configure n_neighbors (2-100) and min_dist (0.0-0.99)

## =¨ Troubleshooting

### Common Issues

**"UnicodeDecodeError" when uploading documents:**
- The app automatically handles most encoding issues
- If problems persist, try saving your document as UTF-8 text

**"Memory error" with large documents:**
- Reduce chunk size in settings
- Process fewer documents simultaneously
- Increase system RAM if possible

**Slow performance:**
- Reduce the number of chunks processed
- Close other resource-intensive applications
- Consider using a machine with more CPU cores

## =æ Dependencies

Key libraries used:
- **Streamlit**: Web interface framework
- **sentence-transformers**: Local embedding generation
- **FAISS**: Vector similarity search
- **scikit-learn**: Machine learning algorithms (t-SNE)
- **umap-learn**: UMAP dimensionality reduction
- **PyPDF2**: PDF document processing
- **plotly**: Interactive visualizations

## > Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## =Ä License

This project is open source. Please check the license file for details.

## <˜ Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify your Python version meets requirements
4. Open an issue on the project repository

---

**Built with d using Streamlit and modern AI technologies**
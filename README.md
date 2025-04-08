# Ministry RAG Q&A System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate and accessible information about Ministry regulations through a chatbot interface. The system strictly adheres to information found in source documents, ensuring factual accuracy and preventing hallucinations.

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
  - [Environment Setup](#environment-setup)
  - [Document Processing](#document-processing)
  - [Running the API](#running-the-api)
  - [Running the CLI](#running-the-cli)
- [Frontend Setup](#frontend-setup)
  - [Environment Configuration](#frontend-environment-configuration)
  - [Development Mode](#development-mode)
  - [Production Build](#production-build)
- [API Reference](#api-reference)
- [Configuration Options](#configuration-options)
- [Adding Documents](#adding-documents)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Adaptive RAG Pipeline**: Intelligent document retrieval and answer generation based on query type and intent
- **Multi-Strategy Retrieval**: Dense, sparse, and hybrid retrieval methods for optimal document matching
- **Self-Reflection**: Answer validation and hallucination detection to ensure response quality
- **Web Search Integration**: Fallback to web search for queries outside the knowledge base scope
- **Document Processing**: Efficient chunking and indexing of PDF documents
- **Rich Formatting**: Responses use markdown with proper headings, lists, tables and LaTeX for mathematical formulas
- **Comprehensive Analysis**: Detailed confidence scoring, evidence tracking, and quality assessment

## System Architecture

The system implements a sophisticated RAG pipeline with these key components:

1. **Query Analysis**
   - Query type classification (factual, analytical, procedural)
   - Entity extraction and relevance assessment
   - Intent recognition to guide processing strategy

2. **Document Retrieval**
   - Dense retrieval using embeddings (semantic search)
   - Sparse retrieval using keyword matching (lexical search)
   - Hybrid retrieval combining both approaches for optimal results
   - Query rewriting to improve retrieval quality

3. **Document Grading**
   - Relevance scoring against the original query
   - Content quality assessment
   - Evidence extraction for supporting the answer
   - Metadata enrichment for improved contextualization

4. **Answer Generation**
   - Context-aware response generation with strict adherence to documents
   - Evidence integration from multiple sources
   - Self-reflection and validation to prevent hallucinations
   - Confidence scoring based on document evidence
   - Rich formatting with markdown and LaTeX support

5. **Web Search Integration**
   - Automatic fallback for out-of-domain queries
   - Result analysis and filtering for relevant information
   - Integration with generated answers

## Project Structure

```text
.
├── src/                    # Core source code
│   ├── nodes/              # Pipeline nodes
│   │   ├── query_analyzer.py   # Query analysis and classification
│   │   ├── retriever.py        # Document retrieval strategies
│   │   ├── grader.py           # Document relevance assessment
│   │   ├── generator.py        # Answer generation and validation
│   │   └── web_searcher.py     # Web search integration
│   ├── pipeline/           # Pipeline orchestration
│   │   └── rag_pipeline.py     # Main pipeline implementation
│   ├── graph/              # Workflow graph
│   │   └── rag_graph.py        # LangGraph implementation
│   └── state/             # State management
│       └── rag_state.py        # RAG state type definitions
├── knowledge_base/        # Document storage
│   ├── raw_data/          # Original PDF documents
│   ├── processed_data/    # Processed chunks and vector store
│   └── scripts/           # Document processing scripts
├── frontend/             # Frontend application
│   ├── apps/             # Frontend applications
│   │   ├── web/          # Web interface
│   │   └── agents/       # Agent definitions
│   ├── package.json      # Frontend dependencies
│   └── README.md         # Frontend documentation
├── tests/                # Test suite
├── config/               # Configuration files
├── api.py               # FastAPI implementation
├── cli.py               # Command-line interface
├── requirements.txt     # Python dependencies
└── .env                 # Environment variables
```

## Prerequisites

- Python 3.9+ for the backend
- Node.js 16+ and npm for the frontend
- Sufficient disk space for vector embeddings
- API keys for language models and web search:
  - Google API key for Gemini
  - Tavily API key for web search (optional)

## Backend Setup

### Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdouENSIA/ministry_rag_q-a.git
   cd ministry_rag_q-a
   ```

2. Create and activate a Python virtual environment:

   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit .env with your API keys
   ```

   Required environment variables:
   - `GOOGLE_API_KEY`: Google API key for Gemini model
   - `TAVILY_API_KEY`: Tavily API key for web search (optional)

### Document Processing

Before running the system, you need to process your documents:

1. Place your PDF documents in the `knowledge_base/raw_data/` directory.

2. Run the document processing script:

   ```bash
   python knowledge_base/scripts/process_documents.py
   ```

   This script will:
   - Load PDF documents from the raw_data directory
   - Split documents into manageable chunks
   - Create embeddings and store them in the vector database
   - Save processed data to the processed_data directory

### Running the API

Start the FastAPI server with:

```bash
python api.py
```

The API will be available at <http://localhost:8000> with the following endpoints:

- `GET /`: API status check
- `POST /api/query`: Main query endpoint

### Running the CLI

For a command-line interface, run:

```bash
python cli.py
```

The CLI provides a rich interactive interface with:

- Colorful output formatting
- Display of metadata and confidence scores
- Command history and suggestion support

## Frontend Setup

### Frontend Environment Configuration

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install NPM dependencies:

   ```bash
   npm install
   ```

3. Configure the frontend environment:

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your configuration
   ```

### Development Mode

Run the frontend in development mode:

```bash
npm run dev
```

This command starts both:

- The web interface on <http://localhost:3000>
- The agents interface on <http://localhost:3001>

### Production Build

Build the frontend for production:

```bash
npm run build
```

The build artifacts will be located in the `frontend/apps/web/dist` and `frontend/apps/agents/dist` directories.

To serve the production build, you can use:

```bash
# Install a simple HTTP server if needed
npm install -g serve

# Serve the web interface
serve -s frontend/apps/web/dist
```

## API Reference

### POST /api/query

**Endpoint:** Main query interface for retrieving answers.

**Request body:**

```json
{
  "query": "What are the key regulations for business registration?",
  "include_source_documents": false
}
```

**Response:**

```json
{
  "answer": "# Business Registration Regulations\n\n...",
  "supporting_evidence": ["Document 1: 'Business registration requires...'"],
  "reasoning_path": "...",
  "confidence_score": 0.85,
  "query_type": "factual",
  "query_intent": "information",
  "processing_time": 1.25,
  "metadata": {
    "sources_used": 3,
    "key_concepts": ["business registration", "regulatory compliance"],
    "confidence_factors": ["Comprehensive document coverage", "Clear regulations"]
  },
  "suggested_followup": [
    "What are the fees associated with business registration?",
    "How long does the business registration process typically take?"
  ],
  "validation": {
    "has_hallucinations": false,
    "answers_question": true,
    "quality_score": 0.85,
    "improvement_needed": ["Could include more specific industry examples"],
    "validation_reasoning": "The answer covers all key aspects of business registration regulations"
  }
}
```

## Configuration Options

Key configuration settings in `.env` and the pipeline configuration:

| Setting | Description | Default |
|---------|-------------|---------|
| `MAX_RETRIEVAL_ATTEMPTS` | Maximum retrieval retry attempts | 2 |
| `MAX_GENERATION_ATTEMPTS` | Maximum answer generation attempts | 1 |
| `MIN_CONFIDENCE_SCORE` | Minimum confidence threshold | 0.7 |
| `REQUESTS_PER_MINUTE` | Rate limit for API requests | 30 |
| `USE_CUDA` | Whether to use GPU acceleration | False |
| `CACHE_TTL` | Cache time-to-live in seconds | 3600 |
| `API_TIMEOUT` | API request timeout in seconds | 30 |
| `API_MAX_RETRIES` | Maximum API request retries | 3 |

## Adding Documents

To add new documents to the knowledge base:

1. Add PDF files to the `knowledge_base/raw_data/` directory
2. Run the document processing script:

   ```bash
   python knowledge_base/scripts/process_documents.py
   ```

3. The system will automatically process and index the new documents

For programmatic document addition:

```python
from langchain_community.document_loaders import PyPDFLoader
from src.pipeline.rag_pipeline import RAGPipeline

# Initialize your pipeline
pipeline = RAGPipeline(...)

# Load documents
loader = PyPDFLoader("path/to/new/document.pdf")
documents = loader.load()

# Update knowledge base
pipeline.update_knowledge_base(documents)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

For specific test categories:

```bash
# Run only retrieval tests
pytest tests/test_retriever.py

# Run only answer generation tests
pytest tests/test_generator.py
```

## Troubleshooting

### Common Issues

1. **Vector store not found error**:
   - Ensure you've run the document processing script
   - Check that the processed_data directory exists and has content

2. **API key errors**:
   - Verify your API keys are correctly set in the .env file
   - Check for any spaces or special characters in the keys

3. **Out of memory issues**:
   - Reduce batch size in configuration
   - Consider processing documents in smaller batches

4. **Slow response times**:
   - Check your internet connection for API calls
   - Consider using a more powerful machine for vector operations
   - Enable CUDA if you have a compatible GPU

### Logs

Log files are located in the project root directory. Check these for troubleshooting:

- For detailed error information, enable debug logging by setting `LOG_LEVEL=DEBUG` in the `.env` file

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

Please follow the coding style and include appropriate tests for new features.

## License

This project is proprietary and confidential. All rights reserved.

---

For questions or support, please contact the development team.

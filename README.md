# Ministry Regulation Q&A System (Arabic Language)

A sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate and accessible information about Ministry regulations through an Arabic-language chatbot interface.

## Features

- **Adaptive RAG Pipeline**: Intelligent document retrieval and answer generation based on query type
- **Multi-Strategy Retrieval**: Dense, sparse, and hybrid retrieval methods
- **Self-Reflection**: Answer validation and hallucination detection
- **Web Search Integration**: Fallback to web search for queries outside the knowledge base
- **Document Processing**: Efficient chunking and indexing of PDF documents
- **Arabic Language Support**: Full support for Arabic text and RTL interface

## Architecture

The system implements a sophisticated RAG pipeline with the following components:

1. **Query Analysis**
   - Query type classification (factual, analytical, procedural)
   - Entity extraction
   - Intent recognition
   - Index relevance assessment

2. **Document Retrieval**
   - Dense retrieval using embeddings
   - Sparse retrieval using keyword matching
   - Hybrid retrieval combining both approaches
   - Query rewriting for better matches

3. **Document Grading**
   - Relevance scoring
   - Quality assessment
   - Evidence extraction
   - Metadata enrichment

4. **Answer Generation**
   - Context-aware response generation
   - Evidence integration
   - Self-reflection and validation
   - Confidence scoring

5. **Web Search Integration**
   - Automatic fallback for out-of-domain queries
   - Result analysis and filtering
   - Integration with generated answers

## Project Structure

```text
.
├── src/
│   ├── nodes/              # Pipeline nodes
│   │   ├── query_analyzer.py
│   │   ├── retriever.py
│   │   ├── grader.py
│   │   ├── generator.py
│   │   └── web_searcher.py
│   ├── pipeline/           # Pipeline orchestration
│   │   └── rag_pipeline.py
│   ├── graph/              # Workflow graph
│   │   └── rag_graph.py
│   └── state/             # State management
│       └── rag_state.py
├── knowledge_base/        # Document storage
│   ├── raw_data/         # Original documents
│   ├── processed_data/   # Processed chunks
│   └── scripts/          # Processing scripts
├── tests/                # Unit tests
├── config/              # Configuration files
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ministry-rag-qa.git
   cd ministry-rag-qa
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Process documents:

   ```bash
   python knowledge_base/scripts/process_documents.py
   ```

## Usage

1. Start the system:

   ```bash
   python main.py
   ```

2. Enter your questions when prompted. The system will:
   - Analyze your query
   - Retrieve relevant documents
   - Generate and validate answers
   - Provide confidence scores and supporting evidence

3. Type 'exit' to quit the system.

## Configuration

Key configuration options in `config/settings.py`:

- `MAX_RETRIEVAL_ATTEMPTS`: Maximum retrieval retry attempts (default: 3)
- `MAX_GENERATION_ATTEMPTS`: Maximum answer generation attempts (default: 2)
- `MIN_CONFIDENCE_SCORE`: Minimum confidence threshold (default: 0.7)
- `CHUNK_SIZE`: Document chunk size
- `CHUNK_OVERLAP`: Chunk overlap size

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is proprietary and confidential. All rights reserved.

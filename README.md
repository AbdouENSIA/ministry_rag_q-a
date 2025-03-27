# RAG Q&A System

A robust Question Answering system built using RAG (Retrieval Augmented Generation) with LangGraph and LangChain. This system implements a sophisticated workflow for processing queries, retrieving relevant information, and generating accurate answers.

## Features

- **Modular Architecture**: Built with SOLID principles and clean architecture
- **LangGraph Workflow**: Orchestrates the RAG process using LangGraph
- **Advanced Components**:
  - Query Analysis
  - Document Retrieval with FAISS
  - Document Grading
  - Answer Generation
- **Configurable**: Easy configuration through YAML files
- **Production-Ready**: Includes logging, error handling, and API endpoints
- **Modern Stack**:
  - LangChain for LLM operations
  - LangGraph for workflow management
  - FAISS for efficient document retrieval
  - HuggingFace embeddings
  - FastAPI for the REST API

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
project_root/
│
├── src/
│   ├── core/           # Core configuration and utilities
│   ├── nodes/          # Individual processing nodes
│   ├── graph/          # LangGraph workflow definition
│   └── pipeline/       # Main pipeline orchestration
│
├── knowledge_base/     # Document storage and processing
├── config/            # Configuration files
└── tests/             # Test suite
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ministry_rag_q-a
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Prepare the knowledge base:
```bash
# Add your documents to knowledge_base/raw_data/documents/
python knowledge_base/scripts/ingest_data.py
python knowledge_base/scripts/build_index.py
```

## Usage

1. Start the API server:
```bash
python main.py
```

2. Send queries to the API:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Your question here"}'
```

## Configuration

The system can be configured through YAML files in the `config/` directory:

- `llm_config.yaml`: LLM settings
- `retriever_config.yaml`: Document retrieval settings

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 guidelines. Format code using:
```bash
black .
isort .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

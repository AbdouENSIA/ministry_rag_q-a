# Ministry RAG Q&A System

<div align="center">

![Ministry RAG](https://img.shields.io/badge/Ministry-RAG%20Q%26A-0F4C81?style=for-the-badge&logo=openai&logoColor=white)
[![Arabic Support](https://img.shields.io/badge/Arabic-RTL%20Support-success?style=for-the-badge&logo=google-translate&logoColor=white)](https://img.shields.io/badge/Arabic-RTL%20Support-success?style=for-the-badge&logo=google-translate&logoColor=white)
[![Made with LangGraph](https://img.shields.io/badge/Made%20with-LangGraph-ff6f00?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)

</div>

A sophisticated Retrieval-Augmented Generation (RAG) system designed to provide accurate and accessible information about Ministry regulations through an intuitive chatbot interface. The system strictly adheres to information found in source documents, ensuring factual accuracy and preventing hallucinations.

Built with modern LLM technologies including LangGraph, Gemini, and a React-based frontend with full RTL support for Arabic content.

## 🌟 Features

<div align="center">
<img src="https://raw.githubusercontent.com/AbdouENSIA/ministry_rag_q-a/main/frontend/apps/web/public/وزارة_التعليم_العالي_والبحث_العلمي.svg.png" alt="Ministry Logo" width="120" />
</div>

- **📊 Advanced RAG Pipeline**: Intelligent document retrieval and answer generation based on automated query analysis
- **🔍 Multi-Strategy Retrieval**: Dense, sparse, and hybrid retrieval strategies that adapt to query type
- **🧠 Self-Reflection & Validation**: Answers are validated for factual accuracy and relevance
- **🔎 Web Search Integration**: Automatic fallback to web search for out-of-domain queries
- **📄 Structured Document Processing**: Efficient chunking and indexing of PDF documents
- **🎨 Rich Formatting**: Responses use Markdown with proper headings, lists, and tables
- **📱 Responsive Interface**: Modern React frontend with full RTL support for Arabic content
- **⚡ Real-time Streaming**: Stream responses as they're generated for a responsive experience
- **🔒 Strict Factuality**: System only provides information found in verified documents

## 🏗️ System Architecture

The system implements a sophisticated RAG pipeline with these key components:

### 1. Query Analysis Engine

- Automatically classifies queries (factual, analytical, procedural)
- Extracts relevant entities and determines query intent
- Guides the retrieval and generation strategies based on query type

### 2. Adaptive Document Retrieval

- **Dense Retrieval**: Uses multilingual embeddings for semantic understanding
- **Sparse Retrieval**: Keywords and BM25 for lexical matching
- **Hybrid Retrieval**: Combines both approaches for optimal results
- **Query Refinement**: Automatically rewrites queries to improve retrieval quality

### 3. Knowledge Synthesis & Generation

- Context-aware response generation with strict adherence to source documents
- Evidence integration from multiple documents
- Self-validation to detect and prevent hallucinations
- Confidence scoring and evidence tracking

### 4. Web Search Integration

- Automatic fallback for queries outside the knowledge base
- Result filtering and integration with document-based answers

## 📂 Project Structure

```text
ministry_rag_q-a/
├── src/                    # Core backend code
│   ├── nodes/              # Pipeline components 
│   │   ├── query_analyzer.py   # Query processing and classification
│   │   ├── retriever.py        # Document retrieval strategies
│   │   ├── generator.py        # Answer generation and validation
│   │   └── web_searcher.py     # Web search integration
│   ├── graph/              # LangGraph workflow
│   │   └── rag_graph.py        # Orchestrates the RAG pipeline
│   ├── state/              # State management
│   │   └── rag_state.py        # Type definitions
│   └── pipeline/           # Pipeline orchestration
│       └── rag_pipeline.py     # Main pipeline implementation
├── frontend/              # Frontend application
│   ├── apps/              # Next.js applications 
│   │   ├── web/           # Main web interface
│   │   └── agents/        # Agent configurations
│   └── package.json       # Frontend dependencies
├── knowledge_base/        # Document storage
│   ├── raw_data/          # Original documents
│   ├── processed_data/    # Processed chunks and embeddings
│   └── scripts/           # Document ingestion scripts
├── api.py                 # FastAPI implementation
├── requirements.txt       # Python dependencies
└── .env                   # Environment configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ and npm/yarn
- API keys:
  - Google API key (for Gemini)
  - Tavily API key (optional, for web search)

### Backend Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AbdouENSIA/ministry_rag_q-a.git
   cd ministry_rag_q-a
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   
   # Linux/macOS
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**

   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit with your API keys
   ```

5. **Process documents:**

   ```bash
   # Add PDF files to knowledge_base/raw_data/
   python knowledge_base/scripts/process_documents.py
   ```

6. **Start the API server:**

   ```bash
   python api.py
   ```

### Frontend Setup

1. **Navigate to the frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start the development server:**

   ```bash
   npm run dev
   ```

4. **Open your browser:**
   - Web interface: <http://localhost:3000>

## 📡 API Reference

### `POST /api/query`

Processes a query and retrieves an answer from the knowledge base.

#### Request Body

```json
{
  "query": "متى تأسست وزارة التعليم العالي والبحث العلمي؟",
  "include_source_documents": false
}
```

#### Response

```json
{
  "answer": "# تأسيس وزارة التعليم العالي والبحث العلمي\n\nتأسست وزارة التعليم العالي والبحث العلمي في...",
  "supporting_evidence": ["Document 1: 'تأسست وزارة التعليم العالي والبحث العلمي...'"],
  "confidence_score": 0.92,
  "query_type": "factual",
  "query_intent": "information",
  "processing_time": 1.25,
  "metadata": {
    "sources_used": 3,
    "key_concepts": ["تأسيس الوزارة", "التعليم العالي"]
  },
  "suggested_followup": [
    "ما هي مهام وزارة التعليم العالي والبحث العلمي؟",
    "كم عدد الجامعات في الجزائر؟"
  ],
  "validation": {
    "has_hallucinations": false,
    "answers_question": true,
    "quality_score": 0.90
  }
}
```

## ⚙️ Configuration

Key configuration options in `.env`:

| Setting | Description | Default |
|---------|-------------|---------|
| `GOOGLE_API_KEY` | API key for Google Gemini | Required |
| `TAVILY_API_KEY` | API key for web search | Optional |
| `USE_CUDA` | Whether to use GPU acceleration | `False` |
| `ENABLE_CACHE` | Enable response caching | `True` |
| `CACHE_TTL` | Cache time-to-live in seconds | `3600` |
| `API_TIMEOUT` | API request timeout | `30` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## 📚 Adding Documents

To add new documents to the knowledge base:

1. Add PDF files to the `knowledge_base/raw_data/` directory
2. Run the document processing script:

   ```bash
   python knowledge_base/scripts/process_documents.py
   ```

## 📊 Performance Tuning

For optimal performance:

- **Memory Usage**: Adjust the chunk size in document processing for balance between context and memory usage
- **Retrieval Quality**: Configure retrieval parameters in `.env` for precision vs. recall trade-offs
- **Speed**: Enable `USE_CUDA=True` if you have a compatible GPU
- **Scalability**: Configure rate limiting parameters for API stability

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

For specific components:

```bash
pytest tests/test_retriever.py
pytest tests/test_generator.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Vector store initialization errors**:
   - Ensure you've processed documents before starting the API
   - Check that the processed_data directory exists and has content

2. **API key errors**:
   - Verify your API keys are correctly set in `.env`
   - Check for any quotation marks or whitespace in the keys

3. **Out of memory issues**:
   - Reduce batch size in configuration
   - Process documents in smaller batches

### Logs

- Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs
- Check `rag_pipeline.log` for pipeline execution details

## 🛡️ License

This project is proprietary software. All rights reserved.

---

<div align="center">
  <p>Built with Professionalism for the Ministry of Higher Education and Scientific Research</p>
</div>

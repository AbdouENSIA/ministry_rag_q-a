"""Configuration settings for the RAG system."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
RAW_DATA_DIR = KNOWLEDGE_BASE_DIR / "raw_data"
PROCESSED_DATA_DIR = KNOWLEDGE_BASE_DIR / "processed_data"
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "vector_store"

# Document Processing
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 750
MAX_CHUNKS_PER_FILE = 1500

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cuda"  # or "cuda" for GPU
EMBEDDING_BATCH_SIZE = 32

# Retrieval
DEFAULT_RETRIEVAL_K = 5
MAX_RETRIEVAL_K = 15
MIN_RELEVANCE_SCORE = 0.75
MAX_RETRIEVAL_ATTEMPTS = 4
HYBRID_RETRIEVAL_WEIGHT = 0.6
RERANKING_ENABLED = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_STRATEGIES = ["dense", "sparse", "hybrid"]

# Generation
MAX_GENERATION_ATTEMPTS = 2
MIN_CONFIDENCE_SCORE = 0.7
TEMPERATURE = 0.3
MAX_TOKENS = 4096

# Web Search
MAX_SEARCH_RESULTS = 5
SEARCH_DEPTH = "advanced"  # basic or advanced
INCLUDE_ANSWER = True
INCLUDE_RAW_CONTENT = True
INCLUDE_IMAGES = False
SEARCH_TIMEOUT = 10  # seconds

# Pipeline
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "logs" / "rag.log"

# Arabic Support
ARABIC_ENABLED = True
RTL_ENABLED = True

# Security
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_FILE_TYPES = [".pdf"]
SECURE_MODE = True

# Performance
BATCH_SIZE = 32
NUM_WORKERS = 4
USE_CUDA = False  # Set to True to use GPU

# Caching
CACHE_DIR = BASE_DIR / "cache"
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB

# API
API_TIMEOUT = 30
API_MAX_RETRIES = 3
API_BACKOFF_FACTOR = 0.3

# Development
DEBUG = False
TESTING = False 
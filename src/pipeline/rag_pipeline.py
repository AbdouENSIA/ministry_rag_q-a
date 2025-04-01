import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import langgraph
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ..graph.rag_graph import RAGGraph
from ..state.rag_state import RAGState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Global rate limiter for API calls."""
    
    # Class-level variables for global rate limiting
    _last_request = 0.0
    _request_times = []
    _lock = asyncio.Lock()
    
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        
    async def wait(self):
        """Wait if needed to respect rate limits."""
        async with self._lock:
            now = time.time()
            
            # Remove request times older than 1 minute
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]
            
            # If we've hit the rate limit, wait
            if len(self._request_times) >= self.requests_per_minute:
                wait_time = self._request_times[0] - cutoff
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._request_times = self._request_times[1:]  # Remove oldest request
            
            # Add current request
            self._request_times.append(now)
            self._last_request = now

class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire retrieval and generation process."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        embeddings: Embeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        logger.info("="*50)
        logger.info("Initializing RAGPipeline...")
        logger.info(f"Config: {config}")
        
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or {}
        
        # Initialize rate limiter
        requests_per_minute = self.config.get("requests_per_minute", 30)
        self.rate_limiter = RateLimiter(requests_per_minute)
        
        # Initialize the graph
        logger.info("Building RAGGraph...")
        start_time = time.time()
        self.rag_graph = RAGGraph(
            vector_store=vector_store,
            llm=llm,
            embeddings=embeddings,
            config=config,
            rate_limiter=self.rate_limiter  # Pass rate limiter to graph
        ).build()
        build_time = time.time() - start_time
        logger.info(f"Graph built successfully in {build_time:.2f} seconds")
        logger.info(f"Graph type: {type(self.rag_graph)}")
        logger.info(f"Graph methods: {dir(self.rag_graph)}")
        logger.info("="*50)
        
    async def process(
        self,
        query: str,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline using the LangGraph workflow.
        
        Args:
            query: User query to process
            config: Optional runtime configuration
            
        Returns:
            Processing results including answer and metadata
        """
        logger.info("="*50)
        logger.info(f"Processing query: {query}")
        start_time = time.time()
        
        # Initialize state with all required fields
        logger.info("Initializing state...")
        initial_state: RAGState = {
            # Input
            "query": query,
            "chat_history": None,
            
            # Query Analysis
            "is_related_to_index": False,
            "query_type": "factual",
            "query_entities": [],
            "query_intent": "unknown",
            
            # Document Retrieval & Processing
            "documents": None,
            "rewritten_query": None,
            "retrieval_strategy": "dense",
            "retrieval_scores": None,
            
            # Document Grading
            "docs_relevant": None,
            "doc_scores": None,
            "doc_metadata": None,
            
            # Generation & Validation
            "answer": None,
            "has_hallucinations": None,
            "answers_question": None,
            "confidence_score": None,
            "supporting_evidence": None,
            
            # Web Search
            "web_search_results": None,
            
            # Metadata & Tracking
            "metadata": {},
            "current_node": "start",
            "retry_count": 0,
            "processing_time": 0.0,
            "error": None
        }
        logger.info("State initialized")
        
        # Execute the graph with initial state
        logger.info("Executing graph workflow...")
        execution_start = time.time()
        final_state = await self.rag_graph.execute(initial_state, config=config or {})
        execution_time = time.time() - execution_start
        logger.info(f"Graph execution completed in {execution_time:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Final state keys: {list(final_state.keys())}")
        logger.info("="*50)
        
        return final_state
        
    def update_knowledge_base(self, documents: list) -> None:
        """
        Update the knowledge base with new documents.
        
        Args:
            documents: List of documents to add to the knowledge base
        """
        logger.info("="*50)
        logger.info(f"Updating knowledge base with {len(documents)} documents")
        start_time = time.time()
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        update_time = time.time() - start_time
        logger.info(f"Knowledge base updated in {update_time:.2f} seconds")
        logger.info("="*50)
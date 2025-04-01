import logging
import time
from typing import Any, Dict, Optional, TypeVar, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, StateGraph

from ..nodes.generator import Generator
from ..nodes.grader import Grader
from ..nodes.query_analyzer import QueryAnalyzer
from ..nodes.retriever import Retriever
from ..nodes.web_searcher import WebSearcher
from ..state.rag_state import RAGState

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type for the state
State = TypeVar("State", bound=RAGState)

class RAGGraph:
    """Builds and configures the LangGraph workflow for the RAG system."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        embeddings: Embeddings,
        config: Optional[Dict[str, Any]] = None,
        rate_limiter: Optional[Any] = None
    ):
        logger.info("="*50)
        logger.info("Initializing RAGGraph...")
        logger.info(f"Config: {config}")
        
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or {}
        self.rate_limiter = rate_limiter
        self.workflow = None
        self.compiled_graph = None
        
        # Initialize nodes
        logger.info("Initializing graph nodes...")
        start_time = time.time()
        
        logger.info("Initializing QueryAnalyzer...")
        self.query_analyzer = QueryAnalyzer(llm=self.llm, rate_limiter=self.rate_limiter)
        
        logger.info("Initializing Retriever...")
        self.retriever = Retriever(
            vector_store=vector_store,
            embeddings=self.embeddings,
            llm=self.llm,
            config=self.config,
            rate_limiter=self.rate_limiter
        )
        
        logger.info("Initializing Grader...")
        self.grader = Grader(llm=self.llm, rate_limiter=self.rate_limiter)
        
        logger.info("Initializing Generator...")
        self.generator = Generator(llm=self.llm, rate_limiter=self.rate_limiter)
        
        logger.info("Initializing WebSearcher...")
        self.web_searcher = WebSearcher(llm=self.llm, config=self.config, rate_limiter=self.rate_limiter)
        
        # Configure retry limits - reduce max attempts
        self.max_retrieval_attempts = min(config.get("max_retrieval_attempts", 2), 2)  # Cap at 2
        self.max_generation_attempts = min(config.get("max_generation_attempts", 1), 1)  # Cap at 1
        
        init_time = time.time() - start_time
        logger.info(f"All nodes initialized in {init_time:.2f} seconds")
        logger.info(f"Max retrieval attempts: {self.max_retrieval_attempts}")
        logger.info(f"Max generation attempts: {self.max_generation_attempts}")
        logger.info("="*50)
        
    def build(self) -> 'RAGGraph':
        """
        Build the LangGraph workflow according to the architecture diagram.
        
        The workflow implements:
        1. Query Analysis with routing to either retrieval or web search
        2. RAG pipeline with document retrieval, grading, and generation
        3. Optimized flow with minimal retries
        
        Returns:
            Self for method chaining
        """
        logger.info("="*50)
        logger.info("Building workflow graph...")
        start_time = time.time()
        
        # Create workflow graph with typed state
        logger.info("Creating StateGraph instance...")
        self.workflow = StateGraph(RAGState)
        
        # Add all nodes
        logger.info("Adding nodes to graph...")
        self.workflow.add_node("query_analyzer", self.query_analyzer.analyze)
        self.workflow.add_node("retriever", self.retriever.retrieve)
        self.workflow.add_node("grader", self.grader.grade)
        self.workflow.add_node("generator", self.generator.generate)
        self.workflow.add_node("web_searcher", self.web_searcher.search)
        logger.info("All nodes added successfully")
        
        # 1. Query Analysis Section
        logger.info("Adding query analysis edges...")
        self.workflow.add_conditional_edges(
            "query_analyzer",
            self.route_from_analysis,
            {
                "retrieve": "retriever",  # Related to index
                "web_search": "web_searcher",  # Unrelated to index
            }
        )
        
        # 2. Document Processing Path - Simplified
        logger.info("Adding document processing edges...")
        self.workflow.add_conditional_edges(
            "retriever",
            self.check_retrieval_attempts,
            {
                "continue": "grader",
                "end": END
            }
        )
        
        # 3. Document Relevance Check - Direct to generator or end
        logger.info("Adding document relevance check edges...")
        self.workflow.add_conditional_edges(
            "grader",
            self.check_docs_relevance,
            {
                "yes": "generator",  # Docs are relevant
                "no": END  # End if docs not relevant
            }
        )
        
        # 4. Web search path
        logger.info("Adding web search path edge...")
        self.workflow.add_edge("web_searcher", "generator")
        
        # 5. Generation - Direct to end
        logger.info("Adding generation edges...")
        self.workflow.add_conditional_edges(
            "generator",
            self.check_generation_quality,
            {
                "continue": END,  # Good quality, end
                "retry": "retriever",  # Poor quality, one retry
                "end": END  # Max attempts or other conditions
            }
        )
        
        # Set entry point
        logger.info("Setting entry point...")
        self.workflow.set_entry_point("query_analyzer")
        
        build_time = time.time() - start_time
        logger.info(f"Graph building completed in {build_time:.2f} seconds")
        logger.info("="*50)
        
        return self

    def compile(self) -> 'RAGGraph':
        """Compile the graph for execution."""
        if not self.workflow:
            raise RuntimeError("Graph must be built before compilation")
        if not self.compiled_graph:
            logger.info("Compiling graph...")
            self.compiled_graph = self.workflow.compile()
            logger.info("Graph compiled successfully")
        return self

    async def execute(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the graph with the given state and config."""
        if not self.compiled_graph:
            self.compile()
        # Use ainvoke instead of invoke for async execution
        return await self.compiled_graph.ainvoke(state, config=config or {})
    
    def route_from_analysis(self, state: State) -> str:
        """Route based on query analysis results."""
        logger.info("Routing from analysis...")
        state = cast(RAGState, state)
        state["current_node"] = "query_analyzer"
        
        if state["is_related_to_index"]:
            logger.info("Query related to index, routing to retrieval")
            return "retrieve"
        elif self.web_searcher.search_client:  # Only use web search if configured
            logger.info("Query unrelated to index, routing to web search")
            return "web_search"
        else:
            logger.info("Web search not available, falling back to retrieval")
            return "retrieve"  # Fallback to retrieval if web search not available
    
    def check_retrieval_attempts(self, state: State) -> str:
        """Check if we should continue retrieving or end due to too many attempts."""
        logger.info("Checking retrieval attempts...")
        state = cast(RAGState, state)
        if state["retry_count"] >= self.max_retrieval_attempts:
            logger.info(f"Max retrieval attempts ({self.max_retrieval_attempts}) reached")
            return "end"
        logger.info(f"Retrieval attempts ({state['retry_count']}) within limit")
        return "continue"
    
    def check_docs_relevance(self, state: State) -> str:
        """Check if retrieved documents are relevant."""
        logger.info("Checking document relevance...")
        state = cast(RAGState, state)
        if state["docs_relevant"]:
            logger.info("Documents found to be relevant")
            return "yes"
        logger.info("Documents found to be irrelevant")
        return "no"
    
    def check_generation_quality(self, state: State) -> str:
        """Combined quality check for generation results."""
        state = cast(RAGState, state)
        
        # Check for too many attempts first
        if state.get("retry_count", 0) >= self.max_generation_attempts:
            logger.info(f"Max generation attempts ({self.max_generation_attempts}) reached")
            return "end"
            
        # Check for hallucinations and answer quality together
        if state.get("has_hallucinations", False) or not state.get("answers_question", True):
            state["retry_count"] = state.get("retry_count", 0) + 1
            logger.info(f"Quality check failed, retry count: {state['retry_count']}")
            return "retry" if state["retry_count"] < self.max_generation_attempts else "end"
            
        logger.info("Generation quality check passed")
        return "continue" 
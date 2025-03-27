from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ..graph.rag_graph import RAGGraph
from ..types.state import RAGState

class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire retrieval and generation process."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or {}
        
        # Initialize the graph
        self.graph = RAGGraph(
            vector_store=vector_store,
            llm=llm,
            config=config
        ).build()
        
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
        # Initialize state with all required fields
        initial_state: RAGState = {
            # Input
            "query": query,
            
            # Query Analysis
            "is_related_to_index": False,
            
            # Document Retrieval & Processing
            "documents": None,
            "rewritten_query": None,
            
            # Document Grading
            "docs_relevant": None,
            
            # Generation & Validation
            "answer": None,
            "has_hallucinations": None,
            "answers_question": None,
            
            # Web Search
            "web_search_results": None,
            
            # Metadata & Tracking
            "metadata": {},
            "current_node": "start",
            "retry_count": 0
        }
        
        # Execute the graph with initial state
        final_state = await self.graph.arun(
            initial_state,
            config=config or {}
        )
        
        return final_state
        
    def update_knowledge_base(self, documents: list) -> None:
        """
        Update the knowledge base with new documents.
        
        Args:
            documents: List of documents to add to the knowledge base
        """
        # Add documents to vector store
        self.vector_store.add_documents(documents)
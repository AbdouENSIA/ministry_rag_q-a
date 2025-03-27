from typing import Dict, Any, Optional, TypeVar, cast
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from ..nodes.query_analyzer import QueryAnalyzer
from ..nodes.retriever import Retriever
from ..nodes.grader import Grader
from ..nodes.generator import Generator
from ..nodes.web_searcher import WebSearcher
from ..types.state import RAGState

# Type for the state
State = TypeVar("State", bound=RAGState)

class RAGGraph:
    """Builds and configures the LangGraph workflow for the RAG system."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or {}
        
        # Initialize nodes
        self.query_analyzer = QueryAnalyzer()
        self.retriever = Retriever()
        self.grader = Grader()
        self.generator = Generator(llm=self.llm)
        self.web_searcher = WebSearcher(search_tool=None)  # Configure search tool in implementation
        
        # Configure retry limits
        self.max_retrieval_attempts = config.get("max_retrieval_attempts", 3)
        self.max_generation_attempts = config.get("max_generation_attempts", 2)
        
    def build(self) -> StateGraph:
        """
        Build the LangGraph workflow according to the architecture diagram.
        
        The workflow implements:
        1. Query Analysis with routing to either retrieval or web search
        2. RAG pipeline with document retrieval, grading, and generation
        3. Self-reflection loops for hallucination detection and answer validation
        
        Returns:
            Configured StateGraph ready for execution
        """
        # Create workflow graph with typed state
        workflow = StateGraph(RAGState)
        
        # Add all nodes
        workflow.add_node("query_analyzer", self.query_analyzer.analyze)
        workflow.add_node("retriever", self.retriever.retrieve)
        workflow.add_node("grader", self.grader.grade)
        workflow.add_node("generator", self.generator.generate)
        workflow.add_node("web_searcher", self.web_searcher.search)
        workflow.add_node("rewrite_question", self._rewrite_question)
        
        # 1. Query Analysis Section
        workflow.add_conditional_edges(
            "query_analyzer",
            self.route_from_analysis,
            {
                "retrieve": "retriever",  # Related to index
                "web_search": "web_searcher",  # Unrelated to index
            }
        )
        
        # 2. Document Processing Path
        # Check retrieval attempts first
        workflow.add_conditional_edges(
            "retriever",
            self.check_retrieval_attempts,
            {
                "continue": "grader",
                "end": END  # End if too many retrieval attempts
            }
        )
        
        # 3. Document Relevance Check ("Docs?")
        workflow.add_conditional_edges(
            "grader",
            self.check_docs_relevance,
            {
                "yes": "generator",  # Docs are relevant
                "no": "rewrite_question"  # Docs not relevant, rewrite query
            }
        )
        
        # 4. Handle rewritten questions
        workflow.add_edge("rewrite_question", "retriever")
        
        # 5. Web search path (optional)
        workflow.add_edge("web_searcher", "generator")
        
        # 6. Generation and Self-Reflection
        # First check for hallucinations
        workflow.add_conditional_edges(
            "generator",
            self.check_hallucinations,
            {
                "yes": "retriever",  # Has hallucinations, retry retrieval
                "no": "answer_quality"  # No hallucinations, check answer quality
            }
        )
        
        # 7. Answer Quality Check
        workflow.add_node("answer_quality", self.check_answer_quality)
        workflow.add_conditional_edges(
            "answer_quality",
            self.route_answer_quality,
            {
                "yes": END,  # Good answer, end
                "no": "generator",  # Poor answer, regenerate
                "end": END  # Too many attempts
            }
        )
        
        # Set entry point
        workflow.set_entry_point("query_analyzer")
        
        return workflow
    
    def route_from_analysis(self, state: State) -> str:
        """Route based on query analysis results."""
        state = cast(RAGState, state)
        state["current_node"] = "query_analyzer"
        
        if state["is_related_to_index"]:
            return "retrieve"
        elif self.web_searcher.search_tool:  # Only use web search if configured
            return "web_search"
        else:
            return "retrieve"  # Fallback to retrieval if web search not available
    
    def check_retrieval_attempts(self, state: State) -> str:
        """Check if we should continue retrieving or end due to too many attempts."""
        state = cast(RAGState, state)
        if state["retry_count"] >= self.max_retrieval_attempts:
            return "end"
        return "continue"
    
    def check_docs_relevance(self, state: State) -> str:
        """Check if retrieved documents are relevant."""
        state = cast(RAGState, state)
        if state["docs_relevant"]:
            return "yes"
        return "no"
    
    def check_hallucinations(self, state: State) -> str:
        """Check if the generated answer contains hallucinations."""
        state = cast(RAGState, state)
        if state.get("has_hallucinations", False):
            state["retry_count"] = state.get("retry_count", 0) + 1
            return "yes"
        return "no"
    
    async def check_answer_quality(self, state: State) -> State:
        """Check if the answer sufficiently addresses the question."""
        state = cast(RAGState, state)
        # This would be implemented with actual quality checking logic
        return state
    
    def route_answer_quality(self, state: State) -> str:
        """Route based on answer quality check."""
        state = cast(RAGState, state)
        
        # Check for too many attempts first
        if state.get("retry_count", 0) >= self.max_generation_attempts:
            return "end"
            
        # Check if answer addresses question
        if state.get("answers_question", False):
            return "yes"
        
        # Increment retry count and try again
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "no"
    
    async def _rewrite_question(self, state: State) -> State:
        """Rewrite the question for better retrieval."""
        state = cast(RAGState, state)
        # This would be implemented in the actual node
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state 
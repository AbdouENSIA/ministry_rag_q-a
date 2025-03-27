from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class RAGState(TypedDict):
    """Type definition for the RAG system state that tracks the complete workflow."""
    
    # Input
    query: str
    
    # Query Analysis
    is_related_to_index: bool  # Tracks if query is related to knowledge base
    
    # Document Retrieval & Processing
    documents: Optional[List[Document]]  # Retrieved documents or web results
    rewritten_query: Optional[str]  # Rewritten query for better retrieval
    
    # Document Grading
    docs_relevant: Optional[bool]  # Whether retrieved docs are relevant
    
    # Generation & Validation
    answer: Optional[str]  # Generated answer
    has_hallucinations: Optional[bool]  # Whether answer contains hallucinations
    answers_question: Optional[bool]  # Whether answer sufficiently addresses question
    
    # Web Search
    web_search_results: Optional[List[str]]  # Results from web search if used
    
    # Metadata & Tracking
    metadata: dict  # Additional metadata and tracking information
    current_node: str  # Tracks current position in workflow
    retry_count: int  # Tracks number of retries for loops 
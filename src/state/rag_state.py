from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    """State type for the RAG pipeline."""
    
    # Input
    query: str
    chat_history: Optional[List[BaseMessage]]
    
    # Query Analysis
    is_related_to_index: bool
    query_type: str  # factual, analytical, procedural
    query_entities: List[str]
    query_intent: str
    
    # Document Retrieval & Processing
    documents: Optional[List[Document]]
    rewritten_query: Optional[str]
    retrieval_strategy: str  # dense, sparse, hybrid
    retrieval_scores: Optional[List[float]]
    
    # Document Grading
    docs_relevant: Optional[bool]
    doc_scores: Optional[List[float]]
    doc_metadata: Optional[List[Dict[str, Any]]]
    
    # Generation & Validation
    answer: Optional[str]
    has_hallucinations: Optional[bool]
    answers_question: Optional[bool]
    confidence_score: Optional[float]
    supporting_evidence: Optional[List[str]]
    
    # Web Search
    web_search_results: Optional[List[Dict[str, Any]]]
    
    # Metadata & Tracking
    metadata: Dict[str, Any]
    current_node: str
    retry_count: int
    processing_time: float
    error: Optional[str]
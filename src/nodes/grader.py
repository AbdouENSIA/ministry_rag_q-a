from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state.rag_state import RAGState


class Grader:
    """Adaptive document grader that evaluates document relevance and quality."""
    
    def __init__(self, llm: BaseChatModel, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.rate_limiter = rate_limiter
        
        # Initialize grading prompts
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at evaluating document relevance for RAG systems.

            EVALUATION CRITERIA:
            1. Semantic Relevance:
               - Core concept alignment with query
               - Coverage of key entities and relationships
               - Contextual appropriateness
            
            2. Information Value:
               - Specificity and detail level
               - Currency and timeliness
               - Authoritativeness of content
            
            3. Query Type Alignment:
               - For factual queries: Look for precise, verifiable information
               - For analytical queries: Look for comprehensive, analytical content
               - For procedural queries: Look for step-by-step instructions
               - For conversational queries: Look for contextually appropriate information
            
            4. Document Quality:
               - Clarity and coherence
               - Information density
               - Source credibility
            
            SCORING GUIDELINES:
            - 0.0-0.2: Completely irrelevant
            - 0.2-0.4: Tangentially related
            - 0.4-0.6: Moderately relevant
            - 0.6-0.8: Highly relevant
            - 0.8-1.0: Perfectly relevant
            
            Respond in JSON format with the following structure:
            {{
                "docs_relevant": bool,  # True if any doc scores > 0.6
                "doc_scores": List[float],  # Individual relevance scores
                "doc_metadata": List[Dict[str, Any]],  # Per-document analysis
                "key_matches": List[str],  # Important matching concepts
                "missing_aspects": List[str],  # Query aspects not covered
                "grading_explanation": str  # Reasoning behind scores
            }}
            """),
            ("user", """Query: {query}
            Query type: {query_type}
            Query intent: {query_intent}
            Analysis confidence: {confidence}
            Analysis reasoning: {reasoning}
            
            Documents:
            {documents}
            """)
        ])
        
    async def grade(self, state: RAGState) -> RAGState:
        """
        Grade retrieved documents for relevance and quality.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with grading results
        """
        # Get required information from state
        query = state["query"]
        query_type = state["query_type"]
        query_intent = state["query_intent"]
        documents = state["documents"]
        
        # Get optional analysis info with defaults
        confidence = state.get("confidence", 0.0)
        reasoning = state.get("reasoning", "No prior analysis reasoning available")
        
        if not documents:
            state.update({
                "docs_relevant": False,
                "current_node": "grader"
            })
            return state
            
        # Format documents for grading
        doc_texts = self._format_documents(documents)
        
        # Run grading
        if self.rate_limiter:
            await self.rate_limiter.wait()
        chain = self.relevance_prompt | self.llm | self.parser
        grading_result = await chain.ainvoke({
            "query": query,
            "query_type": query_type,
            "query_intent": query_intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "documents": doc_texts
        })
        
        # Update state with grading results
        state.update({
            "docs_relevant": grading_result["docs_relevant"],
            "doc_scores": grading_result["doc_scores"],
            "doc_metadata": grading_result["doc_metadata"],
            "current_node": "grader"
        })
        
        return state
        
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for grading prompt."""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
        return "\n".join(formatted_docs) 
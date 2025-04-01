from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state.rag_state import RAGState


class QueryAnalyzer:
    """Analyzes queries to determine their type, intent, and relevance to the knowledge base."""
    
    def __init__(self, llm: BaseChatModel, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.rate_limiter = rate_limiter
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query analyzer for a RAG system, responsible for understanding and classifying user queries with high precision.

            IMPORTANT GUIDELINES:
            1. Consider a query "related to index" in these cases:
               - It asks about specific domain knowledge that might be in the index
               - It's a general query that can be answered using indexed knowledge
               - It's a greeting or general question that should be handled gracefully
            2. Only mark as "unrelated to index" if the query:
               - Explicitly requires real-time data (e.g., current stock prices)
               - Needs information that's definitely not in any knowledge base
               - Requires external API calls or web-specific functionality

            Query Types:
            - factual: Direct questions seeking specific information
            - analytical: Questions requiring analysis, comparison, or reasoning
            - procedural: How-to questions or step-by-step instructions
            - conversational: Greetings, chitchat, or general dialogue
            
            Entity Analysis:
            - Extract both explicit and implicit entities
            - Include relevant context words around entities
            - Note relationships between entities
            
            Intent Classification:
            - information_seeking: User wants to learn something
            - clarification: User needs explanation
            - greeting: User is initiating/continuing conversation
            - task: User wants to accomplish something
            - feedback: User is providing feedback
            
            Respond in JSON format with the following structure:
            {{
                "is_related_to_index": bool,  # True unless explicitly requires external data
                "query_type": str,  # factual, analytical, procedural, or conversational
                "query_entities": List[str],  # Extracted entities and context
                "query_intent": str,  # Detailed intent classification
                "confidence": float,  # 0.0 to 1.0 confidence in analysis
                "reasoning": str  # Brief explanation of classification decisions
            }}
            """),
            ("user", "Analyze this query with the above guidelines: {query}")
        ])
        
    async def analyze(self, state: RAGState) -> RAGState:
        """
        Analyze the query and update the state with analysis results.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with query analysis
        """
        # Get query from state
        query = state["query"]
        
        # Run analysis
        if self.rate_limiter:
            await self.rate_limiter.wait()
        chain = self.analysis_prompt | self.llm | self.parser
        analysis_result = await chain.ainvoke({"query": query})
        
        # Update state with analysis results
        state.update({
            "is_related_to_index": analysis_result["is_related_to_index"],
            "query_type": analysis_result["query_type"],
            "query_entities": analysis_result["query_entities"],
            "query_intent": analysis_result["query_intent"],
            "confidence": analysis_result["confidence"],
            "reasoning": analysis_result["reasoning"],
            "current_node": "query_analyzer"
        })
        
        return state 
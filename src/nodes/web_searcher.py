import asyncio
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from ..state.rag_state import RAGState


class WebSearcher:
    """Performs web searches for queries that need external information."""
    
    def __init__(self, llm: BaseChatModel, config: Optional[Dict[str, Any]] = None, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.config = config or {}
        self.parser = JsonOutputParser()
        self.rate_limiter = rate_limiter
        
        # Initialize Tavily client if API key is provided
        tavily_api_key = self.config.get("tavily_api_key")
        self.search_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing web search results and synthesizing information.
            
            ANALYSIS GUIDELINES:
            1. Evaluate source credibility and relevance
            2. Cross-reference information across sources
            3. Identify key facts and insights
            4. Note any conflicting information
            
            Respond in JSON format with the following structure:
            {{
                "synthesized_info": str,  # Coherent summary of findings
                "key_facts": List[str],  # Important verified facts
                "sources": List[str],  # Relevant source URLs
                "confidence": float  # 0.0 to 1.0 confidence in findings
            }}
            """),
            ("user", """Query: {query}
            Search results: {search_results}
            """)
        ])
        
    async def search(self, state: RAGState) -> RAGState:
        """
        Perform web search and analyze results.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with search results
        """
        if not self.search_client:
            state.update({
                "error": "Web search not configured - missing Tavily API key",
                "current_node": "web_searcher"
            })
            return state
            
        query = state["query"]
        
        try:
            # Run search in executor to avoid blocking
            search_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.search_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5
                )
            )
            
            # Analyze search results
            if self.rate_limiter:
                await self.rate_limiter.wait()
            chain = self.search_prompt | self.llm | self.parser
            analysis_result = await chain.ainvoke({
                "query": query,
                "search_results": search_results
            })
            
            # Update state with results
            state.update({
                "web_search_results": search_results,
                "synthesized_info": analysis_result["synthesized_info"],
                "key_facts": analysis_result["key_facts"],
                "sources": analysis_result["sources"],
                "search_confidence": analysis_result["confidence"],
                "current_node": "web_searcher"
            })
            
        except Exception as e:
            state.update({
                "error": f"Web search failed: {str(e)}",
                "current_node": "web_searcher"
            })
            
        return state 
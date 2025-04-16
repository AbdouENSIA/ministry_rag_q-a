import asyncio
import json
import logging
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from ..state.rag_state import RAGState

# Configure logging
logger = logging.getLogger(__name__)

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
            ("system", """You are an expert at analyzing web search results and synthesizing accurate, comprehensive information.
            
            OUTPUT FORMAT REQUIREMENTS:
            You MUST respond with a valid JSON object containing EXACTLY these fields:
            {{
                "synthesized_info": string,
                "key_facts": array of strings,
                "sources": array of strings,
                "confidence": number (between 0.01 and 0.99)
            }}
            
            If ANY field is missing or incorrectly formatted, the system will fail. ENSURE VALID JSON SYNTAX.
            
            SYNTHESIS GUIDELINES:
            1. Source Evaluation:
               - Assess credibility based on domain authority, publication date, and author expertise
               - Prioritize information from official websites, academic sources, and reputable publications
               - De-prioritize information from forums, personal blogs, or commercial sites with clear bias
               - Compare information across multiple sources to verify consistency
               - Note when sources conflict and explain your reasoning for choosing one over others
            
            2. Information Extraction:
               - Focus on extracting factual, verifiable information responsive to the query
               - Gather comprehensive details covering all aspects of the query
               - Capture nuance, context, and relevant background information
               - Note limitations, exceptions, and qualifications
               - Extract direct quotes only when particularly authoritative or illuminating
            
            3. Synthesis Strategy:
               - Organize information logically by topic, chronology, or importance
               - Combine complementary information from multiple sources
               - Eliminate redundancies while preserving important details
               - Present a complete picture that addresses all aspects of the query
               - Maintain nuance and avoid oversimplification
            
            FIELD-SPECIFIC REQUIREMENTS:
            
            1. "synthesized_info" (String):
               - Write a coherent, comprehensive paragraph summarizing all relevant information
               - 100-300 words of synthesized content
               - Include ALL key information relevant to the query
               - Present a balanced view incorporating all sources
               - Use clear, precise language avoiding unnecessary jargon
               - NEVER say "According to the search results" or similar phrases
               - Structure as a direct response to the query without referencing the search process
            
            2. "key_facts" (Array of Strings):
               - Include 3-7 discrete, important facts extracted from sources
               - Each fact should be a complete, standalone statement (15-30 words)
               - Focus on the most important, verified information
               - Ensure facts directly relate to the query
               - Present in order of importance
               - Start each with a clear subject and active verb
               - Format consistently across all items
            
            3. "sources" (Array of Strings):
               - Include ONLY the full URLs of sources that directly contributed information
               - List in order of importance/relevance
               - Include at minimum 1 source, maximum 5 sources
               - Use exact, complete URLs from the search results
               - Do not include sources that weren't useful or relevant
               - Ensure each URL is properly formatted as a string
            
            4. "confidence" (Number):
               - Provide a score between 0.01 and 0.99 (never exactly 0 or 1)
               - Base on source quality, consistency across sources, and comprehensiveness
               - 0.8-0.99: High-quality sources with consistent information
               - 0.5-0.79: Good sources with mostly consistent information
               - 0.3-0.49: Mixed quality sources or some inconsistencies
               - 0.01-0.29: Poor quality sources, major inconsistencies, or minimal relevant information
            
            HANDLING EDGE CASES:
            - For queries with no good results: Provide best available information with low confidence score
            - For controversial topics: Present balanced view noting different perspectives
            - For technical topics: Focus on accuracy while keeping language accessible
            - For time-sensitive queries: Note recency of information and potential for changes
            
            EXAMPLE OF PROPERLY FORMATTED RESPONSE:
            {{
                "synthesized_info": "Electric vehicles typically have a range of 100-400 miles on a single charge, depending on the model and battery size. Tesla models generally offer the longest range, with the Model S Long Range providing up to 405 miles. Other factors affecting range include driving conditions, speed, temperature, and vehicle age. Most modern EVs can charge to 80% in 30-60 minutes at fast charging stations, while home charging usually takes 6-12 hours for a full charge.",
                "key_facts": [
                    "Tesla Model S Long Range offers the highest EV range at 405 miles per charge according to EPA ratings.",
                    "EV range is typically reduced by 10-40% in cold weather conditions below freezing.",
                    "Fast charging stations (Level 3) can charge most EVs to 80% in 30-60 minutes.",
                    "The average range of mainstream electric vehicles in 2023 is approximately 250 miles per charge."
                ],
                "sources": [
                    "https://www.fueleconomy.gov/feg/evtech.shtml",
                    "https://www.tesla.com/models",
                    "https://www.energy.gov/energysaver/electric-vehicles"
                ],
                "confidence": 0.87
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
        logger.info("\n" + "="*80)
        logger.info(f"[WEB_SEARCHER] Starting web search for query: '{state['query']}'")
        logger.info("-"*80)
        
        if not self.search_client:
            logger.warning("[WEB_SEARCHER] Web search not configured - missing Tavily API key")
            state.update({
                "error": "Web search not configured - missing Tavily API key",
                "current_node": "web_searcher"
            })
            return state
            
        query = state["query"]
        
        try:
            # Run search in executor to avoid blocking
            logger.info(f"[WEB_SEARCHER] Performing Tavily search for query: '{query}'")
            search_results = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.search_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5
                )
            )
            
            # Log search results summary
            result_count = len(search_results.get("results", []))
            logger.info(f"[WEB_SEARCHER] Search returned {result_count} results")
            
            # Log top search results
            for i, result in enumerate(search_results.get("results", [])[:3], 1):
                logger.info(f"[WEB_SEARCHER] Result {i}: {result.get('title', 'No title')} - {result.get('url', 'No URL')}")
                content_preview = result.get('content', '')[:150] + '...' if len(result.get('content', '')) > 150 else result.get('content', '')
                logger.info(f"[WEB_SEARCHER] Content preview: {content_preview}")
            
            # Analyze search results
            if self.rate_limiter:
                await self.rate_limiter.wait()
                
            logger.info("[WEB_SEARCHER] [LLM CALL] Analyzing search results with LLM...")
            
            input_data = {
                "query": query,
                "search_results": search_results
            }
            logger.info(f"[WEB_SEARCHER] [LLM INPUT] Query: '{query}', Search result count: {result_count}")
            
            chain = self.search_prompt | self.llm | self.parser
            analysis_result = await chain.ainvoke(input_data)
            
            logger.info("[WEB_SEARCHER] [LLM OUTPUT] Analysis completed successfully")
            logger.info(f"[WEB_SEARCHER] Synthesized info length: {len(analysis_result.get('synthesized_info', ''))}")
            logger.info(f"[WEB_SEARCHER] Key facts: {len(analysis_result.get('key_facts', []))} items")
            logger.info(f"[WEB_SEARCHER] Sources: {len(analysis_result.get('sources', []))} URLs")
            logger.info(f"[WEB_SEARCHER] Confidence: {analysis_result.get('confidence', 0)}")
            
            # Log synthesized info preview
            info_preview = analysis_result.get('synthesized_info', '')[:150] + '...' if len(analysis_result.get('synthesized_info', '')) > 150 else analysis_result.get('synthesized_info', '')
            logger.info(f"[WEB_SEARCHER] Synthesized info preview: {info_preview}")
            
            # Update state with results
            state.update({
                "web_search_results": search_results,
                "synthesized_info": analysis_result["synthesized_info"],
                "key_facts": analysis_result["key_facts"],
                "sources": analysis_result["sources"],
                "search_confidence": analysis_result["confidence"],
                "current_node": "web_searcher"
            })
            
            logger.info("[WEB_SEARCHER] Successfully updated state with search results")
            
        except Exception as e:
            logger.error(f"[WEB_SEARCHER] Search failed with error: {str(e)}")
            state.update({
                "error": f"Web search failed: {str(e)}",
                "current_node": "web_searcher"
            })
        
        logger.info(f"[WEB_SEARCHER] Web search completed")
        logger.info("="*80)
            
        return state 
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

            OUTPUT FORMAT REQUIREMENTS:
            You MUST respond with a valid JSON object containing EXACTLY these fields:
            {{
                "is_related_to_index": boolean,
                "query_type": string (one of: "factual", "analytical", "procedural", "conversational"),
                "query_entities": array of strings,
                "query_intent": string,
                "confidence": number (between 0.0 and 1.0),
                "reasoning": string
            }}
            
            If ANY field is missing or incorrectly formatted, the system will fail. ENSURE VALID JSON SYNTAX.

            CLASSIFICATION GUIDELINES:
            1. Consider a query "related to index" (is_related_to_index = true) in these cases:
               - It asks about specific domain knowledge that might be in the index
               - It's a general query that can be answered using indexed knowledge
               - It's a greeting or general question that should be handled gracefully
               - DEFAULT TO TRUE if uncertain - it's better to attempt an answer with available knowledge
            
            2. Only mark as "unrelated to index" (is_related_to_index = false) if the query:
               - Explicitly requires real-time data that definitely won't be in static knowledge (e.g., current stock prices)
               - Needs information that's definitely not in any knowledge base (e.g., personal user data)
               - Requires external API calls or web-specific functionality
               - Contains malicious intent or violates ethical guidelines
            
            QUERY TYPE DEFINITIONS (ALWAYS assign one of these exact values for "query_type"):
            - "factual": Direct questions seeking specific information, facts, definitions, or straightforward answers
              Examples: "What is X?", "When was Y invented?", "Who created Z?"
            
            - "analytical": Questions requiring analysis, comparison, reasoning, evaluation, or synthesis
              Examples: "Why did X happen?", "How does Y compare to Z?", "What are the implications of X?"
            
            - "procedural": How-to questions or step-by-step instructions seeking guidance on processes
              Examples: "How do I X?", "What steps should I follow to Y?", "Explain the process of Z"
            
            - "conversational": Greetings, chitchat, general dialogue, or questions about the system itself
              Examples: "Hello", "How are you?", "Can you help me with something?", "What can you do?"
            
            ENTITY EXTRACTION REQUIREMENTS (query_entities):
            - Extract both explicit and implicit entities (minimum 1, maximum 10)
            - Include relevant context words around entities
            - Note relationships between entities
            - For ambiguous queries, extract broader topic areas
            - If no entities are present (e.g., in pure greetings), include an empty array
            
            INTENT CLASSIFICATION GUIDELINES (query_intent):
            - Be specific and detailed about the user's actual goal
            - Choose from these categories but ADD SPECIFICITY beyond the category name:
              - information_seeking: User wants to learn something specific
              - clarification: User needs explanation or disambiguation
              - greeting: User is initiating/continuing conversation
              - task: User wants to accomplish something specific
              - feedback: User is providing feedback or opinions
            - ALWAYS include details on what exact information or action the user seeks
            
            CONFIDENCE SCORING RULES:
            - 0.9-1.0: Very clear, unambiguous query with obvious classification
            - 0.7-0.9: Clear intent but some minor ambiguity
            - 0.5-0.7: Moderate ambiguity but reasonably confident
            - 0.3-0.5: Significant ambiguity with multiple possible interpretations
            - 0.0-0.3: Extremely vague or ambiguous query
            - NEVER leave as 0.0 or 1.0 exactly - always provide a nuanced score
            
            REASONING REQUIREMENTS:
            - Provide clear justification for all classifications
            - Explain any ambiguities or challenges in classification
            - Keep under 100 words but be specific and thorough
            - Focus on WHY you classified as you did rather than restating the classification
            
            HANDLING EDGE CASES:
            - Empty queries: Classify as conversational with low confidence
            - Non-sensical queries: Attempt best classification with low confidence
            - Multi-part queries: Focus on dominant intent, note multiple aspects in reasoning
            - Ambiguous queries: Choose most likely classification, note alternatives in reasoning
            
            EXAMPLES OF PROPERLY FORMATTED RESPONSES:
            For "What is quantum computing?":
            {{
                "is_related_to_index": true,
                "query_type": "factual",
                "query_entities": ["quantum computing"],
                "query_intent": "information_seeking: Learn basic definition and concept of quantum computing",
                "confidence": 0.95,
                "reasoning": "Clear factual question seeking definition and explanation of a specific technical concept."
            }}
            
            For "Hello, how are you today?":
            {{
                "is_related_to_index": true,
                "query_type": "conversational",
                "query_entities": [],
                "query_intent": "greeting: Initiating casual conversation with the system",
                "confidence": 0.98,
                "reasoning": "Standard conversational greeting with no specific information request."
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
        
        # Make sure we pass only the expected parameters to the prompt
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
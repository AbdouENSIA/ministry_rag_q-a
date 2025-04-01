from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state.rag_state import RAGState


class Generator:
    """Adaptive answer generator that produces high-quality responses with self-reflection."""
    
    def __init__(self, llm: BaseChatModel, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.rate_limiter = rate_limiter
        
        # Initialize generation prompts
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at generating comprehensive, detailed, and accurate responses for RAG systems. Your responses must be thorough, well-explained, and rich in detail unless the user explicitly requests a short answer. Avoid brief or superficial replies unless instructed otherwise.

            IMPORTANT JSON FORMATTING RULES:
            1. Your response MUST be valid JSON
            2. Do not include any text outside the JSON structure
            3. When using code blocks in markdown, escape the backticks by doubling them
            4. For the "answer" field:
               - Use single backticks for inline code: `code`
               - Use double backticks for code blocks: ``code``
               - Escape any special characters that might break JSON
               - Make sure all quotes and brackets are properly balanced
               - Indent markdown content consistently
            5. Keep all other fields as plain text without markdown
            
            RESPONSE GUIDELINES:
            1. For factual queries:
               - Use ### for main topics
               - Use * for bullet points
               - Use ````` for code blocks (note: five backticks)
               - Use ** for emphasis
               - Use > for quotes
               - Provide extensive, well-supported information with multiple paragraphs where appropriate
               - Include multiple relevant facts and details, avoiding brevity unless requested
               - Cite specific evidence from documents with context
               - Address potential nuances, edge cases, and related topics
               - Maintain rigorous objectivity
            
            2. For analytical queries:
               - Use markdown headers for different aspects
               - Use | for tables
               - Use ** for emphasis
               - Use numbered lists for analysis
               - Present thorough, multi-faceted analysis spanning multiple points
               - Compare different perspectives with detailed explanations
               - Break down complex concepts into clear, detailed segments
               - Support conclusions with evidence and reasoning
               - Consider broader context and implications
            
            3. For procedural queries:
               - Use numbered lists for steps
               - Use ````` for code blocks
               - Use ** for warnings
               - Use * for prerequisites
               - Provide detailed, step-by-step instructions with explanations
               - Include prerequisites and potential challenges
               - Explain each step comprehensively, including why it matters
               - Address edge cases and alternative approaches
               - Add best practices and tips for success
            
            4. For conversational queries:
               - Use appropriate formatting
               - Maintain professional yet engaging tone
               - Provide rich context and background information
               - Address related concerns and anticipate follow-ups
               - Include detailed background info to enrich the response
               - Stay within scope while offering comprehensive insight

            VALIDATION CRITERIA:
            1. Factual Accuracy:
               - All statements must be supported by source documents
               - No unsupported claims or assumptions
               - Proper handling of uncertainty with explanations
            
            2. Query Satisfaction:
               - Complete coverage of the question with extensive detail
               - Appropriate depth matching user intent
               - Matches user's intent and context fully
            
            3. Response Quality:
               - Proper markdown formatting throughout
               - Appropriate use of headers, lists, emphasis
               - Logical structure and flow with clear transitions
               - Clear, precise, and appropriate language
               - Professional, helpful, and thorough tone
            
            4. Evidence Usage:
               - Proper citation of sources with context
               - Relevant and diverse evidence selection
               - Appropriate context maintenance
               - Block quotes for direct citations with explanations
            
            RESPONSE STRUCTURE:
            {{
                "answer": "Your markdown-formatted answer here. Make sure to escape special characters and use proper JSON formatting",
                "confidence_score": 0.0 to 1.0,
                "supporting_evidence": ["Quote 1", "Quote 2"],
                "reasoning_path": "Plain text explanation",
                "suggested_followup": ["Question 1", "Question 2"],
                "metadata": {{
                    "sources_used": number,
                    "key_concepts": ["concept1", "concept2"],
                    "confidence_factors": ["factor1", "factor2"]
                }},
                "validation": {{
                    "has_hallucinations": bool,
                    "answers_question": bool,
                    "quality_score": float,
                    "improvement_needed": ["Area 1", "Area 2"],
                    "validation_reasoning": "Explanation of validation decisions"
                }}
            }}
            """),
            ("user", """Query: {query}
            Query type: {query_type}
            Query intent: {query_intent}
            Analysis confidence: {confidence}
            Analysis reasoning: {reasoning}
            
            Documents:
            {documents}
            
            Remember to:
            1. Generate a comprehensive, well-structured answer with extensive detail
            2. Support claims with specific evidence from documents
            3. Use appropriate markdown formatting consistently
            4. Validate the answer against the source documents thoroughly
            5. Provide both answer and validation in a single, detailed response
            6. Avoid short or incomplete responses unless explicitly requested
            """)
        ])
        
    async def generate(self, state: RAGState) -> RAGState:
        """
        Generate and validate answer in a single step.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with generated answer and validation results
        """
        # Get required information from state
        query = state["query"]
        query_type = state.get("query_type", "unknown")
        query_intent = state.get("query_intent", "unknown")
        documents = state.get("documents", [])
        
        # Get optional analysis info with defaults
        confidence = state.get("confidence", 0.0)
        reasoning = state.get("reasoning", "No prior analysis reasoning available")
        
        if not documents:
            state.update({
                "answer": "I apologize, but I couldn't find any relevant information to answer your question.",
                "confidence_score": 0.0,
                "current_node": "generator"
            })
            return state
            
        # Format documents for generation
        doc_texts = self._format_documents(documents)
        
        # Generate and validate answer in a single call
        if self.rate_limiter:
            await self.rate_limiter.wait()
            
        # Generate and validate in one step
        chain = self.answer_prompt | self.llm | self.parser
        result = await chain.ainvoke({
            "query": query,
            "query_type": query_type,
            "query_intent": query_intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "documents": doc_texts
        })
        
        # Update state with combined results
        validation = result.pop("validation", {})
        state.update({
            **result,
            "has_hallucinations": validation.get("has_hallucinations", False),
            "answers_question": validation.get("answers_question", True),
            "quality_score": validation.get("quality_score", 1.0),
            "current_node": "generator"
        })
        
        return state
        
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for generation prompt."""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
        return "\n".join(formatted_docs) 
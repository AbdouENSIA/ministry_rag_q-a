from typing import Any, List, Optional
import json

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

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

            IMPORTANT: RESPOND ONLY WITH A VALID JSON OBJECT - DO NOT INCLUDE ANY EXPLANATION OR ADDITIONAL TEXT OUTSIDE THE JSON STRUCTURE.
            ANY TEXT OUTSIDE THE JSON WILL CAUSE A SYSTEM FAILURE.

            OUTPUT FORMAT REQUIREMENTS:
            You MUST respond with a valid JSON object containing EXACTLY these fields:
            {{
                "docs_relevant": boolean,
                "doc_scores": array of numbers (between 0.0 and 1.0),
                "doc_metadata": array of objects,
                "key_matches": array of strings,
                "missing_aspects": array of strings,
                "grading_explanation": string
            }}
            
            The length of "doc_scores" array MUST match the number of documents provided.
            Each object in "doc_metadata" MUST contain "relevance_aspects" (array of strings) and "quality_issues" (array of strings).
            If ANY field is missing or incorrectly formatted, the system will fail. ENSURE VALID JSON SYNTAX.

            DOCUMENT RELEVANCE PRINCIPLES:
            1. Relevance means the document contains information that DIRECTLY addresses the query
            2. A document is considered relevant ONLY if it contains SPECIFIC information that helps answer the query
            3. General background information is less relevant than precise, query-specific details
            4. A document can be high quality but still irrelevant if it doesn't address the specific query
            5. Partial matches should be scored proportionally to how much query-specific information they contain

            EVALUATION CRITERIA:
            1. Semantic Relevance (40% of score):
               - Core concept alignment between document content and query
               - Coverage of key entities mentioned in the query
               - Contextual appropriateness to query intent
               - Conceptual relatedness even if different terminology is used
               - Presence of information that directly answers the query
            
            2. Information Value (30% of score):
               - Specificity and detail level relevant to the query
               - Currency and timeliness of information
               - Authoritativeness and credibility of content
               - Comprehensiveness in covering different aspects of the query
               - Uniqueness of information compared to other documents
            
            3. Query Type Alignment (20% of score):
               - For factual queries: Presence of precise, verifiable information and definitions
               - For analytical queries: Presence of explanations, analyses, and multiple perspectives
               - For procedural queries: Presence of clear steps, requirements, and process details
               - For conversational queries: Presence of contextually appropriate information
            
            4. Document Quality (10% of score):
               - Clarity and coherence of writing
               - Information density and signal-to-noise ratio
               - Internal consistency and logical structure
               - Absence of errors, outdated information, or misleading content
               - Appropriate level of technical language for the query
            
            SCORING GUIDELINES:
            - 0.0-0.19: Completely irrelevant, containing no information related to the query
            - 0.2-0.39: Tangentially related, containing only minimal or indirect connections
            - 0.4-0.59: Moderately relevant, containing some useful but incomplete information
            - 0.6-0.79: Highly relevant, containing substantial information that addresses the query
            - 0.8-1.0: Perfectly relevant, comprehensively addressing all aspects of the query
            
            DOCUMENT RELEVANCE ASSESSMENT METHODOLOGY:
            1. Analyze each document independently against query requirements
            2. For each document, identify SPECIFIC content elements that match query needs
            3. For each document, identify SPECIFIC gaps or quality issues
            4. Note EXACTLY what information is present and what is missing
            5. Set "docs_relevant" to true if ANY document scores 0.6 or higher
            6. Document scores MUST reflect absolute relevance, not just relative ranking
            7. Be strict and objective - a document is only relevant if it ACTUALLY contains information that helps answer the query
            
            FIELD-SPECIFIC REQUIREMENTS:
            
            1. "docs_relevant" (Boolean):
               - true if ANY document scores 0.6 or higher
               - false if ALL documents score below 0.6
            
            2. "doc_scores" (Array of Numbers):
               - One score per document, in the same order as provided documents
               - Each score must be between 0.0 and 1.0
               - Reflect absolute relevance according to scoring guidelines, not relative ranking
               - Must accurately reflect all evaluation criteria in appropriate proportions
               - Base scores ONLY on actual document content and relevance to query
            
            3. "doc_metadata" (Array of Objects):
               - One object per document with this structure:
                 {{
                   "relevance_aspects": ["aspect1", "aspect2", ...],
                   "quality_issues": ["issue1", "issue2", ...]
                 }}
               - "relevance_aspects": 2-5 SPECIFIC content elements that make the document relevant
               - Each relevance aspect should be concrete, citing a specific piece of information
               - "quality_issues": 0-3 problems affecting document usefulness (empty array if none)
               - Each quality issue should point to a specific problem, not generic criticism
            
            4. "key_matches" (Array of Strings):
               - 2-7 specific content elements across all documents that directly match query needs
               - Focus on key concepts, facts, explanations present in documents that satisfy the query
               - Each entry should refer to a SPECIFIC information element (not document names)
               - Be precise about which information elements directly answer the query
               - If no matches, include ["No significant matches found"]
            
            5. "missing_aspects" (Array of Strings):
               - 0-5 important aspects of the query not covered by any documents
               - Focus on informational gaps that would be needed for a complete answer
               - Each entry should identify a specific missing information element
               - Be precise about what information is needed but not present
               - If all aspects covered, include empty array []
            
            6. "grading_explanation" (String):
               - 50-200 word explanation of overall document set relevance
               - Explain key factors that influenced your scoring decisions
               - Highlight strongest and weakest documents with specific reasons
               - Note any particular challenges in assessing this document set
               - Focus on SPECIFIC content elements that determined relevance scores
            
            HANDLING EDGE CASES:
            - Empty document sets: Set "docs_relevant" to false, all scores to 0.0
            - Ambiguous queries: Interpret reasonably based on provided query analysis
            - Technical documents: Focus on informational value despite technical complexity
            - Redundant documents: Reward comprehensive information rather than repetition
            - Short snippets: Evaluate based on available content, not theoretical full documents
            
            CRITICAL: YOU MUST ONLY RETURN VALID JSON. DO NOT INCLUDE ANY TEXT THAT IS NOT PART OF THE JSON STRUCTURE.
            DO NOT INCLUDE MARKDOWN CODE BLOCKS, EXPLANATIONS, OR ANY TEXT OUTSIDE THE JSON OBJECT.
            
            EXAMPLE OF PROPERLY FORMATTED RESPONSE:
            {{
                "docs_relevant": true,
                "doc_scores": [0.85, 0.72, 0.45, 0.31],
                "doc_metadata": [
                    {{
                        "relevance_aspects": ["Contains complete definition of sigmoid function with mathematical formula", "Explains relationship to logistic function", "Provides historical context of use in neural networks"],
                        "quality_issues": []
                    }},
                    {{
                        "relevance_aspects": ["Includes code implementation in Python", "Explains derivative properties", "Shows practical applications in classification"],
                        "quality_issues": ["Uses outdated NumPy methods", "Contains a minor error in derivative formula"]
                    }},
                    {{
                        "relevance_aspects": ["Mentions sigmoid as activation function", "References limited use in modern networks"],
                        "quality_issues": ["Very brief coverage", "Lacks mathematical details", "Missing implementation examples"]
                    }},
                    {{
                        "relevance_aspects": ["Brief mention of sigmoid in activation function comparison"],
                        "quality_issues": ["Mostly focused on ReLU, not sigmoid", "No technical details about sigmoid", "No implementation examples"]
                    }}
                ],
                "key_matches": [
                    "Mathematical definition: f(x) = 1/(1+e^-x)",
                    "Relationship to logistic function in statistics",
                    "Python implementation using NumPy",
                    "Derivative properties for backpropagation",
                    "Historical use in neural networks"
                ],
                "missing_aspects": [
                    "Comparison with modern activation functions",
                    "Vanishing gradient problem associated with sigmoid"
                ],
                "grading_explanation": "The document set provides strong coverage of the sigmoid function's definition, properties, and implementation, with Documents 1 and 2 being particularly relevant. Document 1 offers a comprehensive definition with mathematical formulation and historical context. Document 2 provides valuable implementation details despite using some outdated methods. Documents 3 and 4 have limited relevance, only briefly mentioning sigmoid functions without substantial detail. The set lacks information on the vanishing gradient problem and detailed comparisons with modern alternatives, which would be necessary for a complete understanding of sigmoid's role in contemporary neural networks."
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
        
        # Add a backup formatting function
        self.json_enforcing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a JSON formatting expert. Your task is to convert the given text into a proper JSON object with the exact structure specified below.
            
            Desired JSON structure:
            {{
                "docs_relevant": boolean,
                "doc_scores": array of numbers (between 0.0 and 1.0),
                "doc_metadata": array of objects,
                "key_matches": array of strings,
                "missing_aspects": array of strings,
                "grading_explanation": string
            }}
            
            Each object in "doc_metadata" MUST contain "relevance_aspects" (array of strings) and "quality_issues" (array of strings).
            
            DO NOT add any commentary, explanation, or text outside the JSON structure. Return ONLY valid JSON.
            """),
            ("user", "{text}")
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
                "doc_scores": [],
                "doc_metadata": [],
                "key_matches": ["No documents to evaluate"],
                "missing_aspects": ["Complete information about the query topic"],
                "grading_explanation": "No documents were retrieved for evaluation. The system was unable to find relevant content matching the query.",
                "current_node": "grader"
            })
            return state
            
        # Format documents for grading
        doc_texts = self._format_documents(documents)
        
        # Run grading
        if self.rate_limiter:
            await self.rate_limiter.wait()
            
        chain = self.relevance_prompt | self.llm
        
        try:
            # First try with direct parsing
            raw_result = await chain.ainvoke({
                "query": query,
                "query_type": query_type,
                "query_intent": query_intent,
                "confidence": confidence,
                "reasoning": reasoning,
                "documents": doc_texts
            })
            
            # Try to parse the result
            try:
                grading_result = self.parser.parse(raw_result)
            except OutputParserException:
                # If parsing fails, try to fix the formatting with a second prompt
                if self.rate_limiter:
                    await self.rate_limiter.wait()
                    
                fix_chain = self.json_enforcing_prompt | self.llm
                fixed_raw_result = await fix_chain.ainvoke({"text": raw_result})
                
                # Try to extract JSON from the text if it's still not properly formatted
                try:
                    grading_result = self.parser.parse(fixed_raw_result)
                except OutputParserException:
                    # Last resort: try to extract JSON using string manipulation
                    grading_result = self._extract_json_fallback(fixed_raw_result, documents)
        except Exception as e:
            # If all else fails, create a default response
            grading_result = self._create_default_grading(documents, str(e))
        
        # Update state with grading results
        state.update({
            "docs_relevant": grading_result["docs_relevant"],
            "doc_scores": grading_result["doc_scores"],
            "doc_metadata": grading_result["doc_metadata"],
            "key_matches": grading_result.get("key_matches", ["No significant matches found"]),
            "missing_aspects": grading_result.get("missing_aspects", []),
            "grading_explanation": grading_result.get("grading_explanation", "Document grading completed successfully."),
            "current_node": "grader"
        })
        
        return state
    
    def _extract_json_fallback(self, text: str, documents: List[Document]) -> dict:
        """Attempt to extract JSON from text or create default grading."""
        try:
            # Try to find JSON in the text between braces
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                return self._create_default_grading(documents, "Could not extract valid JSON")
        except:
            return self._create_default_grading(documents, "JSON extraction failed")
            
    def _create_default_grading(self, documents: List[Document], error: str) -> dict:
        """Create a default grading response when all parsing attempts fail."""
        doc_count = len(documents)
        return {
            "docs_relevant": True,  # Assume documents are relevant as a fallback
            "doc_scores": [0.7] * doc_count,  # Assign moderate scores to all docs
            "doc_metadata": [
                {
                    "relevance_aspects": ["Document content potentially relevant to query"],
                    "quality_issues": []
                } for _ in range(doc_count)
            ],
            "key_matches": ["Potential document match to query"],
            "missing_aspects": [],
            "grading_explanation": f"Default grading applied due to parsing error. {doc_count} documents were evaluated but detailed grading information could not be generated. The documents are assumed to be potentially relevant to the query. Error: {error}"
        }
        
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for grading prompt."""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}\n")
        return "\n".join(formatted_docs) 
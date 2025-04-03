from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from ..state.rag_state import RAGState


class Retriever:
    """Adaptive document retriever that selects optimal retrieval strategy based on query type."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embeddings: Embeddings,
        llm: BaseChatModel,
        config: Optional[Dict[str, Any]] = None,
        rate_limiter: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.config = config or {}
        self.rate_limiter = rate_limiter
        
        # Configure retrieval parameters
        self.default_k = config.get("default_k", 5)
        self.max_k = config.get("max_k", 10)
        self.min_score = config.get("min_score", 0.7)
        
        # Initialize query rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at rewriting queries for optimal retrieval in RAG systems.

            TASK DESCRIPTION:
            Your task is to rewrite the given query to maximize retrieval effectiveness while preserving its original intent.
            You MUST return ONLY the rewritten query as plain text with no additional commentary, explanations, or JSON formatting.
            
            REWRITING GUIDELINES:
            1. Focus on these specific techniques:
               - Expand acronyms and abbreviations into full terms
               - Add synonyms and related terms separated by OR
               - Include technical terminology relevant to the domain
               - Specify document types that might contain the answer (e.g., "policy document", "technical specification")
               - Add context terms that would appear in relevant documents
               - Remove filler words, pleasantries, and irrelevant context
               - Break complex questions into key conceptual components
               - Convert implicit questions to explicit information needs
            
            2. Query Type-Specific Strategies:
               - For factual queries: Focus on key entities, their attributes, and precise terminology
               - For analytical queries: Include key concepts, relationships, and analytical frameworks
               - For procedural queries: Add step-related terms, tool names, and action verbs
               - For conversational queries: Extract and emphasize the core information need
            
            3. Format Requirements:
               - Keep the rewritten query under 50 words
               - Use natural language rather than search operators
               - Maintain readable sentence or phrase structure
               - Preserve all original entities and key terms
               - Add new terms in a way that enhances rather than dilutes relevance
            
            4. Critical Rules:
               - NEVER invent new query intents or change the fundamental question
               - NEVER use placeholder or example text without replacing it
               - NEVER add interpretations that weren't implied in the original query
               - NEVER add special search syntax, boolean operators, or other non-natural language constructs
               - NEVER return anything other than the rewritten query text
            
            5. Handling Edge Cases:
               - For very short queries (1-3 words): Expand with likely related terms and context
               - For vague queries: Add clarifying terms while maintaining breadth
               - For multi-part queries: Focus on key concepts from each part
               - For already-specific queries: Make minimal changes, focusing on synonyms and term expansion
            
            EXAMPLES OF GOOD REWRITES:
            
            Original: "What's ML?"
            Rewritten: "What is machine learning artificial intelligence AI deep learning statistical models"
            
            Original: "How do I configure the system?"
            Rewritten: "How to configure system setup installation settings parameters step by step guide tutorial documentation"
            
            Original: "Tell me about recent developments"
            Rewritten: "recent developments latest updates new features changes improvements advances innovations progress"
            
            REMEMBER: Return ONLY the rewritten query with no additional text, explanation, or formatting.
            """),
            ("user", "Original query: {query}\nQuery type: {query_type}\nQuery intent: {query_intent}")
        ])
        
    async def retrieve(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using adaptive strategies based on query analysis.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved documents
        """
        # Get query and analysis from state
        query = state["query"]
        query_type = state["query_type"]
        
        # Determine retrieval strategy based on query type
        strategy = self._select_retrieval_strategy(query_type)
        state["retrieval_strategy"] = strategy
        
        # Rewrite query if needed
        if strategy in ["hybrid", "sparse"]:
            if self.rate_limiter:
                await self.rate_limiter.wait()
            rewritten_query = await self._rewrite_query(state)
            state["rewritten_query"] = rewritten_query
        else:
            rewritten_query = query
            
        # Perform retrieval based on strategy
        if strategy == "dense":
            docs, scores = await self._dense_retrieval(rewritten_query)
        elif strategy == "sparse":
            docs, scores = await self._sparse_retrieval(rewritten_query)
        else:  # hybrid
            dense_docs, dense_scores = await self._dense_retrieval(rewritten_query)
            sparse_docs, sparse_scores = await self._sparse_retrieval(rewritten_query)
            docs, scores = self._merge_results(dense_docs, dense_scores, sparse_docs, sparse_scores)
            
        # Update state with results
        state.update({
            "documents": docs,
            "retrieval_scores": scores,
            "current_node": "retriever"
        })
        
        return state
        
    def _select_retrieval_strategy(self, query_type: str) -> str:
        """Select optimal retrieval strategy based on query type."""
        if query_type == "factual":
            return "dense"  # Better for exact matches
        elif query_type == "analytical":
            return "hybrid"  # Combine both for comprehensive results
        else:  # procedural
            return "sparse"  # Better for keyword-based retrieval
            
    async def _rewrite_query(self, state: RAGState) -> str:
        """Rewrite query for better retrieval."""
        chain = self.rewrite_prompt | self.llm
        result = await chain.ainvoke({
            "query": state["query"],
            "query_type": state["query_type"],
            "query_intent": state["query_intent"]
        })
        return result.content
        
    async def _dense_retrieval(self, query: str) -> Tuple[List[Document], List[float]]:
        """Perform dense retrieval using embeddings."""
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.default_k
        )
        docs, scores = zip(*results)
        return list(docs), list(scores)
        
    async def _sparse_retrieval(self, query: str) -> Tuple[List[Document], List[float]]:
        """Perform sparse retrieval using keyword matching."""
        # For now, fallback to dense retrieval until proper sparse retrieval is implemented
        return await self._dense_retrieval(query)
        
    def _merge_results(
        self,
        dense_docs: List[Document],
        dense_scores: List[float],
        sparse_docs: List[Document],
        sparse_scores: List[float]
    ) -> Tuple[List[Document], List[float]]:
        """Merge results from different retrieval strategies using reciprocal rank fusion."""
        # Create a map to track best scores for each document
        doc_scores = {}
        
        # Process dense results
        for doc, score in zip(dense_docs, dense_scores):
            doc_key = doc.page_content  # Use content as key for deduplication
            doc_scores[doc_key] = {
                'doc': doc,
                'score': max(score, doc_scores.get(doc_key, {}).get('score', 0.0))
            }
            
        # Process sparse results
        for doc, score in zip(sparse_docs, sparse_scores):
            doc_key = doc.page_content
            doc_scores[doc_key] = {
                'doc': doc,
                'score': max(score, doc_scores.get(doc_key, {}).get('score', 0.0))
            }
            
        # Sort by score and split back into docs and scores
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        merged_docs = [item['doc'] for item in sorted_results]
        merged_scores = [item['score'] for item in sorted_results]
        
        # Limit to max_k results
        return merged_docs[:self.max_k], merged_scores[:self.max_k] 
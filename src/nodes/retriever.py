from typing import Any, Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
import json

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from ..state.rag_state import RAGState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetadataFilters:
    """Metadata filters extracted from query."""
    decision_number: Optional[str] = None
    year: Optional[str] = None
    chunk_type: Optional[str] = None
    official_bulletin: Optional[str] = None

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
        self.min_score = config.get("min_score", 0.5)
        
        # Initialize query rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت خبير في إعادة صياغة الاستعلامات لتحقيق الاسترجاع الأمثل في أنظمة RAG المتخصصة في المجال التعليمي والأكاديمي.

            وصف المهمة:
            مهمتك هي تحليل الاستعلام وإعادة صياغته لتحقيق أفضل نتائج استرجاع ممكنة من قاعدة البيانات التعليمية. يجب عليك:
            1. تحديد وإزالة أفعال التوجيه والطلب مثل:
               - أفعال البحث: "استخرج"، "ابحث"، "جد"، "اعرض"، "أرني"
               - أدوات الاستفهام: "ما"، "ما هو"، "كيف"، "أين"
               - عبارات الطلب: "من فضلك"، "هل يمكن"
            
            2. فهم وتوسيع المصطلحات المؤسسية:
               أ. البرنامج البيداغوجي:
                  - المعنى الأساسي: "جداول وملاحق البرنامج الدراسي"
                  - المصطلحات المرتبطة: "المقررات الدراسية"، "المواد التعليمية"، "الخطة الدراسية"
                  - الوثائق المرتبطة: "ملحق"، "جدول"، "مصفوفة المقررات"
               
               ب. التخصصات والشهادات:
                  - صيغ التخصص: "تخصص"، "فرع"، "مسار"، "شعبة"
                  - أنواع الشهادات: "شهادة"، "دبلوم"، "مؤهل"، "إجازة"
                  - المستويات: "ليسانس"، "ماستر"، "دكتوراه"
            
            3. معالجة المصطلحات التخصصية:
               - الحفاظ على المصطلحات التقنية كما هي (مثل "تسيير الغابات")
               - إضافة المصطلحات المرادفة الشائعة في نفس المجال
               - ربط المصطلحات بسياقها المؤسسي والأكاديمي
            
            4. قواعد إعادة الصياغة:
               أ. الأولوية للمصطلحات الرئيسية:
                  - تقديم اسم التخصص والبرنامج
                  - الحفاظ على المصطلحات التقنية
                  - إزالة الكلمات غير الضرورية
               
               ب. هيكل الاستعلام المعاد صياغته:
                  - البدء بالمصطلح الأكثر أهمية
                  - ترتيب المصطلحات حسب الأهمية
                  - استخدام الصيغ المباشرة والموجزة
               
               ج. تجنب:
                  - الجمل الطويلة والمركبة
                  - الكلمات العامة وغير المحددة
                  - التكرار غير الضروري
            
            5. أمثلة على إعادة الصياغة:
               - الأصل: "استخرج البرنامج البيداغوجي لنيل شهادة في تخصص تسيير الغابات"
                 المعاد: "البرنامج البيداغوجي تخصص تسيير الغابات جداول ملاحق المقررات"
               
               - الأصل: "أريد معرفة المواد التي يدرسها طلاب تخصص علوم البيئة"
                 المعاد: "المقررات الدراسية تخصص علوم البيئة البرنامج البيداغوجي"
            
            تذكر:
            1. أعد الاستعلام كنص عادي فقط
            2. ركز على المصطلحات التي تظهر في الوثائق المستهدفة
            3. احتفظ بالمعنى الأساسي مع تحسين قابلية البحث
            4. لا تضف أي تنسيق أو علامات خاصة للنص المعاد صياغته
            5. أعد فقط النص المعاد صياغته بدون أي تعليقات أو شروحات إضافية
            6. تأكد من أن النص المعاد صياغته يتوافق مع نوع الاستعلام والغرض منه
            
            نوع الاستعلام: {query_type}
            نية الاستعلام: {query_intent}
            الاستعلام الأصلي: {query}
            """),
            ("user", "{query}")
        ])
        
    def _extract_metadata_filters(self, query: str) -> MetadataFilters:
        """Extract metadata filters from query."""
        filters = MetadataFilters()
        
        # Extract decision number (e.g., "رقم 24-15")
        decision_match = re.search(r'رقم\s+(\d+[-\s]?\d+)', query)
        if decision_match:
            filters.decision_number = decision_match.group(1).replace(' ', '-')
            
        # Extract year (4 digits)
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters.year = year_match.group(1)
            
        # Extract document type
        type_patterns = {
            "قرار وزاري مشترك": r'قرار\s+وزاري\s+مشترك',
            "قرار": r'\bقرار\b',
            "مقــرر": r'\bمقرر\b',
            "منشور": r'\bمنشور\b',
            "النصوص الصادرة في الجريدة الرسمية": r'النصوص\s+الصادرة\s+في\s+الجريدة\s+الرسمية',
            "الاتــفاقـــــــيات": r'\bالاتفاقيات\b'
        }
        
        for doc_type, pattern in type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters.chunk_type = doc_type
                break
                
        # Extract official bulletin reference
        bulletin_match = re.search(r'النشرة الرسمية (?:الثلاثي )?(\d+[-\s]?\d{4})', query)
        if bulletin_match:
            filters.official_bulletin = f"النشرة الرسمية الثلاثي {bulletin_match.group(1)}"
            
        return filters
        
    async def retrieve(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using adaptive strategies based on query analysis.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved documents
        """
        logger.info("\n" + "="*80)
        logger.info(f"[RETRIEVER] Starting retrieval for query: '{state['query']}'")
        logger.info("-"*80)
        
        # Get query and analysis from state
        query = state["query"]
        query_type = state["query_type"]
        
        # Extract metadata filters from query
        metadata_filters = self._extract_metadata_filters(query)
        logger.info(f"[RETRIEVER] Extracted metadata filters: {metadata_filters}")
        
        # Determine retrieval strategy based on query type
        strategy = self._select_retrieval_strategy(query_type)
        state["retrieval_strategy"] = strategy
        logger.info(f"[RETRIEVER] Selected retrieval strategy: {strategy}")
        
        # Rewrite query if needed
        if strategy in ["hybrid", "sparse"]:
            if self.rate_limiter:
                await self.rate_limiter.wait()
            logger.info(f"[RETRIEVER] Rewriting query with LLM for {strategy} retrieval...")
            rewritten_query = await self._rewrite_query(state)
            state["rewritten_query"] = rewritten_query
            logger.info(f"[RETRIEVER] Original query: '{query}'")
            logger.info(f"[RETRIEVER] Rewritten query: '{rewritten_query}'")
        else:
            rewritten_query = query
            logger.info(f"[RETRIEVER] Using original query without rewriting for {strategy} retrieval")
            
        # Perform retrieval based on strategy
        logger.info(f"[RETRIEVER] Performing {strategy} retrieval...")
        if strategy == "dense":
            docs, scores = await self._dense_retrieval(rewritten_query, metadata_filters)
        elif strategy == "sparse":
            docs, scores = await self._sparse_retrieval(rewritten_query, metadata_filters)
        else:  # hybrid
            logger.info("[RETRIEVER] Starting dense retrieval for hybrid approach...")
            dense_docs, dense_scores = await self._dense_retrieval(rewritten_query, metadata_filters)
            logger.info(f"[RETRIEVER] Found {len(dense_docs)} documents with dense retrieval")
            
            logger.info("[RETRIEVER] Starting sparse retrieval for hybrid approach...")
            sparse_docs, sparse_scores = await self._sparse_retrieval(rewritten_query, metadata_filters)
            logger.info(f"[RETRIEVER] Found {len(sparse_docs)} documents with sparse retrieval")
            
            logger.info("[RETRIEVER] Merging dense and sparse retrieval results...")
            docs, scores = self._merge_results(dense_docs, dense_scores, sparse_docs, sparse_scores)
            logger.info(f"[RETRIEVER] Merged results: {len(docs)} documents")
            
        # Log retrieval results
        self._log_retrieval_results(docs, scores)
            
        # Update state with results
        state.update({
            "documents": docs,
            "retrieval_scores": scores,
            "current_node": "retriever"
        })
        
        logger.info(f"[RETRIEVER] Completed retrieval with {len(docs)} documents")
        logger.info("="*80)
        
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
        logger.info(f"[RETRIEVER] [LLM CALL] Rewriting query using prompt template...")
        
        input_data = {
            "query": state["query"],
            "query_type": state["query_type"],
            "query_intent": state["query_intent"]
        }
        logger.info(f"[RETRIEVER] [LLM INPUT] {json.dumps(input_data, ensure_ascii=False)}")
        
        chain = self.rewrite_prompt | self.llm
        result = await chain.ainvoke(input_data)
        
        logger.info(f"[RETRIEVER] [LLM OUTPUT] Query rewriting result: {result.content}")
        return result.content
        
    async def _search_articles_and_appendices(self, query: str) -> Tuple[List[Document], List[float]]:
        """
        Search through articles and appendices content when metadata filters yield no results.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (documents, scores)
        """
        logger.info("[RETRIEVER] Performing deep search in articles and appendices content")
        
        # First get documents that have articles or appendices
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.max_k * 2  # Get more results since we'll filter
        )
        
        enhanced_results = []
        best_subsection_score = 0
        
        for idx, (doc, score) in enumerate(results):
            try:
                # Parse articles and appendices from metadata
                articles = json.loads(doc.metadata.get('articles', '[]'))
                appendices = json.loads(doc.metadata.get('appendices', '[]'))
                
                # Track matched items for inclusion in metadata
                matched_articles = []
                matched_appendices = []
                
                # Calculate similarity scores for each article and appendix
                article_scores = []
                for article in articles:
                    article_text = f"{article.get('article_number', '')} {article.get('text', '')}"
                    article_embedding = self.embeddings.embed_query(article_text)
                    query_embedding = self.embeddings.embed_query(query)
                    similarity = self._cosine_similarity(query_embedding, article_embedding)
                    if similarity >= self.min_score:
                        article_scores.append(similarity)
                        matched_articles.append({
                            **article,  # Include all original article fields
                            'similarity_score': similarity
                        })
                
                appendix_scores = []
                for appendix in appendices:
                    appendix_text = f"{appendix.get('title', '')} {appendix.get('text', '')}"
                    appendix_embedding = self.embeddings.embed_query(appendix_text)
                    query_embedding = self.embeddings.embed_query(query)
                    similarity = self._cosine_similarity(query_embedding, appendix_embedding)
                    if similarity >= self.min_score:
                        appendix_scores.append(similarity)
                        matched_appendices.append({
                            **appendix,  # Include all original appendix fields
                            'similarity_score': similarity
                        })
                
                # If we found good matches in articles or appendices
                if matched_articles or matched_appendices:
                    best_subscore = max(article_scores + appendix_scores + [0])
                    enhanced_score = 0.7 * score + 0.3 * best_subscore  # Weight original and subsection scores
                    
                    # Keep track of the document with the best subsection match
                    if best_subscore > best_subsection_score:
                        best_subsection_score = best_subscore
                        best_doc_index = idx
                    
                    # Create a new document with enhanced metadata
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,  # Keep original metadata
                            'matched_articles': matched_articles,
                            'matched_appendices': matched_appendices,
                            'original_score': score,
                            'subsection_score': best_subscore
                        }
                    )
                    enhanced_results.append((enhanced_doc, enhanced_score))
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error processing document subsections: {str(e)}")
                enhanced_results.append((doc, score))  # Keep original score if error
                
        # Sort by enhanced scores
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum score
        filtered_results = [
            (doc, score) for doc, score in enhanced_results
            if score >= self.min_score
        ][:self.default_k]
        
        if not filtered_results:
            return [], []
        
        # Log the matched articles and appendices for debugging
        logger.info("\n=== Matched Articles and Appendices ===")
        for doc_idx, (doc, score) in enumerate(filtered_results):
            logger.info(f"\nDocument Score: {score:.3f}")
            
            # Note if this document has the best subsection match
            is_best_match = doc.metadata.get('subsection_score', 0) == best_subsection_score
            if is_best_match:
                logger.info("★ This document contains the best matching article/appendix ★")
            
            if 'matched_articles' in doc.metadata:
                logger.info("\nMatched Articles:")
                for article in doc.metadata['matched_articles']:
                    logger.info(f"- Article {article.get('article_number', 'N/A')}")
                    logger.info(f"  Similarity: {article.get('similarity_score', 0):.3f}")
                    logger.info(f"  Text Preview: {article.get('text', '')[:100]}...")
                    
            if 'matched_appendices' in doc.metadata:
                logger.info("\nMatched Appendices:")
                for appendix in doc.metadata['matched_appendices']:
                    logger.info(f"- Appendix")  # Removed direct access to appendix_number
                    logger.info(f"  Similarity: {appendix.get('similarity_score', 0):.3f}")
                    logger.info(f"  Text Preview: {appendix.get('text', '')[:100]}...")
                    
        return zip(*filtered_results)  # Unzip into separate doc and score lists
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
        
    async def _dense_retrieval(self, query: str, filters: MetadataFilters) -> Tuple[List[Document], List[float]]:
        """
        Perform dense retrieval using embeddings and metadata filters.
        
        Args:
            query: Query string
            filters: Metadata filters extracted from query
            
        Returns:
            Tuple of (documents, scores)
        """
        logger.info(f"[RETRIEVER] Performing dense retrieval for query: '{query}'")
        
        # Check if all metadata filters are None
        all_filters_none = all(
            value is None for value in [
                filters.decision_number,
                filters.year,
                filters.chunk_type,
                filters.official_bulletin
            ]
        )
        
        # If all filters are None, try searching through articles and appendices
        if all_filters_none:
            logger.info("[RETRIEVER] No metadata filters found, searching through articles and appendices")
            docs, scores = await self._search_articles_and_appendices(query)
            if docs:  # If we found results in articles/appendices, return them
                
                # Find document with highest subsection score
                highest_subsection_score = 0
                highest_doc_index = 0
                
                for i, doc in enumerate(docs):
                    subsection_score = doc.metadata.get('subsection_score', 0)
                    if subsection_score > highest_subsection_score:
                        highest_subsection_score = subsection_score
                        highest_doc_index = i
                
                # For all documents except the one with highest subsection score,
                # remove matched_articles and matched_appendices to keep only
                # the most relevant matches
                for i, doc in enumerate(docs):
                    if i != highest_doc_index:
                        if 'matched_articles' in doc.metadata:
                            logger.info(f"[RETRIEVER] Removing {len(doc.metadata['matched_articles'])} matched articles from document {i+1} to focus on most relevant matches")
                            del doc.metadata['matched_articles']
                        if 'matched_appendices' in doc.metadata:
                            logger.info(f"[RETRIEVER] Removing {len(doc.metadata['matched_appendices'])} matched appendices from document {i+1} to focus on most relevant matches")
                            del doc.metadata['matched_appendices']
                
                logger.info(f"[RETRIEVER] Keeping only matched articles and appendices from document {highest_doc_index+1} which has highest subsection score of {highest_subsection_score:.3f}")
                return list(docs), list(scores)
                
        # Continue with normal metadata filtering if articles/appendices search failed or filters exist
        # Build metadata filter dict
        filter_conditions = []
        if filters.decision_number:
            filter_conditions.append({"decision_number": filters.decision_number})
        if filters.year:
            filter_conditions.append({"year": filters.year})
        if filters.chunk_type:
            filter_conditions.append({"chunk_type": filters.chunk_type})
        if filters.official_bulletin:
            filter_conditions.append({"official_bulletin": filters.official_bulletin})
            
        # Combine filters with $and operator if there are multiple conditions
        filter_dict = (
            {"$and": filter_conditions} if len(filter_conditions) > 1
            else filter_conditions[0] if filter_conditions
            else None
        )
        
        logger.info(f"[RETRIEVER] Using metadata filter: {filter_dict}")
            
        # Perform similarity search with metadata filters and score threshold
        logger.info(f"[RETRIEVER] Performing vector similarity search with k={self.max_k}...")
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.max_k,  # Get more results initially for filtering
            filter=filter_dict
        )
        
        logger.info(f"[RETRIEVER] Vector search returned {len(results)} initial results")
        
        # Filter by minimum score and sort by relevance
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= self.min_score
        ]
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"[RETRIEVER] Filtered to {len(filtered_results)} results with minimum score {self.min_score}")
        
        # Limit to default_k results
        filtered_results = filtered_results[:self.default_k]
        logger.info(f"[RETRIEVER] Limited to top {min(self.default_k, len(filtered_results))} results")
        
        if not filtered_results:
            logger.info("[RETRIEVER] No results found after filtering")
            return [], []
            
        docs, scores = zip(*filtered_results)
        return list(docs), list(scores)
        
    async def _sparse_retrieval(self, query: str, filters: MetadataFilters) -> Tuple[List[Document], List[float]]:
        """
        Perform sparse retrieval using keyword matching and metadata filters.
        
        Args:
            query: Query string
            filters: Metadata filters extracted from query
            
        Returns:
            Tuple of (documents, scores)
        """
        # For now, fallback to dense retrieval until proper sparse retrieval is implemented
        logger.info("[RETRIEVER] Using dense retrieval as fallback for sparse retrieval")
        return await self._dense_retrieval(query, filters)
        
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
        
        # Process dense results with higher weight (0.7)
        for doc, score in zip(dense_docs, dense_scores):
            doc_key = doc.page_content
            doc_scores[doc_key] = {
                'doc': doc,
                'score': 0.7 * score
            }
            
        # Process sparse results with lower weight (0.3)
        for doc, score in zip(sparse_docs, sparse_scores):
            doc_key = doc.page_content
            if doc_key in doc_scores:
                doc_scores[doc_key]['score'] += 0.3 * score
            else:
                doc_scores[doc_key] = {
                    'doc': doc,
                    'score': 0.3 * score
                }
            
        # Sort by combined score and split back into docs and scores
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Filter by minimum score and limit to max_k results
        filtered_results = [
            item for item in sorted_results 
            if item['score'] >= self.min_score
        ][:self.max_k]
        
        if not filtered_results:
            return [], []
            
        merged_docs = [item['doc'] for item in filtered_results]
        merged_scores = [item['score'] for item in filtered_results]
        
        # Find document with highest subsection score to keep only its matches
        highest_subsection_score = 0
        highest_doc_index = 0
        
        for i, doc in enumerate(merged_docs):
            subsection_score = doc.metadata.get('subsection_score', 0)
            if subsection_score > highest_subsection_score:
                highest_subsection_score = subsection_score
                highest_doc_index = i
        
        # For all documents except the one with highest subsection score,
        # remove matched_articles and matched_appendices to keep only
        # the most relevant matches
        for i, doc in enumerate(merged_docs):
            if i != highest_doc_index:
                if 'matched_articles' in doc.metadata:
                    logger.info(f"[RETRIEVER] Removing {len(doc.metadata['matched_articles'])} matched articles from merged document {i+1} to focus on most relevant matches")
                    del doc.metadata['matched_articles']
                if 'matched_appendices' in doc.metadata:
                    logger.info(f"[RETRIEVER] Removing {len(doc.metadata['matched_appendices'])} matched appendices from merged document {i+1} to focus on most relevant matches")
                    del doc.metadata['matched_appendices']
        
        if highest_subsection_score > 0:
            logger.info(f"[RETRIEVER] Keeping only matched articles and appendices from merged document {highest_doc_index+1} which has highest subsection score of {highest_subsection_score:.3f}")
        
        return merged_docs, merged_scores
        
    def _log_retrieval_results(self, docs: List[Document], scores: List[float]) -> None:
        """Log retrieval results with document previews and metadata."""
        logger.info("\n=== Retrieval Results ===")
        
        # Find the document with highest subsection score for reference
        highest_subsection_score = 0
        highest_doc_index = -1
        for i, doc in enumerate(docs):
            subsection_score = doc.metadata.get('subsection_score', 0)
            if subsection_score > highest_subsection_score:
                highest_subsection_score = subsection_score
                highest_doc_index = i
        
        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Score: {score:.3f}")
            logger.info(f"Content Preview: {doc.page_content[:100]}...")
            logger.info("Metadata:")
            
            # Track if we found articles or appendices
            has_matches = False
            
            # Note whether this is the document with the best matches
            is_best_match = (i-1 == highest_doc_index and highest_doc_index >= 0)
            if is_best_match:
                logger.info("\n  ★ This document contains the most relevant matches ★")
            
            # First log matched articles if any
            if 'matched_articles' in doc.metadata and doc.metadata['matched_articles']:
                has_matches = True
                logger.info("\n  Matched Articles:")
                for article in doc.metadata['matched_articles']:
                    logger.info(f"\n    Article {article.get('article_number', 'N/A')}:")
                    logger.info(f"    Similarity Score: {article.get('similarity_score', 0):.3f}")
                    logger.info(f"    Chunk Type: {article.get('chunk_type', 'N/A')}")
                    logger.info(f"    Text: {article.get('text', '')}")
                    if article.get('table_data'):
                        logger.info("    Table Data:")
                        logger.info(f"      {json.dumps(article.get('table_data'), ensure_ascii=False, indent=2)}")
                    # Log any additional fields
                    for key, value in article.items():
                        if key not in ['similarity_score', 'chunk_type', 'article_number', 'text', 'table_data']:
                            logger.info(f"    {key}: {value}")
            
            # Then log matched appendices if any
            if 'matched_appendices' in doc.metadata and doc.metadata['matched_appendices']:
                has_matches = True
                logger.info("\n  Matched Appendices:")
                for appendix in doc.metadata['matched_appendices']:
                    logger.info(f"\n    Appendix:")
                    logger.info(f"    Similarity Score: {appendix.get('similarity_score', 0):.3f}")
                    logger.info(f"    Chunk Type: {appendix.get('chunk_type', 'N/A')}")
                    logger.info(f"    Text: {appendix.get('text', '')}")
                    if appendix.get('table_data'):
                        logger.info("    Table Data:")
                        logger.info(f"      {json.dumps(appendix.get('table_data'), ensure_ascii=False, indent=2)}")
                    # Log any additional fields
                    for key, value in appendix.items():
                        if key not in ['similarity_score', 'chunk_type', 'text', 'table_data']:
                            logger.info(f"    {key}: {value}")
            
            # Log other metadata
            logger.info("\n  Other Metadata:")
            for key, value in doc.metadata.items():
                if key not in ['matched_articles', 'matched_appendices', 'articles', 'appendices']:
                    if isinstance(value, str) and len(value) > 100:
                        logger.info(f"    {key}: {value[:100]}...")
                    else:
                        logger.info(f"    {key}: {value}")
            
            # Special log message when document has no matches
            if not has_matches:
                if i-1 == highest_doc_index:
                    logger.info("  No matching articles or appendices found in this document (but it's still the highest-scoring document)")
                    # If this is the highest-scoring document and it has articles or appendices but they didn't match, 
                    # still pass them along with zero similarity scores to ensure the generator has access to them
                    if 'articles' in doc.metadata and doc.metadata['articles']:
                        try:
                            articles_data = json.loads(doc.metadata['articles']) if isinstance(doc.metadata['articles'], str) else doc.metadata['articles']
                            if articles_data and len(articles_data) > 0:
                                # Create matched_articles with zero similarity scores
                                doc.metadata['matched_articles'] = [
                                    {**article, 'similarity_score': 0.0}
                                    for article in articles_data
                                ]
                                logger.info(f"  Added {len(doc.metadata['matched_articles'])} articles with zero similarity scores from highest-scoring document")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"  Error processing articles for document without matches: {str(e)}")
                    
                    if 'appendices' in doc.metadata and doc.metadata['appendices']:
                        try:
                            appendices_data = json.loads(doc.metadata['appendices']) if isinstance(doc.metadata['appendices'], str) else doc.metadata['appendices']
                            if appendices_data and len(appendices_data) > 0:
                                # Create matched_appendices with zero similarity scores
                                doc.metadata['matched_appendices'] = [
                                    {**appendix, 'similarity_score': 0.0}
                                    for appendix in appendices_data
                                ]
                                logger.info(f"  Added {len(doc.metadata['matched_appendices'])} appendices with zero similarity scores from highest-scoring document")
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"  Error processing appendices for document without matches: {str(e)}")
                else:
                    logger.info("  No matching articles or appendices found in this document - focusing on highest-scoring document instead")
                
        logger.info("=" * 50) 
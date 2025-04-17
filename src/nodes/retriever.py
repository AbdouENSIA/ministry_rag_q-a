from typing import Any, Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
import json
import math

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
            ("system", """أنت خبير متخصص في استرجاع المعلومات (IR) وتحسين استعلامات البحث للأنظمة العربية المتقدمة في المجال التعليمي والأكاديمي.

            وصف المهمة:
            مهمتك الأساسية هي إعادة صياغة استعلامات المستخدمين لتحقيق أقصى درجات التشابه الدلالي (Semantic Similarity) مع الوثائق المطلوبة. يجب عليك:
            
            1. تحليل عميق للاستعلام:
               - تحديد الغرض الرئيسي للاستعلام (وثيقة محددة، معلومة عامة، إجراء معين، إلخ)
               - تحديد المفاهيم الأساسية والكلمات الرئيسية
               - فهم السياق المؤسسي والأكاديمي للاستعلام
            
            2. إزالة العناصر غير المفيدة للاسترجاع:
               - أفعال البحث والتوجيه: "استخرج"، "ابحث"، "جد"، "أرني"
               - أدوات الاستفهام: "ما"، "كيف"، "أين"، "هل"
               - عبارات الطلب: "من فضلك"، "لو سمحت"، "هل يمكن"
               - كلمات الوصل غير الضرورية: "و"، "أو"، "ثم"، "بعد ذلك"
            
            3. توسيع المصطلحات الأكاديمية والإدارية:
               أ. المصطلحات الأكاديمية:
                  - "البرنامج البيداغوجي" ← ["جداول الدراسة"، "ملاحق البرنامج"، "المقررات"، "المواد التعليمية"]
                  - "التخصص" ← ["فرع"، "مسار"، "شعبة"، "مجال الدراسة"]
                  - "الشهادة" ← ["الإجازة"، "الدبلوم"، "الليسانس"، "الماستر"، "الدكتوراه"]
               
               ب. الوثائق الإدارية:
                  - "قرار" ← ["قرار وزاري"، "مرسوم"، "قرار تنفيذي"]
                  - "منشور" ← ["تعميم"، "بلاغ"، "مذكرة توضيحية"]
                  - "ملحق" ← ["مرفق"، "جدول"، "مصفوفة"]
            
            4. تقنيات التحسين المتقدمة:
               أ. مطابقة المصطلحات مع محتوى قاعدة البيانات:
                  - استخدام المصطلحات الرسمية بالصيغة المستخدمة في الوثائق
                  - الحفاظ على الأرقام والمعرفات كما هي (مثل أرقام القرارات)
                  - ترتيب المصطلحات حسب أهميتها في النظام
               
               ب. تقنيات تحسين التشابه الدلالي:
                  - إضافة مرادفات وكلمات ذات صلة للمفاهيم الرئيسية
                  - استخدام الاختصارات والتعابير الرسمية المعتمدة
                  - إضافة مؤشرات السياق (مثل السنة الدراسية أو نوع المؤسسة)
               
               ج. تقنيات تكامل الاستعلام:
                  - دمج المفاهيم المترابطة في عبارات متماسكة
                  - تكرار المصطلحات المهمة في سياقات مختلفة
                  - إضافة علاقات مفاهيمية بين المصطلحات
            
            5. معايير قياس جودة إعادة الصياغة:
               - مدى زيادة التشابه الدلالي مع الوثائق المستهدفة
               - درجة الشمولية وتغطية كافة المفاهيم المهمة
               - فعالية في إزالة الضوضاء والعناصر غير المفيدة
               - تحسين القدرة على التمييز بين الوثائق ذات الصلة وغير ذات الصلة
            
            6. استراتيجيات متقدمة لأنواع مختلفة من الاستعلامات:
               - للاستعلامات الوثائقية: التركيز على المعرفات ونوع الوثيقة
               - للاستعلامات المفاهيمية: التوسع في المصطلحات وإطارها المعرفي
               - للاستعلامات الإجرائية: التركيز على الخطوات والعمليات ونتائجها المتوقعة
            
            7. أمثلة عالية الجودة:
               - الأصل: "أريد استخراج البرنامج البيداغوجي الخاص بتخصص علوم البيئة لسنة 2022"
                 المعاد: "البرنامج البيداغوجي علوم البيئة 2022 مقررات جداول ملاحق المواد التخصص البيئي خطة الدراسة"
               
               - الأصل: "هل يمكن البحث عن القرار الوزاري رقم 24-15 المتعلق بالتسجيل في الدكتوراه؟"
                 المعاد: "القرار الوزاري 24-15 التسجيل الدكتوراه قبول الطلبة الدراسات العليا شروط تنظيم مسابقة الالتحاق"
               
               - الأصل: "كيف أعرف المواد المقررة في تخصص الهندسة المدنية للسنة الثانية؟"
                 المعاد: "المقررات الدراسية الهندسة المدنية السنة الثانية البرنامج البيداغوجي المواد وحدات التعليم الأساسية"
            
            تعليمات نهائية:
            1. أنتج فقط النص المعاد صياغته بدون أي شروحات
            2. ركز على تعظيم التشابه الدلالي مع محتوى قاعدة البيانات
            3. اجعل الاستعلام المعاد أكثر شمولاً وغنى بالمفاهيم والمصطلحات ذات الصلة
            4. تجنب الجمل والتراكيب اللغوية المعقدة
            5. ضع المصطلحات الأكثر أهمية في بداية الاستعلام
            6. احرص على أن يكون الاستعلام المعاد متوافقاً مع طبيعته والغرض منه
            
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
            "قرار": r'\b(?:ال)?قرار\b',
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
        # Enhanced to better handle different query types with more nuanced approach
        if query_type == "factual":
            return "hybrid"  # Changed from dense to hybrid for better article/appendix coverage
        elif query_type == "analytical":
            return "hybrid"  # Continue using hybrid for comprehensive results
        else:  # procedural or others
            return "hybrid"  # Changed from sparse to hybrid for more comprehensive results
            
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
        Advanced deep search through articles and appendices content with intelligent scoring.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (documents, scores)
        """
        logger.info("[RETRIEVER] Performing advanced deep search in articles and appendices content")
        
        # Enhanced: Use larger initial document pool for comprehensive analysis
        results = self.vector_store.similarity_search_with_score(
            query,
            k=self.max_k * 5  # Increased from 3x to 5x to handle larger document collections
        )
        
        if not results:
            logger.info("[RETRIEVER] No initial documents found for articles/appendices search")
            return [], []
            
        logger.info(f"[RETRIEVER] Found {len(results)} initial documents for analyzing articles and appendices")
        
        # Initialize tracking variables
        enhanced_results = []
        best_subsection_score = 0
        best_doc_index = -1
        best_doc_id = None
        query_embedding = self.embeddings.embed_query(query)
        
        # Create a hashmap to track documents by ID to avoid duplicates
        doc_map = {}
        
        # Track the distribution of scores to normalize effectively
        all_subsection_scores = []
        
        # Enhanced: Extract query keywords for better term matching
        query_keywords = self._extract_query_keywords(query)
        logger.info(f"[RETRIEVER] Extracted query keywords: {query_keywords}")
        
        # First pass: Analyze all articles and appendices to gather statistics
        logger.info("[RETRIEVER] First pass: Analyzing all articles and appendices for scoring normalization")
        for idx, (doc, score) in enumerate(results):
            doc_id = doc.metadata.get('id', f"doc_{idx}")
            
            # Enhanced: Early filtering - skip documents with very low scores
            if score < self.min_score * 0.8:  # Allow some threshold below min_score for articles that might match better
                continue
                
            try:
                # Parse articles and appendices from metadata with better error handling
                articles = []
                appendices = []
                
                try:
                    if isinstance(doc.metadata.get('articles'), str):
                        articles = json.loads(doc.metadata.get('articles', '[]'))
                    elif isinstance(doc.metadata.get('articles'), list):
                        articles = doc.metadata.get('articles', [])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing articles for document {doc_id}: {str(e)}")
                
                try:
                    if isinstance(doc.metadata.get('appendices'), str):
                        appendices = json.loads(doc.metadata.get('appendices', '[]'))
                    elif isinstance(doc.metadata.get('appendices'), list):
                        appendices = doc.metadata.get('appendices', [])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing appendices for document {doc_id}: {str(e)}")
                
                # Calculate subsection scores and collect them for normalization
                for article in articles:
                    # Enhance article text by combining multiple fields
                    article_text = f"{article.get('article_number', '')} {article.get('title', '')} {article.get('text', '')}"
                    
                    # Enhanced: Use keyword matching first to filter non-promising articles before embedding
                    if not self._contains_keywords(article_text, query_keywords):
                        continue  # Skip articles without any keyword matches to improve efficiency
                        
                    article_embedding = self.embeddings.embed_query(article_text)
                    similarity = self._cosine_similarity(query_embedding, article_embedding)
                    all_subsection_scores.append(similarity)
                    
                for appendix in appendices:
                    # Enhance appendix text by combining multiple fields
                    appendix_text = f"{appendix.get('title', '')} {appendix.get('appendix_number', '')} {appendix.get('text', '')}"
                    
                    # Enhanced: Use keyword matching first to filter non-promising appendices before embedding
                    if not self._contains_keywords(appendix_text, query_keywords):
                        continue  # Skip appendices without any keyword matches to improve efficiency
                        
                    appendix_embedding = self.embeddings.embed_query(appendix_text)
                    similarity = self._cosine_similarity(query_embedding, appendix_embedding)
                    all_subsection_scores.append(similarity)
                    
            except Exception as e:
                logger.warning(f"Unexpected error during first pass analysis for document {doc_id}: {str(e)}")
        
        # Calculate score thresholds for better filtering
        if all_subsection_scores:
            avg_score = sum(all_subsection_scores) / len(all_subsection_scores)
            # Enhanced: More sophisticated adaptive threshold with percentile-based approach
            all_subsection_scores.sort(reverse=True)
            top_20_percent_threshold = all_subsection_scores[max(0, min(len(all_subsection_scores) // 5, len(all_subsection_scores) - 1))]
            score_variance = sum((s - avg_score) ** 2 for s in all_subsection_scores) / len(all_subsection_scores)
            score_std = score_variance ** 0.5
            
            # Use a dynamic threshold based on data distribution
            adaptive_threshold = max(
                self.min_score, 
                min(avg_score + 0.5 * score_std, 
                    (avg_score + top_20_percent_threshold) / 2, 
                    0.75)
            )
            logger.info(f"[RETRIEVER] Calculated adaptive threshold for subsections: {adaptive_threshold:.3f} (avg: {avg_score:.3f}, std: {score_std:.3f}, top20%: {top_20_percent_threshold:.3f})")
        else:
            adaptive_threshold = self.min_score
            logger.info(f"[RETRIEVER] Using default threshold for subsections: {adaptive_threshold:.3f}")
        
        # Enhanced: Different thresholds for different content types
        article_threshold = adaptive_threshold * 0.95  # Slightly lower threshold for articles
        appendix_threshold = adaptive_threshold * 0.9  # Even lower threshold for appendices which may be more diverse
        
        # Second pass: Detailed analysis with the adaptive threshold
        logger.info("[RETRIEVER] Second pass: Detailed analysis with adaptive thresholds (articles: {:.3f}, appendices: {:.3f})".format(
            article_threshold, appendix_threshold
        ))
        
        # Enhanced: Track keyword term frequency for weighting
        term_frequency = self._calculate_term_frequency(query_keywords, results)
        
        # Track document precision (ratio of matched subsections to total subsections)
        doc_precision_scores = {}
        
        for idx, (doc, score) in enumerate(results):
            doc_id = doc.metadata.get('id', f"doc_{idx}")
            
            # Enhanced: Early filtering - skip documents with very low scores
            if score < self.min_score * 0.8:
                continue
                
            try:
                # Parse articles and appendices from metadata
                articles = []
                appendices = []
                
                try:
                    if isinstance(doc.metadata.get('articles'), str):
                        articles = json.loads(doc.metadata.get('articles', '[]'))
                    elif isinstance(doc.metadata.get('articles'), list):
                        articles = doc.metadata.get('articles', [])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing articles for document {doc_id}: {str(e)}")
                
                try:
                    if isinstance(doc.metadata.get('appendices'), str):
                        appendices = json.loads(doc.metadata.get('appendices', '[]'))
                    elif isinstance(doc.metadata.get('appendices'), list):
                        appendices = doc.metadata.get('appendices', [])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing appendices for document {doc_id}: {str(e)}")
                
                # Track matched items with improved matching
                matched_articles = []
                matched_appendices = []
                
                # Enhanced: Track precision metrics
                total_articles = len(articles)
                total_appendices = len(appendices)
                matched_article_count = 0
                matched_appendix_count = 0
                
                # Calculate similarity scores with context-aware boosting
                article_scores = []
                for article in articles:
                    # Build a more complete article text
                    article_text = f"{article.get('article_number', '')} {article.get('title', '')} {article.get('text', '')}"
                    
                    # Enhanced: Skip articles without any keyword matches for efficiency
                    if not self._contains_keywords(article_text, query_keywords):
                        continue
                    
                    # Enhanced: More sophisticated boosting based on multiple factors
                    boost_factor = 1.0
                    
                    # Boost articles with specific attributes
                    if 'is_key_article' in article and article['is_key_article']:
                        boost_factor *= 1.25  # Increased from 1.2
                    if 'importance' in article:
                        boost_factor *= (1.0 + float(article.get('importance', 0)) / 8)  # Adjusted from /10
                        
                    # Enhanced: Boost by article number if query contains numbers
                    if article.get('article_number') and any(kw.isdigit() for kw in query_keywords):
                        for kw in query_keywords:
                            if kw.isdigit() and kw in str(article.get('article_number')):
                                boost_factor *= 1.5
                                break
                    
                    # Enhanced: Term frequency boost
                    term_boost = self._calculate_term_match_boost(article_text, term_frequency)
                    boost_factor *= (1.0 + term_boost)
                    
                    article_embedding = self.embeddings.embed_query(article_text)
                    similarity = self._cosine_similarity(query_embedding, article_embedding) * boost_factor
                    
                    if similarity >= article_threshold:
                        article_scores.append(similarity)
                        matched_article_count += 1
                        
                        # Add highlight feature to show relevant text
                        article_copy = article.copy()
                        article_copy['similarity_score'] = similarity
                        article_copy['is_high_match'] = similarity >= (adaptive_threshold + 0.1)
                        article_copy['boost_factor'] = boost_factor
                        
                        # Extract snippets for highlighting with enhanced keyword context
                        if 'text' in article:
                            snippets = self._extract_context_snippets(article['text'], query_keywords)
                            if snippets:
                                article_copy['highlighted_snippets'] = snippets
                        
                        matched_articles.append(article_copy)
                
                appendix_scores = []
                for appendix in appendices:
                    # Build a more complete appendix text
                    appendix_text = f"{appendix.get('title', '')} {appendix.get('appendix_number', '')} {appendix.get('text', '')}"
                    
                    # Enhanced: Skip appendices without any keyword matches for efficiency
                    if not self._contains_keywords(appendix_text, query_keywords):
                        continue
                    
                    # Enhanced: More sophisticated boosting based on multiple factors
                    boost_factor = 1.0
                    
                    # Boost appendices with tables or important data
                    if 'has_table' in appendix and appendix['has_table']:
                        boost_factor *= 1.25  # Increased from 1.2
                    if 'is_key_appendix' in appendix and appendix['is_key_appendix']:
                        boost_factor *= 1.4  # Increased from 1.3
                        
                    # Enhanced: Boost by title match
                    if appendix.get('title'):
                        title_match_count = sum(1 for kw in query_keywords if kw.lower() in appendix.get('title', '').lower())
                        if title_match_count > 0:
                            boost_factor *= (1.0 + (title_match_count * 0.2))
                    
                    # Enhanced: Term frequency boost
                    term_boost = self._calculate_term_match_boost(appendix_text, term_frequency)
                    boost_factor *= (1.0 + term_boost)
                    
                    appendix_embedding = self.embeddings.embed_query(appendix_text)
                    similarity = self._cosine_similarity(query_embedding, appendix_embedding) * boost_factor
                    
                    if similarity >= appendix_threshold:
                        appendix_scores.append(similarity)
                        matched_appendix_count += 1
                        
                        # Add highlight feature to show relevant text
                        appendix_copy = appendix.copy()
                        appendix_copy['similarity_score'] = similarity
                        appendix_copy['is_high_match'] = similarity >= (adaptive_threshold + 0.1)
                        appendix_copy['boost_factor'] = boost_factor
                        
                        # Extract snippets for highlighting with enhanced keyword context
                        if 'text' in appendix:
                            snippets = self._extract_context_snippets(appendix['text'], query_keywords)
                            if snippets:
                                appendix_copy['highlighted_snippets'] = snippets
                        
                        matched_appendices.append(appendix_copy)
                
                # Calculate precision metrics
                non_zero_totals = (total_articles > 0) or (total_appendices > 0)
                if non_zero_totals:
                    precision = (matched_article_count + matched_appendix_count) / (total_articles + total_appendices) if (total_articles + total_appendices) > 0 else 0
                    doc_precision_scores[doc_id] = precision
                
                # Calculate comprehensive document score using weighted approach
                has_matches = bool(matched_articles or matched_appendices)
                
                if has_matches:
                    # Enhanced: More sophisticated scoring that considers multiple factors
                    max_article_score = max(article_scores) if article_scores else 0
                    max_appendix_score = max(appendix_scores) if appendix_scores else 0
                    
                    # Calculate match diversity (more diverse matches are better)
                    article_coverage = len(matched_articles) / total_articles if total_articles > 0 else 0
                    appendix_coverage = len(matched_appendices) / total_appendices if total_appendices > 0 else 0
                    diversity_score = min(1.0, (article_coverage + appendix_coverage) / 2.0 + 
                                        (0.05 * (len(matched_articles) + len(matched_appendices))))  # Reward more matches
                    
                    # Calculate document score boosted by subsection matches
                    # Enhanced: Weighted averaging of article and appendix scores
                    article_weight = 0.6
                    appendix_weight = 0.4
                    
                    # Adjust weights based on what matched better
                    if max_article_score > max_appendix_score * 1.5:
                        article_weight = 0.8
                        appendix_weight = 0.2
                    elif max_appendix_score > max_article_score * 1.5:
                        article_weight = 0.2
                        appendix_weight = 0.8
                    
                    # Calculate subsection score with balanced weighting
                    if max_article_score > 0 and max_appendix_score > 0:
                        subsection_score = (article_weight * max_article_score) + (appendix_weight * max_appendix_score)
                    else:
                        subsection_score = max(max_article_score, max_appendix_score)
                    
                    # Enhanced: Precision factor (ratio of matched to total subsections)
                    precision_factor = doc_precision_scores.get(doc_id, 0.5)  # Default to 0.5 if not calculated
                    
                    # Weighted score combining document-level and subsection-level matches with precision
                    enhanced_score = (0.2 * score) + (0.6 * subsection_score) + (0.1 * diversity_score) + (0.1 * precision_factor)
                    
                    # Enhanced: Apply metadata bonuses for exact matches
                    # If document has exact metadata matches with query entities, boost its score
                    metadata_bonus = self._calculate_metadata_match_bonus(doc.metadata, query, query_keywords)
                    enhanced_score += metadata_bonus
                    
                    # Track the best matching document with subsections
                    if subsection_score > best_subsection_score:
                        best_subsection_score = subsection_score
                        best_doc_index = idx
                        best_doc_id = doc_id
                    
                    # Flag the document as having best matches
                    has_best_match = (subsection_score == best_subsection_score)
                    
                    # Create enhanced document with improved metadata
                    enhanced_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,  # Keep original metadata
                            'matched_articles': matched_articles,
                            'matched_appendices': matched_appendices,
                            'original_score': score,
                            'subsection_score': subsection_score,
                            'precision_score': precision_factor,
                            'diversity_score': diversity_score,
                            'metadata_bonus': metadata_bonus,
                            'has_best_match': has_best_match,
                            'total_matches': len(matched_articles) + len(matched_appendices),
                            'match_coverage': (article_coverage + appendix_coverage) / 2.0 if non_zero_totals else 0
                        }
                    )
                    
                    # Add to results with document metadata to avoid duplicates
                    doc_map[doc_id] = (enhanced_doc, enhanced_score)
                    enhanced_results.append((enhanced_doc, enhanced_score))
                    
            except Exception as e:
                logger.warning(f"Error processing document {doc_id} subsections: {str(e)}")
        
        # Enhanced: Apply document re-ranking based on best subsection matches
        if best_doc_id and best_doc_id in doc_map:
            # Get the document with the best subsection match
            best_doc, best_score = doc_map[best_doc_id]
            
            # Boost other documents that are similar to the best document
            if best_doc and hasattr(best_doc, 'page_content'):
                best_doc_embedding = self.embeddings.embed_query(best_doc.page_content[:1000])  # Use first 1000 chars
                
                # Re-rank other documents based on similarity to best document
                for doc_id, (doc, score) in doc_map.items():
                    if doc_id != best_doc_id and hasattr(doc, 'page_content'):
                        doc_embedding = self.embeddings.embed_query(doc.page_content[:1000])
                        doc_similarity = self._cosine_similarity(best_doc_embedding, doc_embedding)
                        
                        # If document is similar to best document, boost its score
                        if doc_similarity > 0.7:  # High similarity threshold
                            boosted_score = score * (1 + (doc_similarity - 0.7) * 0.5)  # Boost by up to 15%
                            # Update the score in both doc_map and enhanced_results
                            doc_map[doc_id] = (doc, boosted_score)
                            
                            # Find and update in enhanced_results
                            for i, (d, s) in enumerate(enhanced_results):
                                if d.page_content == doc.page_content:
                                    enhanced_results[i] = (d, boosted_score)
                                    break
                            
                            logger.info(f"[RETRIEVER] Boosted document {doc_id} score from {score:.3f} to {boosted_score:.3f} due to similarity to best document")
        
        # Sort by enhanced scores
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # Enhanced: Remove duplicate content
        # Sometimes vector stores return near-duplicate documents
        unique_results = []
        seen_content = set()
        
        for doc, score in enhanced_results:
            # Create a content signature using truncated content
            content_sig = doc.page_content[:200].strip()
            if content_sig not in seen_content:
                unique_results.append((doc, score))
                seen_content.add(content_sig)
                
        enhanced_results = unique_results
        
        # Filter by minimum score and limit results
        filtered_results = [(doc, score) for doc, score in enhanced_results if score >= self.min_score][:self.default_k]
        
        if not filtered_results:
            logger.info("[RETRIEVER] No matching documents found after filtering articles and appendices")
            return [], []
        
        # Log the matched articles and appendices with improved formatting
        logger.info(f"\n=== Found {len(filtered_results)} matched documents with articles/appendices ===")
        logger.info(f"Best document match: ID {best_doc_id} with subsection score {best_subsection_score:.3f}")
        
        for doc_idx, (doc, score) in enumerate(filtered_results):
            logger.info(f"\nDocument {doc_idx+1}: Score: {score:.3f}")
            
            # Note if this document has the best subsection match
            is_best_match = doc.metadata.get('has_best_match', False)
            if is_best_match:
                logger.info("★★★ Document with BEST MATCHING content ★★★")
            
            # Log matched articles with improved details
            if 'matched_articles' in doc.metadata and doc.metadata['matched_articles']:
                articles = doc.metadata['matched_articles']
                logger.info(f"\n  Matched Articles: {len(articles)}")
                
                # Sort articles by similarity score
                sorted_articles = sorted(articles, key=lambda a: a.get('similarity_score', 0), reverse=True)
                
                # Show only top articles if there are many
                display_articles = sorted_articles[:5] if len(sorted_articles) > 5 else sorted_articles
                
                for article in display_articles:
                    is_high_match = article.get('is_high_match', False)
                    logger.info(f"  - Article {article.get('article_number', 'N/A')}{' ⭐' if is_high_match else ''}")
                    logger.info(f"    Score: {article.get('similarity_score', 0):.3f}")
                    
                    # Show highlighted snippets if available
                    if 'highlighted_snippets' in article:
                        logger.info("    Highlighted snippets:")
                        for snippet in article['highlighted_snippets']:
                            logger.info(f"      • ...{snippet}...")
                    else:
                        logger.info(f"    Text Preview: {article.get('text', '')[:100]}...")
            
            # Log matched appendices with improved details
            if 'matched_appendices' in doc.metadata and doc.metadata['matched_appendices']:
                appendices = doc.metadata['matched_appendices']
                logger.info(f"\n  Matched Appendices: {len(appendices)}")
                
                # Sort appendices by similarity score
                sorted_appendices = sorted(appendices, key=lambda a: a.get('similarity_score', 0), reverse=True)
                
                # Show only top appendices if there are many
                display_appendices = sorted_appendices[:5] if len(sorted_appendices) > 5 else sorted_appendices
                
                for appendix in display_appendices:
                    is_high_match = appendix.get('is_high_match', False)
                    logger.info(f"  - Appendix{' ⭐' if is_high_match else ''}")
                    logger.info(f"    Score: {appendix.get('similarity_score', 0):.3f}")
                    
                    # Show highlighted snippets if available
                    if 'highlighted_snippets' in appendix:
                        logger.info("    Highlighted snippets:")
                        for snippet in appendix['highlighted_snippets']:
                            logger.info(f"      • ...{snippet}...")
                    else:
                        logger.info(f"    Text Preview: {appendix.get('text', '')[:100]}...")
        
        # Add special handling for best document to ensure it's always included
        best_match_included = any(doc.metadata.get('has_best_match', False) for doc, _ in filtered_results)
        
        if not best_match_included and best_doc_id and best_doc_id in doc_map:
            logger.info(f"[RETRIEVER] Ensuring best matching document {best_doc_id} is included in results")
            
            # Add the best document to results if not already included
            best_doc, best_score = doc_map[best_doc_id]
            
            # Replace the lowest scoring document with the best match if needed
            if len(filtered_results) >= self.default_k:
                # Find lowest scoring document
                lowest_idx = min(range(len(filtered_results)), key=lambda i: filtered_results[i][1])
                filtered_results[lowest_idx] = (best_doc, best_score)
                logger.info(f"[RETRIEVER] Replaced lowest scoring document with best match document")
            else:
                filtered_results.append((best_doc, best_score))
                logger.info(f"[RETRIEVER] Added best match document to results")
            
            # Resort after addition
            filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"[RETRIEVER] Returning {len(filtered_results)} documents with matched articles and appendices")
        
        if not filtered_results:
            return [], []
            
        docs, scores = zip(*filtered_results)
        return list(docs), list(scores)
    
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
        
        # Count non-None metadata filters
        non_none_filters_count = sum(
            value is not None for value in [
                filters.decision_number,
                filters.year,
                filters.chunk_type,
                filters.official_bulletin
            ]
        )
        
        # Check if we have at least 2 non-None metadata filters
        # If less than 2, use the article/appendices search method instead
        if non_none_filters_count < 2:
            logger.info(f"[RETRIEVER] Only {non_none_filters_count} metadata filters found (less than 2), searching through articles and appendices")
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
                
                # Ensure the document with highest subsection score has all its articles and appendices
                if highest_doc_index >= 0 and highest_doc_index < len(docs):
                    self._ensure_complete_articles_and_appendices(docs[highest_doc_index])
                
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
                
        # Continue with metadata filtering only if we have at least 2 non-None filters
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
        
        logger.info(f"[RETRIEVER] Using metadata filter with {non_none_filters_count} conditions: {filter_dict}")
            
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
        
        # Ensure the document with highest subsection score has all its articles and appendices
        self._ensure_complete_articles_and_appendices(merged_docs[highest_doc_index])
        
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
        
    def _ensure_complete_articles_and_appendices(self, doc: Document) -> None:
        """
        Ensure that the document with the highest subsection score includes all of its articles and appendices
        to provide a complete answer to the user.
        
        Args:
            doc: Document with highest subsection score
        """
        try:
            # Enhanced: First check if we need to extract subsections from structured data
            self._extract_subsections_from_structured_data(doc)
            
            # If the document doesn't have matched articles/appendices, but has original articles/appendices,
            # add all of them as matched with zero similarity scores
            # For articles
            if 'articles' in doc.metadata and (
                'matched_articles' not in doc.metadata or 
                not doc.metadata['matched_articles']
            ):
                try:
                    articles_data = []
                    if isinstance(doc.metadata['articles'], str):
                        articles_data = json.loads(doc.metadata['articles'])
                    elif isinstance(doc.metadata['articles'], list):
                        articles_data = doc.metadata['articles']
                    
                    if articles_data:
                        # Add all articles as matched with zero similarity scores
                        doc.metadata['matched_articles'] = [
                            {**article, 'similarity_score': 0.0, 'is_high_match': False}
                            for article in articles_data
                        ]
                        logger.info(f"[RETRIEVER] Added all {len(doc.metadata['matched_articles'])} articles from best document as matched with zero similarity scores")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"[RETRIEVER] Error processing articles for complete document: {str(e)}")
            
            # For appendices
            if 'appendices' in doc.metadata and (
                'matched_appendices' not in doc.metadata or 
                not doc.metadata['matched_appendices']
            ):
                try:
                    appendices_data = []
                    if isinstance(doc.metadata['appendices'], str):
                        appendices_data = json.loads(doc.metadata['appendices'])
                    elif isinstance(doc.metadata['appendices'], list):
                        appendices_data = doc.metadata['appendices']
                    
                    if appendices_data:
                        # Add all appendices as matched with zero similarity scores
                        doc.metadata['matched_appendices'] = [
                            {**appendix, 'similarity_score': 0.0, 'is_high_match': False}
                            for appendix in appendices_data
                        ]
                        logger.info(f"[RETRIEVER] Added all {len(doc.metadata['matched_appendices'])} appendices from best document as matched with zero similarity scores")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"[RETRIEVER] Error processing appendices for complete document: {str(e)}")
                    
            # If the document has matched articles, make sure all articles from the original document are included
            if 'matched_articles' in doc.metadata and 'articles' in doc.metadata:
                try:
                    # Get the original complete articles list
                    articles_data = []
                    if isinstance(doc.metadata['articles'], str):
                        articles_data = json.loads(doc.metadata['articles'])
                    elif isinstance(doc.metadata['articles'], list):
                        articles_data = doc.metadata['articles']
                    
                    if articles_data:
                        # Enhanced: Create a lookup of article numbers with more robust handling
                        matched_article_lookup = {}
                        for article in doc.metadata['matched_articles']:
                            # Create a lookup key that accounts for potential format differences
                            key = str(article.get('article_number', '')).strip().lower()
                            if key:
                                matched_article_lookup[key] = article
                        
                        # Add any articles not already in matched_articles
                        for article in articles_data:
                            article_number = str(article.get('article_number', '')).strip().lower()
                            if not article_number:
                                # If no article number, generate a fallback key from text
                                if 'text' in article and article['text']:
                                    article_number = f"text_{hash(article['text'][:50])}"
                                else:
                                    continue  # Skip articles with no identifiable features
                                    
                            if article_number not in matched_article_lookup:
                                article_copy = article.copy()
                                article_copy['similarity_score'] = 0.0
                                article_copy['is_high_match'] = False
                                doc.metadata['matched_articles'].append(article_copy)
                                matched_article_lookup[article_number] = article_copy
                        
                        # Sort matched articles by article number for better presentation
                        doc.metadata['matched_articles'] = sorted(
                            doc.metadata['matched_articles'],
                            key=lambda a: self._extract_numeric_part(str(a.get('article_number', '')))
                        )
                        
                        logger.info(f"[RETRIEVER] Ensured all {len(articles_data)} articles from original document are included and sorted")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"[RETRIEVER] Error ensuring complete articles: {str(e)}")
            
            # Same for appendices with enhanced handling
            if 'matched_appendices' in doc.metadata and 'appendices' in doc.metadata:
                try:
                    # Get the original complete appendices list
                    appendices_data = []
                    if isinstance(doc.metadata['appendices'], str):
                        appendices_data = json.loads(doc.metadata['appendices'])
                    elif isinstance(doc.metadata['appendices'], list):
                        appendices_data = doc.metadata['appendices']
                    
                    if appendices_data:
                        # Enhanced: More robust lookup using combination of identifiers
                        matched_appendix_lookup = {}
                        for appendix in doc.metadata['matched_appendices']:
                            # Try to create a reliable identifier
                            title = str(appendix.get('title', '')).strip().lower()
                            app_num = str(appendix.get('appendix_number', '')).strip().lower()
                            
                            if title:
                                matched_appendix_lookup[f"title_{title}"] = appendix
                            if app_num:
                                matched_appendix_lookup[f"num_{app_num}"] = appendix
                            # If both title and appendix number failed, use text hash
                            if not title and not app_num and 'text' in appendix and appendix['text']:
                                text_key = f"text_{hash(appendix['text'][:50])}"
                                matched_appendix_lookup[text_key] = appendix
                        
                        # Add any appendices not already in matched_appendices
                        for appendix in appendices_data:
                            # Try all possible keys
                            title = str(appendix.get('title', '')).strip().lower()
                            app_num = str(appendix.get('appendix_number', '')).strip().lower()
                            
                            # Check if this appendix is already matched
                            is_matched = False
                            
                            if title and f"title_{title}" in matched_appendix_lookup:
                                is_matched = True
                            elif app_num and f"num_{app_num}" in matched_appendix_lookup:
                                is_matched = True
                            elif 'text' in appendix and appendix['text']:
                                text_key = f"text_{hash(appendix['text'][:50])}"
                                if text_key in matched_appendix_lookup:
                                    is_matched = True
                            
                            if not is_matched:
                                appendix_copy = appendix.copy()
                                appendix_copy['similarity_score'] = 0.0
                                appendix_copy['is_high_match'] = False
                                doc.metadata['matched_appendices'].append(appendix_copy)
                                
                                # Also add to lookup to avoid duplicates
                                if title:
                                    matched_appendix_lookup[f"title_{title}"] = appendix_copy
                                if app_num:
                                    matched_appendix_lookup[f"num_{app_num}"] = appendix_copy
                        
                        # Sort matched appendices by appendix number if possible
                        doc.metadata['matched_appendices'] = sorted(
                            doc.metadata['matched_appendices'],
                            key=lambda a: self._extract_numeric_part(str(a.get('appendix_number', '')))
                        )
                        
                        logger.info(f"[RETRIEVER] Ensured all {len(appendices_data)} appendices from original document are included and sorted")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"[RETRIEVER] Error ensuring complete appendices: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"[RETRIEVER] Unexpected error ensuring complete articles and appendices: {str(e)}")
        
    def _extract_subsections_from_structured_data(self, doc: Document) -> None:
        """Extract articles and appendices from subsections if they exist."""
        try:
            if 'subsections' in doc.metadata:
                subsections_data = None
                
                # Try to parse subsections
                if isinstance(doc.metadata['subsections'], str):
                    try:
                        subsections_data = json.loads(doc.metadata['subsections'])
                    except json.JSONDecodeError:
                        logger.warning(f"[RETRIEVER] Cannot parse subsections JSON: {doc.metadata['subsections'][:100]}...")
                elif isinstance(doc.metadata['subsections'], dict):
                    subsections_data = doc.metadata['subsections']
                
                if subsections_data:
                    # Extract articles from subsections
                    if 'articles' in subsections_data and subsections_data['articles']:
                        if 'articles' not in doc.metadata or not doc.metadata['articles']:
                            # Add articles to document metadata
                            doc.metadata['articles'] = subsections_data['articles']
                            logger.info(f"[RETRIEVER] Extracted {len(subsections_data['articles'])} articles from subsections")
                    
                    # Extract appendices from subsections
                    if 'appendices' in subsections_data and subsections_data['appendices']:
                        if 'appendices' not in doc.metadata or not doc.metadata['appendices']:
                            # Add appendices to document metadata
                            doc.metadata['appendices'] = subsections_data['appendices']
                            logger.info(f"[RETRIEVER] Extracted {len(subsections_data['appendices'])} appendices from subsections")
        except Exception as e:
            logger.warning(f"[RETRIEVER] Error extracting subsections: {str(e)}")
            
    def _extract_numeric_part(self, text: str) -> int:
        """Extract numeric part from text for sorting."""
        try:
            # Find all numbers in the text
            numbers = re.findall(r'\d+', text)
            if numbers:
                # Use the first numeric part for sorting
                return int(numbers[0])
        except (ValueError, TypeError):
            pass
        return float('inf')  # Put items without numbers at the end
    
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
    
    # Helper methods for enhanced article and appendix search
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query for matching."""
        # Remove common stopwords in Arabic and split
        arabic_stopwords = {"من", "في", "على", "و", "أو", "ثم", "أن", "إلى", "عن", "مع", "هل", "كم", "متى", "أين", "لماذا"}
        
        # Process query to extract keywords
        query_lower = query.lower()
        query_words = re.findall(r'\b\w+\b', query_lower)
        keywords = [word for word in query_words if word not in arabic_stopwords and len(word) > 2]
        
        # Extract numbers which might be references to article numbers
        numbers = re.findall(r'\d+', query)
        keywords.extend(numbers)
        
        # Add special handling for decision/article numbers with format XX-XX
        special_refs = re.findall(r'\b\d+-\d+\b', query)
        keywords.extend(special_refs)
        
        # Add special handling for whole phrases that might be important
        phrases = []
        for i in range(len(query_words) - 1):
            if len(query_words[i]) > 3 and len(query_words[i+1]) > 3:
                phrases.append(f"{query_words[i]} {query_words[i+1]}")
        
        keywords.extend(phrases)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        if not keywords or not text:
            return False
            
        text_lower = text.lower()
        # Check if any keyword is in the text
        return any(kw.lower() in text_lower for kw in keywords)
    
    def _calculate_term_frequency(self, keywords: List[str], documents: List[Tuple[Document, float]]) -> Dict[str, float]:
        """Calculate the frequency of each keyword across all documents."""
        term_counts = {kw: 0 for kw in keywords}
        doc_count = len(documents)
        
        if doc_count == 0 or not keywords:
            return term_counts
            
        # Count occurrences across documents
        for doc, _ in documents:
            doc_text = doc.page_content.lower()
            for kw in keywords:
                if kw.lower() in doc_text:
                    term_counts[kw] += 1
        
        # Convert to inverse document frequency
        term_frequency = {}
        for kw, count in term_counts.items():
            # Avoid division by zero
            if count > 0:
                # Terms that appear in fewer documents are more valuable
                term_frequency[kw] = math.log(doc_count / count) 
            else:
                term_frequency[kw] = 0
                
        # Normalize to 0-1 range
        max_freq = max(term_frequency.values()) if term_frequency.values() else 1
        for kw in term_frequency:
            term_frequency[kw] = term_frequency[kw] / max_freq if max_freq > 0 else 0
            
        return term_frequency
    
    def _calculate_term_match_boost(self, text: str, term_frequency: Dict[str, float]) -> float:
        """Calculate boost factor based on matching terms and their frequency."""
        if not text or not term_frequency:
            return 0
            
        text_lower = text.lower()
        boost = 0
        
        # Sum the frequency values of matching terms
        for term, freq in term_frequency.items():
            if term.lower() in text_lower:
                boost += freq
                
        # Scale boost to a reasonable range
        return min(0.5, boost / 3)  # Cap at 0.5 (50% boost)
    
    def _extract_context_snippets(self, text: str, keywords: List[str], max_snippets: int = 3, context_size: int = 60) -> List[str]:
        """Extract snippets from text that contain keywords with surrounding context."""
        if not text or not keywords:
            return []
            
        text_lower = text.lower()
        snippets = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if len(keyword) < 3:  # Skip very short keywords
                continue
                
            # Find all occurrences of the keyword
            start_idx = 0
            while len(snippets) < max_snippets:
                pos = text_lower.find(keyword_lower, start_idx)
                if pos == -1:
                    break
                    
                # Extract snippet with context
                start = max(0, pos - context_size)
                end = min(len(text), pos + len(keyword) + context_size)
                
                # Try to start and end at word boundaries
                while start > 0 and text[start].isalnum():
                    start -= 1
                    
                while end < len(text) - 1 and text[end].isalnum():
                    end += 1
                
                snippet = text[start:end]
                
                # Add ellipsis for truncated text
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet = snippet + "..."
                    
                # Check if this snippet overlaps with existing ones
                # If there's significant overlap, skip it
                is_duplicate = False
                for existing in snippets:
                    if self._calculate_overlap(existing, snippet) > 0.7:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    snippets.append(snippet)
                    
                start_idx = pos + len(keyword)
                
        # Ensure we return at most max_snippets
        return snippets[:max_snippets]
    
    def _calculate_overlap(self, str1: str, str2: str) -> float:
        """Calculate the overlap ratio between two strings."""
        if not str1 or not str2:
            return 0
            
        # Use character 5-grams as the basis for comparison
        def get_ngrams(text, n=5):
            return [text[i:i+n] for i in range(len(text) - n + 1)]
            
        ngrams1 = set(get_ngrams(str1))
        ngrams2 = set(get_ngrams(str2))
        
        if not ngrams1 or not ngrams2:
            return 0
            
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union)
    
    def _calculate_metadata_match_bonus(self, metadata: Dict, query: str, keywords: List[str]) -> float:
        """Calculate bonus for exact metadata matches with query entities."""
        bonus = 0
        
        # Check for decision number match
        if 'decision_number' in metadata and metadata['decision_number']:
            for kw in keywords:
                if kw in str(metadata['decision_number']):
                    bonus += 0.2
                    break
                    
        # Check for year match
        if 'year' in metadata and metadata['year']:
            year_pattern = r'\b(20\d{2})\b'
            query_years = re.findall(year_pattern, query)
            if query_years and str(metadata['year']) in query_years:
                bonus += 0.15
                
        # Check for chunk_type match
        if 'chunk_type' in metadata and metadata['chunk_type']:
            chunk_type = str(metadata['chunk_type']).lower()
            if any(chunk_type in kw.lower() for kw in keywords):
                bonus += 0.1
                
        # Check for official bulletin match
        if 'official_bulletin' in metadata and metadata['official_bulletin']:
            official_bulletin = str(metadata['official_bulletin']).lower()
            if any(kw.lower() in official_bulletin for kw in keywords):
                bonus += 0.1
                
        return min(0.4, bonus)  # Cap at 0.4 (40% bonus) 
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
            ("system", """أنت خبير في إعادة صياغة الاستعلامات لتحقيق الاسترجاع الأمثل في أنظمة RAG.

            وصف المهمة:
            مهمتك هي إعادة صياغة الاستعلام المقدم لتعظيم فعالية الاسترجاع مع الحفاظ على نيته الأصلية.
            يجب أن تعيد فقط الاستعلام المعاد صياغته كنص عادي بدون تعليقات إضافية أو شروحات أو تنسيق JSON.
            
            إرشادات إعادة الصياغة:
            1. ركز على هذه التقنيات المحددة:
               - توسيع الاختصارات والمختصرات إلى مصطلحات كاملة
               - إضافة المرادفات والمصطلحات ذات الصلة مفصولة بـ OR
               - تضمين المصطلحات التقنية ذات الصلة بالمجال
               - تحديد أنواع المستندات التي قد تحتوي على الإجابة (مثل "وثيقة السياسة"، "المواصفات التقنية")
               - إضافة مصطلحات السياق التي قد تظهر في المستندات ذات الصلة
               - إزالة كلمات الحشو والمجاملات والسياق غير ذي الصلة
               - تقسيم الأسئلة المعقدة إلى مكونات مفاهيمية رئيسية
               - تحويل الأسئلة الضمنية إلى احتياجات معلومات صريحة
            
            2. استراتيجيات خاصة بنوع الاستعلام:
               - للاستعلامات الواقعية: ركز على الكيانات الرئيسية وسماتها والمصطلحات الدقيقة
               - للاستعلامات التحليلية: تضمين المفاهيم الرئيسية والعلاقات وأطر التحليل
               - للاستعلامات الإجرائية: أضف المصطلحات المتعلقة بالخطوات وأسماء الأدوات وأفعال العمل
               - للاستعلامات المحادثة: استخراج وتأكيد الحاجة الأساسية للمعلومات
            
            3. متطلبات التنسيق:
               - حافظ على الاستعلام المعاد صياغته تحت 50 كلمة
               - استخدم اللغة الطبيعية بدلاً من عوامل البحث
               - حافظ على بنية الجملة أو العبارة المقروءة
               - حافظ على جميع الكيانات الأصلية والمصطلحات الرئيسية
               - أضف مصطلحات جديدة بطريقة تعزز الصلة بدلاً من تخفيفها
            
            4. القواعد الحاسمة:
               - لا تبتكر أبدًا نوايا استعلام جديدة أو تغير السؤال الأساسي
               - لا تستخدم أبدًا نصًا عامًا أو نصًا مثالًا دون استبداله
               - لا تضيف أبدًا تفسيرات لم تكن مضمنة في الاستعلام الأصلي
               - لا تضيف أبدًا بناء خاص للبحث أو عوامل منطقية أو بنية أخرى غير طبيعية
               - لا تعيد أبدًا أي شيء غير نص الاستعلام المعاد صياغته
            
            5. التعامل مع الحالات الخاصة:
               - للاستعلامات القصيرة جدًا (1-3 كلمات): توسيع مع المصطلحات ذات الصلة المحتملة والسياق
               - للاستعلامات الغامضة: أضف مصطلحات توضيحية مع الحفاظ على الاتساع
               - للاستعلامات متعددة الأجزاء: ركز على المفاهيم الرئيسية من كل جزء
               - للاستعلامات المحددة بالفعل: قم بتغييرات طفيفة، مع التركيز على المرادفات وتوسيع المصطلحات
            
            تذكر: أعد فقط الاستعلام المعاد صياغته بدون نص إضافي أو شرح أو تنسيق.
            """),
            ("user", "الاستعلام الأصلي: {query}\nنوع الاستعلام: {query_type}\nنية الاستعلام: {query_intent}")
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
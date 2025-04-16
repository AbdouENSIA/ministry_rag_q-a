from typing import Any, Optional
import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state.rag_state import RAGState

# Configure logging
logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyzes queries to determine their type, intent, and relevance to the knowledge base."""
    
    def __init__(self, llm: BaseChatModel, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.parser = JsonOutputParser()
        self.rate_limiter = rate_limiter
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت محلل استعلامات خبير لنظام RAG، مسؤول عن فهم وتصنيف استفسارات المستخدم بدقة عالية.

            متطلبات تنسيق المخرجات:
            يجب أن ترد بكائن JSON صالح يحتوي على هذه الحقول بالضبط:
            {{
                "is_related_to_index": boolean,
                "query_type": string (واحد من: "factual", "analytical", "procedural", "conversational"),
                "query_entities": array of strings,
                "query_intent": string,
                "confidence": number (بين 0.0 و 1.0),
                "reasoning": string
            }}
            
            إذا كان أي حقل مفقودًا أو منسقًا بشكل غير صحيح، فسيفشل النظام. تأكد من صحة بناء JSON.

            إرشادات التصنيف:
            1. اعتبر الاستعلام "متعلق بالفهرس" (is_related_to_index = true) في هذه الحالات:
               - إذا كان يسأل عن معرفة مجال محددة قد تكون في الفهرس
               - إذا كان استعلامًا عامًا يمكن الإجابة عليه باستخدام المعرفة المفهرسة
               - إذا كان تحية أو سؤالًا عامًا يجب التعامل معه بلطف
               - اعتبره صحيحًا إذا كنت غير متأكد - من الأفضل محاولة الإجابة بالمعرفة المتاحة
            
            2. ضع علامة "غير متعلق بالفهرس" (is_related_to_index = false) فقط إذا كان الاستعلام:
               - يتطلب صراحة بيانات في الوقت الفعلي بالتأكيد لن تكون في المعرفة الثابتة
               - يحتاج إلى معلومات بالتأكيد ليست في أي قاعدة معرفة
               - يتطلب استدعاءات واجهة برمجة تطبيقات خارجية أو وظائف خاصة بالويب
               - يحتوي على نوايا ضارة أو ينتهك المبادئ التوجيهية الأخلاقية
            
            تعريفات نوع الاستعلام (يجب دائمًا تعيين إحدى هذه القيم لـ "query_type"):
            - "factual": أسئلة مباشرة تبحث عن معلومات محددة، حقائق، تعريفات، أو إجابات مباشرة
              أمثلة: "ما هو X؟"، "متى تم اختراع Y؟"، "من أنشأ Z؟"
            
            - "analytical": أسئلة تتطلب تحليل، مقارنة، تفكير، تقييم، أو تركيب
              أمثلة: "لماذا حدث X؟"، "كيف يقارن Y بـ Z؟"، "ما هي آثار X؟"
            
            - "procedural": أسئلة كيفية أو تعليمات خطوة بخطوة تطلب إرشادات حول العمليات
              أمثلة: "كيف أقوم بـ X؟"، "ما هي الخطوات التي يجب اتباعها لـ Y؟"، "اشرح عملية Z"
            
            - "conversational": تحيات، نظام ، حوار عام، أو أسئلة حول النظام نفسه
              أمثلة: "مرحبا"، "كيف حالك؟"، "هل يمكنك مساعدتي في شيء ما؟"، "ماذا يمكنك أن تفعل؟"
            
            متطلبات استخراج الكيانات (query_entities):
            - استخرج الكيانات الصريحة والضمنية (الحد الأدنى 1، الحد الأقصى 10)
            - تضمين كلمات السياق ذات الصلة حول الكيانات
            - ملاحظة العلاقات بين الكيانات
            - للاستعلامات الغامضة، استخرج مجالات الموضوع الأوسع
            - إذا لم تكن هناك كيانات (مثل التحيات الخالصة)، قم بتضمين مصفوفة فارغة
            
            إرشادات تصنيف النوايا (query_intent):
            - كن محددًا ومفصلًا حول هدف المستخدم الفعلي
            - اختر من هذه الفئات ولكن أضف تفاصيل محددة تتجاوز اسم الفئة:
              - information_seeking: يريد المستخدم أن يتعلم شيئًا محددًا
              - clarification: يحتاج المستخدم إلى شرح أو توضيح
              - greeting: المستخدم يبدأ/يواصل المحادثة
              - task: يريد المستخدم إنجاز شيء محدد
              - feedback: المستخدم يقدم ملاحظات أو آراء
            - دائمًا تضمين تفاصيل حول ما هي المعلومات أو الإجراء الذي يبحث عنه المستخدم بالضبط
            
            قواعد تقييم الثقة:
            - 0.9-1.0: استعلام واضح جدًا لا لبس فيه مع تصنيف واضح
            - 0.7-0.9: نية واضحة ولكن مع بعض الغموض البسيط
            - 0.5-0.7: غموض معتدل ولكن ثقة معقولة
            - 0.3-0.5: غموض كبير مع تفسيرات محتملة متعددة
            - 0.0-0.3: استعلام غامض جدًا أو غامض للغاية
            - لا تترك أبدًا 0.0 أو 1.0 بالضبط - دائمًا قدم درجة دقيقة
            
            متطلبات التفكير:
            - قدم تبريرًا واضحًا لجميع التصنيفات
            - اشرح أي غموض أو تحديات في التصنيف
            - أبقِه تحت 100 كلمة ولكن كن محددًا وشاملاً
            - ركز على سبب تصنيفك كما فعلت بدلاً من إعادة صياغة التصنيف
            """),
            ("user", "حلل هذا الاستعلام وفقًا للإرشادات أعلاه: {query}")
        ])
        
    async def analyze(self, state: RAGState) -> RAGState:
        """
        Analyze the query and update the state with analysis results.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with query analysis
        """
        logger.info("\n" + "="*80)
        logger.info(f"[QUERY_ANALYZER] Starting analysis of query: '{state['query']}'")
        logger.info("-"*80)
        
        # Get query from state
        query = state["query"]
        
        # Run analysis
        if self.rate_limiter:
            await self.rate_limiter.wait()
        
        logger.info("[QUERY_ANALYZER] [LLM CALL] Analyzing query with LLM...")
        
        # Make sure we pass only the expected parameters to the prompt
        input_data = {"query": query}
        logger.info(f"[QUERY_ANALYZER] [LLM INPUT] {json.dumps(input_data, ensure_ascii=False)}")
        
        chain = self.analysis_prompt | self.llm | self.parser
        analysis_result = await chain.ainvoke(input_data)
        
        logger.info(f"[QUERY_ANALYZER] [LLM OUTPUT] Analysis result: {json.dumps(analysis_result, ensure_ascii=False)}")
        
        # Log key insights from analysis
        logger.info(f"[QUERY_ANALYZER] Query identified as '{analysis_result['query_type']}' type")
        logger.info(f"[QUERY_ANALYZER] Is related to index: {analysis_result['is_related_to_index']}")
        logger.info(f"[QUERY_ANALYZER] Query intent: {analysis_result['query_intent']}")
        logger.info(f"[QUERY_ANALYZER] Analysis confidence: {analysis_result['confidence']}")
        logger.info(f"[QUERY_ANALYZER] Entities: {', '.join(analysis_result['query_entities'])}")
        logger.info(f"[QUERY_ANALYZER] Reasoning: {analysis_result['reasoning']}")
        
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
        
        logger.info(f"[QUERY_ANALYZER] Analysis completed")
        logger.info("="*80)
        
        return state 
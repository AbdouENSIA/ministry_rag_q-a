from typing import Any, List, Optional
import json
import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..state.rag_state import RAGState

# Configure logging
logger = logging.getLogger(__name__)

class Generator:
    """Adaptive answer generator that produces high-quality responses with self-reflection."""
    
    def __init__(self, llm: BaseChatModel, rate_limiter: Optional[Any] = None):
        self.llm = llm
        self.parser = JsonOutputParser(
            pydantic_object=None,  # No validation against Pydantic
            include_raw_response=True,  # Include raw response in case of parsing failures
            streaming=False  # Don't stream the JSON output
        )
        self.rate_limiter = rate_limiter
        
        # Initialize generation prompts
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت خبير نخبة في إنشاء إجابات شاملة بشكل استثنائي، ودقيقة بدقة، ودقيقة تمامًا لأنظمة RAG. يجب أن تكون إجاباتك شاملة بشكل شامل، وموضحة بشكل احترافي، وغنية بشكل لا يصدق بالتفاصيل ما لم يطلب المستخدم صراحةً إجابة قصيرة.

            ضع في اعتبارك أن جميع الإجابات يجب أن تكون باللغة العربية بشكل أساسي وتستخدم المصطلحات العربية بشكل صحيح. يجب تقديم الإجابات بلغة عربية سليمة وواضحة مع مراعاة قواعد اللغة العربية الصحيحة.

            متطلبات تنسيق ماركداون متقدمة:
            1. استخدم تنسيق ماركداون واسع النطاق لتحقيق أقصى قدر من الوضوح وإمكانية القراءة:
                - استخدم عناوين هرمية (# عنوان رئيسي، ## عنوان فرعي، ### عنوان فرعي ثانوي)
                - أنشئ أقسامًا مميزة بوضوح مع عناوين وصفية
                - استخدم ** نص غامق ** للمصطلحات والمفاهيم والنقاط الرئيسية المهمة
                - استخدم * نص مائل * للتأكيد والفروق الدقيقة
                - استخدم `رمز` للمصطلحات الفنية أو الأوامر أو بناء الجملة
                - استخدم > اقتباس للاقتباسات المهمة من المستندات
                - استخدم خطوط أفقية (---) لفصل الأقسام الرئيسية عند الاقتضاء
                - استخدم جداول مع محاذاة مناسبة للأعمدة للبيانات المقارنة
                - استخدم المسافة البادئة المناسبة للقوائم المتداخلة والمعلومات الهرمية
                - لا تقم بتضمين أشكال أو رسوم بيانية أو مراجع صور في إجاباتك
                        
            2. التنظيم الهيكلي المتقدم:
                - ابدأ بعنوان واضح ومفيد باستخدام عنوان #
                - اتبعه بملخص تنفيذي موجز للنتائج الرئيسية
                - استخدم هيكل هرمي منطقي مع مستويات عنوان مناسبة
                - قم بتجميع المعلومات ذات الصلة تحت عناوين الأقسام المناسبة
                - قدم المعلومات بترتيب الأهمية أو التسلسل المنطقي
                - استخدم أنماط تنسيق متسقة عبر المستند
                - انتهِ بملخص موجز أو استنتاج عند الاقتضاء
                - قم بتضمين قسم "المراجع" يسرد مصادر المستندات عندما يكون ذلك مفيدًا
            
            3. التنظيم المرئي:
                - استخدم نقاط للقوائم غير المرتبة
                - استخدم قوائم مرقمة للخطوات المتسلسلة أو العناصر ذات الأولوية
                - استخدم قوائم متداخلة للمعلومات الهرمية
                - أنشئ جداول للبيانات المنظمة والمقارنة
                - استخدم كتل الكود مع تمييز صيغة الكود للمحتوى الفني
                - قم بتطبيق مسافات متسقة بين الأقسام
                - استخدم التأكيد (غامق، مائل) باستمرار للمصطلحات المهمة

            4. تنسيق الجداول:
                - قم دائماً بإنشاء جداول مناسبة للمواد والملاحق المذكورة في المستندات
                - حافظ على تنسيق الجداول بشكل كامل مع محاذاة الأعمدة
                - تضمين جميع حقول البيانات والمعلومات التي تظهر في المستندات المصدر
                - استخدم العناوين المناسبة للأعمدة التي تعكس محتوى البيانات
                - نسق بيانات المواد كجدول كامل مع أرقام المواد ونصوصها
                - نسق بيانات الملاحق بشكل كامل مع أرقامها وعناوينها ونصوصها
                - لا تحذف أي معلومات أو حقول من الجداول الأصلية

            أولوية محتوى الإجابة:
            1. حقل "الإجابة" هو الأولوية المطلقة الرئيسية - يجب أن يكون مفصلاً بشكل شامل واستثنائي وشامل وشامل بشكل استثنائي
            2. يجب أن تقف الإجابة وحدها كرد كامل وموثوق وذو جودة احترافية على الاستعلام
            3. يجب أن تحتوي الإجابة على جميع المعلومات ذات الصلة من المستندات، مركبة ومنظمة لتحقيق أقصى قدر من الوضوح
            4. جميع الحقول الأخرى ثانوية ومكملة للإجابة الرئيسية
            6. لا تحجب أبدًا معلومات مهمة من الإجابة لوضعها في مكان آخر
            7. اجعل الإجابة شاملة بحيث تكون الأدلة الداعمة والحقول الأخرى مجرد تأكيد
            8. قم بهيكلة الإجابة بعناوين واضحة ونقاط وقوائم مرقمة وجداول وما إلى ذلك حسب الاقتضاء
            9. لأي استعلام يحتوي على معلومات واقعية في المستندات، قدم استجابة واسعة ومفصلة
            10. حتى للأسئلة البسيطة، قدم السياق والخلفية والشروحات الكاملة من المستندات
            11. لا تقم بتضمين أشكال أو رسوم بيانية أو مخططات أو صور لأنه لا يمكن عرضها بشكل صحيح
            12. عند وجود جداول للمواد أو الملاحق، قم بعرضها بشكل كامل مع جميع عناصرها دون حذف أي معلومات

            متطلبات خاصة للجداول والبيانات المنظمة:
            1. عندما تحتوي المستندات على بيانات المواد (Articles) أو الملاحق (Appendices):
               - قم دائماً بعرضها في جداول ماركداون منسقة بشكل صحيح
               - تأكد من تضمين جميع الحقول والأعمدة الموجودة في المصدر
               - حافظ على الترقيم الأصلي للمواد والملاحق
               - استخدم ترتيب منطقي لعرض البيانات
               - قم بتمييز المواد الرئيسية أو الملاحق الهامة
            2. عند وجود أقسام أو فئات متعددة في المستندات:
               - قسم الإجابة إلى أقسام وعناوين فرعية متميزة
               - رتب المعلومات بشكل منطقي مع مراعاة التدرج الهرمي
               - استخدم الجداول للمعلومات المتعلقة ببعضها
               - استخدم القوائم للمعلومات المتسلسلة أو المرتبطة
            3. الحفاظ على اكتمال المعلومات:
               - لا تحذف أي عناصر أو حقول من الجداول الأصلية
               - تأكد من عرض جميع بيانات المواد والملاحق بشكل كامل
               - لا تختصر المعلومات الطويلة حتى عند كتابتها في جداول
               - إذا لزم الأمر، قسم الجداول الكبيرة إلى أقسام منطقية
            
            متطلبات خاصة للمحتوى الكامل من المواد والملاحق:
            1. نص مواد القرارات والمراسيم:
               - يجب عرض النص الكامل للمواد ذات الصلة بالاستعلام دون اختصار
               - شمل رقم المادة ونصها الكامل في الإجابة
               - نظم المواد في جداول مرتبة حسب أرقامها
               - لا تلخص أو تختصر نص المواد مهما كان طويلاً
            2. محتوى الملاحق:
               - اعرض النص الكامل والمحتوى الكامل للملاحق ذات الصلة
               - قم بتضمين جميع البيانات والجداول الموجودة في الملاحق
               - حافظ على التنسيق الأصلي للملاحق قدر الإمكان
               - لا تختصر أبداً محتوى الملاحق حتى لو كان طويلاً
            3. الإجابة على استعلامات محددة:
               - إذا كان السؤال يتعلق بمادة أو ملحق محدد، أبرز هذا المحتوى في بداية الإجابة
               - ثم قدم المعلومات السياقية والإضافية من باقي المستند
               - استخدم عناوين واضحة لتمييز أقسام الإجابة
               - قم بإعطاء الأولوية للنصوص والمواد والملاحق التي تجيب مباشرة على الاستعلام
            
            مثال تنسيق المصدر:
             "تستخدم الحوسبة الكمومية كيوبتات يمكن أن توجد في حالات متعددة في وقت واحد بسبب التراكب. على عكس البتات الكلاسيكية، يمكن أن تكون الكيوبتات متشابكة، مما يسمح لها بمشاركة الحالات الكمومية بغض النظر عن المسافة."
            
            قواعد استخدام المستندات:
            1. استخدم فقط المعلومات الموجودة صراحة في المستندات المقدمة التي تجيب مباشرة على السؤال
            2. لا تقم أبدًا بتضمين حقائق أو بيانات أو مطالبات غير مدعومة بالمستندات حتى لو كانت ذات صلة
            3. لا تستخدم أبدًا معرفتك العامة لملء الفجوات في المستندات
            4. لا تقدم أبدًا معلومات بناءً على افتراضات أو استنتاجات غير موجودة مباشرة في المستندات
            5. إذا كانت المستندات لا تحتوي على معلومات مطلوبة للإجابة على الاستعلام:
                - اذكر بوضوح: "المستندات المقدمة لا تحتوي على معلومات حول [جانب محدد]."
                - لا تقدم معلومات بديلة أو ذات صلة
                - لا تحاول الإجابة باستخدام معلومات مشابهة ولكن غير مباشرة
                - اقترح إعادة صياغة السؤال أو طرح سؤال مختلف
            6. ارفض تمامًا أي رغبة في أن تكون مفيدًا بإضافة معلومات تتجاوز ما هو مطلوب بدقة في السؤال
            7. ضع مراجع المستندات في قسم الأدلة الداعمة فقط للمعلومات المستخدمة في الإجابة المباشرة
            8. لا تضمن معلومات إضافية "للسياق" أو "للفائدة" إذا لم تكن ضرورية مباشرة للإجابة
            9. قم بتجميع المعلومات عبر المستندات فقط إذا كانت تجيب مباشرة على السؤال

            تعليمات نهائية هامة:
            1. لا تقم أبداً باختصار نص المواد أو الملاحق عند عرضها في الإجابة
            2. قم بإدراج النص الكامل للمواد وتعرض النص الكامل للملاحق ذات الصلة
            3. حتى إذا كان النص طويلاً جداً، أدرجه بالكامل
            4. لا تكتفِ بتلخيص المحتوى أو الإشارة إليه، بل أدرجه كاملاً
            5. أعطِ الأولوية للنصوص الكاملة للمواد والملاحق ذات الصلة المباشرة بالاستعلام
            6. تأكد من إبراز المواد أو الملاحق التي تحتوي على معايير أو ضوابط أو شروط مطلوبة في الاستعلام
            7. نظِّم المحتوى بطريقة تسهل على القارئ فهم الإجابة الكاملة على سؤاله

            ضع في اعتبارك أن جميع الإجابات يجب أن تكون باللغة العربية بشكل أساسي وتستخدم المصطلحات العربية بشكل صحيح. يجب تقديم الإجابات بلغة عربية سليمة وواضحة مع مراعاة قواعد اللغة العربية الصحيحة.

            !!! CRITICAL JSON FORMATTING REQUIREMENTS !!!
            You MUST follow these exact JSON formatting rules:
            1. The response MUST be a valid JSON object
            2. All string values containing markdown MUST escape newlines with \\n
            3. All string values containing quotes MUST escape them with \"
            4. All JSON object keys MUST be in double quotes
            5. All JSON values MUST be properly typed (strings, numbers, booleans, arrays, or objects)
            6. Arrays and objects MUST use square brackets [] and curly braces {{}} respectively
            7. NEVER use single quotes for JSON strings, only double quotes
            8. NEVER use unescaped newlines within JSON strings
            9. The response MUST contain ALL of these required fields:
                - "answer": (string) The complete markdown-formatted answer
                - "confidence_score": (number) Between 0 and 1
                - "supporting_evidence": (array of strings) Evidence from documents
                - "reasoning_path": (string) Explanation of answer construction
                - "suggested_followup": (array of strings) Follow-up questions
                - "metadata": (object) Additional metadata
                - "validation": (object) Self-validation results
            10. The validation object MUST contain:
                - "has_hallucinations": (boolean)
                - "answers_question": (boolean) 
                - "quality_score": (number) Between 0 and 1
                - "improvement_needed": (array of strings)
                - "validation_reasoning": (string)

            !!! SPECIAL TABLE HANDLING RULES !!!
            For tables in markdown, you MUST follow these specific rules:
            1. ALL tables must be properly escaped in the JSON string
            2. NEVER use the actual newline character within JSON strings for tables
            3. Use \\n to represent newlines in tables and preserve the table structure
            4. Ensure all | (pipe) characters are properly included
            5. Format each row of a table as a single line using \\n between rows
            6. ALWAYS check that rows and columns align properly after escaping
            7. SIMPLIFY complex tables if they have many columns (>8) or rows (>50) 
            8. For very large tables, consider summarizing them or only showing key portions
            9. If a table is causing formatting issues, convert it to a more compact format

            EXAMPLE OF PROPERLY FORMATTED JSON RESPONSE:
{{
    "answer": "# عنوان رئيسي\\n\\n## ملخص تنفيذي\\n\\nهذا مثال على إجابة منسقة بشكل صحيح.\\n\\n## القسم الأول\\n\\nمحتوى القسم الأول هنا.\\n\\n### مثال على جدول بسيط\\n\\n| العمود الأول | العمود الثاني | العمود الثالث |\\n|-------------|-------------|-------------|\\n| قيمة 1 | قيمة 2 | قيمة 3 |\\n| قيمة 4 | قيمة 5 | قيمة 6 |",
    "confidence_score": 0.87,
    "supporting_evidence": [
        "Document 1: نص الدليل الأول",
        "Document 2: نص الدليل الثاني"
    ],
    "reasoning_path": "شرح كيفية بناء الإجابة خطوة بخطوة",
    "suggested_followup": [
        "سؤال المتابعة الأول؟",
        "سؤال المتابعة الثاني؟"
    ],
    "metadata": {{
        "sources_used": 2,
        "key_concepts": ["المفهوم الأول", "المفهوم الثاني"],
        "confidence_factors": ["العامل الأول", "العامل الثاني"]
    }},
    "validation": {{
        "has_hallucinations": false,
        "answers_question": true,
        "quality_score": 0.85,
        "improvement_needed": [
            "النقطة الأولى للتحسين",
            "النقطة الثانية للتحسين"
        ],
        "validation_reasoning": "شرح سبب صحة الإجابة وجودتها"
    }}
}}"""),
            ("user", """Query: {query}
            Query type: {query_type}
            Query intent: {query_intent}
            Analysis confidence: {confidence}
            Analysis reasoning: {reasoning}
            
            Documents:
            {documents}
            
            !!! CRITICAL INSTRUCTION !!!
            يجب عليك الإجابة فقط باستخدام المعلومات من المستندات أعلاه.
            إذا كانت المستندات لا تحتوي على الإجابة، اذكر ذلك بوضوح.
            لا تستخدم أبداً معرفتك العامة لملء الفجوات في المستندات.
            عند عرض الجداول، تأكد من عرض كل المعلومات بشكل شامل وكامل بدون اختصار أي بيانات.
            يجب عرض النص الكامل للمواد والملاحق ذات الصلة بالاستعلام دون أي اختصار.
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
        logger.info("\n" + "="*80)
        logger.info(f"[GENERATOR] Starting answer generation for query: '{state['query']}'")
        logger.info("-"*80)
        
        # Get required information from state
        query = state["query"]
        query_type = state.get("query_type", "unknown")
        query_intent = state.get("query_intent", "unknown")
        documents = state.get("documents", [])
        
        # Get optional analysis info with defaults
        confidence = state.get("confidence", 0.0)
        reasoning = state.get("reasoning", "No prior analysis reasoning available")
        
        if not documents:
            logger.info("[GENERATOR] No documents available for generation")
            state.update({
                "answer": "I apologize, but I couldn't find any relevant documents to answer your question. Please rephrase your query or ask about a different topic that might be covered in our knowledge base.",
                "confidence_score": 0.0,
                "supporting_evidence": ["No relevant documents found"],
                "has_hallucinations": False,
                "answers_question": False,
                "quality_score": 0.0,
                "docs_relevant": False,  # Added for grader functionality
                "metadata": {
                    "sources_used": 0,
                    "generation_approach": "default_no_documents_response"
                },
                "current_node": "generator"
            })
            logger.info("[GENERATOR] Returning default 'no documents' response")
            return state
            
        # Format documents for generation
        doc_texts = self._format_documents(documents)
        logger.info(f"[GENERATOR] Formatted {len(documents)} documents for generation")
        
        # Generate answer and validation
        if self.rate_limiter:
            await self.rate_limiter.wait()
            
        logger.info("[GENERATOR] [LLM CALL] Generating answer with self-validation...")
        
        input_data = {
            "query": query,
            "query_type": query_type,
            "query_intent": query_intent,
            "confidence": confidence,
            "reasoning": reasoning,
            "documents": doc_texts
        }
        logger.info(f"[GENERATOR] [LLM INPUT] Query: '{query}', Type: {query_type}, Doc count: {len(documents)}")
        
        try:
            # Call LLM for generation and validation
            chain = self.answer_prompt | self.llm | self.parser
            result = await chain.ainvoke(input_data)
            
            # If response contains raw_response (from parser with include_raw_response=True), extract it
            if isinstance(result, dict) and "raw_response" in result:
                raw_response = result.pop("raw_response", "")
                # Check if we have a properly parsed result
                if not result or "answer" not in result:
                    # Attempt to manually parse the raw response if automatic parsing failed
                    logger.warning("[GENERATOR] Automatic JSON parsing failed, attempting manual parsing")
                    try:
                        # Try to fix and parse the JSON manually
                        cleaned_json = self._clean_json_response(raw_response)
                        import json
                        result = json.loads(cleaned_json)
                        logger.info("[GENERATOR] Manual JSON parsing succeeded")
                    except Exception as manual_parse_error:
                        logger.error(f"[GENERATOR] Manual JSON parsing failed: {str(manual_parse_error)}")
                        # Use a simpler approach to extract just the answer if JSON parsing completely fails
                        answer = self._extract_answer_from_raw_text(raw_response)
                        result = {
                            "answer": answer,
                            "confidence_score": 0.7,
                            "supporting_evidence": ["Extracted from raw text due to parsing issues"],
                            "reasoning_path": "Direct extraction from LLM response due to JSON parsing failure",
                            "suggested_followup": [],
                            "metadata": {
                                "sources_used": len(documents),
                                "generation_approach": "fallback_extraction"
                            },
                            "validation": {
                                "has_hallucinations": False,
                                "answers_question": True,
                                "quality_score": 0.7,
                                "improvement_needed": ["JSON parsing failed, simplified response provided"],
                                "validation_reasoning": "Fallback extraction due to JSON parsing issues"
                            }
                        }
                        logger.info("[GENERATOR] Used fallback text extraction")
            
            logger.info(f"[GENERATOR] [LLM OUTPUT] Generation completed successfully")
            logger.info(f"[GENERATOR] Answer length: {len(result.get('answer', ''))}")
            logger.info(f"[GENERATOR] Validation result: has_hallucinations={result.get('validation', {}).get('has_hallucinations', False)}, answers_question={result.get('validation', {}).get('answers_question', True)}")
            
            # Extract validation from result
            validation = result.get("validation", {})
            
            # Determine document relevance based on quality score
            quality_score = validation.get("quality_score", 0.75)
            docs_relevant = quality_score >= 0.6  # Same threshold previously used in grader
            
            # Update state with combined results
            state.update({
                **result,
                "has_hallucinations": validation.get("has_hallucinations", False),
                "answers_question": validation.get("answers_question", True),
                "quality_score": quality_score,
                "docs_relevant": docs_relevant,  # Added for grader functionality
                "improvement_needed": validation.get("improvement_needed", []),
                "validation_reasoning": validation.get("validation_reasoning", ""),
                "current_node": "generator"
            })
            
            logger.info(f"[GENERATOR] Generation completed with quality_score: {quality_score}, docs_relevant: {docs_relevant}")
            logger.info("="*80)
            
            return state
            
        except Exception as e:
            logger.error(f"[GENERATOR] Generation failed with error: {str(e)}")
            # Return a default response in case of error
            state.update({
                "answer": "I apologize, but I encountered an error while generating the response. Please try rephrasing your question.",
                "confidence_score": 0.0,
                "supporting_evidence": ["Error during generation"],
                "has_hallucinations": False,
                "answers_question": False,
                "quality_score": 0.0,
                "docs_relevant": False,
                "metadata": {
                    "sources_used": len(documents),
                    "generation_approach": "error_fallback"
                },
                "current_node": "generator",
                "error": str(e)
            })
            return state
        
    def _clean_json_response(self, raw_text: str) -> str:
        """Clean and fix JSON response that might have formatting issues."""
        # Remove Markdown code block markers
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        
        # Try to find the actual JSON content
        import re
        json_pattern = r'\{.*\}'
        json_matches = re.search(json_pattern, cleaned_text, re.DOTALL)
        
        if json_matches:
            cleaned_text = json_matches.group(0)
        
        # Try to simplify complex tables that might be causing issues
        cleaned_text = self._simplify_tables_in_json(cleaned_text)
        
        # Fix common JSON escaping issues with tables
        cleaned_text = cleaned_text.replace('\n', '\\n')
        
        # Fix double escaping if it happened
        cleaned_text = cleaned_text.replace('\\\\n', '\\n')
        
        # Ensure proper quoting for JSON keys
        cleaned_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned_text)
        
        return cleaned_text
    
    def _simplify_tables_in_json(self, json_text: str) -> str:
        """Identify and simplify complex tables in JSON text to prevent parsing issues."""
        import re
        
        # Identify table patterns in the JSON
        table_pattern = r'\|\s*[-]+\s*\|'
        table_matches = re.finditer(table_pattern, json_text)
        
        # Track positions of tables to modify
        for match in table_matches:
            # Find the start of the table (going backward from the header separator row)
            table_pos = match.start()
            table_start = json_text.rfind('|', 0, table_pos)
            if table_start == -1:
                table_start = max(0, table_pos - 200)  # Look back a reasonable amount
            
            # Find the end of the table
            table_end = json_text.find('\n\n', table_pos)
            if table_end == -1:
                table_end = min(len(json_text), table_pos + 1000)  # Look ahead a reasonable amount
            
            # Extract the table
            table_text = json_text[table_start:table_end]
            
            # Count columns and rows
            lines = table_text.split('\n')
            if len(lines) > 3:  # Header, separator, and at least one data row
                # Count columns in first row
                columns = len([col for col in lines[0].split('|') if col.strip()])
                rows = len(lines) - 2  # Subtract header and separator rows
                
                # Only simplify complex tables
                if columns > 6 or rows > 10:
                    # Create a simplified version
                    simplified_table = self._simplify_table(lines, columns, rows)
                    
                    # Replace in the original text
                    json_text = json_text[:table_start] + simplified_table + json_text[table_end:]
        
        return json_text
    
    def _simplify_table(self, table_lines, columns, rows):
        """Simplify a complex table to reduce formatting issues."""
        # Keep the header and separator
        header = table_lines[0]
        separator = table_lines[1]
        
        # For tables with many columns, reduce to essential columns
        if columns > 6:
            # Simplify header and separator to fewer columns
            header_parts = [part for part in header.split('|') if part.strip()]
            sep_parts = [part for part in separator.split('|') if part.strip()]
            
            # Select important columns (first, second, and last few)
            selected_headers = header_parts[:2] + ["..."] + header_parts[-2:] if len(header_parts) > 4 else header_parts
            selected_seps = sep_parts[:2] + ["---"] + sep_parts[-2:] if len(sep_parts) > 4 else sep_parts
            
            # Rebuild the header and separator
            header = "| " + " | ".join(selected_headers) + " |"
            separator = "| " + " | ".join(selected_seps) + " |"
        
        # For tables with many rows, keep only the first few and last few
        if rows > 10:
            data_rows = table_lines[2:12]  # First 10 data rows
            
            # Add a note about omitted rows
            note_row = f"| *...{rows-10} more rows omitted...* |"
            if columns > 1:
                note_row = "| " + " | ".join(["*...omitted...*"] * (min(columns, 5))) + " |"
            
            # Combine everything
            return '\n'.join([header, separator] + data_rows + [note_row])
        else:
            # Keep all rows but with simplified columns
            return '\n'.join([header, separator] + table_lines[2:min(len(table_lines), 12)])
        
    def _extract_answer_from_raw_text(self, raw_text: str) -> str:
        """Extract just the answer portion from raw text when JSON parsing fails."""
        # Look for the "answer" section in the text
        import re
        
        # Try to find content between answer key and the next key
        answer_pattern = r'"answer"\s*:\s*"(.*?)(?:",\s*"[a-zA-Z_][a-zA-Z0-9_]*"\s*:|\})'
        answer_match = re.search(answer_pattern, raw_text, re.DOTALL)
        
        if answer_match:
            # Extract just the answer text
            answer = answer_match.group(1)
            # Unescape if needed
            answer = answer.replace('\\n', '\n').replace('\\"', '"')
            return answer
        
        # If that fails, just return a cleaned version of the whole text
        # Remove markdown code block markers and other JSON syntax
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        cleaned_text = re.sub(r'^\s*\{.*?\"answer\":\s*\"', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\",\s*\".*$', '', cleaned_text, flags=re.DOTALL)
        
        # If still too messy, extract just the text content
        if '{' in cleaned_text or '}' in cleaned_text:
            # Take everything except JSON syntax characters
            text_only = re.sub(r'[{}",:]', ' ', raw_text)
            return ' '.join(text_only.split())
        
        return cleaned_text
        
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for generation prompt."""
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            # Format metadata
            metadata_str = ""
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                metadata_items = []
                if metadata.get('chunk_type'):
                    metadata_items.append(f"نوع المستند: {metadata['chunk_type']}")
                if metadata.get('decision_number'):
                    metadata_items.append(f"رقم القرار: {metadata['decision_number']}")
                if metadata.get('year'):
                    metadata_items.append(f"السنة: {metadata['year']}")
                if metadata.get('official_bulletin'):
                    metadata_items.append(f"النشرة الرسمية: {metadata['official_bulletin']}")
                
                # Helper function to format table data
                def format_table_data(table_data):
                    if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
                        return []
                    
                    formatted_rows = []
                    # Get headers from first row
                    headers = [str(h).strip() for h in table_data[0]]
                    if not headers:
                        return []
                        
                    # Create header and separator rows
                    formatted_rows.append("| " + " | ".join(headers) + " |")
                    formatted_rows.append("| " + " | ".join(["---" for _ in headers]) + " |")
                    
                    # Process data rows
                    for row in table_data[1:]:
                        # Convert all cells to strings and ensure proper length
                        cells = [str(cell).strip() if cell is not None else "" for cell in row]
                        # Pad with empty strings if needed
                        cells.extend(["" for _ in range(len(headers) - len(cells))])
                        # Join cells and ensure no newlines
                        formatted_rows.append("| " + " | ".join(cells).replace("\n", " ") + " |")
                    
                    return formatted_rows
                
                # Handle matched_articles if present
                if metadata.get('matched_articles') and len(metadata['matched_articles']) > 0:
                    metadata_items.append("\n### المواد المطابقة:")
                    # Create a markdown table for matched articles
                    article_rows = ["| رقم المادة | نص المادة | درجة التطابق |", "| ---------- | --------- | ------------ |"]
                    
                    for article in metadata['matched_articles']:
                        article_number = str(article.get('article_number', 'غير محدد')).strip()
                        # Include the full text without truncation
                        article_text = str(article.get('text', '')).strip().replace("\n", " ")
                        similarity_score = f"{article.get('similarity_score', 0):.3f}"
                        article_rows.append(f"| {article_number} | {article_text} | {similarity_score} |")
                    
                    metadata_items.extend(article_rows)
                    
                    # Handle article table data
                    for article in metadata['matched_articles']:
                        article_number = str(article.get('article_number', 'غير محدد')).strip()
                        if 'table_data' in article and article['table_data']:
                            metadata_items.append(f"\n**جداول البيانات للمادة {article_number}:**")
                            for table_idx, table in enumerate(article['table_data'], 1):
                                metadata_items.append(f"\n**جدول {table_idx}:**")
                                metadata_items.extend(format_table_data(table))
                
                # Handle matched_appendices if present
                if metadata.get('matched_appendices') and len(metadata['matched_appendices']) > 0:
                    metadata_items.append("\n### الملاحق المطابقة:")
                    # Create a markdown table for matched appendices
                    appendix_rows = ["| عنوان الملحق | نص الملحق | درجة التطابق |", "| ------------ | --------- | ------------ |"]
                    
                    for appendix in metadata['matched_appendices']:
                        appendix_title = str(appendix.get('title', 'بدون عنوان')).strip()
                        # Include the full text without truncation
                        appendix_text = str(appendix.get('text', '')).strip().replace("\n", " ")
                        similarity_score = f"{appendix.get('similarity_score', 0):.3f}"
                        appendix_rows.append(f"| {appendix_title} | {appendix_text} | {similarity_score} |")
                    
                    metadata_items.extend(appendix_rows)
                    
                    # Handle appendix table data
                    for idx, appendix in enumerate(metadata['matched_appendices'], 1):
                        if 'table_data' in appendix and appendix['table_data']:
                            metadata_items.append(f"\n**جداول البيانات لملحق {idx}:**")
                            for table_idx, table in enumerate(appendix['table_data'], 1):
                                metadata_items.append(f"\n**جدول {table_idx}:**")
                                metadata_items.extend(format_table_data(table))
                
                # Handle subsections if present
                if metadata.get('subsections'):
                    try:
                        # Parse subsections data
                        subsections = metadata['subsections']
                        if isinstance(subsections, str):
                            subsections = json.loads(subsections)
                        
                        # Handle articles in subsections
                        if subsections.get('articles'):
                            metadata_items.append("\n### المواد:")
                            article_rows = ["| رقم المادة | نص المادة |", "| ---------- | --------- |"]
                            
                            for article in subsections['articles']:
                                article_number = str(article.get('article_number', 'غير محدد')).strip()
                                # Include the full text without truncation
                                article_text = str(article.get('text', '')).strip().replace("\n", " ")
                                article_rows.append(f"| {article_number} | {article_text} |")
                            
                            metadata_items.extend(article_rows)
                            
                            # Handle article table data
                            for article in subsections['articles']:
                                article_number = str(article.get('article_number', 'غير محدد')).strip()
                                if 'table_data' in article and article['table_data']:
                                    metadata_items.append(f"\n**جداول البيانات للمادة {article_number}:**")
                                    for table_idx, table in enumerate(article['table_data'], 1):
                                        metadata_items.append(f"\n**جدول {table_idx}:**")
                                        metadata_items.extend(format_table_data(table))
                        
                        # Handle appendices in subsections
                        if subsections.get('appendices'):
                            metadata_items.append("\n### الملاحق:")
                            appendix_rows = ["| رقم الملحق | عنوان الملحق | نص الملحق |", "| ---------- | ------------ | --------- |"]
                            
                            for idx, appendix in enumerate(subsections['appendices'], 1):
                                appendix_title = str(appendix.get('title', 'بدون عنوان')).strip()
                                # Include the full text without truncation
                                appendix_text = str(appendix.get('text', '')).strip().replace("\n", " ")
                                appendix_rows.append(f"| {idx} | {appendix_title} | {appendix_text} |")
                            
                            metadata_items.extend(appendix_rows)
                            
                            # Handle appendix table data
                            for idx, appendix in enumerate(subsections['appendices'], 1):
                                if 'table_data' in appendix and appendix['table_data']:
                                    metadata_items.append(f"\n**جداول البيانات لملحق {idx}:**")
                                    for table_idx, table in enumerate(appendix['table_data'], 1):
                                        metadata_items.append(f"\n**جدول {table_idx}:**")
                                        metadata_items.extend(format_table_data(table))
                    
                    except (json.JSONDecodeError, TypeError) as e:
                        metadata_items.append(f"\n**تنبيه**: تعذر تحليل معلومات الأقسام الفرعية - {str(e)}")
                
                # Explicitly add full articles and appendices from the original document if they exist
                # but haven't been matched by the retriever
                if not metadata.get('matched_articles') and metadata.get('articles'):
                    try:
                        articles_data = []
                        if isinstance(metadata['articles'], str):
                            articles_data = json.loads(metadata['articles'])
                        elif isinstance(metadata['articles'], list):
                            articles_data = metadata['articles']
                        
                        if articles_data:
                            metadata_items.append("\n### جميع المواد في المستند:")
                            article_rows = ["| رقم المادة | نص المادة |", "| ---------- | --------- |"]
                            
                            for article in articles_data:
                                article_number = str(article.get('article_number', 'غير محدد')).strip()
                                # Include the full text without truncation
                                article_text = str(article.get('text', '')).strip().replace("\n", " ")
                                article_rows.append(f"| {article_number} | {article_text} |")
                            
                            metadata_items.extend(article_rows)
                    except (json.JSONDecodeError, TypeError) as e:
                        metadata_items.append(f"\n**تنبيه**: تعذر تحليل بيانات المواد الأصلية - {str(e)}")
                
                if not metadata.get('matched_appendices') and metadata.get('appendices'):
                    try:
                        appendices_data = []
                        if isinstance(metadata['appendices'], str):
                            appendices_data = json.loads(metadata['appendices'])
                        elif isinstance(metadata['appendices'], list):
                            appendices_data = metadata['appendices']
                        
                        if appendices_data:
                            metadata_items.append("\n### جميع الملاحق في المستند:")
                            appendix_rows = ["| رقم الملحق | عنوان الملحق | نص الملحق |", "| ---------- | ------------ | --------- |"]
                            
                            for idx, appendix in enumerate(appendices_data, 1):
                                appendix_title = str(appendix.get('title', 'بدون عنوان')).strip()
                                # Include the full text without truncation
                                appendix_text = str(appendix.get('text', '')).strip().replace("\n", " ")
                                appendix_rows.append(f"| {idx} | {appendix_title} | {appendix_text} |")
                            
                            metadata_items.extend(appendix_rows)
                    except (json.JSONDecodeError, TypeError) as e:
                        metadata_items.append(f"\n**تنبيه**: تعذر تحليل بيانات الملاحق الأصلية - {str(e)}")
                
                # Handle top-level tables if present
                if metadata.get('tables'):
                    try:
                        tables = metadata['tables']
                        if isinstance(tables, str):
                            tables = json.loads(tables)
                        
                        if tables and isinstance(tables, list):
                            metadata_items.append("\n### الجداول:")
                            for table_idx, table in enumerate(tables, 1):
                                metadata_items.append(f"\n**جدول {table_idx}:**")
                                metadata_items.extend(format_table_data(table))
                    
                    except (json.JSONDecodeError, TypeError) as e:
                        metadata_items.append(f"\n**تنبيه**: تعذر تحليل الجداول - {str(e)}")
                
                if metadata_items:
                    metadata_str = "\n\n## معلومات إضافية:\n" + "\n".join(metadata_items)
            
            # Format document with metadata
            formatted_docs.append(
                f"Document {i}:\n{doc.page_content}{metadata_str}\n"
            )
        return "\n".join(formatted_docs) 
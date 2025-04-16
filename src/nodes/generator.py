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
        self.parser = JsonOutputParser()
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

            ضع في اعتبارك أن جميع الإجابات يجب أن تكون باللغة العربية بشكل أساسي وتستخدم المصطلحات العربية بشكل صحيح. يجب تقديم الإجابات بلغة عربية سليمة وواضحة مع مراعاة قواعد اللغة العربية الصحيحة.

            EXAMPLE OF PROPERLY FORMATTED RESPONSE FOR REFERENCE:
            ```json
            {{{{
                "answer": "# Quantum Computing Fundamentals\\n\\n## Executive Summary\\n\\nQuantum computing represents a paradigm shift in computational technology, leveraging quantum mechanical phenomena to achieve processing capabilities beyond classical computers for specific problems. The documents provide comprehensive information on core quantum principles, technical implementations, current limitations, and future applications.\\n\\n## Quantum Computing Basics\\n\\nQuantum computing leverages the principles of quantum mechanics to process information in ways that classical computers cannot. The fundamental unit is the **qubit**, which unlike classical bits, can exist in multiple states simultaneously due to superposition.\\n\\nKey concepts in quantum computing include:\\n\\n* **Superposition**: Qubits exist in multiple states at once, allowing quantum computers to process vast amounts of possibilities simultaneously\\n* **Entanglement**: When qubits become correlated, sharing quantum states regardless of distance, creating powerful computational connections\\n* **Quantum Gates**: Operations that manipulate qubit states, similar to how classical computers use logic gates but with quantum mechanical properties\\n\\nUnlike classical computers that use transistors representing 0 or 1, quantum computers can represent many values simultaneously, offering exponential processing capability for certain problems.\\n\\n### Mathematical Foundation\\n\\nIn quantum computing, a qubit's state is represented mathematically as:\\n\\n$$|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$$\\n\\nWhere $\\alpha$ and $\\beta$ are complex numbers satisfying:\\n\\n$$|\\alpha|^2 + |\\beta|^2 = 1$$\\n\\nThis formula represents the probability amplitudes for measuring either 0 or 1. When multiple qubits are entangled, their states cannot be described independently, leading to the powerful phenomenon that enables quantum computing's advantages.\\n\\n## Applications of Quantum Computing\\n\\nQuantum computers excel at:\\n\\n1. **Factoring large numbers** - Breaking encryption using Shor's algorithm\\n2. **Quantum system simulation** - Modeling complex molecular and physical systems for drug discovery and material science\\n3. **Optimization problems** - Finding optimal solutions to complex problems with many variables\\n4. **Machine learning** - Enhancing certain algorithms with quantum speedup\\n\\n### Cryptographic Implications\\n\\nShor's algorithm represents a significant threat to current public-key cryptography. The algorithm can factor large numbers in polynomial time, which would break RSA encryption. The mathematical advantage comes from quantum Fourier transforms operating in superposition, as described by:\\n\\n$$\\text{{QFT}}|j\\rangle = \\frac{{1}}{{\\sqrt{{N}}}}\\sum_{{k=0}}^{{N-1}}e^{{2\\pi ijk/N}}|k\\rangle$$\\n\\nThis transformation allows quantum computers to find factors exponentially faster than classical algorithms.\\n\\n## Current Limitations\\n\\nDespite the theoretical promise, quantum computers face significant challenges:\\n\\n* **Decoherence** - Quantum states are extremely fragile and easily disturbed by environmental noise\\n* **Error rates** - Current quantum operations have high error rates requiring correction\\n* **Scalability** - Building machines with enough stable qubits remains difficult\\n\\n### Error Correction Strategies\\n\\nQuantum error correction codes work fundamentally differently from classical error correction. The approach involves encoding a logical qubit across multiple physical qubits to protect against errors. This requires techniques like surface codes that can detect and correct errors without collapsing the quantum state through measurement.\\n\\n## Quantum Programming\\n\\nMost quantum algorithms are implemented using specialized languages and frameworks:\\n\\n```python\\n# Simplified representation of qubit states\\nfrom qiskit import QuantumCircuit\\nqc = QuantumCircuit(2, 2)\\nqc.h(0)  # Put qubit 0 in superposition\\nqc.cx(0, 1)  # Entangle qubits 0 and 1\\n```\\n\\nResearchers are actively working on error correction codes and more stable qubit designs. Despite current limitations, quantum computing shows enormous potential for revolutionizing fields from cryptography to drug discovery and materials science in the coming decades.\\n\\n## Comparison: Classical vs. Quantum Computing\\n\\n| Aspect | Classical Computing | Quantum Computing |\\n|--------|-------------------|-------------------|\\n| Basic unit | Bit (0 or 1) | Qubit (superposition of 0 and 1) |\\n| Processing | Sequential | Parallel through superposition |\\n| Algorithms | Deterministic | Probabilistic |\\n| Error handling | Robust | Highly susceptible to noise |\\n| Current status | Mature technology | Emerging technology |\\n| Best applications | General-purpose computing | Specialized problems (factoring, simulation) |\\n\\nThis comparison highlights the complementary nature of classical and quantum approaches, with each excelling in different domains.\\n\\n## References\\n\\n* Document 1 - Basic quantum computing principles and applications\\n* Document 2 - Quantum programming and entanglement concepts\\n* Document 3 - Current limitations and error correction in quantum systems",
                "confidence_score": 0.87,
                "supporting_evidence": [
                    "Document 1: 'Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously due to superposition.'",
                    "Document 2: 'Unlike classical bits, qubits can be entangled, allowing them to share quantum states regardless of distance.'",
                    "Document 3: 'Quantum gates manipulate qubit states, similar to how logic gates process classical bits.'",
                    "Document 1: 'Quantum computers excel at factoring large numbers, simulating quantum systems, and solving certain optimization problems.'",
                    "Document 3: 'Current quantum computers are limited by high error rates and quantum decoherence, requiring error correction techniques.'"
                ],
                "reasoning_path": "Started with fundamental quantum computing concepts, explained key quantum principles, detailed applications, outlined current limitations, provided programming examples, and discussed future prospects - covering all essential aspects mentioned in the documents.",
                "suggested_followup": [
                    "What are the most promising approaches to solving the quantum decoherence problem?",
                    "How does Shor's algorithm use quantum computing for integer factorization?",
                    "What companies are currently leading in quantum computing hardware development?"
                ],
                "metadata": {{{{
                    "sources_used": 3,
                    "key_concepts": ["qubits", "superposition", "entanglement", "quantum gates", "quantum limitations"],
                    "confidence_factors": ["Comprehensive definitions across multiple documents", "Technical details with code examples", "Clear explanations of limitations", "Limited coverage of recent developments"]
                }}}},
                "validation": {{{{
                    "has_hallucinations": false,
                    "answers_question": true,
                    "quality_score": 0.85,
                    "improvement_needed": [
                        "Documents lack information on recent quantum hardware developments",
                        "No comparison between different quantum computing approaches is available in the documents"
                    ],
                    "validation_reasoning": "The answer accurately explains quantum computing fundamentals based solely on the provided documents. All technical concepts are directly supported by specific document references. The code example is from Document 2. The answer acknowledges limitations mentioned in Document 3. There are no statements that go beyond what's explicitly in the documents."
                }}}}
            }}}}
            ```
            """),
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
                "current_node": "generator"
            })
            return state
        
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
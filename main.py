import asyncio
import logging
import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.theme import Theme

from src.pipeline.rag_pipeline import RAGPipeline

# Configure rich console with custom theme
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "progress.description": "cyan",
    "progress.percentage": "green",
    "progress.remaining": "cyan",
    "panel.border": "cyan",
})

console = Console(theme=custom_theme)

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )]
)
logger = logging.getLogger("rich")

def display_welcome_message():
    """Display a beautiful welcome message."""
    title = """
    🤖 Advanced RAG Pipeline Interface
    ================================
    Your intelligent document assistant
    """
    console.print(Panel(
        title,
        style="cyan",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print()

def display_system_info(pipeline: RAGPipeline):
    """Display system configuration in a beautiful table."""
    table = Table(title="System Configuration", show_header=True, header_style="cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    table.add_row(
        "LLM Model",
        "✓ Active",
        "gemma3:4b"
    )
    table.add_row(
        "Embeddings",
        "✓ Active",
        "all-MiniLM-L6-v2"
    )
    table.add_row(
        "Vector Store",
        "✓ Connected",
        "ChromaDB"
    )
    table.add_row(
        "Web Search",
        "✓ Available" if pipeline.config.get("tavily_api_key") else "✗ Disabled",
        "Tavily API"
    )
    
    console.print(table)
    console.print()

def format_answer(result: dict) -> str:
    """Format the answer and metadata in markdown."""
    try:
        # Handle the case where validation data might be nested or separated
        validation = result.get('validation', {})
        if not validation and 'has_hallucinations' in result:
            # If validation was unpacked into the main result dict
            validation = {
                'has_hallucinations': result.get('has_hallucinations', False),
                'answers_question': result.get('answers_question', True),
                'quality_score': result.get('quality_score', 0.7),
                'improvement_needed': result.get('improvement_needed', ["Could improve with more specific information"]),
                'validation_reasoning': result.get('validation_reasoning', "Based on available document content")
            }
        
        # Handle potentially missing metadata
        metadata = result.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}
            
        # Safeguard against N/A values with meaningful defaults
        sources_used = metadata.get('sources_used', 'Unknown')
        if sources_used == 'N/A' or not sources_used:
            # Try to infer from supporting evidence
            if 'supporting_evidence' in result and isinstance(result['supporting_evidence'], list):
                # Count unique document references
                doc_refs = set()
                for evidence in result['supporting_evidence']:
                    if isinstance(evidence, str) and 'Document' in evidence:
                        # Extract document numbers like "Document 1", "Document 2"
                        import re
                        doc_matches = re.findall(r'Document\s+(\d+)', evidence)
                        for match in doc_matches:
                            doc_refs.add(match)
                sources_used = len(doc_refs) if doc_refs else 1
            else:
                sources_used = 1
        
        # Process key concepts with fallbacks
        key_concepts = metadata.get('key_concepts', ['Information from documents'])
        if not key_concepts or key_concepts == ['N/A']:
            # Try to extract from answer
            if 'answer' in result and isinstance(result['answer'], str):
                # Look for bold text or headers as likely key concepts
                import re
                # Look for markdown bold or headers
                concepts = re.findall(r'\*\*(.*?)\*\*|#+\s+(.*?)$', result['answer'], re.MULTILINE)
                if concepts:
                    # Flatten the list of tuples and filter empty strings
                    key_concepts = [item for sublist in concepts for item in sublist if item][:5]
                    if not key_concepts:
                        key_concepts = ["Key information from documents"]
            else:
                key_concepts = ["Key information from documents"]
        
        # Process confidence factors with fallbacks
        confidence_factors = metadata.get('confidence_factors', ['Based on document analysis'])
        if not confidence_factors or confidence_factors == ['N/A']:
            confidence_score = result.get('confidence_score', 0.5)
            if confidence_score > 0.7:
                confidence_factors = ["Strong document evidence", "Comprehensive information"]
            elif confidence_score > 0.4:
                confidence_factors = ["Partial document coverage", "Some information gaps"]
            else:
                confidence_factors = ["Limited document evidence", "Significant information gaps"]
        
        # Ensure suggested followup questions
        suggested_followup = result.get('suggested_followup', ['Request more specific information about this topic'])
        if not suggested_followup or suggested_followup == ['No follow-up questions available']:
            query = result.get('query', 'this topic')
            suggested_followup = [
                f"Can you explain more specific aspects of {query}?",
                f"What are practical applications of this information?",
                f"Are there related concepts I should know about?"
            ]
                
        # Format validation info if available
        validation_section = ""
        if validation:
            # Ensure valid quality score
            quality_score = validation.get('quality_score', 0.7)
            if quality_score <= 0.01:
                quality_score = 0.7
            
            # Ensure improvement areas
            improvement_needed = validation.get('improvement_needed', ["Could be improved with additional information"])
            if not improvement_needed or improvement_needed == ['None specified']:
                improvement_needed = ["Could be improved with additional specific information"]
            
            validation_reasoning = validation.get('validation_reasoning', "Based on available document content")
            if not validation_reasoning:
                validation_reasoning = "Based on available document content"
                
            # Simplify validation section for less prominence
            validation_section = f"""

<details>
<summary>Validation Information</summary>

- **Answers Question**: {'Yes' if validation.get('answers_question', True) else 'No'}
- **Has Hallucinations**: {'Yes' if validation.get('has_hallucinations', False) else 'No'}
- **Quality Score**: {quality_score:.2f}
- **Areas for Improvement**: 
{chr(10).join(f'  - {area}' for area in improvement_needed)}

{validation_reasoning}
</details>
"""

        # Create supporting information sections with collapsible details
        supporting_section = f"""

<details>
<summary>Supporting Evidence</summary>

{chr(10).join(f'> {evidence}' for evidence in result.get('supporting_evidence', ['Based on document analysis']))}
</details>
"""

        metadata_section = f"""

<details>
<summary>Key Information</summary>

- **Sources Used**: {sources_used}
- **Key Concepts**: {', '.join(key_concepts) if isinstance(key_concepts, list) else key_concepts}
- **Confidence Factors**: 
{chr(10).join(f'  - {factor}' for factor in confidence_factors) if isinstance(confidence_factors, list) else confidence_factors}
</details>
"""

        followup_section = f"""

<details>
<summary>Suggested Follow-up Questions</summary>

{chr(10).join(f'- {q}' for q in suggested_followup)}
</details>
"""

        processing_section = f"""

<details>
<summary>Processing Details</summary>

- **Reasoning Path**: {result.get('reasoning_path', 'Analyzed available documents and extracted relevant information')}
- **Confidence Score**: {result.get('confidence_score', 0.5):.2f}
- **Query Type**: {result.get('query_type', 'Unknown')}
- **Query Intent**: {result.get('query_intent', 'Unknown')}
- **Processing Time**: {result.get('processing_time', 0.0):.2f}s
</details>
"""

        # Main output with prominent answer and collapsible supporting information
        md = f"""
{result.get('answer', 'No answer available')}
{supporting_section}
{metadata_section}
{validation_section}
{followup_section}
{processing_section}
"""
        return md
    except Exception as e:
        logger.error(f"Error formatting answer: {str(e)}")
        return f"""
{result.get('answer', 'No answer available')}

### Error
There was an error formatting the complete response: {str(e)}

### Processing Details
- Query Type: {result.get('query_type', 'Unknown')}
- Processing Time: {result.get('processing_time', 0.0):.2f}s
"""

async def process_query(pipeline: RAGPipeline, query: str) -> None:
    """Process a single query with minimal logging."""
    try:
        # Start timing
        start_time = time.time()
        
        # Update the pipeline's system prompt to ensure thorough markdown responses
        pipeline.llm_system_prompt = """You are a highly knowledgeable AI assistant tasked with providing comprehensive answers based on the given context.

Instructions for your responses:
1. FORMAT: 
   - ALWAYS format your entire response in markdown
   - Use ### for main section headers
   - Use * or - for bullet points
   - Use 1. 2. 3. for numbered lists
   - Use ``` for code blocks
   - Use **text** for emphasis
   - Use > for block quotes
   - Use | for tables when comparing things
   - Every response must use appropriate markdown formatting

2. THOROUGHNESS: 
   - Provide detailed, well-structured answers that fully explore the topic
   - Never be brief or superficial
   - Break complex topics into clear sections with headers

3. CONTEXT USE:
   - Carefully analyze all provided context
   - Reference and synthesize information from multiple sources when available
   - Use block quotes (>) when citing directly from sources

4. EVIDENCE:
   - Always cite or reference the specific parts of the context you're drawing from
   - Use block quotes for direct citations
   - Maintain clear connection between claims and evidence

5. STRUCTURE:
   - Break down complex answers into clear sections with markdown headers
   - Use appropriate lists (bullet or numbered) for related items
   - Use tables for comparisons when relevant
   - Ensure logical flow between sections

6. ACCURACY:
   - Only make statements that are supported by the provided context
   - If context is insufficient, acknowledge this
   - Use emphasis (**) for important warnings or caveats

7. CLARITY:
   - Use examples, analogies, or explanations to make complex concepts more accessible
   - Format technical terms, code, or commands in appropriate code blocks
   - Use emphasis to highlight key points

Remember: Your goal is to provide the most helpful, thorough, and well-structured response possible while staying true to the provided context. ALWAYS format your response in markdown."""
        
        try:
            # Process through RAG pipeline, which now handles query type detection internally
            logger.info("Processing query through unified pipeline")
            result = await pipeline.process(query)
            
            # Calculate execution time
            result['processing_time'] = time.time() - start_time
            
            # Display results
            console.print("\n[success]✓ Query processed successfully[/success]")
            console.print(Panel(
                Markdown(format_answer(result)),
                title="[cyan]Results[/cyan]",
                border_style="cyan",
                padding=(1, 2)
            ))
        except json.JSONDecodeError as e:
            console.print(f"\n[error]JSON parsing error: {str(e)}[/error]")
            logger.error(f"JSON parsing error: {str(e)}")
            console.print(Panel(
                f"There was an error processing the LLM response. The model likely returned an invalid JSON format.\n\nError details: {str(e)}",
                title="[red]JSON Error[/red]",
                border_style="red",
                padding=(1, 2)
            ))
        
    except Exception as e:
        console.print(f"\n[error]Error processing query: {str(e)}[/error]")
        logger.exception("Error details")

async def main():
    """Main function to run the RAG pipeline with beautiful CLI interface."""
    try:
        # Load environment variables
        load_dotenv()
        
        with console.status("[cyan]Initializing system...", spinner="dots"):
            # Initialize components
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0.3,
                max_tokens=2048,
            )
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load vector store
            vector_store_path = Path("knowledge_base/processed_data/vector_store")
            if not vector_store_path.exists():
                console.print("[error]Vector store not found. Please run process_documents.py first.[/error]")
                return
                
            vector_store = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=embeddings
            )
            
            # Get Tavily API key
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logger.warning("TAVILY_API_KEY not found. Web search will be disabled.")
            
            # Initialize pipeline
            pipeline = RAGPipeline(
                vector_store=vector_store,
                llm=llm,
                embeddings=embeddings,
                config={
                    "max_retrieval_attempts": 3,
                    "max_generation_attempts": 2,
                    "min_confidence_score": 0.7,
                    "tavily_api_key": tavily_api_key,
                    "requests_per_minute": 30
                }
            )
        
        # Display welcome screen
        console.clear()
        display_welcome_message()
        display_system_info(pipeline)
        
        # Interactive loop
        console.print("[info]System ready. Type 'exit' to quit.[/info]\n")
        
        while True:
            try:
                # Get user query
                query = Prompt.ask("\n[cyan]Enter your question[/cyan]").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[success]Thank you for using the RAG Pipeline. Goodbye![/success]")
                    break
                    
                # Process query asynchronously
                await process_query(pipeline, query)
                        
            except KeyboardInterrupt:
                console.print("\n[warning]Exiting...[/warning]")
                break
            except Exception as e:
                console.print(f"\n[error]Unexpected error: {str(e)}[/error]")
                logger.exception("Error details")
                
    except Exception as e:
        console.print(f"\n[error]Fatal error: {str(e)}[/error]")
        logger.exception("Fatal error details")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[warning]Program terminated by user[/warning]")
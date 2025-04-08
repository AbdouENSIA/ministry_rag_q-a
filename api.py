import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from src.pipeline.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API", version="1.0.0", 
              description="Retrieval Augmented Generation API with strict adherence to source documents")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=2048,
)

embeddings = HuggingFaceEmbeddings(
    model_name="aubmindlab/bert-base-arabertv02"
)

# Load vector store
vector_store_path = Path("knowledge_base/processed_data/vector_store")
if not vector_store_path.exists():
    raise ValueError(f"Vector store not found at {vector_store_path}. Please run process_documents.py first.")

vector_store = Chroma(
    persist_directory=str(vector_store_path),
    embedding_function=embeddings
)

# Initialize pipeline
pipeline = RAGPipeline(
    vector_store=vector_store,
    llm=llm,
    embeddings=embeddings,
    config={
        "max_retrieval_attempts": 2,  # Reduced from 3 to align with graph settings
        "max_generation_attempts": 1,  # Reduced from 2 to align with graph settings
        "min_confidence_score": 0.9,
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "requests_per_minute": 30
    }
)

class QueryRequest(BaseModel):
    query: str
    include_source_documents: bool = Field(default=False, description="Whether to include the actual source documents in the response")

class ValidationInfo(BaseModel):
    has_hallucinations: bool = Field(default=False, description="Whether the answer contains hallucinations")
    answers_question: bool = Field(default=True, description="Whether the answer addresses the query")
    quality_score: float = Field(default=0.0, description="Quality score of the answer (0.01-0.99)")
    improvement_needed: List[str] = Field(default_factory=list, description="Areas where the answer could be improved")
    validation_reasoning: str = Field(default="", description="Reasoning behind the validation")

class MetadataInfo(BaseModel):
    sources_used: int = Field(default=0, description="Number of sources used in the answer")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts from the documents")
    confidence_factors: List[str] = Field(default_factory=list, description="Factors affecting confidence in the answer")

class QueryResponse(BaseModel):
    answer: str
    supporting_evidence: List[str]
    reasoning_path: Optional[str] = None
    confidence_score: float
    query_type: str
    query_intent: str
    processing_time: float
    metadata: MetadataInfo
    suggested_followup: List[str]
    validation: ValidationInfo
    source_documents: Optional[List[Dict[str, Any]]] = None

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "واجهة برمجة تطبيقات RAG قيد التشغيل", 
        "version": "1.0.0",
        "strict_mode": "الإجابات مقتصرة بشكل صارم على المعلومات الموجودة في المستندات المسترجعة",
        "language": "Arabic",
        "rtl_support": True
    }

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        start_time = time.time()
        
        # Process the query
        result = await pipeline.process(request.query)
        
        # Add processing time to result
        result["processing_time"] = time.time() - start_time
        
        # Format and validate the response
        formatted_result = format_response(result, include_source_documents=request.include_source_documents)
        
        return formatted_result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_response(result: Dict[str, Any], include_source_documents: bool = False) -> Dict[str, Any]:
    """Format and validate the response to ensure it matches the QueryResponse model."""
    
    # Extract validation data if it's embedded in the result
    validation = {
        'has_hallucinations': result.get('has_hallucinations', False),
        'answers_question': result.get('answers_question', True),
        'quality_score': result.get('quality_score', 0.7),
        'improvement_needed': result.get('improvement_needed', ["Could be improved with more specific information"]),
        'validation_reasoning': result.get('validation_reasoning', "Based on available document content")
    }
    
    # Handle potentially missing metadata
    metadata = result.get('metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    
    # Process source documents if requested
    source_documents = None
    if include_source_documents and result.get('documents'):
        # Convert Document objects to dict representation
        source_documents = []
        for doc in result.get('documents', []):
            # Only include page_content and metadata fields
            source_documents.append({
                'page_content': doc.page_content if hasattr(doc, 'page_content') else str(doc),
                'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
            })
    
    # Format final response
    response = {
        'answer': result.get('answer', 'No answer available'),
        'supporting_evidence': result.get('supporting_evidence', []),
        'reasoning_path': result.get('reasoning_path', ''),
        'confidence_score': result.get('confidence_score', 0.5),
        'query_type': result.get('query_type', 'unknown'),
        'query_intent': result.get('query_intent', 'unknown'),
        'processing_time': result.get('processing_time', 0.0),
        'metadata': {
            'sources_used': metadata.get('sources_used', 0),
            'key_concepts': metadata.get('key_concepts', []),
            'confidence_factors': metadata.get('confidence_factors', [])
        },
        'suggested_followup': result.get('suggested_followup', []),
        'validation': validation
    }
    
    # Add source documents if requested
    if include_source_documents:
        response['source_documents'] = source_documents
    
    return response

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
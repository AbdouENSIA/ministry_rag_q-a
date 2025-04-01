from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uvicorn
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

from src.pipeline.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=2048,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store
vector_store_path = Path("knowledge_base/processed_data/vector_store")
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
        "max_retrieval_attempts": 3,
        "max_generation_attempts": 2,
        "min_confidence_score": 0.7,
        "tavily_api_key": os.getenv("TAVILY_API_KEY"),
        "requests_per_minute": 30
    }
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    supporting_evidence: List[str]
    reasoning_path: str
    confidence_score: float
    quality_score: Optional[float]
    metadata: Dict
    suggested_followup: List[str]
    query_type: str
    query_intent: str
    processing_time: float

@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG API is running"}

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Update system prompt for markdown formatting
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
   - Break complex topics into clear sections with headers

3. CONTEXT USE:
   - Carefully analyze all provided context
   - Reference and synthesize information from multiple sources when available
   - Use block quotes (>) when citing directly from sources"""

        # Process the query
        result = await pipeline.process(request.query)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
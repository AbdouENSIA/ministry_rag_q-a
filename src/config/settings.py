from pydantic import BaseSettings, Field


class RAGSettings(BaseSettings):
    """Configuration settings for the RAG system."""
    
    # Vector store settings
    embedding_model: str = Field(..., description="Name or path of the embedding model")
    vector_store_path: str = Field(..., description="Path to vector store")
    
    # LLM settings
    llm_model: str = Field(..., description="Name or path of the language model")
    temperature: float = Field(0.7, description="Temperature for text generation")
    
    # Retrieval settings
    max_documents: int = Field(5, description="Maximum number of documents to retrieve")
    similarity_threshold: float = Field(0.7, description="Minimum similarity score for retrieval")
    
    # Web search settings
    search_api_key: str = Field(..., description="API key for web search")
    max_search_results: int = Field(3, description="Maximum number of web search results")
    
    class Config:
        env_file = ".env"
        case_sensitive = False 
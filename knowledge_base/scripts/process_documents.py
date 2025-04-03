import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and index documents from the raw_data directory."""
    
    def __init__(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 2500,
        chunk_overlap: int = 750
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create processed data directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def process_documents(self) -> None:
        """Process all documents in the raw_data directory."""
        # Get all PDF files
        pdf_files = list(self.raw_data_dir.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_data_dir}")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each file
        for pdf_file in pdf_files:
            try:
                self._process_file(pdf_file)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                
    def _process_file(self, file_path: Path) -> None:
        """Process a single PDF file."""
        logger.info(f"Processing {file_path}")
        
        # Load PDF
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Add metadata
        chunks = self._add_metadata(chunks, file_path)
        
        # Save chunks to processed data directory
        self._save_chunks(chunks, file_path)
        
        # Create or update vector store
        self._update_vector_store(chunks)
        
        logger.info(f"Successfully processed {file_path}")
        
    def _add_metadata(self, chunks: List[Document], file_path: Path) -> List[Document]:
        """Add metadata to document chunks."""
        for chunk in chunks:
            chunk.metadata.update({
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix
            })
        return chunks
        
    def _save_chunks(self, chunks: List[Document], file_path: Path) -> None:
        """Save processed chunks to disk."""
        # Create output directory structure
        rel_path = file_path.relative_to(self.raw_data_dir)
        output_dir = self.processed_data_dir / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        output_file = output_dir / f"{file_path.stem}_chunks.json"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.json() + "\n")
                
    def _update_vector_store(self, chunks: List[Document]) -> None:
        """Update the vector store with new chunks."""
        # Initialize or load vector store
        vector_store_path = self.processed_data_dir / "vector_store"
        if vector_store_path.exists():
            vector_store = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=self.embeddings
            )
            vector_store.add_documents(chunks)
        else:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(vector_store_path)
            )
        vector_store.persist()

def main():
    """Main function to process documents."""
    # Get paths
    current_dir = Path(__file__).parent.parent
    raw_data_dir = current_dir / "raw_data"
    processed_data_dir = current_dir / "processed_data"
    
    # Initialize processor
    processor = DocumentProcessor(
        raw_data_dir=str(raw_data_dir),
        processed_data_dir=str(processed_data_dir)
    )
    
    # Process documents
    processor.process_documents()

if __name__ == "__main__":
    main() 
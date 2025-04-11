import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import re
import argparse
import os
import json
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process and index DOCX documents from the raw_data directory."""
    
    def __init__(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        embedding_model: str = "aubmindlab/bert-base-arabertv02",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        language: str = "arabic"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        
        # Initialize embeddings with Arabic-specific model if language is Arabic
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize text splitter with Arabic-friendly settings
        separators = self._get_language_specific_separators()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
        
        # Create processed data directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_language_specific_separators(self) -> List[str]:
        """Get language-specific separators for text splitting."""
        if self.language.lower() == "arabic":
            # Arabic-specific separators including Arabic punctuation
            return ["\n\n", "\n", ".", "!", "؟", ":", "؛", " ", ""]
        else:
            # Default separators for other languages
            return ["\n\n", "\n", " ", ""]
    
    def _clean_arabic_text(self, text: str) -> str:
        """Clean Arabic text by handling special characters and normalization."""
        # Replace multiple Arabic commas with a single one
        text = re.sub(r'،+', '،', text)
        # Replace multiple periods with a single one
        text = re.sub(r'\.+', '.', text)
        # Remove sequences of underscores
        text = re.sub(r'_+', '', text)
        # Remove Arabic tatweel (elongation character)
        text = re.sub(r'ـ+', '', text)
        # Remove Quranic citation marks
        text = re.sub(r'﴾|﴿', '', text)
        # Remove Arabic diacritics
        text = re.sub(r'[ًٌٍَُِّْٰٓ]', '', text)
        # Remove '*-*' patterns
        text = re.sub(r'\*-\*', '', text)
        
        if self.language.lower() == "arabic":
            # Normalize alef variants to simple alef
            text = re.sub(r'[إأآا]', 'ا', text)
            
            # Normalize yeh variants
            text = re.sub(r'[يى]', 'ي', text)
        
        return text
    
    def extract_text_from_docx(self, path: Path) -> List[Dict[str, Any]]:
        """Extract text from DOCX file with metadata per page."""
        try:
            doc = DocxDocument(str(path))
            
            # Extract document properties when available
            doc_properties = {
                "title": getattr(doc.core_properties, "title", "") or "",
                "author": getattr(doc.core_properties, "author", "") or "",
                "created": getattr(doc.core_properties, "created", "") or "",
                "modified": getattr(doc.core_properties, "modified", "") or "",
                "language": self.language,
                "source": str(path),
                "file_name": path.name,
                "file_type": "docx"
            }
            
            # Extract text
            full_text = ""
            
            for child in doc.element.body.iterchildren():
                if child.tag.endswith('}p'):
                    para = Paragraph(child, doc)
                    raw_text = para.text.strip()
                    if raw_text:
                        full_text += self._clean_arabic_text(raw_text) + "\n\n"
                elif child.tag.endswith('}tbl'):
                    table = Table(child, doc)
                    for row in table.rows:
                        row_text = " | ".join([self._clean_arabic_text(cell.text.strip()) 
                                              for cell in row.cells if cell.text.strip()])
                        if row_text:
                            full_text += row_text + "\n"
                    full_text += "\n"
            
            # Create a document with the extracted text
            return [{"page_content": full_text.strip(), "metadata": doc_properties}]
            
        except Exception as e:
            logger.error(f"Error extracting text from {path}: {str(e)}")
            raise
        
    def process_documents(self) -> None:
        """Process all DOCX documents in the raw_data directory."""
        # Get all DOCX files
        docx_files = list(self.raw_data_dir.glob("**/*.docx"))
        
        if not docx_files:
            logger.warning(f"No DOCX files found in {self.raw_data_dir}")
            return
            
        logger.info(f"Found {len(docx_files)} DOCX files to process")
        
        # Process each file
        all_chunks = []
        for docx_file in docx_files:
            try:
                file_chunks = self._process_file(docx_file)
                all_chunks.extend(file_chunks)
            except Exception as e:
                logger.error(f"Error processing {docx_file}: {str(e)}")
        
        # Update vector store with all chunks at once for better efficiency
        if all_chunks:
            self._update_vector_store(all_chunks)
                
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single DOCX file."""
        logger.info(f"Processing {file_path}")
        
        # Extract text from DOCX
        raw_documents = self.extract_text_from_docx(file_path)
        
        # Convert to LangChain Document objects
        documents = [Document(**doc) for doc in raw_documents]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Save chunks to processed data directory
        self._save_chunks(chunks, file_path)
        
        logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")
        return chunks
        
    def _save_chunks(self, chunks: List[Document], file_path: Path) -> None:
        """Save processed chunks to disk."""
        # Create output directory structure
        try:
            rel_path = file_path.relative_to(self.raw_data_dir)
        except ValueError:
            # Fall back to using just the file name if it's not under raw_data_dir
            rel_path = Path(file_path.name)
            
        output_dir = self.processed_data_dir / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        output_file = output_dir / f"{file_path.stem}_chunks.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                # Convert Document to dict for JSON serialization
                chunk_dict = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False, default=str) + "\n")
                
    def _update_vector_store(self, chunks: List[Document]) -> None:
        """Update the vector store with new chunks."""
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return
            
        # Initialize or load vector store
        vector_store_path = self.processed_data_dir / "vector_store"
        
        try:
            if vector_store_path.exists():
                logger.info(f"Updating existing vector store at {vector_store_path}")
                vector_store = Chroma(
                    persist_directory=str(vector_store_path),
                    embedding_function=self.embeddings
                )
                vector_store.add_documents(chunks)
            else:
                logger.info(f"Creating new vector store at {vector_store_path}")
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=str(vector_store_path)
                )
            vector_store.persist()
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")

def main():
    """Main function to process documents."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process DOCX documents for RAG system")
    parser.add_argument("--raw-data-dir", type=str, help="Directory containing raw DOCX files")
    parser.add_argument("--processed-data-dir", type=str, help="Directory to store processed data")
    parser.add_argument("--language", type=str, default="arabic", help="Language of documents (default: arabic)")
    parser.add_argument("--embedding-model", type=str, default="aubmindlab/bert-base-arabertv02", 
                        help="HuggingFace embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=800, 
                        help="Size of text chunks for processing")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                        help="Overlap between text chunks")
    args = parser.parse_args()
    
    # Get paths
    current_dir = Path(__file__).parent.parent
    raw_data_dir = args.raw_data_dir if args.raw_data_dir else current_dir / "raw_data"
    processed_data_dir = args.processed_data_dir if args.processed_data_dir else current_dir / "processed_data"
    
    # Initialize processor
    processor = DocumentProcessor(
        raw_data_dir=str(raw_data_dir),
        processed_data_dir=str(processed_data_dir),
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        language=args.language
    )
    
    # Log language processing information
    logger.info(f"Processing documents in {args.language} language")
    logger.info(f"Using embedding model: {args.embedding_model}")
    logger.info(f"Chunk size: {args.chunk_size}, Chunk overlap: {args.chunk_overlap}")
    if args.language.lower() == "arabic":
        logger.info("Arabic language processing enabled with special text cleaning and normalization")
    
    # Process documents
    processor.process_documents()

if __name__ == "__main__":
    main()

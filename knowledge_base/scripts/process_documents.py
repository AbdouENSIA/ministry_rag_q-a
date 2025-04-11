import logging
from pathlib import Path
from typing import List
import re
import argparse
import os
import json
from docx import Document as dc
from docx.table import Table
from docx.text.paragraph import Paragraph

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
    
    def extract_cleaned_text_from_docx(self, path: str) -> str:
        doc = dc(path)
        text = ""

        def clean_text(text):
            text = re.sub(r'،+', '،', text)  # Replace multiple Arabic commas with a single one
            text = re.sub(r'\.+', '.', text)  # Replace multiple periods with a single one
            text = re.sub(r'_+', '', text)    # Remove sequences of underscores
            text = re.sub(r'ـ+', '', text)    # Remove Arabic tatweel (elongation character)
            text = re.sub(r'﴾|﴿', '', text)   # Remove Quranic citation marks
            text = re.sub(r'[ًٌٍَُِّْٰٓ]', '', text)  # Remove Arabic diacritics
            text = re.sub(r'\*-\*', '', text)  # Remove '*-*' patterns
            return text

        for child in doc.element.body.iterchildren():
            if child.tag.endswith('}p'):
                para = Paragraph(child, doc)
                raw_text = para.text.strip()
                if raw_text:
                    text += clean_text(raw_text) + "\n\n"
            elif child.tag.endswith('}tbl'):
                table = Table(child, doc)
                for row in table.rows:
                    row_text = " | ".join([clean_text(cell.text.strip()) for cell in row.cells if cell.text.strip()])
                    if row_text:
                        text += row_text + "\n"
                text += "\n"

        return text.strip()
        
    def process_documents(self) -> None:
        """Process all documents in the raw_data directory."""
        # Get all DOCX files
        docx_files = list(self.raw_data_dir.glob("**/*.docx"))
        
        if not docx_files:
            logger.warning(f"No DOCX files found in {self.raw_data_dir}")
            return
            
        logger.info(f"Found {len(docx_files)} DOCX files to process")
        
        # Process each file
        for docx_file in docx_files:
            try:
                self._process_file(docx_file)
            except Exception as e:
                logger.error(f"Error processing {docx_file}: {str(e)}")
                
    def _process_file(self, file_path: Path) -> None:
        """Process a single DOCX file."""
        logger.info(f"Processing {file_path}")

        text = self.extract_cleaned_text_from_docx(file_path)

        text_chunks = self.text_splitter.split_text(text)
        chunks = [{"id":None, "page_content": chunk, "metadata": {"source": file_path},"type":"Document"} for chunk in text_chunks]
        
        # Add metadata
        chunks = self._add_metadata(chunks, file_path)
        
        # Save chunks to processed data directory
        self._save_chunks(chunks, file_path)
        
        for chunk in chunks:
            if isinstance(chunk, dict):
                chunk.pop("id", None)
        
        clean_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                clean_chunks.append(Document(**chunk))
            else:
                clean_chunks.append(chunk)

        # Create or update vector store
        self._update_vector_store(clean_chunks)
        
        logger.info(f"Successfully processed {file_path}")
        
    def _add_metadata(self, chunks: List[Document], file_path: Path) -> List[Document]:
        """Add metadata to document chunks."""
        for chunk in chunks:
            chunk["metadata"].update({
                "source": str(file_path),
                "file_name": file_path.replace(".docx", ""),
                "file_type": "docx",
                "language": "arabic",
                "total_pages":558,
                "page_label":"1",
                "page":0 ,
                "producer":"www.ilovepdf.com",
                "creator":"Microsoft® Word 2016",
                "creationdate":"2025-04-03T13:04:52+00:00",
                "author":"ency-education.com",
                "moddate":"2025-04-03T13:05:03+00:00" 
            })
        return chunks
        
    def _save_chunks(self, chunks: List[Document], file_path: str) -> None:
        """Save processed chunks to disk."""
        raw_data_dir_str = str(self.raw_data_dir)
        if file_path.startswith(raw_data_dir_str):
            rel_path = os.path.relpath(file_path, raw_data_dir_str)
        else:
            rel_path = os.path.basename(file_path)
        
        rel_parent = os.path.dirname(rel_path)
        output_dir = os.path.join(str(self.processed_data_dir), rel_parent)
        os.makedirs(output_dir, exist_ok=True)        
        file_stem = os.path.splitext(os.path.basename(file_path))[0]
        
        output_file = os.path.join(output_dir, f"{file_stem}_chunks.json")

        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    parser.add_argument("--language", type=str, default="arabic", help="Language of documents (default: arabic)")
    parser.add_argument("--embedding-model", type=str, default="aubmindlab/bert-base-arabertv02", 
                        help="HuggingFace embedding model to use")
    args = parser.parse_args()
    
    # Get paths
    current_dir = Path(__file__).parent.parent
    raw_data_dir = current_dir / "raw_data"
    processed_data_dir = current_dir / "processed_data"
    
    # Initialize processor
    processor = DocumentProcessor(
        raw_data_dir=str(raw_data_dir),
        processed_data_dir=str(processed_data_dir),
        embedding_model=args.embedding_model,
        language=args.language
    )
    
    # Log language processing information
    logger.info(f"Processing documents in {args.language} language")
    logger.info(f"Using embedding model: {args.embedding_model}")
    if args.language.lower() == "arabic":
        logger.info("Arabic language processing enabled with special text cleaning and normalization")
    
    # Process documents
    processor.process_documents()

if __name__ == "__main__":
    main() 

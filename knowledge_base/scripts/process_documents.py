import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import argparse
import html
import unicodedata

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and index Arabic documents from the raw_data directory."""
    
    def __init__(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        embedding_model: str = "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka",
        language: str = "arabic"
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.documents_dir = self.raw_data_dir / "documents"
        self.processed_data_dir = Path(processed_data_dir)
        self.embedding_model = embedding_model
        self.language = language
        
        # Initialize embeddings with Arabic-specific model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Create output directory if it doesn't exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean Arabic text by:
        - Removing diacritics (تشكيل: Unicode ranges 0617-061A and 064B-0652)
        - Removing tatweel characters (ـ)
        - Normalizing Arabic letters (replacing various forms of alef, yaa, etc.)
        - Removing unwanted punctuation (keeping Arabic letters, digits and common punctuation)
        - Normalizing whitespace
        - Handling HTML entities and special characters
        """
        if not text:
            return ""
        
        # Handle HTML entities
        text = html.unescape(text)
        
        # Normalize Unicode forms (NFC normalization)
        text = unicodedata.normalize('NFC', text)
        
        # Remove Arabic diacritics (harakat)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        
        # Remove tatweel (kashida)
        text = re.sub(r'ـ+', '', text)
        
        # Normalize Arabic letters
        replacements = {
            # Alef variations to regular alef
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
            # Yaa variations
            'ى': 'ي', 'ئ': 'ي',
            # Hamza variations
            'ؤ': 'و',
            # Taa marbuta to haa
            'ة': 'ه',
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        
        # Remove unwanted symbols but preserve common punctuation
        # Keep Arabic letters (\u0600-\u06FF), Persian characters (\u0750-\u077F), 
        # Arabic presentation forms (\uFB50-\uFDFF), Latin letters, digits, and
        # common punctuation (،؛.:-–_()[]{}"'、،؛/|)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\w\d\s،؛\.\:\-\–\_\(\)\[\]\{\}\"\'\、\،\؛\/\|]', '', text)
        
        # Handle multiple dots/periods and make spacing consistent
        text = re.sub(r'\.{2,}', '...', text)  # Replace multiple dots with ellipsis
        
        # Normalize spaces around punctuation
        text = re.sub(r'\s*([،؛:.،؛\)\]\}])\s*', r'\1 ', text)  # No space before, one after
        text = re.sub(r'\s*([\(\[\{])\s*', r' \1', text)  # One space before, none after
        
        # Remove zero-width characters and other invisible separators
        text = re.sub(r'[\u200B-\u200F\u061C\u202A-\u202E\u2066-\u2069]', '', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def iter_block_items(self, parent: Document) -> Generator[Paragraph | Table, None, None]:
        """
        Yield Paragraph and Table objects as they appear in the document.
        Adapted from: https://github.com/python-openxml/python-docx/issues/40
        """
        parent_elm = parent.element.body
        for child in parent_elm.iterchildren():
            if child.tag == qn('w:p'):
                yield Paragraph(child, parent)
            elif child.tag == qn('w:tbl'):
                yield Table(child, parent)
    
    def extract_table_data(self, table: Table) -> List[List[str]]:
        """
        Extract table data as a list of rows, where each row is a list of cleaned cell texts.
        """
        table_data = []
        for row in table.rows:
            cells = [self.clean_text(cell.text) for cell in row.cells if self.clean_text(cell.text)]
            if cells:
                table_data.append(cells)
        return table_data
    
    def extract_year_from_filename(self, filename: str) -> str:
        """
        Extract a 4-digit year from the filename (e.g., "2024" from "النشرة الرسمية الثلاثي 1-2024.docx").
        """
        match = re.search(r'(\d{4})', filename)
        return match.group(1) if match else ""
    
    def extract_official_bulletin_from_filename(self, filename: str) -> str:
        """
        Extract the bulletin identifier from the filename.
        """
        return filename.replace('.docx', '')
    
    def extract_decision_number(self, text: str) -> str:
        """
        Extract the decision number from text.
        Looks for patterns like "قرار رقم 32" or "قرار وزاري مشترك رقم42".
        """
        match = re.search(r'قرار(?:\s+وزاري\s+مشترك)?\s+رقم\s*(\d+)', text, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def detect_top_level_chunk(self, text: str) -> Optional[str]:
        """
        Detect if the text signals a new top-level chunk.
        Valid types:
          - "قرار وزاري مشترك"
          - "قرار"
          - "مقــرر"
          - "منشور"
          - "النصوص الصادرة في الجريدة الرسمية"
          - "الاتــفاقـــــــيات"
        Returns the detected type as a string or None.
        """
        text = text.strip()
        patterns = {
            "قرار وزاري مشترك": r'^قرار\s+وزاري\s+مشترك',
            "قرار": r'^قرار\b',
            "مقــرر": r'^مقرر\b',
            "منشور": r'^منشور\b',
            "النصوص الصادرة في الجريدة الرسمية": r'^النصوص\s+الصادرة\s+في\s+الجريدة\s+الرسمية',
            "الاتــفاقـــــــيات": r'^الاتفاقيات'
        }
        for typ, pat in patterns.items():
            if re.search(pat, text, re.IGNORECASE):
                return typ
        return None
    
    def detect_subchunk_type(self, text: str) -> Optional[str]:
        """
        Detect if the text signals a sub-chunk.
        Sub-chunks:
          - "مادة" for articles
          - "ملحق" for appendices
        Returns the type or None.
        """
        if re.search(r'^(المادة|مادة)\s+\d+', text):
            return "مادة"
        if re.search(r'(ملحق)', text, re.IGNORECASE):
            return "ملحق"
        return None
    
    def split_official_text(self, text: str) -> List[str]:
        """
        Splits the input text of type "النصوص الصادرة في الجريدة الرسمية" based on page markers.
        The pattern matches tokens like (ص37), (ص3..21), etc.
        Returns a list of text segments. Each segment will have the page marker at the end.
        """
        # The regex captures the page marker pattern: \(ص\d+(?:\.\.\d+)?\)
        parts = re.split(r'(\(ص\d+(?:\.\.\d+)?\))', text)
        segments = []
        temp = ""
        for part in parts:
            if re.fullmatch(r'\(ص\d+(?:\.\.\d+)?\)', part):
                temp += " " + part
                segments.append(temp.strip())
                temp = ""
            else:
                temp += " " + part
        if temp.strip():
            segments.append(temp.strip())
        return segments
    
    def chunk_document(self, filename: str) -> Dict[str, Any]:
        """
        Process a DOCX file into a structured dictionary with chunks.
        """
        logger.info(f"Chunking document...")
        file_path = self.documents_dir / filename
        doc = Document(str(file_path))
        blocks = list(self.iter_block_items(doc))
        
        document_data = {
            "filename": filename,
            "year": self.extract_year_from_filename(filename),
            "official_bulletin": self.extract_official_bulletin_from_filename(filename),
            "chunks": []
        }
        
        current_chunk = None
        last_subchunk = None  # Active sub-chunk pointer for "مادة" or "ملحق"
        
        logger.info(f"Analyzing document structure and extracting chunks...")
        for block in blocks:
            if isinstance(block, Paragraph):
                raw_text = block.text
                text = self.clean_text(raw_text)
                if not text:
                    continue

                # Check if block starts a new top-level chunk
                top_type = self.detect_top_level_chunk(text)
                if top_type:
                    logger.debug(f"Detected top-level chunk type: {top_type}")
                    # If the previous chunk is of "النصوص الصادرة في الجريدة الرسمية" type,
                    # process its text to split it by the page marker pattern.
                    if current_chunk and current_chunk.get("chunk_type") == "النصوص الصادرة في الجريدة الرسمية":
                        segments = self.split_official_text(current_chunk["text"])
                        document_data["chunks"].pop()
                        for seg in segments:
                            new_chunk = {
                                "chunk_type": "النصوص الصادرة في الجريدة الرسمية",
                                "decision_number": "",
                                "text": seg,
                                "articles": [],
                                "appendices": [],
                                "tables": []
                            }
                            document_data["chunks"].append(new_chunk)
                    # Create a new top-level chunk.
                    current_chunk = {
                        "chunk_type": top_type,
                        "decision_number": self.extract_decision_number(text) if top_type in ["قرار", "قرار وزاري مشترك"] else "",
                        "text": text,
                        "articles": [],
                        "appendices": [],
                        "tables": []
                    }
                    document_data["chunks"].append(current_chunk)
                    last_subchunk = None
                    continue

                # Check for sub-chunk marker ("مادة" or "ملحق")
                sub_type = self.detect_subchunk_type(text)
                if sub_type and current_chunk is not None:
                    logger.debug(f"Detected sub-chunk type: {sub_type}")
                    if sub_type == "مادة":
                        match = re.search(r'(المادة\s+\d+)', text, re.IGNORECASE)
                        article_number = match.group(1) if match else ""
                        article = {
                            "chunk_type": "مادة",
                            "article_number": article_number,
                            "text": text,
                            "table_data": []
                        }
                        current_chunk["articles"].append(article)
                        last_subchunk = article
                    elif sub_type == "ملحق":
                        appendix = {
                            "chunk_type": "ملحق",
                            "text": text,
                            "table_data": []
                        }
                        current_chunk["appendices"].append(appendix)
                        last_subchunk = appendix
                    continue

                # Append text to the active sub-chunk or to the current chunk.
                if last_subchunk is not None:
                    last_subchunk["text"] += " " + text
                elif current_chunk is not None:
                    current_chunk["text"] += " " + text

            elif isinstance(block, Table):
                table_data = self.extract_table_data(block)
                if not table_data:
                    continue
                # Attach table data to the active sub-chunk if present.
                if last_subchunk is not None:
                    last_subchunk["table_data"].append(table_data)
                elif current_chunk is not None:
                    current_chunk["tables"].append(table_data)

        return document_data
    
    def convert_to_langchain_documents(self, chunks: List[Dict[str, Any]]) -> List[LangchainDocument]:
        """
        Convert the structured chunks to LangChain Document objects for vectorization.
        Each chunk from the JSON file becomes exactly one LangChain Document.
        Complex metadata is serialized to JSON strings to comply with Chroma's requirements.
        """
        langchain_docs = []
        
        # Process each chunk as a single document
        for chunk in chunks:
            # Create metadata dictionary with flattened/serialized fields
            metadata = {
                "chunk_type": chunk["chunk_type"],
                "decision_number": chunk["decision_number"],
                "official_bulletin": chunk["official_bulletin"],
                "year": chunk["year"],
                "filename": chunk["filename"],
                # Serialize complex structures to JSON strings
                "articles": json.dumps(chunk["articles"], ensure_ascii=False),
                "appendices": json.dumps(chunk["appendices"], ensure_ascii=False),
                "tables": json.dumps(chunk["tables"], ensure_ascii=False)
            }
            
            # Create a single document for the entire chunk
            doc = LangchainDocument(
                page_content=chunk["text"],
                metadata=metadata
            )
            langchain_docs.append(doc)
            
            # Print the first chunk in detail
            if len(langchain_docs) == 1:
                logger.info("\n=== First Chunk Structure ===")
                logger.info("Content:")
                logger.info(doc.page_content)
                logger.info("\nMetadata:")
                for key, value in doc.metadata.items():
                    logger.info(f"{key}: {value[:100] + '...' if isinstance(value, str) and len(value) > 100 else value}")
                logger.info("========================\n")
        
        return langchain_docs
        
    def process_documents(self) -> None:
        """Process all DOCX files in the raw_data/documents directory."""
        # Check if documents directory exists
        if not self.documents_dir.exists():
            logger.error(f"Documents directory not found: {self.documents_dir}")
            return
            
        # Get all DOCX files
        docx_files = list(self.documents_dir.glob("**/*.docx"))
        
        if not docx_files:
            logger.warning(f"No DOCX files found in {self.documents_dir}")
            return
            
        logger.info(f"Found {len(docx_files)} DOCX files to process")
        
        all_langchain_docs = []
        
        # Process each file
        for docx_file in docx_files:
            try:
                logger.info(f"Processing document...")
                
                # Extract filename
                filename = docx_file.name
                
                # Chunk the document
                doc_data = self.chunk_document(filename)
                
                # Add metadata to each chunk
                for chunk in doc_data["chunks"]:
                    chunk["official_bulletin"] = doc_data["official_bulletin"]
                    chunk["year"] = doc_data["year"]
                    chunk["filename"] = doc_data["filename"]
                
                # Save chunked data directly to processed_data directory
                json_filename = filename.replace(".docx", ".json")
                output_path = self.processed_data_dir / json_filename
                
                logger.info(f"Saving chunked data...")
                with open(output_path, "w", encoding="utf-8") as outfile:
                    json.dump(doc_data["chunks"], outfile, ensure_ascii=False, indent=4)
                
                # Convert chunks to LangChain documents
                langchain_docs = self.convert_to_langchain_documents(doc_data["chunks"])
                all_langchain_docs.extend(langchain_docs)
                
                logger.info(f"Successfully processed document with {len(doc_data['chunks'])} chunks and created {len(langchain_docs)} vector entries")
                
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
        
        # Create or update vector store
        if all_langchain_docs:
            self._update_vector_store(all_langchain_docs)
    
    def _update_vector_store(self, chunks: List[LangchainDocument]) -> None:
        """Update the vector store with new chunks."""
        logger.info(f"Updating vector store with {len(chunks)} documents")
        
        # Print first chunk embedding
        if chunks:
            first_chunk = chunks[0]
            logger.info("\n=== First Chunk Embedding ===")
            embedding = self.embeddings.embed_documents([first_chunk.page_content])[0]
            logger.info(f"Embedding shape: {len(embedding)} dimensions")
            logger.info(f"First 10 values: {embedding[:10]}")
            logger.info("========================\n")

        # Initialize or load vector store
        vector_store_path = self.processed_data_dir / "vector_store"
        if vector_store_path.exists():
            logger.info("Adding documents to existing vector store")
            vector_store = Chroma(
                persist_directory=str(vector_store_path),
                embedding_function=self.embeddings
            )
            vector_store.add_documents(chunks)
            print(chunks[0])
        else:
            logger.info("Creating new vector store")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(vector_store_path)
            )
        
        logger.info("Vector store update complete")


def main():
    """Main function to process documents."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    parser.add_argument("--language", type=str, default="arabic", help="Language of documents (default: arabic)")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                        help="HuggingFace embedding model to use")
    args = parser.parse_args()
    
    # Get paths
    current_dir = Path(__file__).parent.parent
    raw_data_dir = current_dir / "raw_data"
    processed_data_dir = current_dir / "processed_data"
    
    # Log startup information
    logger.info(f"Starting document processing")
    logger.info(f"Using embedding model: {args.embedding_model}")
    logger.info(f"Language setting: {args.language}")
    
    # Initialize processor
    processor = DocumentProcessor(
        raw_data_dir=str(raw_data_dir),
        processed_data_dir=str(processed_data_dir),
        embedding_model=args.embedding_model,
        language=args.language
    )
    
    # Process documents
    processor.process_documents()
    
    logger.info("Document processing complete. Files saved to 'processed_data' directory.")


if __name__ == "__main__":
    main() 
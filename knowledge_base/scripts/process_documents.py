import logging
from pathlib import Path
from typing import List, Dict, Any
import re
import argparse
import json
from datetime import datetime

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
        chunk_size: int = 1200,
        chunk_overlap: int = 400,
        language: str = "arabic",
        split_by_decisions: bool = True
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.split_by_decisions = split_by_decisions
        
        # Initialize embeddings with Arabic-specific model
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
            # Arabic-specific separators including Arabic punctuation and decision markers
            return [
                "\n\nقرار رقم",
                "\nقرار رقم",
                "\n\n",
                "\n",
                ".",
                "!",
                "؟",
                ":",
                "؛",
                " ",
                ""
            ]
        else:
            return ["\n\n", "\n", " ", ""]
    
    def _clean_arabic_text(self, text: str) -> str:
        """Clean Arabic text by handling special characters and normalization."""
        if self.language.lower() == "arabic":
            # Remove diacritics (tashkeel)
            text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
            
            # Normalize alef variants to simple alef
            text = re.sub(r'[إأآا]', 'ا', text)
            
            # Normalize yeh variants
            text = re.sub(r'[يى]', 'ي', text)
            
            # Remove tatweel (kashida)
            text = re.sub(r'\u0640', '', text)
            
            # Normalize Arabic numbers
            text = re.sub(r'[٠-٩]', lambda m: str('٠١٢٣٤٥٦٧٨٩'.index(m.group())), text)
            
            # Remove multiple spaces
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep Arabic punctuation
            text = re.sub(r'[^\u0600-\u06FF\s\.\,\!\?\;\:\(\)\-\d]', '', text)
        
        return text.strip()
    
    def _extract_decision_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from decision text."""
        metadata = {
            "decision_number": "unknown",
            "date": "unknown",
            "has_annexes": False,
            "decision_type": "unknown",
            "ministry": "unknown"
        }
        
        # Decision number patterns
        number_patterns = [
            r'قرار\s+رقم\s+(\d+/\d+)',
            r'قرار\s+(\d+/\d+)',
            r'رقم\s+(\d+/\d+)'
        ]
        
        # Date patterns
        date_patterns = [
            r'بتاريخ\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'في\s+(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        # Ministry patterns
        ministry_patterns = [
            r'وزارة\s+([^،\.\n]+)',
            r'وزير\s+([^،\.\n]+)'
        ]
        
        # Decision type patterns
        type_patterns = [
            r'قرار\s+([^،\.\n]+?)\s+رقم',
            r'قرار\s+([^،\.\n]+?)\s+في'
        ]
        
        # Extract decision number
        for pattern in number_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["decision_number"] = match.group(1)
                break
        
        # Extract date
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["date"] = match.group(1)
                break
        
        # Extract ministry
        for pattern in ministry_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["ministry"] = match.group(1).strip()
                break
        
        # Extract decision type
        for pattern in type_patterns:
            match = re.search(pattern, text)
            if match:
                metadata["decision_type"] = match.group(1).strip()
                break
        
        # Check for annexes
        annex_patterns = [
            r'ملحق\s+رقم',
            r'ملحقات',
            r'الملحق',
            r'الملحقات'
        ]
        
        for pattern in annex_patterns:
            if re.search(pattern, text):
                metadata["has_annexes"] = True
                break
        
        return metadata
    
    def _identify_decisions(self, text: str) -> List[Dict[str, Any]]:
        """
        Identify individual decisions in the text.
        Returns a list of dictionaries containing decision text and metadata.
        """
        decisions = []
        
        # Main decision pattern
        decision_pattern = r'(?:قرار\s+رقم\s+\d+/\d+|قرار\s+\d+/\d+)'
        
        # Find all decision boundaries
        decision_matches = list(re.finditer(decision_pattern, text))
        
        if not decision_matches:
            # If no decisions found, treat the entire text as one decision
            metadata = self._extract_decision_metadata(text)
            decisions.append({
                "text": text,
                **metadata
            })
            return decisions
        
        # Process each decision
        for i, match in enumerate(decision_matches):
            start_pos = match.start()
            # Determine end position (either next decision or end of text)
            end_pos = decision_matches[i + 1].start() if i + 1 < len(decision_matches) else len(text)
            
            decision_text = text[start_pos:end_pos].strip()
            
            # Extract metadata
            metadata = self._extract_decision_metadata(decision_text)
            
            # Add the decision text and metadata
            decisions.append({
                "text": decision_text,
                **metadata
            })
        
        return decisions
        
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
        loader = PyPDFLoader(
            str(file_path),
            extract_images=False  # Skip image extraction as it might cause issues with Arabic PDFs
        )
        pages = loader.load()
        
        # Clean text if working with Arabic
        if self.language.lower() == "arabic":
            for page in pages:
                page.page_content = self._clean_arabic_text(page.page_content)
        
        # Combine all pages into a single text
        full_text = "\n\n".join([page.page_content for page in pages])
        
        if self.split_by_decisions:
            # Split into individual decisions
            decisions = self._identify_decisions(full_text)
            logger.info(f"Identified {len(decisions)} decisions in {file_path}")
            
            # Process each decision
            all_chunks = []
            for i, decision in enumerate(decisions):
                decision_text = decision["text"]
                decision_metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "language": self.language,
                    "decision_number": decision["decision_number"],
                    "decision_date": decision["date"],
                    "has_annexes": decision["has_annexes"],
                    "decision_type": decision["decision_type"],
                    "ministry": decision["ministry"],
                    "decision_index": i,
                    "processing_date": datetime.now().isoformat()
                }
                
                # Create a document for the decision
                decision_doc = Document(
                    page_content=decision_text,
                    metadata=decision_metadata
                )
                
                # Split decision into chunks if it's too long
                if len(decision_text) > self.chunk_size:
                    decision_chunks = self.text_splitter.split_documents([decision_doc])
                else:
                    decision_chunks = [decision_doc]
                
                # Add metadata to chunks
                decision_chunks = self._add_metadata(decision_chunks, file_path, decision_metadata)
                
                all_chunks.extend(decision_chunks)
                
                # Save individual decision
                self._save_decision(decision, file_path, i)
            
            # Save all chunks to processed data directory
            self._save_chunks(all_chunks, file_path)
            
            # Create or update vector store
            self._update_vector_store(all_chunks)
        else:
            # Original processing without splitting by decisions
            chunks = self.text_splitter.split_documents(pages)
            
            # Add metadata
            chunks = self._add_metadata(chunks, file_path)
            
            # Save chunks to processed data directory
            self._save_chunks(chunks, file_path)
            
            # Create or update vector store
            self._update_vector_store(chunks)
        
        logger.info(f"Successfully processed {file_path}")
    
    def _save_decision(self, decision: Dict[str, Any], file_path: Path, index: int) -> None:
        """Save an individual decision to disk."""
        # Create output directory structure
        rel_path = file_path.relative_to(self.raw_data_dir)
        output_dir = self.processed_data_dir / rel_path.parent / "decisions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decision
        output_file = output_dir / f"{file_path.stem}_decision_{index}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        
    def _add_metadata(self, chunks: List[Document], file_path: Path, additional_metadata: Dict[str, Any] = None) -> List[Document]:
        """Add metadata to document chunks."""
        for chunk in chunks:
            chunk.metadata.update({
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "language": self.language
            })
            
            # Add additional metadata if provided
            if additional_metadata:
                chunk.metadata.update(additional_metadata)
                
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process documents for RAG system")
    parser.add_argument("--language", type=str, default="arabic", help="Language of documents (default: arabic)")
    parser.add_argument("--embedding-model", type=str, default="aubmindlab/bert-base-arabertv02", 
                        help="HuggingFace embedding model to use")
    parser.add_argument("--split-by-decisions", action="store_true", 
                        help="Split documents into individual decisions with their annexes")
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
        language=args.language,
        split_by_decisions=args.split_by_decisions
    )
    
    # Log language processing information
    logger.info(f"Processing documents in {args.language} language")
    logger.info(f"Using embedding model: {args.embedding_model}")
    if args.language.lower() == "arabic":
        logger.info("Arabic language processing enabled with special text cleaning and normalization")
    if args.split_by_decisions:
        logger.info("Splitting documents into individual decisions with their annexes")
    
    # Process documents
    processor.process_documents()

if __name__ == "__main__":
    main() 
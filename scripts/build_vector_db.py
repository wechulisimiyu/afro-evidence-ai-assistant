import os
import sys
import fitz  # PyMuPDF
import json
from pathlib import Path
import re
from loguru import logger
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import embeddings

# Add the parent directory to Python path to import app modules
sys.path.append(str(Path(__file__).parent.parent))


def extract_title_footer_doi_keywords(pdf_path):
    """Extract metadata from PDF file."""
    doc = fitz.open(pdf_path)

    # -------- TITLE EXTRACTION --------
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    title_candidates = []

    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                font_size = span["size"]
                # Filter for longer text, avoid headers/metadata
                if text and len(text.split()) > 3 and not re.search(r'(journal|doi|volume|issue|page \d+)', text, re.IGNORECASE):
                    title_candidates.append((text, font_size, span["bbox"][1]))

    # Sort by font size descending, then y-position (top of page)
    title_candidates.sort(key=lambda x: (-x[1], x[2]))
    title = title_candidates[0][0] if title_candidates else "Unknown Title"

    # -------- CITATION INFO (Footer) EXTRACTION --------
    footer_blocks = first_page.get_text("blocks")
    footer_blocks.sort(key=lambda b: b[1], reverse=True)

    # Look for citation-style text among bottom blocks
    citation_candidates = [
        block[4].strip() for block in footer_blocks
        if re.search(r'(journal|volume|issue|vol\.|no\.|doi|issn)', block[4], re.IGNORECASE)
    ]

    citation_info = citation_candidates[0] if citation_candidates else (
        footer_blocks[0][4].strip() if footer_blocks else "Unknown Citation Info"
    )

    # -------- DOI + KEYWORDS --------
    text_to_search = ""
    for i in range(min(2, len(doc))):  # First 2 pages
        text_to_search += doc[i].get_text()

    # Extract DOI
    doi_match = re.search(r'\b(10\.\d{4,9}/[^\s"\'<>]*)', text_to_search)
    doi = f"http://dx.doi.org/{doi_match.group(1).rstrip('.,;')}" if doi_match else "DOI not found"

    # Extract keywords (supporting Keywords, Key words, Index Terms)
    keywords_match = re.search(r'(Keywords|Key words|Index Terms)\s[:\-–]?\s(.+)', text_to_search, re.IGNORECASE)
    if keywords_match:
        raw_keywords = keywords_match.group(2).split('\n')[0]  # Take only the first line in case of line breaks
        keywords = [kw.strip().strip('.') for kw in re.split(r',|;', raw_keywords) if kw.strip()]
    else:
        keywords = []

    return title, citation_info, doi, keywords

def process_pdf(pdf_path):
    """Process a single PDF file and return its content and metadata."""
    try:
        doc = fitz.open(pdf_path)
        content = ""
        for page in doc:
            content += page.get_text()

        # Clean the text: remove multiple newlines and replace them with a single space
        cleaned_content = re.sub(r'\s*\n\s*', ' ', content)
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()
        
        title, citation_info, doi, keywords = extract_title_footer_doi_keywords(pdf_path)
        
        return {
            "title": title,
            "content": cleaned_content,
            "doi": doi,
            "keywords": keywords,
            "metadata": {
                "citation_info": citation_info,
                "source_file": os.path.basename(pdf_path)
            }
        }
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def build_vector_db_for_type(pdf_dir, vector_db_dir, doc_type):
    """Build vector database for a specific document type."""
    logger.info(f"Building vector database for {doc_type} documents...")
    
    # Create vector store directory if it doesn't exist
    os.makedirs(vector_db_dir, exist_ok=True)
    
    # Process all PDFs and create documents
    documents = []
    metadata = []
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            logger.info(f"Processing {pdf_file}...")
            
            doc_data = process_pdf(pdf_path)
            if doc_data:
                documents.append(doc_data['content'])
                metadata.append({
                    'title': doc_data['title'],
                    'doi': doc_data['doi'],
                    'keywords': doc_data['keywords'],
                    'citation_info': doc_data['metadata']['citation_info'],
                    'source_file': doc_data['metadata']['source_file'],
                    'doc_type': doc_type
                })
    
    if documents:
        # Create and save the vector store
        vector_store = FAISS.from_texts(
            documents,
            embeddings,
            metadatas=metadata
        )
        
        # Save the vector store
        vector_store.save_local(vector_db_dir)
        
        # Save metadata separately for easy access
        with open(os.path.join(vector_db_dir, 'documents.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.success(f"Successfully built vector database for {doc_type} documents")
    else:
        logger.warning(f"No documents processed for {doc_type}")

def main():
    """Main function to build vector databases."""
    # Setup logging
    logger.add("vector_db_build.log", rotation="500 MB")
    
    try:
        # Set up paths relative to script location
        base_dir = Path(__file__).parent.parent
        
        # Define source and destination paths
        pdf_source_dir = base_dir / "scripts" / "pdfs"
        vector_store_dir = base_dir / "data"
        
        # Ensure directories exist
        (pdf_source_dir / "guidelines").mkdir(parents=True, exist_ok=True)
        (pdf_source_dir / "journals").mkdir(parents=True, exist_ok=True)
        (vector_store_dir / "guidelines_vector_store").mkdir(parents=True, exist_ok=True)
        (vector_store_dir / "journals_vector_store").mkdir(parents=True, exist_ok=True)
        
        # Build vector database for guidelines
        build_vector_db_for_type(
            str(pdf_source_dir / "guidelines"),
            str(vector_store_dir / "guidelines_vector_store"),
            "guidelines"
        )
        
        # Build vector database for journals
        build_vector_db_for_type(
            str(pdf_source_dir / "journals"),
            str(vector_store_dir / "journals_vector_store"),
            "journals"
        )
        
        logger.success("Vector database building completed successfully")
        
    except Exception as e:
        logger.error(f"Error building vector databases: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

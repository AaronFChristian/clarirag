"""
================================================================================
FILE: src/ingestion/loader.py
================================================================================
WHAT THIS FILE DOES:
    Loads PDF files from the data/raw/ folder and extracts raw text from every
    page, along with metadata (file name, page number, section headers).

WHY IT EXISTS:
    Before we can chunk or embed anything, we need clean text out of the PDFs.
    PyMuPDF (fitz) is used because it preserves page numbers accurately — which
    is critical for our citation system later. When ClariRAG says "see page 12
    of obesity_management_guidelines.pdf", that page number must be real.

INPUT:
    - A folder path containing one or more PDF files (e.g. data/raw/)
    - OR a single PDF file path

OUTPUT:
    - A list of Page objects, where each Page contains:
        {
            "doc_name"   : "obesity_management_guidelines.pdf",
            "page_number": 12,
            "text"       : "...raw text from that page...",
            "file_path"  : "/absolute/path/to/file.pdf"
        }

CONCEPT:
    Think of this as the "document reader" step. It doesn't chunk, embed, or
    analyse — it just opens every PDF and extracts one dictionary per page.
    Downstream steps (chunker.py) will split these pages into smaller pieces.

KEY LIBRARY:
    PyMuPDF (imported as `fitz`) — fast, accurate PDF parser that gives us
    page-level text extraction with metadata.
================================================================================
"""

import fitz  # PyMuPDF — the library for reading PDFs
import os
from pathlib import Path
from typing import List, Dict
import logging

# Set up logging so we can see what's happening when this runs
# This prints messages like "Loading diabetes.pdf... done (45 pages)"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load a single PDF file and extract text from every page.

    Args:
        file_path (str): Absolute or relative path to a .pdf file

    Returns:
        List[Dict]: One dictionary per page with keys:
                    doc_name, page_number, text, file_path

    Raises:
        FileNotFoundError: If the PDF path doesn't exist
        ValueError: If the file is not a PDF
    """

    # Validate the file exists
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    # Validate it's actually a PDF
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages = []  # We'll collect all page dicts here

    logger.info(f"Loading: {path.name}")

    # Open the PDF using PyMuPDF
    # fitz.open() loads the whole document into memory
    doc = fitz.open(file_path)

    logger.info(f"  → {len(doc)} pages found in {path.name}")

    # Loop through every page in the document
    for page_num in range(len(doc)):

        # Load the specific page (0-indexed internally, we store as 1-indexed)
        page = doc[page_num]

        # Extract all text from this page as a plain string
        # "text" mode gives clean UTF-8 text, stripping image data
        raw_text = page.get_text("text")

        # Skip pages that are essentially empty (cover pages, blank pages, images)
        # Strip whitespace before checking — some pages are just "\n\n\n"
        if len(raw_text.strip()) < 50:
            logger.debug(f"  → Skipping page {page_num + 1} (too short, likely blank/image)")
            continue

        # Build the page dictionary — this is the core data unit for all downstream steps
        page_data = {
            "doc_name"   : path.name,            # e.g. "obesity_management_guidelines.pdf"
            "page_number": page_num + 1,          # 1-indexed (matches real PDF page numbers)
            "text"       : raw_text.strip(),      # clean text, leading/trailing whitespace removed
            "file_path"  : str(path.absolute()),  # full path for debugging
            "char_count" : len(raw_text.strip())  # useful for QA and filtering
        }

        pages.append(page_data)

    # Close the document to free memory
    doc.close()

    logger.info(f"  → Extracted {len(pages)} usable pages from {path.name}")

    return pages


def load_all_pdfs(folder_path: str) -> List[Dict]:
    """
    Load ALL PDF files from a folder and return all pages combined.

    This is the main function called by the ingestion pipeline.
    It finds every .pdf in the folder, loads each one, and returns
    a flat list of all pages across all documents.

    Args:
        folder_path (str): Path to the folder containing PDF files
                           (e.g. "data/raw/")

    Returns:
        List[Dict]: All pages from all PDFs, each with doc_name,
                    page_number, text, file_path, char_count

    Example:
        >>> pages = load_all_pdfs("data/raw/")
        >>> print(f"Loaded {len(pages)} pages from {folder_path}")
        >>> print(pages[0]["doc_name"])   # "clinical_trials_best_practices.pdf"
        >>> print(pages[0]["page_number"])  # 1
    """

    folder = Path(folder_path)

    # Make sure the folder actually exists
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Find all PDF files in the folder (non-recursive — only top level)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {folder_path}")

    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")

    all_pages = []  # Accumulate pages from all documents here

    # Process each PDF one at a time
    for pdf_path in pdf_files:
        try:
            pages = load_pdf(str(pdf_path))
            all_pages.extend(pages)  # Add this doc's pages to the master list
        except Exception as e:
            # Don't crash the whole pipeline if one PDF fails
            # Log the error and continue with the rest
            logger.error(f"Failed to load {pdf_path.name}: {e}")
            continue

    # Summary statistics for QA
    logger.info(f"\n{'='*50}")
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"  Total PDFs processed : {len(pdf_files)}")
    logger.info(f"  Total pages extracted: {len(all_pages)}")
    avg_chars = sum(p["char_count"] for p in all_pages) / max(len(all_pages), 1)
    logger.info(f"  Avg chars per page   : {int(avg_chars)}")
    logger.info(f"{'='*50}\n")

    return all_pages


def get_corpus_summary(pages: List[Dict]) -> Dict:
    """
    Print and return a summary of the loaded corpus.
    Useful for verifying the load worked correctly before moving to chunking.

    Args:
        pages: Output from load_all_pdfs()

    Returns:
        Dict with per-document page counts and total stats
    """

    # Group pages by document name
    doc_stats = {}
    for page in pages:
        doc = page["doc_name"]
        if doc not in doc_stats:
            doc_stats[doc] = {"pages": 0, "total_chars": 0}
        doc_stats[doc]["pages"] += 1
        doc_stats[doc]["total_chars"] += page["char_count"]

    summary = {
        "total_pages"    : len(pages),
        "total_documents": len(doc_stats),
        "documents"      : doc_stats
    }

    # Pretty print the summary
    print("\n" + "="*60)
    print("CORPUS SUMMARY")
    print("="*60)
    for doc_name, stats in doc_stats.items():
        print(f"  {doc_name}")
        print(f"    Pages: {stats['pages']}  |  "
              f"Chars: {stats['total_chars']:,}  |  "
              f"Avg/page: {stats['total_chars']//max(stats['pages'],1):,}")
    print("-"*60)
    print(f"  TOTAL: {len(pages)} pages across {len(doc_stats)} documents")
    print("="*60 + "\n")

    return summary


# ============================================================
# QUICK TEST — run this file directly to verify it works
# From your project root: python src/ingestion/loader.py
# ============================================================
if __name__ == "__main__":
    import sys

    # Default to data/raw/ but allow override from command line
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    print(f"\nRunning loader test on: {folder}\n")

    # Load all PDFs
    pages = load_all_pdfs(folder)

    # Print corpus summary
    get_corpus_summary(pages)

    # Show a sample page to verify text extraction looks right
    if pages:
        sample = pages[5] if len(pages) > 5 else pages[0]
        print("SAMPLE PAGE PREVIEW:")
        print(f"  Document : {sample['doc_name']}")
        print(f"  Page     : {sample['page_number']}")
        print(f"  Chars    : {sample['char_count']}")
        print(f"  Text preview:\n")
        # Print first 400 characters of the sample page
        print("  " + sample["text"][:400].replace("\n", "\n  "))
        print("\n[loader.py test complete — no errors]\n")

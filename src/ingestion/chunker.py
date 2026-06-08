"""
================================================================================
FILE: src/ingestion/chunker.py
================================================================================
WHAT THIS FILE DOES:
    Takes the list of page dictionaries produced by loader.py and splits each
    page's text into smaller, overlapping chunks ready for embedding.
    This is the "chunking" step of the RAG pipeline — the bridge between
    raw PDF text and the vector database.

INPUT:
    - A list of page dicts from loader.py, each containing:
        {
            "doc_name"   : "obesity_management_guidelines.pdf",
            "page_number": 12,
            "text"       : "...raw page text...",
            "file_path"  : "/absolute/path/to/file.pdf",
            "char_count" : 2400
        }

OUTPUT:
    - A flat list of chunk dicts, each containing:
        {
            "chunk_id"            : "obesity_management_guidelines.pdf_p12_c3",
            "doc_name"            : "obesity_management_guidelines.pdf",
            "page_number"         : 12,
            "file_path"           : "/absolute/path/to/file.pdf",
            "text"                : "...chunk text (~512 chars)...",
            "chunk_index"         : 3,
            "total_chunks_in_page": 5,
            "char_count"          : 498
        }

WHY 512 CHARACTERS / 64 CHARACTER OVERLAP:
    - 512 chars ≈ 100-130 tokens, well within embedding model limits
    - Each chunk stays semantically focused on one idea
    - 64 char overlap (~12%) ensures sentences split across a boundary
      appear fully in at least one chunk — retrieval never misses a key
      phrase just because it straddles a cut point
    - Smaller chunks (<256) lose context; larger (>1024) dilute relevance

WHY RecursiveCharacterTextSplitter OVER SIMPLE SPLITTING:
    - Simple text[i:i+512] cuts mid-word, mid-sentence, mid-paragraph
    - RecursiveCharacterTextSplitter tries split order: paragraph breaks
      "\n\n" → newlines "\n" → spaces " " → individual characters
    - Only falls back to finer boundaries when coarser ones produce
      oversized chunks — result is chunks that end at natural language
      boundaries, preserving semantic coherence for better embeddings
================================================================================
"""

import logging
from typing import List, Dict
from collections import defaultdict

# FIXED IMPORT: In LangChain v0.2+, text splitters moved to langchain_text_splitters
# Old path: from langchain.text_splitter import RecursiveCharacterTextSplitter  ← breaks
# New path: from langchain_text_splitters import RecursiveCharacterTextSplitter  ← correct
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Consistent logging format across all ingestion files
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def chunk_page(page_dict: Dict, chunk_size: int = 512, chunk_overlap: int = 64) -> List[Dict]:
    """
    Split a single page dict into a list of overlapping chunk dicts.

    The splitter tries to cut at paragraph breaks first, then newlines,
    then spaces — so chunks almost always end at a natural sentence boundary.

    Args:
        page_dict    : One page dict as returned by loader.load_pdf()
        chunk_size   : Max characters per chunk (default 512)
        chunk_overlap: Characters repeated at start of next chunk (default 64)

    Returns:
        List of chunk dicts. Each inherits parent page metadata plus
        chunk-specific fields. Returns [] if no valid chunks produced.
    """

    # Extract fields from the parent page
    text        = page_dict["text"]
    doc_name    = page_dict["doc_name"]
    page_number = page_dict["page_number"]
    file_path   = page_dict["file_path"]

    # Build the splitter with our settings
    # length_function=len means we measure in characters, not tokens
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )

    # Split the page text into raw string fragments
    raw_splits = splitter.split_text(text)

    # Filter out fragments too short to be meaningful
    # (stray headers, page numbers, footnote artifacts after splitting)
    valid_splits = [s for s in raw_splits if len(s.strip()) >= 30]

    # Nothing useful after filtering — return empty
    if not valid_splits:
        return []

    total_chunks = len(valid_splits)
    chunks = []

    for idx, chunk_text in enumerate(valid_splits):

        # Build a unique, traceable ID for this chunk
        # Format: "filename.pdf_p12_c3" → document, page 12, chunk index 3
        # This ID is what gets stored in Pinecone and used for citations
        chunk_id = f"{doc_name}_p{page_number}_c{idx}"

        chunk_dict = {
            "chunk_id"            : chunk_id,
            "doc_name"            : doc_name,
            "page_number"         : page_number,
            "file_path"           : file_path,
            "text"                : chunk_text.strip(),
            "chunk_index"         : idx,           # position within this page
            "total_chunks_in_page": total_chunks,  # how many chunks this page made
            "char_count"          : len(chunk_text.strip()),
        }

        chunks.append(chunk_dict)

    return chunks


def chunk_all_pages(
    pages: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
    """
    Chunk every page in the corpus into a single flat list of all chunks.

    This is the main function called by the ingestion pipeline.
    Processes all pages from all documents and returns everything combined.

    Args:
        pages        : Full list of page dicts from loader.load_all_pdfs()
        chunk_size   : Characters per chunk (default 512)
        chunk_overlap: Overlap characters between chunks (default 64)

    Returns:
        Flat list of all chunk dicts across every page and document
    """

    all_chunks: List[Dict] = []

    # Track per-document counts for the end summary
    doc_chunk_counts: Dict[str, int] = defaultdict(int)

    logger.info(
        f"Starting chunking: {len(pages)} pages | "
        f"chunk_size={chunk_size} | overlap={chunk_overlap}"
    )

    for page_idx, page in enumerate(pages):

        # Log progress every 50 pages so we can watch it run
        if page_idx > 0 and page_idx % 50 == 0:
            logger.info(
                f"  Progress: {page_idx}/{len(pages)} pages | "
                f"{len(all_chunks)} chunks so far"
            )

        # Chunk this single page
        page_chunks = chunk_page(page, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Add to the master flat list
        all_chunks.extend(page_chunks)

        # Tally for per-document summary
        doc_chunk_counts[page["doc_name"]] += len(page_chunks)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_chunks = len(all_chunks)
    avg_size = (
        sum(c["char_count"] for c in all_chunks) / total_chunks
        if total_chunks > 0 else 0
    )

    logger.info(f"\n{'='*52}")
    logger.info("CHUNKING COMPLETE")
    logger.info(f"  Total chunks   : {total_chunks}")
    logger.info(f"  Avg chunk size : {avg_size:.0f} chars")
    logger.info("  Chunks per document:")
    for doc_name, count in sorted(doc_chunk_counts.items()):
        logger.info(f"    {doc_name}: {count} chunks")
    logger.info(f"{'='*52}\n")

    return all_chunks


def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """
    Print a detailed per-document summary table and return a stats dict.
    Useful for verifying chunking worked before moving to embedding.

    Args:
        chunks: Output from chunk_all_pages()

    Returns:
        Dict with total_chunks, total_documents, avg_chunk_size, documents
    """

    if not chunks:
        print("No chunks to summarise.")
        return {"total_chunks": 0, "total_documents": 0, "documents": {}}

    # Aggregate per-document stats in one pass
    doc_stats: Dict[str, Dict] = defaultdict(lambda: {"chunk_count": 0, "total_chars": 0})

    for chunk in chunks:
        doc = chunk["doc_name"]
        doc_stats[doc]["chunk_count"] += 1
        doc_stats[doc]["total_chars"]  += chunk["char_count"]

    # Corpus-wide totals
    total_chunks   = len(chunks)
    total_chars    = sum(c["char_count"] for c in chunks)
    avg_chunk_size = total_chars / total_chunks

    # Align columns to the longest document name
    col_width = max(len(d) for d in doc_stats) + 2

    print("\n" + "="*70)
    print("CHUNK STATS SUMMARY")
    print("="*70)
    print(f"  {'Document':<{col_width}}  {'Chunks':>8}  {'Avg Chars':>10}")
    print(f"  {'-'*col_width}  {'--------':>8}  {'----------':>10}")

    for doc_name, stats in sorted(doc_stats.items()):
        avg = stats["total_chars"] / max(stats["chunk_count"], 1)
        print(f"  {doc_name:<{col_width}}  {stats['chunk_count']:>8}  {avg:>10.0f}")

    print("-"*70)
    print(f"  {'TOTAL':<{col_width}}  {total_chunks:>8}  {avg_chunk_size:>10.0f}")
    print("="*70 + "\n")

    return {
        "total_chunks"   : total_chunks,
        "total_documents": len(doc_stats),
        "avg_chunk_size" : round(avg_chunk_size, 1),
        "documents"      : {
            doc: {
                "chunk_count": stats["chunk_count"],
                "avg_chars"  : round(
                    stats["total_chars"] / max(stats["chunk_count"], 1), 1
                ),
            }
            for doc, stats in doc_stats.items()
        },
    }


# ============================================================
# QUICK TEST — run directly to verify
# From project root: python src/ingestion/chunker.py
# ============================================================
if __name__ == "__main__":
    # Import loader from the project — make sure you run from clarirag/ root
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from src.ingestion.loader import load_all_pdfs

    RAW_FOLDER = "data/raw"

    print(f"\nLoading PDFs from: {RAW_FOLDER}\n")
    pages = load_all_pdfs(RAW_FOLDER)

    print(f"Chunking {len(pages)} pages (512 chars, 64 overlap)...\n")
    chunks = chunk_all_pages(pages)

    # Print the detailed stats table
    get_chunk_stats(chunks)

    # ── Overlap verification ──────────────────────────────────────────────
    # Find two consecutive chunks from the same page and show their
    # boundary — the end of chunk A should appear at the start of chunk B
    print("OVERLAP VERIFICATION")
    print("="*70)
    print("Finding two consecutive chunks from the same page...\n")

    demo_pair = None
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        # Adjacent chunks from the same doc + page
        if a["doc_name"] == b["doc_name"] and a["page_number"] == b["page_number"]:
            demo_pair = (a, b)
            break

    if demo_pair is None:
        print("  No page produced more than one chunk — corpus may be too small.\n")
    else:
        a, b = demo_pair
        print(f"  Document : {a['doc_name']}")
        print(f"  Page     : {a['page_number']}")
        print()

        # Show the last 80 chars of chunk A and first 80 of chunk B
        # The overlapping text should be visible at the boundary
        tail_a = a["text"][-80:]
        head_b = b["text"][:80]

        print(f"  Chunk {a['chunk_index']} — last 80 chars:")
        print(f"    ...{tail_a}")
        print()
        print(f"  Chunk {b['chunk_index']} — first 80 chars:")
        print(f"    {head_b}...")
        print()
        print("  ↑ The end of chunk A should appear at the start of chunk B.")
        print("    That repeated text is the overlap — confirming it works.")
        print("="*70)

    print("\n[chunker.py test complete — no errors]\n")

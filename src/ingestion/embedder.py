"""
================================================================================
FILE: src/ingestion/embedder.py
================================================================================
WHAT THIS FILE DOES:
    Takes all chunk dicts from chunker.py, generates vector embeddings using
    a FREE local sentence-transformers model (no API cost), then upserts
    everything into your Pinecone index.

    This version uses 'all-MiniLM-L6-v2' which runs entirely on your machine.
    No OpenAI API calls. No cost. No quota issues.

INPUT:
    - List of chunk dicts from chunker.chunk_all_pages()

OUTPUT:
    - All chunks upserted into Pinecone index "clarirag"
    - Each vector has: ID (chunk_id), values (384-dim embedding), metadata
      (doc_name, page_number, text, char_count — used for citations)

MODEL: all-MiniLM-L6-v2
    - 384-dimensional embeddings (vs OpenAI's 1536)
    - Runs 100% locally — no API key needed
    - Industry-standard model for RAG applications
    - Fast: ~1000 chunks/minute on CPU

PINECONE INDEX REQUIREMENT:
    Your Pinecone index must be created with dimension=384
    (Delete the old 1536-dim index and create a new one)
================================================================================
"""

import os
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load all keys from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "clarirag")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
BATCH_SIZE          = 64    # sentence-transformers handles batches efficiently
UPSERT_BATCH_SIZE   = 100   # Pinecone recommends ≤100 vectors per upsert call


def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model locally.
    First run downloads ~90MB to ~/.cache/huggingface/
    Subsequent runs load from cache instantly.

    Returns:
        SentenceTransformer model ready for encoding
    """
    logger.info(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    logger.info("(First run downloads ~90MB — subsequent runs load from cache)")

    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(f"Model loaded. Output dimension: {model.get_sentence_embedding_dimension()}")
    return model


def get_pinecone_index(create_if_missing: bool = True):
    """
    Connect to Pinecone and return the index object.
    Creates the index if it doesn't exist yet.

    IMPORTANT: Index must have dimension=384 to match all-MiniLM-L6-v2.

    Args:
        create_if_missing: Auto-create index if not found (default True)

    Returns:
        Pinecone Index object ready for upsert and query
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file.")

    # Connect to Pinecone
    pc = Pinecone(api_key=api_key)

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        if not create_if_missing:
            raise ValueError(f"Index '{PINECONE_INDEX_NAME}' not found.")

        logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIMENSION})...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # 384 for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )

        # Wait for index to be ready
        logger.info("Waiting for index to initialize...")
        while True:
            status = pc.describe_index(PINECONE_INDEX_NAME).status
            if status.get("ready", False):
                break
            time.sleep(2)

        logger.info(f"Index '{PINECONE_INDEX_NAME}' ready.")
    else:
        logger.info(f"Connected to existing index '{PINECONE_INDEX_NAME}'.")

    return pc.Index(PINECONE_INDEX_NAME)


def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> List[Dict]:
    """
    Generate embedding vectors for all chunks using the local model.

    Processes chunks in batches for efficiency. Each chunk gets a new
    "embedding" key containing a list of 384 floats.

    Args:
        chunks: List of chunk dicts from chunker.chunk_all_pages()
        model : Loaded SentenceTransformer model

    Returns:
        Same chunks with "embedding" key added to each dict
    """
    logger.info(f"Embedding {len(chunks)} chunks locally (batch size: {BATCH_SIZE})...")

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    embedded_chunks = []

    for batch_idx in range(0, len(chunks), BATCH_SIZE):

        # Slice this batch
        batch = chunks[batch_idx : batch_idx + BATCH_SIZE]
        batch_texts = [chunk["text"] for chunk in batch]
        current_batch = (batch_idx // BATCH_SIZE) + 1

        # Log progress every 5 batches
        if current_batch % 5 == 1 or current_batch == total_batches:
            logger.info(f"  Batch {current_batch}/{total_batches} ({len(batch)} chunks)...")

        # Generate embeddings for this batch
        # show_progress_bar=False keeps output clean
        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True   # returns numpy array — faster than list
        )

        # Attach embedding to each chunk dict
        for chunk, embedding in zip(batch, embeddings):
            chunk_with_embedding = {**chunk}
            # Convert numpy array to plain Python list for Pinecone
            chunk_with_embedding["embedding"] = embedding.tolist()
            embedded_chunks.append(chunk_with_embedding)

    logger.info(f"Embedding complete. {len(embedded_chunks)} chunks embedded.")
    return embedded_chunks


def upsert_to_pinecone(embedded_chunks: List[Dict], index) -> int:
    """
    Upsert all embedded chunks into the Pinecone index.

    Each Pinecone vector contains:
        id       — chunk_id for traceability
        values   — the 384-dim embedding vector
        metadata — doc_name, page_number, text, char_count
                   (returned with search results for citations)

    Args:
        embedded_chunks: Chunks with "embedding" key from embed_chunks()
        index          : Pinecone Index object

    Returns:
        Total number of vectors upserted
    """
    logger.info(f"Upserting {len(embedded_chunks)} vectors to Pinecone...")

    total_upserted = 0
    total_batches = (len(embedded_chunks) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    for batch_idx in range(0, len(embedded_chunks), UPSERT_BATCH_SIZE):

        batch = embedded_chunks[batch_idx : batch_idx + UPSERT_BATCH_SIZE]
        current_batch = (batch_idx // UPSERT_BATCH_SIZE) + 1

        # Build the vector list Pinecone expects
        vectors = []
        for chunk in batch:
            vectors.append({
                "id"      : chunk["chunk_id"],
                "values"  : chunk["embedding"],
                "metadata": {
                    # These come back with every search result
                    # Critical for showing citations to the user
                    "doc_name"   : chunk["doc_name"],
                    "page_number": chunk["page_number"],
                    "text"       : chunk["text"],
                    "char_count" : chunk["char_count"],
                    "file_path"  : chunk["file_path"],
                }
            })

        # Send to Pinecone
        index.upsert(vectors=vectors)
        total_upserted += len(vectors)

        logger.info(
            f"  Batch {current_batch}/{total_batches} upserted "
            f"({total_upserted}/{len(embedded_chunks)} total)"
        )

    logger.info(f"Upsert complete. {total_upserted} vectors in Pinecone.")
    return total_upserted


def verify_index(index) -> None:
    """
    Check Pinecone index stats to confirm vectors were stored correctly.
    """
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    print("\n" + "="*50)
    print("PINECONE INDEX VERIFICATION")
    print("="*50)
    print(f"  Index name    : {PINECONE_INDEX_NAME}")
    print(f"  Total vectors : {total_vectors}")
    print(f"  Dimension     : {EMBEDDING_DIMENSION}")
    print("="*50)

    if total_vectors == 0:
        print("\nWARNING: No vectors found — upsert may have failed.\n")
    else:
        print(f"\nSUCCESS: {total_vectors} vectors ready for semantic search.\n")


def test_search(index, model: SentenceTransformer) -> None:
    """
    Run a quick test query to confirm search is working end-to-end.
    Embeds a sample clinical question and retrieves the top 3 matches.
    """
    test_query = "What are the recommended treatments for type 2 diabetes?"

    print("="*50)
    print("TEST SEARCH")
    print("="*50)
    print(f"Query: {test_query}\n")

    # Embed the query using the same model
    query_embedding = model.encode(test_query).tolist()

    # Search Pinecone for top 3 most similar chunks
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True   # get text + doc info back
    )

    for i, match in enumerate(results["matches"]):
        meta = match["metadata"]
        print(f"Result {i+1} — Score: {match['score']:.4f}")
        print(f"  Document : {meta['doc_name']}")
        print(f"  Page     : {meta['page_number']}")
        print(f"  Excerpt  : {meta['text'][:150]}...")
        print()

    print("[Search test complete]\n")


# ============================================================
# MAIN PIPELINE
# From project root: PYTHONPATH=. python src/ingestion/embedder.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    from src.ingestion.loader import load_all_pdfs
    from src.ingestion.chunker import chunk_all_pages

    RAW_FOLDER = "data/raw"

    # Step 1: Load PDFs
    print("\n" + "="*50)
    print("STEP 1: Loading PDFs")
    print("="*50)
    pages = load_all_pdfs(RAW_FOLDER)
    print(f"Loaded {len(pages)} pages\n")

    # Step 2: Chunk pages
    print("="*50)
    print("STEP 2: Chunking pages")
    print("="*50)
    chunks = chunk_all_pages(pages)
    print(f"Created {len(chunks)} chunks\n")

    # Step 3: Load local embedding model
    print("="*50)
    print("STEP 3: Loading embedding model")
    print("="*50)
    model = load_embedding_model()
    print()

    # Step 4: Connect to Pinecone
    print("="*50)
    print("STEP 4: Connecting to Pinecone")
    print("="*50)
    pinecone_index = get_pinecone_index(create_if_missing=True)
    print()

    # Step 5: Embed all chunks locally
    print("="*50)
    print("STEP 5: Embedding chunks (local, no API cost)")
    print("="*50)
    embedded = embed_chunks(chunks, model)
    print()

    # Step 6: Upsert to Pinecone
    print("="*50)
    print("STEP 6: Upserting to Pinecone")
    print("="*50)
    total = upsert_to_pinecone(embedded, pinecone_index)
    print()

    # Step 7: Verify
    print("="*50)
    print("STEP 7: Verifying index")
    print("="*50)
    verify_index(pinecone_index)

    # Step 8: Test search
    test_search(pinecone_index, model)

    print("[embedder.py complete — Pinecone loaded and search verified]\n")

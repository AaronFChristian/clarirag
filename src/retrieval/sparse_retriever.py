"""
================================================================================
FILE: src/retrieval/sparse_retriever.py
================================================================================
WHAT THIS FILE DOES:
    Builds and queries a BM25 keyword-based search index over all chunks.
    This is the "sparse" half of our hybrid retrieval system.

WHY WE NEED THIS (the problem with dense-only search):
    Pinecone (dense/semantic search) finds chunks that are *conceptually*
    similar to a query. But in clinical guidelines, exact terminology matters:
    - "Section 4.2 dosage threshold" needs exact keyword matching
    - Drug names, disease codes, procedure names must match precisely
    - Dense search embeds "hypertension" and "high blood pressure" as similar
      — great for general queries, but misses exact regulatory references

    BM25 solves this: it scores chunks by exact term frequency and rarity.
    A chunk containing "metformin" scores high for the query "metformin dosage"
    even if no dense model would think they're semantically close.

INPUT:
    - List of chunk dicts from chunker.chunk_all_pages()

OUTPUT:
    - A BM25Index object saved to disk as data/bm25_index.pkl
    - Query function that returns top-K chunks by BM25 score

CONCEPT — How BM25 works:
    BM25 (Best Match 25) is a ranking function used by search engines like
    Elasticsearch. For each chunk it calculates a score based on:
    1. Term Frequency (TF): how often query words appear in the chunk
    2. Inverse Document Frequency (IDF): how rare the word is across ALL chunks
       (rare words like "metformin" score higher than common words like "the")
    3. Length normalization: shorter chunks with the same term count score higher

    Result: chunks that contain rare, query-specific terms rank highest.

HOW IT COMBINES WITH PINECONE:
    Dense (Pinecone) + Sparse (BM25) results are merged in hybrid_retriever.py
    using Reciprocal Rank Fusion — giving us the best of both worlds:
    semantic understanding + exact keyword precision.

PERSISTENCE:
    The BM25 index is saved as a pickle file so we don't rebuild it on every
    query. Rebuilding from 1,911 chunks takes ~1 second, but at query time
    we want sub-100ms response — so we load from disk instead.
================================================================================
"""

import os
import pickle
import logging
from typing import List, Dict, Tuple
from pathlib import Path

from rank_bm25 import BM25Okapi  # industry-standard BM25 implementation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Path where the BM25 index is saved/loaded from disk
BM25_INDEX_PATH = "data/bm25_index.pkl"


def tokenize(text: str) -> List[str]:
    """
    Convert a text string into a list of lowercase tokens for BM25.

    Simple whitespace + punctuation tokenization. BM25 works on token lists,
    not raw strings. Lowercasing ensures "Diabetes" and "diabetes" match.

    Args:
        text: Raw text string

    Returns:
        List of lowercase word tokens with punctuation stripped

    Example:
        tokenize("Type 2 Diabetes management.")
        → ["type", "2", "diabetes", "management"]
    """
    # Lowercase everything for case-insensitive matching
    text = text.lower()

    # Remove common punctuation that would split words incorrectly
    # Keep hyphens (anti-inflammatory) and numbers (type-2)
    for char in ".,;:!?()[]{}\"'\\/@#$%^&*+=<>|~`":
        text = text.replace(char, " ")

    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if len(token) > 0]

    return tokens


class BM25Index:
    """
    Wrapper around rank_bm25.BM25Okapi that stores chunk metadata
    alongside the index for easy retrieval of full chunk dicts.

    Attributes:
        bm25      : The underlying BM25Okapi model
        chunks    : Original chunk dicts in the same order as BM25 corpus
        corpus    : Tokenized texts (parallel list to chunks)
    """

    def __init__(self, chunks: List[Dict]):
        """
        Build the BM25 index from a list of chunk dicts.

        Args:
            chunks: All chunk dicts from chunker.chunk_all_pages()
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks...")

        # Store the original chunks — we'll return these on search
        self.chunks = chunks

        # Tokenize every chunk's text for BM25
        # BM25Okapi expects a list of token lists: [["word1","word2"], ["word3",...], ...]
        logger.info("  Tokenizing all chunks...")
        self.corpus = [tokenize(chunk["text"]) for chunk in chunks]

        # Build the BM25 model over the tokenized corpus
        # This computes IDF scores for every unique token across all chunks
        logger.info("  Building BM25 model (computing IDF scores)...")
        self.bm25 = BM25Okapi(self.corpus)

        logger.info(f"BM25 index built. Vocabulary size: {len(self.bm25.idf)} unique terms.")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search the BM25 index and return the top-K most relevant chunks.

        Args:
            query : Natural language query string
            top_k : Number of top results to return (default 10)

        Returns:
            List of result dicts, each containing:
                - All original chunk fields (chunk_id, doc_name, page_number, text, etc.)
                - "bm25_score": the raw BM25 relevance score
                - "bm25_rank" : rank position (1 = highest score)

        Example:
            results = index.search("metformin dosage diabetes", top_k=5)
            print(results[0]["doc_name"])   # "ncd_diabetes_policy.pdf"
            print(results[0]["bm25_score"]) # 12.34
        """
        # Tokenize the query the same way we tokenized the corpus
        query_tokens = tokenize(query)

        if not query_tokens:
            logger.warning("Query produced no tokens after tokenization.")
            return []

        # Get BM25 scores for every chunk in the corpus
        # Returns a numpy array of scores, one per chunk, in corpus order
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-K highest scores using argsort
        # argsort returns ascending order, so we reverse with [::-1]
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])

            # Skip chunks with zero score — no query terms matched at all
            if score <= 0:
                continue

            # Build result dict: original chunk + BM25-specific fields
            result = {
                **self.chunks[idx],   # all original chunk fields
                "bm25_score": score,
                "bm25_rank" : rank + 1,  # 1-indexed rank
            }
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Return basic statistics about the index."""
        return {
            "total_chunks"  : len(self.chunks),
            "vocabulary_size": len(self.bm25.idf),
            "avg_doc_length": float(self.bm25.avgdl),
        }


def build_and_save_bm25_index(chunks: List[Dict], save_path: str = BM25_INDEX_PATH) -> BM25Index:
    """
    Build a BM25 index from chunks and save it to disk as a pickle file.

    Saves to disk so we can load instantly at query time without rebuilding.

    Args:
        chunks   : All chunk dicts from chunker.chunk_all_pages()
        save_path: Where to save the pickle file (default: data/bm25_index.pkl)

    Returns:
        The built BM25Index object
    """
    # Build the index
    index = BM25Index(chunks)

    # Make sure the output directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to disk as pickle
    logger.info(f"Saving BM25 index to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump(index, f)

    file_size_mb = Path(save_path).stat().st_size / (1024 * 1024)
    logger.info(f"BM25 index saved. File size: {file_size_mb:.1f} MB")

    return index


def load_bm25_index(load_path: str = BM25_INDEX_PATH) -> BM25Index:
    """
    Load a previously saved BM25 index from disk.

    Called at query time — much faster than rebuilding from scratch.

    Args:
        load_path: Path to the saved pickle file

    Returns:
        Loaded BM25Index object ready for search

    Raises:
        FileNotFoundError: If the index file doesn't exist yet
                           (run build_and_save_bm25_index first)
    """
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"BM25 index not found at {load_path}. "
            "Run build_and_save_bm25_index() first."
        )

    logger.info(f"Loading BM25 index from {load_path}...")
    with open(load_path, "rb") as f:
        index = pickle.load(f)

    logger.info(f"BM25 index loaded. {len(index.chunks)} chunks, "
                f"{len(index.bm25.idf)} unique terms.")
    return index


# ============================================================
# QUICK TEST — run directly to verify
# From project root: PYTHONPATH=. python src/retrieval/sparse_retriever.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    from src.ingestion.loader import load_all_pdfs
    from src.ingestion.chunker import chunk_all_pages

    RAW_FOLDER = "data/raw"

    # Load and chunk the corpus
    print("\nLoading and chunking corpus for BM25 index...\n")
    pages  = load_all_pdfs(RAW_FOLDER)
    chunks = chunk_all_pages(pages)

    # Build and save the BM25 index
    print("\nBuilding BM25 index...\n")
    bm25_index = build_and_save_bm25_index(chunks)

    # Print index stats
    stats = bm25_index.get_stats()
    print("\n" + "="*55)
    print("BM25 INDEX STATS")
    print("="*55)
    print(f"  Total chunks indexed : {stats['total_chunks']}")
    print(f"  Vocabulary size      : {stats['vocabulary_size']} unique terms")
    print(f"  Avg chunk length     : {stats['avg_doc_length']:.1f} tokens")
    print("="*55)

    # ── Test Query 1: Keyword-specific medical term ───────────────────────
    print("\n\nTEST 1 — Exact keyword query: 'metformin type 2 diabetes dosage'")
    print("-"*55)
    results = bm25_index.search("metformin type 2 diabetes dosage", top_k=3)
    if results:
        for r in results:
            print(f"  Rank {r['bm25_rank']} | Score: {r['bm25_score']:.3f} | "
                  f"{r['doc_name']} p.{r['page_number']}")
            print(f"  Excerpt: {r['text'][:120]}...")
            print()
    else:
        print("  No results found.")

    # ── Test Query 2: Clinical terminology ───────────────────────────────
    print("\nTEST 2 — Clinical query: 'body mass index obesity intervention'")
    print("-"*55)
    results = bm25_index.search("body mass index obesity intervention", top_k=3)
    if results:
        for r in results:
            print(f"  Rank {r['bm25_rank']} | Score: {r['bm25_score']:.3f} | "
                  f"{r['doc_name']} p.{r['page_number']}")
            print(f"  Excerpt: {r['text'][:120]}...")
            print()
    else:
        print("  No results found.")

    # ── Test save/load round-trip ─────────────────────────────────────────
    print("\nTEST 3 — Save/load round-trip verification")
    print("-"*55)
    loaded_index = load_bm25_index()
    test_results = loaded_index.search("clinical trial randomized", top_k=2)
    print(f"  Loaded index search returned {len(test_results)} results.")
    if test_results:
        print(f"  Top result: {test_results[0]['doc_name']} "
              f"p.{test_results[0]['page_number']} "
              f"(score: {test_results[0]['bm25_score']:.3f})")

    print("\n[sparse_retriever.py test complete — BM25 index saved to data/bm25_index.pkl]\n")

"""
================================================================================
FILE: src/agents/nodes/retriever_node.py
================================================================================
WHAT THIS FILE DOES:
    Node 3 of the LangGraph pipeline. Takes the expanded queries from the
    Expander node and runs ALL of them through the hybrid retriever, then
    merges and deduplicates the results into one ranked list.

INPUT  (from GraphState):
    - state["expanded_queries"] : list of 2-3 query strings
    - state["query"]            : original query (fallback)

OUTPUT (written back to GraphState):
    - state["retrieved_chunks"] : top-K deduplicated chunks ranked by rerank score

WHY RUN ALL EXPANDED QUERIES:
    Each query variant may surface different chunks. Running all 3 and
    merging gives us the best possible candidate set before the judge
    decides if it's sufficient to answer the question.
================================================================================
"""

import os
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from src.agents.state import GraphState
from src.retrieval.hybrid_retriever import HybridRetriever

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Singleton retriever — loaded once, reused across all queries
_retriever_instance = None

def get_retriever() -> HybridRetriever:
    """Return the shared HybridRetriever instance (lazy init)."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("[RetrieverNode] Initialising HybridRetriever...")
        _retriever_instance = HybridRetriever()
    return _retriever_instance


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node 3: Run hybrid retrieval for all expanded queries.

    Runs each expanded query through BM25+Pinecone+RRF+reranker,
    merges results, deduplicates by chunk_id, and returns top-10.

    Args:
        state: GraphState with "expanded_queries" and "query"

    Returns:
        Dict with "retrieved_chunks" key
    """
    expanded_queries = state.get("expanded_queries", [])
    original_query   = state.get("query", "")

    # Fall back to original query if expander didn't run
    if not expanded_queries:
        expanded_queries = [original_query]

    logger.info(f"[RetrieverNode] Running retrieval for {len(expanded_queries)} queries")

    retriever = get_retriever()

    # Run hybrid retrieval for each query variant
    all_chunks = []
    for q in expanded_queries:
        try:
            results = retriever.retrieve(q, top_k=5)
            all_chunks.extend(results)
            logger.info(f"[RetrieverNode] Query '{q[:50]}' → {len(results)} chunks")
        except Exception as e:
            logger.error(f"[RetrieverNode] Retrieval failed for query '{q[:50]}': {e}")
            continue

    if not all_chunks:
        logger.warning("[RetrieverNode] No chunks retrieved for any query.")
        return {"retrieved_chunks": []}

    # Deduplicate by chunk_id — keep highest rerank_score for each chunk
    seen: Dict[str, Dict] = {}
    for chunk in all_chunks:
        cid   = chunk["chunk_id"]
        score = chunk.get("rerank_score", 0.0)
        if cid not in seen or score > seen[cid].get("rerank_score", 0.0):
            seen[cid] = chunk

    # Sort by rerank_score descending and keep top 10
    deduped = sorted(seen.values(), key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:10]

    logger.info(f"[RetrieverNode] {len(deduped)} unique chunks after dedup (from {len(all_chunks)} total)")

    return {"retrieved_chunks": deduped}


# ============================================================
# QUICK TEST
# From project root: PYTHONPATH=. python src/agents/nodes/retriever_node.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    state = {
        "query": "What is the BMI threshold for obesity in adults?",
        "expanded_queries": [
            "What is the BMI threshold for obesity in adults?",
            "What body mass index cutoff defines adult obesity classification?",
            "At what BMI value are adults categorized as obese per clinical criteria?",
        ]
    }

    print("\n[RetrieverNode Test]")
    result = retriever_node(state)
    chunks = result["retrieved_chunks"]
    print(f"\nRetrieved {len(chunks)} unique chunks:\n")
    for i, c in enumerate(chunks[:3]):
        print(f"  {i+1}. {c['doc_name']} p.{c['page_number']} "
              f"(rerank: {c.get('rerank_score', 0):.4f})")
        print(f"     {c['text'][:120]}...\n")

    print("[retriever_node.py test complete]\n")

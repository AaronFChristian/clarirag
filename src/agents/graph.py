"""
================================================================================
FILE: src/agents/graph.py
================================================================================
WHAT THIS FILE DOES:
    Wires all 5 nodes into a complete LangGraph StateGraph with a conditional
    retry edge. This is the top-level pipeline that takes a query string and
    returns a fully grounded RAGResponse.

THE GRAPH STRUCTURE:

    START
      │
      ▼
    analyser_node       → sets query_type, entities
      │
      ▼
    expander_node       → sets expanded_queries
      │
      ▼
    retriever_node      → sets retrieved_chunks
      │
      ▼
    judge_node          → sets is_context_sufficient, retry_count
      │
      ├─── sufficient=True ──────────────────────────────────────┐
      │                                                          │
      └─── sufficient=False AND retry_count < 2 ──► retriever   │
                                                    (retry loop) │
      └─── sufficient=False AND retry_count >= 2 ───────────────┤
                                                                 ▼
                                                         generator_node
                                                                 │
                                                                END

INPUT:  {"query": "What is the BMI threshold for obesity?"}
OUTPUT: RAGResponse with answer, citations, confidence, eval_scores

USAGE:
    from src.agents.graph import run_pipeline
    response = run_pipeline("What is the BMI threshold for obesity?")
    print(response.answer)
    for citation in response.citations:
        print(citation.doc_name, citation.page_number)
================================================================================
"""

import os
import logging
from typing import Dict, Any

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from src.agents.state import GraphState, RAGResponse, INSUFFICIENT_CONTEXT_RESPONSE
from src.agents.nodes.analyser import analyser_node
from src.agents.nodes.expander import expander_node
from src.agents.nodes.retriever_node import retriever_node
from src.agents.nodes.judge_node import judge_node
from src.agents.nodes.generator_node import generator_node

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── Conditional edge function ─────────────────────────────────────────────────

def route_after_judge(state: GraphState) -> str:
    """
    Conditional edge called after judge_node.

    Decides where to route based on sufficiency and retry count:
    - sufficient=True              → go to generator
    - sufficient=False, retries<2  → go back to retriever (retry loop)
    - sufficient=False, retries>=2 → go to generator (will return safe refusal)

    Returns:
        "generator" or "retriever" — must match node names in the graph
    """
    is_sufficient = state.get("is_context_sufficient", False)
    retry_count   = state.get("retry_count", 0)

    if is_sufficient:
        logger.info(f"[Router] Context sufficient → generator")
        return "generator"
    elif retry_count < MAX_RETRIES:
        logger.info(f"[Router] Context insufficient, retry {retry_count}/{MAX_RETRIES} → retriever")
        return "retriever"
    else:
        logger.info(f"[Router] Max retries reached → generator (safe refusal)")
        return "generator"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Build and compile the ClariRAG LangGraph pipeline.

    Returns a compiled graph ready to invoke with:
        graph.invoke({"query": "your question"})

    The graph is compiled once and reused — it holds all the node
    functions but NOT the model instances (those are singletons in
    each node module).
    """
    # Create the state graph with our GraphState TypedDict
    graph = StateGraph(GraphState)

    # ── Add nodes ─────────────────────────────────────────────────────────
    # Each node is a plain Python function that takes GraphState and
    # returns a dict of fields to update
    graph.add_node("analyser" , analyser_node)
    graph.add_node("expander" , expander_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("judge"    , judge_node)
    graph.add_node("generator", generator_node)

    # ── Add edges — the fixed linear path ────────────────────────────────
    graph.add_edge(START      , "analyser")   # always start with analyser
    graph.add_edge("analyser" , "expander")   # analyser → expander
    graph.add_edge("expander" , "retriever")  # expander → retriever

    # ── Add the conditional edge after judge ──────────────────────────────
    # This is the retry loop — the most important edge in the graph
    graph.add_conditional_edges(
        "judge",              # from this node
        route_after_judge,    # call this function to decide where to go
        {
            "retriever": "retriever",   # if returns "retriever" → go to retriever
            "generator": "generator",   # if returns "generator" → go to generator
        }
    )

    # ── Fixed edges after retriever and generator ─────────────────────────
    graph.add_edge("retriever", "judge")    # retriever always goes to judge
    graph.add_edge("generator", END)        # generator always ends the graph

    # Compile the graph — validates structure and returns runnable
    compiled = graph.compile()

    logger.info("ClariRAG graph compiled successfully.")
    return compiled


# ── Singleton compiled graph ──────────────────────────────────────────────────

_graph_instance = None

def get_graph():
    """Return the shared compiled graph instance (lazy init)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph()
    return _graph_instance


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(query: str) -> RAGResponse:
    """
    Run the full ClariRAG pipeline for a single query.

    This is the function called by FastAPI and by tests.
    It initialises the graph (once), invokes it with the query,
    and returns the final RAGResponse.

    Args:
        query: The user's natural language question

    Returns:
        RAGResponse with answer, citations, confidence, eval_scores
        Returns INSUFFICIENT_CONTEXT_RESPONSE on any unrecoverable error.
    """
    if not query or not query.strip():
        return INSUFFICIENT_CONTEXT_RESPONSE

    logger.info(f"\n{'='*55}")
    logger.info(f"PIPELINE START: {query[:80]}")
    logger.info(f"{'='*55}")

    try:
        graph = get_graph()

        # Invoke the graph with the initial state
        # LangGraph fills in all other fields as nodes run
        final_state = graph.invoke({
            "query"      : query.strip(),
            "retry_count": 0,   # start with 0 retries
        })

        # Extract the final response from state
        response = final_state.get("final_response")

        if response is None:
            logger.warning("[Pipeline] No final_response in state — returning safe refusal")
            return INSUFFICIENT_CONTEXT_RESPONSE

        logger.info(f"PIPELINE COMPLETE | grounded={response.is_grounded} | "
                    f"confidence={response.confidence:.2f} | "
                    f"citations={len(response.citations)}")

        return response

    except Exception as e:
        logger.error(f"[Pipeline] Unrecoverable error: {e}")
        return INSUFFICIENT_CONTEXT_RESPONSE


# ============================================================
# QUICK TEST — run the full end-to-end pipeline
# From project root: PYTHONPATH=. python src/agents/graph.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    test_queries = [
        "What is the BMI threshold for obesity in adults?",
        "What are the key principles of randomization in clinical trials?",
        "What is the price of gold in Dubai?",   # out of scope → safe refusal
    ]

    print("\n" + "="*60)
    print("FULL PIPELINE END-TO-END TEST")
    print("="*60)

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

        response = run_pipeline(query)

        print(f"\n  is_grounded        : {response.is_grounded}")
        print(f"  confidence         : {response.confidence:.2f}")
        print(f"  query_type         : {response.query_type}")
        print(f"  citations          : {len(response.citations)}")
        print(f"  retrieval_attempts : {response.retrieval_attempts}")
        print(f"\n  ANSWER:\n  {response.answer[:350]}...")

        if response.citations:
            print(f"\n  CITATIONS:")
            for c in response.citations[:2]:
                print(f"    • {c.doc_name} p.{c.page_number} "
                      f"(relevance: {c.relevance:.2f})")
                print(f"      {c.excerpt[:80]}...")

    print(f"\n{'='*60}")
    print("FULL PIPELINE TEST COMPLETE")
    print("="*60 + "\n")

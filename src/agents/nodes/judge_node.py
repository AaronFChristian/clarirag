"""
================================================================================
FILE: src/agents/nodes/judge_node.py
================================================================================
WHAT THIS FILE DOES:
    Node 4 of the LangGraph pipeline. The "sufficiency judge" — evaluates
    whether the retrieved chunks contain enough information to answer the
    original query reliably.

    This is the node that controls the RETRY LOOP. If context is insufficient
    and retry_count < 2, it reformulates the query and loops back to retrieval.
    If context is still insufficient after 2 retries, the pipeline returns
    the safe INSUFFICIENT_CONTEXT_RESPONSE instead of hallucinating.

INPUT  (from GraphState):
    - state["query"]            : original query
    - state["retrieved_chunks"] : chunks from retriever node
    - state["retry_count"]      : how many retries so far (starts at 0)

OUTPUT (written back to GraphState):
    - state["is_context_sufficient"] : True → proceed to generator
    - state["retry_count"]           : incremented if retrying
    - state["missing_aspects"]       : what's missing (used to reformulate query)
    - state["query"]                 : potentially reformulated for retry

CONDITIONAL EDGE LOGIC (in graph.py):
    if is_context_sufficient == True  → go to GeneratorNode
    if is_context_sufficient == False and retry_count < 2 → go back to RetrieverNode
    if is_context_sufficient == False and retry_count >= 2 → go to GeneratorNode
      (GeneratorNode will detect insufficient context and return safe refusal)
================================================================================
"""

import os
import json
import logging
from typing import Dict, Any

from anthropic import Anthropic
from dotenv import load_dotenv

from src.agents.state import GraphState

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_MODEL   = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
MAX_RETRIES = 2


def judge_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node 4: Evaluate whether retrieved context is sufficient.

    Asks Claude to judge: "Given these chunks, can we answer this question
    reliably without hallucinating?" Returns a structured verdict.

    Args:
        state: GraphState with "query", "retrieved_chunks", "retry_count"

    Returns:
        Dict with "is_context_sufficient", "retry_count", "missing_aspects",
        and optionally a reformulated "query" for the retry
    """
    query           = state.get("query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    retry_count     = state.get("retry_count", 0)

    logger.info(f"[Judge] Evaluating {len(retrieved_chunks)} chunks "
                f"(retry {retry_count}/{MAX_RETRIES})")

    # If no chunks at all — immediately insufficient
    if not retrieved_chunks:
        logger.warning("[Judge] No chunks to evaluate — insufficient.")
        return {
            "is_context_sufficient": False,
            "retry_count"          : retry_count + 1,
            "missing_aspects"      : "No relevant chunks were retrieved.",
        }

    # Build a context string from top-5 chunks for the judge prompt
    context_preview = ""
    for i, chunk in enumerate(retrieved_chunks[:5]):
        context_preview += (
            f"\n[Chunk {i+1}] {chunk['doc_name']} p.{chunk['page_number']}\n"
            f"{chunk['text'][:300]}\n"
        )

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""You are a RAG quality judge for a clinical guidelines system.

QUESTION: "{query}"

RETRIEVED CONTEXT:
{context_preview}

Evaluate whether the retrieved context contains enough information to answer 
the question ACCURATELY and COMPLETELY without hallucinating.

Respond with ONLY a valid JSON object — no explanation, no markdown:

{{
  "is_sufficient": true or false,
  "confidence": 0.0 to 1.0,
  "missing_aspects": "<what specific information is missing, or 'none' if sufficient>",
  "reformulated_query": "<a better search query to find the missing info, or '' if sufficient>"
}}

Be strict: if the context only partially answers the question, mark as false.
Only mark true if the context clearly and directly addresses the question."""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        parsed = json.loads(raw_text)

        is_sufficient      = bool(parsed.get("is_sufficient", False))
        missing_aspects    = parsed.get("missing_aspects", "")
        reformulated_query = parsed.get("reformulated_query", "").strip()
        confidence         = float(parsed.get("confidence", 0.5))

        logger.info(f"[Judge] sufficient={is_sufficient} | confidence={confidence:.2f}")
        if not is_sufficient:
            logger.info(f"[Judge] Missing: {missing_aspects[:80]}")

        # Build return dict
        updates: Dict[str, Any] = {
            "is_context_sufficient": is_sufficient,
            "missing_aspects"      : missing_aspects,
            "retry_count"          : retry_count + (0 if is_sufficient else 1),
        }

        # If not sufficient and we have a reformulated query — use it for retry
        if not is_sufficient and reformulated_query and retry_count < MAX_RETRIES:
            logger.info(f"[Judge] Reformulated query: '{reformulated_query[:80]}'")
            updates["query"]            = reformulated_query
            updates["expanded_queries"] = [reformulated_query]

        return updates

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[Judge] Parse error: {e}")
        # On parse failure — assume sufficient to avoid infinite retry loop
        return {
            "is_context_sufficient": True,
            "missing_aspects"      : "",
            "retry_count"          : retry_count,
        }

    except Exception as e:
        logger.error(f"[Judge] Error: {e}")
        return {
            "is_context_sufficient": True,
            "missing_aspects"      : "",
            "retry_count"          : retry_count,
            "error"                : str(e),
        }


# ============================================================
# QUICK TEST
# From project root: PYTHONPATH=. python src/agents/nodes/judge_node.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()

    test_cases = [
        {
            "name" : "Good context — should be sufficient",
            "query": "What is the BMI threshold for obesity in adults?",
        },
        {
            "name" : "Out of scope — should be insufficient",
            "query": "What is the recommended dosage of aspirin for children?",
        },
    ]

    print("\n" + "="*60)
    print("JUDGE NODE TEST")
    print("="*60)

    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Query: {case['query']}")

        # Get real chunks first
        chunks = retriever.retrieve(case["query"], top_k=5)

        state = {
            "query"           : case["query"],
            "retrieved_chunks": chunks,
            "retry_count"     : 0,
        }

        result = judge_node(state)
        print(f"  is_sufficient : {result['is_context_sufficient']}")
        print(f"  retry_count   : {result['retry_count']}")
        print(f"  missing       : {result.get('missing_aspects', '')[:80]}")

    print("\n[judge_node.py test complete]\n")

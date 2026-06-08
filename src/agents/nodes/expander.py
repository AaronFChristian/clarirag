"""
================================================================================
FILE: src/agents/nodes/expander.py
================================================================================
WHAT THIS FILE DOES:
    Node 2 of the LangGraph pipeline. Takes the original query plus the
    entities extracted by the Analyser, and generates 2 alternative
    phrasings of the same question.

    These expanded queries are passed to the hybrid retriever to improve
    recall — especially for BM25, which relies on exact term matching.

INPUT  (from GraphState):
    - state["query"]    : original user question
    - state["entities"] : key terms from analyser (e.g. ["BMI", "obesity"])
    - state["query_type"]: query classification from analyser

OUTPUT (written back to GraphState):
    - state["expanded_queries"] : list of 2-3 query strings including original

WHY QUERY EXPANSION MATTERS:
    Clinical documents use inconsistent terminology. A user asking:
        "What is the BMI threshold for obesity?"
    might miss chunks that say:
        "Body mass index classification for overweight adults"
    or
        "WHO cutoffs for adiposity measurement"

    By generating variants, we cast a wider net across both BM25 (exact
    terms) and Pinecone (semantic meaning) — improving hit rate before
    the reranker picks the best results.

    The original query is ALWAYS included as the first expanded query
    so we never lose the user's exact intent.
================================================================================
"""

import os
import json
import logging
from typing import Dict, Any, List

from anthropic import Anthropic
from dotenv import load_dotenv

from src.agents.state import GraphState

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")


def expander_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node 2: Generate alternative query phrasings for better recall.

    Takes the original query + extracted entities and generates 2 variants
    using different terminology that might appear in clinical documents.
    The original query is always kept as the first element.

    Args:
        state: GraphState with "query", "entities", "query_type"

    Returns:
        Dict with "expanded_queries" key containing list of 2-3 query strings
    """

    query      = state.get("query", "")
    entities   = state.get("entities", [])
    query_type = state.get("query_type", "factual")

    logger.info(f"[Expander] Expanding query: '{query[:60]}'")

    if not query.strip():
        return {"expanded_queries": [query]}

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Build entity context string for the prompt
    entity_context = ", ".join(entities) if entities else "none extracted"

    prompt = f"""You are a clinical search query expander for a medical RAG system.

ORIGINAL QUERY: "{query}"
QUERY TYPE: {query_type}
KEY ENTITIES: {entity_context}

Generate exactly 2 alternative phrasings of this query.
Each variant should use DIFFERENT clinical terminology that might appear in WHO guidelines,
medical textbooks, or clinical practice documents.

Rules:
- Keep the same meaning and intent as the original
- Use different words/synonyms where possible
- Keep each variant under 15 words
- Focus on terms likely to appear in clinical guideline documents
- Do NOT add new questions or change the topic

Respond with ONLY a valid JSON array of 2 strings — no explanation, no markdown:
["<variant 1>", "<variant 2>"]"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        # Parse the JSON array
        variants = json.loads(raw_text)

        # Validate — must be a list of strings
        if not isinstance(variants, list):
            raise ValueError("Response is not a JSON array")

        variants = [str(v).strip() for v in variants if v and len(str(v).strip()) > 5]

        # Always include original query first — never lose the user's intent
        # Deduplicate while preserving order
        expanded = [query]
        for v in variants:
            if v.lower() != query.lower() and v not in expanded:
                expanded.append(v)

        # Cap at 3 total (original + 2 variants)
        expanded = expanded[:3]

        logger.info(f"[Expander] Generated {len(expanded)} queries:")
        for i, q in enumerate(expanded):
            logger.info(f"  [{i}] {q}")

        return {"expanded_queries": expanded}

    except (json.JSONDecodeError, ValueError) as e:
        # Parsing failed — fall back to original query only
        logger.warning(f"[Expander] Parse error: {e}. Using original query only.")
        return {"expanded_queries": [query]}

    except Exception as e:
        logger.error(f"[Expander] Error: {e}")
        return {"expanded_queries": [query], "error": str(e)}


# ============================================================
# QUICK TEST
# From project root: PYTHONPATH=. python src/agents/nodes/expander.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    test_cases = [
        {
            "query"     : "What is the BMI threshold to classify obesity in adults?",
            "entities"  : ["BMI", "obesity", "adults"],
            "query_type": "factual",
        },
        {
            "query"     : "How should clinical trials handle randomization?",
            "entities"  : ["clinical trials", "randomization"],
            "query_type": "procedural",
        },
        {
            "query"     : "What are the recommended treatments for type 2 diabetes?",
            "entities"  : ["type 2 diabetes", "treatment"],
            "query_type": "factual",
        },
    ]

    print("\n" + "="*60)
    print("EXPANDER NODE TEST")
    print("="*60)

    for case in test_cases:
        print(f"\nOriginal : {case['query']}")
        result = expander_node(case)
        expanded = result.get("expanded_queries", [])
        print(f"Expanded queries ({len(expanded)} total):")
        for i, q in enumerate(expanded):
            label = "ORIGINAL" if i == 0 else f"VARIANT {i}"
            print(f"  [{label}] {q}")

    print("\n[expander.py test complete]\n")

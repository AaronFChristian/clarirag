"""
================================================================================
FILE: src/agents/nodes/analyser.py
================================================================================
WHAT THIS FILE DOES:
    Node 1 of the LangGraph pipeline. Takes the raw user query and:
    1. Classifies it into one of 4 types (factual/comparative/procedural/definitional)
    2. Extracts key clinical/medical entities from it

    This runs BEFORE retrieval so the downstream nodes know what kind of
    question they're dealing with and what terms to focus on.

INPUT  (from GraphState):
    - state["query"] : the raw user question string

OUTPUT (written back to GraphState):
    - state["query_type"] : "factual" | "comparative" | "procedural" | "definitional"
    - state["entities"]   : list of key terms e.g. ["BMI", "obesity", "adults"]

WHY QUERY CLASSIFICATION MATTERS:
    - Factual queries need precise, direct answers with a single citation
    - Comparative queries need chunks from multiple documents or sections
    - Procedural queries need step-by-step chunks in the right order
    - Definitional queries need glossary/definition chunks specifically

    The classification isn't used for routing in this version — but it's
    stored in the final RAGResponse so users and developers can see what
    the system "thought" the question was asking for. It also improves
    the query expansion step.

WHY ENTITY EXTRACTION MATTERS:
    Extracted entities ("BMI", "obesity") are appended to expanded queries
    in the next node — this improves BM25 recall for exact clinical terms
    that might not appear in a semantically-phrased query variant.
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

# Use the model name from .env or default to claude-sonnet
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5")


def analyser_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node 1: Classify query type and extract entities.

    Called by LangGraph with the current state dict. Returns ONLY the
    fields this node is responsible for — LangGraph merges them back.

    Args:
        state: Current GraphState (must contain "query")

    Returns:
        Dict with "query_type" and "entities" keys
        On error: returns safe defaults so the pipeline continues
    """

    query = state.get("query", "")
    logger.info(f"[Analyser] Processing query: '{query[:80]}'")

    if not query.strip():
        logger.warning("[Analyser] Empty query received.")
        return {"query_type": "factual", "entities": []}

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Prompt ────────────────────────────────────────────────────────────
    # We ask Claude to return ONLY valid JSON — no preamble, no markdown.
    # This makes parsing reliable and deterministic.
    prompt = f"""You are a clinical query analyser. Analyse this medical/clinical query:

QUERY: "{query}"

Respond with ONLY a valid JSON object — no explanation, no markdown, no backticks.
Use exactly this structure:

{{
  "query_type": "<one of: factual | comparative | procedural | definitional>",
  "entities": ["<entity1>", "<entity2>", "<entity3>"]
}}

QUERY TYPE DEFINITIONS:
- factual      : asks for a specific fact, number, threshold, or recommendation
                 e.g. "What is the BMI threshold for obesity?"
- comparative  : compares two or more treatments, interventions, or approaches
                 e.g. "How does bariatric surgery compare to lifestyle intervention?"
- procedural   : asks for a sequence of steps or a process
                 e.g. "What steps should be followed for diabetes screening?"
- definitional : asks for the meaning or definition of a term
                 e.g. "What is metabolic syndrome?"

ENTITIES: Extract 2-5 key clinical/medical terms from the query.
Focus on: disease names, drug names, procedures, measurements, biomarkers.
Keep entities short (1-3 words each).

QUERY: "{query}"
JSON:"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=200,      # short response — just JSON
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the text content from the response
        raw_text = response.content[0].text.strip()
        logger.debug(f"[Analyser] Raw response: {raw_text}")

        # Strip markdown code fences if Claude added them despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()

        # Parse the JSON response
        parsed = json.loads(raw_text)

        query_type = parsed.get("query_type", "factual")
        entities   = parsed.get("entities", [])

        # Validate query_type is one of the expected values
        valid_types = {"factual", "comparative", "procedural", "definitional"}
        if query_type not in valid_types:
            logger.warning(f"[Analyser] Unexpected query_type '{query_type}', defaulting to 'factual'")
            query_type = "factual"

        # Ensure entities is a list of strings
        if not isinstance(entities, list):
            entities = []
        entities = [str(e) for e in entities if e][:5]  # max 5 entities

        logger.info(f"[Analyser] query_type={query_type} | entities={entities}")

        return {
            "query_type": query_type,
            "entities"  : entities,
        }

    except json.JSONDecodeError as e:
        # Claude returned something that's not valid JSON
        logger.error(f"[Analyser] JSON parse error: {e}. Raw: {raw_text[:100]}")
        return {"query_type": "factual", "entities": []}

    except Exception as e:
        # Any other error — log and return safe defaults
        logger.error(f"[Analyser] Error: {e}")
        return {"query_type": "factual", "entities": [], "error": str(e)}


# ============================================================
# QUICK TEST
# From project root: PYTHONPATH=. python src/agents/nodes/analyser.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    test_queries = [
        "What is the recommended BMI threshold to classify obesity in adults?",
        "How does bariatric surgery compare to lifestyle intervention for type 2 diabetes?",
        "What steps should a clinician follow when screening for hypertension?",
        "What is metabolic syndrome?",
    ]

    print("\n" + "="*60)
    print("ANALYSER NODE TEST")
    print("="*60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = analyser_node({"query": query})
        print(f"  type    : {result['query_type']}")
        print(f"  entities: {result.get('entities', [])}")

    print("\n[analyser.py test complete]\n")

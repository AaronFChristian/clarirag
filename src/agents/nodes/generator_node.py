"""
================================================================================
FILE: src/agents/nodes/generator_node.py
================================================================================
WHAT THIS FILE DOES:
    Node 5 of the LangGraph pipeline. The answer generator — takes the
    retrieved chunks and produces a grounded, cited answer using Claude.

    This node has two modes:
    1. NORMAL: context is sufficient → generate answer with citations
    2. REFUSAL: context insufficient after retries → return safe refusal

    Every factual claim in the answer is linked to a specific Citation
    (doc_name + page_number + excerpt). If Claude tries to cite a document
    NOT in the retrieved chunks, the citation is rejected. This is the
    core anti-hallucination guardrail.

INPUT  (from GraphState):
    - state["query"]                 : original user question
    - state["retrieved_chunks"]      : top chunks from retriever
    - state["is_context_sufficient"] : from judge node
    - state["retry_count"]           : if >= MAX_RETRIES and not sufficient → refuse
    - state["query_type"]            : factual/comparative/procedural/definitional

OUTPUT (written back to GraphState):
    - state["answer"]         : the generated answer string
    - state["citations"]      : list of Citation objects
    - state["confidence"]     : 0.0–1.0
    - state["is_grounded"]    : True if answer uses real context
    - state["final_response"] : complete RAGResponse object
================================================================================
"""

import os
import json
import logging
from typing import Dict, Any, List

from anthropic import Anthropic
from dotenv import load_dotenv

from src.agents.state import GraphState, Citation, RAGResponse, INSUFFICIENT_CONTEXT_RESPONSE

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LLM_MODEL   = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
MAX_RETRIES = 2


def _build_context_string(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk is labelled with its source so Claude can cite it correctly.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        context += (
            f"\n[SOURCE {i+1}] Document: {chunk['doc_name']} | Page: {chunk['page_number']}\n"
            f"{chunk['text']}\n"
            f"{'─'*60}\n"
        )
    return context


def _validate_citations(citations: List[Dict], retrieved_chunks: List[Dict]) -> List[Citation]:
    """
    Validate that every citation refers to a document actually in the
    retrieved chunks. Reject any citation Claude invented from nowhere.

    This is the hallucination guardrail — if Claude cites a document that
    wasn't in the context, we drop that citation silently.

    Args:
        citations       : Raw citation dicts from Claude's response
        retrieved_chunks: Chunks actually passed to Claude

    Returns:
        List of validated Citation objects
    """
    # Build a set of valid (doc_name, page_number) pairs from retrieved chunks
    valid_sources = {
        (c["doc_name"], c["page_number"]) for c in retrieved_chunks
    }

    validated = []
    for cite in citations:
        doc  = cite.get("doc_name", "")
        page = int(cite.get("page_number", 0))

        if (doc, page) in valid_sources:
            # Valid citation — build Citation object
            validated.append(Citation(
                doc_name    = doc,
                page_number = page,
                excerpt     = cite.get("excerpt", "")[:300],
                relevance   = float(cite.get("relevance", 1.0))
            ))
        else:
            # Citation not in retrieved context — reject it
            logger.warning(
                f"[Generator] Rejected hallucinated citation: "
                f"{doc} p.{page} (not in retrieved chunks)"
            )

    return validated


def generator_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph Node 5: Generate a grounded answer with validated citations.

    Args:
        state: GraphState with query, retrieved_chunks, is_context_sufficient,
               retry_count, query_type

    Returns:
        Dict with answer, citations, confidence, is_grounded, final_response
    """
    query                 = state.get("query", "")
    retrieved_chunks      = state.get("retrieved_chunks", [])
    is_context_sufficient = state.get("is_context_sufficient", False)
    retry_count           = state.get("retry_count", 0)
    query_type            = state.get("query_type", "factual")

    logger.info(f"[Generator] Generating answer | sufficient={is_context_sufficient} "
                f"| retries={retry_count} | chunks={len(retrieved_chunks)}")

    # ── REFUSAL PATH ──────────────────────────────────────────────────────
    # Return safe refusal if context is insufficient after max retries
    if not is_context_sufficient and retry_count >= MAX_RETRIES:
        logger.info("[Generator] Context insufficient after max retries — returning safe refusal.")
        return {
            "answer"        : INSUFFICIENT_CONTEXT_RESPONSE.answer,
            "citations"     : [],
            "confidence"    : 0.0,
            "is_grounded"   : False,
            "final_response": INSUFFICIENT_CONTEXT_RESPONSE,
        }

    if not retrieved_chunks:
        logger.warning("[Generator] No chunks — returning safe refusal.")
        return {
            "answer"        : INSUFFICIENT_CONTEXT_RESPONSE.answer,
            "citations"     : [],
            "confidence"    : 0.0,
            "is_grounded"   : False,
            "final_response": INSUFFICIENT_CONTEXT_RESPONSE,
        }

    # ── GENERATION PATH ───────────────────────────────────────────────────
    context_str = _build_context_string(retrieved_chunks)
    client      = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""You are a precise clinical information assistant. Answer questions using ONLY the provided context from WHO clinical guidelines.

QUESTION: "{query}"
QUESTION TYPE: {query_type}

RETRIEVED CONTEXT:
{context_str}

INSTRUCTIONS:
1. Answer ONLY using information from the context above
2. For every factual claim, cite the source using [SOURCE N] where N matches the source number above
3. If the context doesn't fully address the question, say so explicitly
4. Do NOT invent, infer, or add information not in the context
5. Be concise but complete

Respond with ONLY a valid JSON object:
{{
  "answer": "<your answer with [SOURCE N] inline citations>",
  "citations": [
    {{
      "doc_name": "<exact filename from context>",
      "page_number": <integer>,
      "excerpt": "<the specific text you cited, max 150 chars>",
      "relevance": <0.0 to 1.0>
    }}
  ],
  "confidence": <0.0 to 1.0>,
  "answer_is_grounded": true or false
}}"""

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1000,
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

        answer      = parsed.get("answer", "").strip()
        raw_cites   = parsed.get("citations", [])
        confidence  = float(parsed.get("confidence", 0.5))
        is_grounded = bool(parsed.get("answer_is_grounded", True))

        # Validate citations — remove any that weren't in retrieved chunks
        validated_citations = _validate_citations(raw_cites, retrieved_chunks)

        logger.info(f"[Generator] Answer length: {len(answer)} chars | "
                    f"Citations: {len(validated_citations)}/{len(raw_cites)} valid | "
                    f"Confidence: {confidence:.2f}")

        # Build the complete RAGResponse
        final_response = RAGResponse(
            answer             = answer,
            citations          = validated_citations,
            query_type         = query_type,
            confidence         = confidence,
            is_grounded        = is_grounded and len(validated_citations) > 0,
            eval_scores        = {},   # filled by eval node
            retrieval_attempts = retry_count + 1,
        )

        return {
            "answer"        : answer,
            "citations"     : validated_citations,
            "confidence"    : confidence,
            "is_grounded"   : final_response.is_grounded,
            "final_response": final_response,
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"[Generator] JSON parse error: {e}")
        return {
            "answer"        : INSUFFICIENT_CONTEXT_RESPONSE.answer,
            "citations"     : [],
            "confidence"    : 0.0,
            "is_grounded"   : False,
            "final_response": INSUFFICIENT_CONTEXT_RESPONSE,
        }

    except Exception as e:
        logger.error(f"[Generator] Error: {e}")
        return {
            "answer"        : INSUFFICIENT_CONTEXT_RESPONSE.answer,
            "citations"     : [],
            "confidence"    : 0.0,
            "is_grounded"   : False,
            "final_response": INSUFFICIENT_CONTEXT_RESPONSE,
            "error"         : str(e),
        }


# ============================================================
# QUICK TEST
# From project root: PYTHONPATH=. python src/agents/nodes/generator_node.py
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath("."))

    from src.retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever()

    test_cases = [
        {
            "name"   : "Good query — should produce grounded answer",
            "query"  : "What is the BMI threshold for obesity in adults?",
            "sufficient": True,
        },
        {
            "name"   : "Insufficient context — should produce safe refusal",
            "query"  : "What is the price of insulin in the USA?",
            "sufficient": False,
        },
    ]

    print("\n" + "="*65)
    print("GENERATOR NODE TEST")
    print("="*65)

    for case in test_cases:
        print(f"\nTest : {case['name']}")
        print(f"Query: {case['query']}")

        chunks = retriever.retrieve(case["query"], top_k=5)

        state = {
            "query"                : case["query"],
            "retrieved_chunks"     : chunks,
            "is_context_sufficient": case["sufficient"],
            "retry_count"          : 0 if case["sufficient"] else 2,
            "query_type"           : "factual",
        }

        result = generator_node(state)

        print(f"\n  is_grounded : {result['is_grounded']}")
        print(f"  confidence  : {result['confidence']:.2f}")
        print(f"  citations   : {len(result['citations'])}")
        print(f"\n  Answer:\n  {result['answer'][:300]}...")

        if result["citations"]:
            print(f"\n  First citation:")
            c = result["citations"][0]
            print(f"    {c.doc_name} p.{c.page_number}")
            print(f"    Excerpt: {c.excerpt[:100]}...")

    print("\n[generator_node.py test complete]\n")

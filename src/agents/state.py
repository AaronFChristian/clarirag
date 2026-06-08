"""
================================================================================
FILE: src/agents/state.py
================================================================================
WHAT THIS FILE DOES:
    Defines the shared state object that flows through every node in the
    LangGraph pipeline. Think of it as the "memory" of one query's journey
    through the agent — every node reads from it and writes back to it.

WHY THIS EXISTS:
    LangGraph works by passing a single state dict between nodes. Each node
    receives the full state, does its job, and returns ONLY the fields it
    changed. LangGraph merges those changes back into the state automatically.

    Without a well-defined state schema:
    - Nodes can't communicate with each other
    - You can't debug what went wrong at which step
    - The conditional retry edge (judge → retriever) has no "retry_count" to check

THE FLOW:
    GraphState starts mostly empty. As the query moves through nodes:

    [START] → query filled in
    [Analyser] → query_type, entities filled in
    [Expander] → expanded_queries filled in
    [Retriever] → retrieved_chunks filled in
    [Judge]     → is_context_sufficient, retry_count updated
    [Generator] → answer, citations, confidence filled in
    [Eval]      → eval_scores filled in
    [END]       → full RAGResponse returned to API

INPUT:  A query string from the user (entered at graph invocation)
OUTPUT: A complete RAGResponse with answer, citations, and eval scores

PYDANTIC MODELS DEFINED HERE:
    - Citation      : one source reference (doc, page, excerpt)
    - RAGResponse   : the final structured answer returned to the user
    - GraphState    : the full LangGraph state TypedDict
================================================================================
"""

from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Citation Model ────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """
    Represents a single source reference attached to an answer.

    Every factual claim in the generated answer should be backed by
    at least one Citation. This is what makes ClariRAG auditable —
    users can verify every claim against the original document.

    Example:
        Citation(
            doc_name    = "obesity_management_guidelines.pdf",
            page_number = 51,
            excerpt     = "BMI is a weight-for-height index...",
            relevance   = 0.92
        )
    """

    doc_name    : str   = Field(description="Source PDF filename")
    page_number : int   = Field(description="Page number in the source document (1-indexed)")
    excerpt     : str   = Field(description="The specific text excerpt this citation refers to")
    relevance   : float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Relevance score of this chunk to the query (0.0 to 1.0)"
    )


# ── RAGResponse Model ─────────────────────────────────────────────────────────

class RAGResponse(BaseModel):
    """
    The final structured response returned by the ClariRAG pipeline.

    This is what the FastAPI endpoint returns as JSON. Every field is
    typed and validated by Pydantic — no raw strings, no silent failures.

    The API consumer gets:
    - A grounded answer (or a safe refusal)
    - Full citations for every claim
    - Query classification for transparency
    - Eval scores so the quality is always visible
    - A trace_id for debugging in LangSmith
    """

    # The answer itself
    answer      : str  = Field(description="The generated answer grounded in retrieved context")

    # Source citations — parallel to claims in the answer
    citations   : List[Citation] = Field(
        default_factory=list,
        description="List of source citations supporting the answer"
    )

    # Query metadata
    query_type  : str  = Field(
        default="unknown",
        description="Classified query type: factual | comparative | procedural | definitional"
    )

    # Confidence and quality signals
    confidence  : float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Model confidence in the answer (0.0 = refused, 1.0 = fully grounded)"
    )

    # Whether the system had sufficient context to answer
    is_grounded : bool  = Field(
        default=False,
        description="True if the answer is grounded in retrieved context. "
                    "False means the system returned a safe refusal."
    )

    # Ragas evaluation scores (filled in by the eval node)
    eval_scores : Dict[str, float] = Field(
        default_factory=dict,
        description="Ragas quality scores: faithfulness, answer_relevancy, context_precision"
    )

    # LangSmith trace ID for observability
    trace_id    : Optional[str] = Field(
        default=None,
        description="LangSmith trace ID for debugging this specific query"
    )

    # How many retrieval iterations were needed
    retrieval_attempts : int = Field(
        default=1,
        description="Number of retrieval attempts (1 = first try, 2 = retried once)"
    )


# ── GraphState TypedDict ──────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    The shared state object passed between all LangGraph nodes.

    TypedDict means it's a dict with known, typed keys.
    total=False means ALL keys are optional — nodes only need to set
    the fields they're responsible for. LangGraph merges partial updates.

    FIELD LIFECYCLE:
        query            → set at graph invocation (the user's question)
        query_type       → set by AnalyserNode
        entities         → set by AnalyserNode
        expanded_queries → set by ExpanderNode
        retrieved_chunks → set by RetrieverNode
        is_context_sufficient → set by JudgeNode
        retry_count      → incremented by JudgeNode on each retry
        answer           → set by GeneratorNode
        citations        → set by GeneratorNode
        confidence       → set by GeneratorNode
        is_grounded      → set by GeneratorNode
        eval_scores      → set by EvalNode
        error            → set by any node that catches an exception
        final_response   → set by GeneratorNode — the complete RAGResponse
    """

    # ── Input ─────────────────────────────────────────────────────────────
    query           : str
    # The original user query — set at graph invocation, never modified

    # ── Analyser outputs ──────────────────────────────────────────────────
    query_type      : str
    # One of: "factual" | "comparative" | "procedural" | "definitional"
    # Factual: "What is the recommended BMI threshold?"
    # Comparative: "How does bariatric surgery compare to lifestyle intervention?"
    # Procedural: "What steps should be followed for diabetes screening?"
    # Definitional: "What is metabolic syndrome?"

    entities        : List[str]
    # Key medical/clinical entities extracted from the query
    # e.g. ["BMI", "obesity", "adults"] for the BMI threshold query
    # Used to improve query expansion in the next node

    # ── Expander outputs ──────────────────────────────────────────────────
    expanded_queries : List[str]
    # 2-3 alternative phrasings of the original query
    # e.g. ["BMI classification obesity", "body mass index adult threshold WHO"]
    # These are passed to the hybrid retriever for broader recall

    # ── Retriever outputs ─────────────────────────────────────────────────
    retrieved_chunks : List[Dict[str, Any]]
    # Top-K chunk dicts from hybrid_retriever.py
    # Each has: chunk_id, doc_name, page_number, text, rerank_score, etc.

    # ── Judge outputs ─────────────────────────────────────────────────────
    is_context_sufficient : bool
    # True  → proceed to GeneratorNode
    # False → retry retrieval (if retry_count < 2) OR refuse to answer

    retry_count     : int
    # How many times retrieval has been retried for this query
    # Starts at 0, max 2 retries before falling back to safe refusal

    missing_aspects : str
    # If judge says insufficient, what specifically is missing?
    # Used to reformulate the query for the retry

    # ── Generator outputs ─────────────────────────────────────────────────
    answer          : str
    citations       : List[Citation]
    confidence      : float
    is_grounded     : bool

    # ── Eval outputs ──────────────────────────────────────────────────────
    eval_scores     : Dict[str, float]
    # {"faithfulness": 0.85, "answer_relevancy": 0.79, "context_precision": 0.81}

    # ── Error handling ────────────────────────────────────────────────────
    error           : Optional[str]
    # Set if any node throws an exception — allows graceful degradation

    # ── Final output ──────────────────────────────────────────────────────
    final_response  : Optional[RAGResponse]
    # The complete RAGResponse object — read by the FastAPI endpoint


# ── Safe refusal constant ─────────────────────────────────────────────────────

INSUFFICIENT_CONTEXT_RESPONSE = RAGResponse(
    answer      = (
        "I don't have sufficient information in the knowledge base to answer "
        "this question reliably. Please rephrase your question or ask about "
        "topics covered in the clinical guidelines (diabetes, obesity, "
        "cardiovascular disease, clinical trials, COVID-19 management)."
    ),
    citations   = [],
    query_type  = "unknown",
    confidence  = 0.0,
    is_grounded = False,
    eval_scores = {},
)
# This constant is returned by the GeneratorNode when:
# 1. Judge says context is insufficient after 2 retries
# 2. The LLM tries to cite a document not in retrieved_chunks
# 3. Any unrecoverable error occurs during generation


# ============================================================
# QUICK TEST — verify models instantiate correctly
# From project root: PYTHONPATH=. python src/agents/state.py
# ============================================================
if __name__ == "__main__":

    print("\nTesting Citation model...")
    c = Citation(
        doc_name    = "obesity_management_guidelines.pdf",
        page_number = 51,
        excerpt     = "BMI is a weight-for-height index that is commonly used...",
        relevance   = 0.92
    )
    print(f"  Citation created: {c.doc_name} p.{c.page_number} (relevance: {c.relevance})")

    print("\nTesting RAGResponse model...")
    r = RAGResponse(
        answer      = "The BMI threshold for obesity in adults is 30 kg/m² or above.",
        citations   = [c],
        query_type  = "factual",
        confidence  = 0.95,
        is_grounded = True,
        eval_scores = {"faithfulness": 0.88, "answer_relevancy": 0.91},
    )
    print(f"  RAGResponse created:")
    print(f"    Answer     : {r.answer[:60]}...")
    print(f"    Citations  : {len(r.citations)}")
    print(f"    Grounded   : {r.is_grounded}")
    print(f"    Confidence : {r.confidence}")
    print(f"    Eval scores: {r.eval_scores}")

    print("\nTesting INSUFFICIENT_CONTEXT_RESPONSE constant...")
    print(f"  is_grounded: {INSUFFICIENT_CONTEXT_RESPONSE.is_grounded}")
    print(f"  confidence : {INSUFFICIENT_CONTEXT_RESPONSE.confidence}")
    print(f"  answer     : {INSUFFICIENT_CONTEXT_RESPONSE.answer[:60]}...")

    print("\nTesting GraphState structure...")
    # GraphState is a TypedDict — test that it accepts the right fields
    sample_state: GraphState = {
        "query"                : "What is the BMI threshold for obesity?",
        "query_type"           : "factual",
        "entities"             : ["BMI", "obesity", "adults"],
        "expanded_queries"     : ["BMI classification adults", "body mass index obesity threshold"],
        "retrieved_chunks"     : [],
        "is_context_sufficient": True,
        "retry_count"          : 0,
        "missing_aspects"      : "",
        "answer"               : "",
        "citations"            : [],
        "confidence"           : 0.0,
        "is_grounded"          : False,
        "eval_scores"          : {},
        "error"                : None,
        "final_response"       : None,
    }
    print(f"  GraphState created with {len(sample_state)} fields")
    print(f"  query        : {sample_state['query']}")
    print(f"  query_type   : {sample_state['query_type']}")
    print(f"  entities     : {sample_state['entities']}")
    print(f"  retry_count  : {sample_state['retry_count']}")

    print("\n[state.py test complete — all models valid]\n")

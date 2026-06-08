"""
================================================================================
FILE: src/api/main.py
================================================================================
WHAT THIS FILE DOES:
    The FastAPI application that exposes the ClariRAG pipeline as a REST API.
    This is what gets deployed to AWS and what the React UI calls.

ENDPOINTS:
    POST /query          → runs the full pipeline, returns RAGResponse
    GET  /health         → service health check
    GET  /docs           → auto-generated OpenAPI docs (FastAPI built-in)

INPUT  (POST /query):
    {"query": "What is the BMI threshold for obesity in adults?"}

OUTPUT (POST /query):
    {
        "answer"      : "The BMI threshold for obesity...",
        "citations"   : [{"doc_name": "obesity.pdf", "page_number": 52, ...}],
        "query_type"  : "factual",
        "confidence"  : 1.0,
        "is_grounded" : true,
        "eval_scores" : {}
    }

HOW TO RUN:
    From project root:
    PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

    Then visit: http://localhost:8000/docs for interactive API docs
================================================================================
"""

import os
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.graph import run_pipeline
from src.agents.state import RAGResponse

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ClariRAG API",
    description = (
        "Production-grade Agentic RAG over clinical guidelines. "
        "Hybrid retrieval (BM25 + Pinecone) + LangGraph agent + "
        "citation validation + Ragas evals."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(
        ...,
        min_length  = 3,
        max_length  = 500,
        description = "The clinical question to answer",
        example     = "What is the BMI threshold for obesity in adults?"
    )


class CitationResponse(BaseModel):
    """A single source citation returned in the response"""
    doc_name    : str
    page_number : int
    excerpt     : str
    relevance   : float


class QueryResponse(BaseModel):
    """Full response from POST /query"""
    answer             : str
    citations          : List[CitationResponse]
    query_type         : str
    confidence         : float
    is_grounded        : bool
    eval_scores        : dict
    retrieval_attempts : int
    latency_ms         : float   # how long the pipeline took


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status     : str
    version    : str
    model      : str
    index_name : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    response_model = QueryResponse,
    summary        = "Answer a clinical question",
    description    = (
        "Runs the full ClariRAG pipeline: query analysis → expansion → "
        "hybrid retrieval → sufficiency judge → answer generation with citations."
    )
)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main endpoint — takes a clinical question and returns a grounded answer.

    The pipeline runs:
    1. Analyse query type and extract entities
    2. Expand into 2-3 variants
    3. Hybrid retrieval (BM25 + Pinecone + reranker)
    4. Judge sufficiency (retry up to 2x if needed)
    5. Generate answer with validated citations

    Returns a safe refusal if the question is outside the corpus scope.
    """
    query = request.query.strip()
    logger.info(f"POST /query | query='{query[:60]}'")

    start_time = time.time()

    try:
        # Run the full LangGraph pipeline
        response: RAGResponse = run_pipeline(query)

        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"POST /query | latency={latency_ms:.0f}ms | "
                    f"grounded={response.is_grounded} | "
                    f"citations={len(response.citations)}")

        # Convert Citation objects to response dicts
        citations = [
            CitationResponse(
                doc_name    = c.doc_name,
                page_number = c.page_number,
                excerpt     = c.excerpt,
                relevance   = c.relevance,
            )
            for c in response.citations
        ]

        return QueryResponse(
            answer             = response.answer,
            citations          = citations,
            query_type         = response.query_type,
            confidence         = response.confidence,
            is_grounded        = response.is_grounded,
            eval_scores        = response.eval_scores,
            retrieval_attempts = response.retrieval_attempts,
            latency_ms         = round(latency_ms, 1),
        )

    except Exception as e:
        logger.error(f"POST /query | error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
)
async def health_endpoint() -> HealthResponse:
    """
    Health check endpoint. Returns service status and configuration.
    Used by deployment infrastructure to verify the service is running.
    """
    return HealthResponse(
        status     = "healthy",
        version    = "1.0.0",
        model      = os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
        index_name = os.getenv("PINECONE_INDEX_NAME", "clarirag"),
    )


@app.get("/", summary="Root")
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "message"    : "ClariRAG API is running",
        "docs"       : "/docs",
        "health"     : "/health",
        "query"      : "POST /query",
    }


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1,    # 1 worker because our models are loaded as singletons
    )

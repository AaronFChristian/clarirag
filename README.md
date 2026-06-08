# ClariRAG

*Production-grade Agentic RAG over WHO Clinical Guidelines*

Hybrid retrieval (BM25 + Pinecone) + LangGraph agent + citation validation + MCP server. Built to demonstrate what "advanced RAG" actually looks like in production, not in a tutorial.

*Live demo:* [clarirag-ui.vercel.app](https://clarirag-ui.vercel.app)

---

## Results

| Metric | Before | After |
|--------|--------|-------|
| Retrieval Hit Rate @5 | 58% (dense only) | 81% (hybrid + rerank) |
| Hallucinated citations | Present | 0 (validated guardrail) |
| Out-of-scope refusal | None | Safe refusal on every out-of-scope query |
| Ragas Faithfulness | N/A | 0.86 |

---

## What it does

A user asks a clinical question in plain English. The system:

1.⁠ ⁠Classifies the query type (factual / comparative / procedural / definitional)
2.⁠ ⁠Expands it into 2-3 variants for broader recall
3.⁠ ⁠Runs hybrid retrieval: BM25 (keyword) + Pinecone (semantic) merged via Reciprocal Rank Fusion
4.⁠ ⁠Re-scores candidates with a cross-encoder reranker
5.⁠ ⁠Judges whether context is sufficient (retries up to 2x with a reformulated query)
6.⁠ ⁠Generates a grounded answer with validated citations (doc name + page + excerpt)
7.⁠ ⁠Returns a safe refusal if the question is outside the corpus scope

Every factual claim is linked to a specific source. Citations that reference documents not in the retrieved context are rejected before the answer reaches the user.

---

## Architecture


User query
    |
    v
[Analyser]      classifies query type, extracts entities
    |
    v
[Expander]      generates 2-3 query variants
    |
    v
[Retriever]     BM25 + Pinecone + RRF + CrossEncoder reranker
    |
    v
[Judge]         sufficient? --> retry loop (max 2x) or proceed
    |
    v
[Generator]     grounded answer + validated citations (Pydantic)
    |
    v
FastAPI /query  returns RAGResponse JSON
    |
    v
React UI        chat interface with citation viewer
    +
MCP Server      exposes KB as tool for Claude Desktop / any agent


---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Orchestration | LangGraph | Explicit state machine -- debuggable in production vs AgentExecutor |
| LLM | Anthropic Claude | Best structured JSON output for citation-heavy tasks |
| Dense retrieval | Pinecone (all-MiniLM-L6-v2) | Managed, production-grade, 1911 vectors |
| Sparse retrieval | BM25 (rank_bm25) | Exact term matching for regulatory terminology |
| Fusion | Reciprocal Rank Fusion | Combines ranked lists without score normalization |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint (query, chunk) scoring adds ~12% precision@5 |
| Evals | Ragas | Faithfulness + answer relevancy + context precision |
| Observability | LangSmith | Full node-level trace with token cost and latency |
| MCP | FastMCP | Exposes KB as callable tool for any MCP-compatible agent |
| Backend | FastAPI | Async, auto-docs, Pydantic native |
| Frontend | React + Vite | Deployed to Vercel |
| CI/CD | GitHub Actions | Eval gate blocks merges on quality regression |

---

## Corpus

5 WHO clinical guideline PDFs (299 pages, 1911 chunks):

•⁠  ⁠⁠ clinical_trials_best_practices.pdf ⁠ -- WHO best practices for trial design and randomization
•⁠  ⁠⁠ covid19_clinical_management.pdf ⁠ -- WHO COVID-19 clinical management policy
•⁠  ⁠⁠ ncd_diabetes_policy.pdf ⁠ -- WHO diabetes prevention and NCD control resolution
•⁠  ⁠⁠ ncd_prevention_interventions.pdf ⁠ -- WHO NCD prevention policy options
•⁠  ⁠⁠ obesity_management_guidelines.pdf ⁠ -- Australian clinical guidelines for obesity management (BMI, interventions, bariatric surgery)

---

## Project Structure


clarirag/
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # PDF loading with page metadata
│   │   ├── chunker.py          # 512-char chunks with 64-char overlap
│   │   └── embedder.py         # Embed + upsert to Pinecone
│   ├── retrieval/
│   │   ├── sparse_retriever.py # BM25 index (rank_bm25)
│   │   └── hybrid_retriever.py # RRF + CrossEncoder reranker
│   ├── agents/
│   │   ├── state.py            # GraphState TypedDict + Pydantic models
│   │   ├── graph.py            # LangGraph pipeline with retry edge
│   │   └── nodes/
│   │       ├── analyser.py     # Node 1: query classification
│   │       ├── expander.py     # Node 2: query expansion
│   │       ├── retriever_node.py # Node 3: hybrid retrieval
│   │       ├── judge_node.py   # Node 4: sufficiency + retry
│   │       └── generator_node.py # Node 5: answer + citations
│   ├── api/
│   │   └── main.py             # FastAPI endpoints
│   └── mcp/
│       └── server.py           # FastMCP server (3 tools)
├── evals/
│   └── retrieval_test_set.json # Labeled test set for CI gate
├── clarirag-ui/                # React + Vite frontend
├── data/
│   └── raw/                    # WHO PDFs (not committed)
├── DECISIONS.md                # 10 architectural decision records
└── .github/workflows/          # CI eval gate


---

## Key Design Decisions

See [DECISIONS.md](./DECISIONS.md) for full architectural decision records. Key choices:

*LangGraph over LangChain AgentExecutor* -- explicit state transitions make the retry loop debuggable and auditable. Every node's input/output is inspectable in LangSmith.

*Hybrid retrieval over dense-only* -- clinical guidelines use exact regulatory terminology. BM25 recovers lexical precision that dense embeddings miss. Hybrid + rerank improved hit rate from 58% to 81%.

*Citation validation guardrail* -- every citation is validated against the set of actually-retrieved chunks. Citations referencing documents not in the context are rejected before the answer is returned. Caught and rejected fabricated citations in testing.

*Sufficiency judge with retry loop* -- forces deliberate evaluation before generation. The LLM judges whether retrieved context actually answers the question, reformulates the query on failure, and retries up to 2 times. Falls back to a safe refusal rather than hallucinating.

*MCP server* -- exposes the knowledge base as a standard MCP tool. Any MCP-compatible agent (Claude Desktop, Cursor) can query it without custom integration.

---

## Quick Start

⁠ bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/clarirag.git
cd clarirag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set environment variables
cp .env.template .env
# Add: ANTHROPIC_API_KEY, PINECONE_API_KEY, OPENAI_API_KEY (for embeddings)

# Build the ingestion pipeline
PYTHONPATH=. python src/ingestion/embedder.py

# Start the API
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

# Start the UI (separate terminal)
cd clarirag-ui && npm install && npm run dev
 ⁠

Then open ⁠ http://localhost:5173 ⁠

---

## API

*POST /query*
⁠ json
{
  "query": "What is the BMI threshold for obesity in adults?"
}
 ⁠

Response:
⁠ json
{
  "answer": "According to WHO classifications, the BMI threshold for obesity in adults is 30.0 kg/m2 or above...",
  "citations": [
    {
      "doc_name": "obesity_management_guidelines.pdf",
      "page_number": 52,
      "excerpt": "BMI 30.0-34.9 Obesity I, 35.0-39.9 Obesity II, >= 40.0 Obesity III. Source: WHO (2000).",
      "relevance": 1.0
    }
  ],
  "is_grounded": true,
  "confidence": 1.0,
  "query_type": "factual",
  "latency_ms": 4823.1
}
 ⁠

*GET /health* -- service status

*Full API docs:* ⁠ http://localhost:8000/docs ⁠

---

## Connect to Claude Desktop (MCP)

Add to ⁠ ~/Library/Application Support/Claude/claude_desktop_config.json ⁠:

⁠ json
{
  "mcpServers": {
    "clarirag": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/clarirag",
      "env": { "PYTHONPATH": "/path/to/clarirag" }
    }
  }
}
 ⁠

Available tools: ⁠ search_knowledge_base ⁠, ⁠ list_documents ⁠, ⁠ get_health ⁠

---

## Evals and Observability

*Retrieval eval* (runs on every push via GitHub Actions):
•⁠  ⁠Hit Rate @5: dense 58% vs hybrid+rerank 81%
•⁠  ⁠Mean Reciprocal Rank tracked per commit

*Ragas end-to-end eval:*
•⁠  ⁠Faithfulness: 0.86
•⁠  ⁠Answer Relevancy: 0.79
•⁠  ⁠Context Precision: 0.81

*LangSmith tracing:* every query logged with full node-level trace, token cost, and latency per step.

---

## What this demonstrates

•⁠  ⁠Production RAG architecture (not a demo or tutorial clone)
•⁠  ⁠Advanced retrieval: hybrid search + RRF + cross-encoder reranker
•⁠  ⁠Agentic reasoning: LangGraph state machine with conditional retry edge
•⁠  ⁠Eval discipline: CI-gated quality gate, labeled test set, Ragas scoring
•⁠  ⁠AI observability: full trace with LangSmith
•⁠  ⁠MCP integration: knowledge base exposed as agent-ready tool
•⁠  ⁠Full-stack shipping: FastAPI + React + Vercel + Docker-ready

---

## Author

*Aaron Christian*
MS Information Systems, San Diego State University (GPA 3.7)
B.Tech Computer Science and Business Systems

[LinkedIn](www.linkedin.com/in/aaronchristi7n) | [Portfolio](https://clarirag-ui.vercel.app)

---

Built in 3 days as part of a portfolio sprint targeting AI Engineer, LLM Engineer, and Analytics Engineer roles in the USA.

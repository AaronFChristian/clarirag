# ClariRAG

**The agent that shows its work, and knows when to stay quiet.**

An agentic RAG system over WHO clinical guidelines. Every claim is tied to a page number. Every citation is checked before it reaches you. If the answer is not in the corpus, it says so instead of guessing.

Live demo: [clarirag-ui.vercel.app](https://clarirag-ui.vercel.app)
Source: [github.com/AaronFChristian/clarirag](https://github.com/AaronFChristian/clarirag)

---

## The problem

Ask a general purpose LLM a clinical question and it will answer fluently, confidently, and sometimes wrong, with no way to check where the information came from. In a clinical context that gap between confidence and correctness is the whole problem. ClariRAG closes it by forcing every answer through retrieval, a sufficiency check, and a citation audit before it is allowed to reach the user.

## What it does, in one line

Ask a question about WHO clinical guidelines. ClariRAG retrieves the relevant passages using both keyword and semantic search, decides whether it actually knows enough to answer, generates a response with page level citations, and rejects any citation that does not point to a real retrieved source.

---

## How it works

A single query moves through five agent steps, with a built in retry loop when context is not good enough.

![ClariRAG data flow](diagrams/dataflow.svg)

1. **Analyser** classifies the query (factual, comparative, procedural, definitional) and extracts entities
2. **Expander** generates 2 to 3 phrasings of the same question to widen recall
3. **Hybrid retriever** runs BM25 and Pinecone in parallel, merges results with reciprocal rank fusion, and reranks with a cross encoder
4. **Sufficiency judge** asks: does this context actually answer the question? If not, it rewrites the query and loops back, up to twice
5. **Generator** writes the answer and attaches citations, each one checked against the retrieved chunks before being returned

---

## The trust pipeline

This is the part that makes ClariRAG different from a typical RAG demo. Two independent search strategies are fused, reranked, and then every resulting citation is checked against what was actually retrieved. Anything fabricated gets dropped before it reaches the answer.

![The ClariRAG trust pipeline](diagrams/trust_pipeline.svg)

BM25 catches exact clinical terminology (section numbers, drug names, diagnostic codes) that embeddings tend to blur together. Pinecone catches the cases where the right answer uses different words than the question. Fusing both, reranking with a cross encoder, and validating citations afterward turned a 58% retrieval hit rate into 81%, and reduced hallucinated citations to zero in testing.

---

## System architecture

![ClariRAG system architecture](diagrams/architecture.svg)

The React UI and Claude Desktop are two front doors into the same agent. Claude Desktop talks to it through a FastMCP server exposing three tools (search_knowledge_base, list_documents, get_health), so any MCP compatible agent can use ClariRAG as a tool without custom integration code.

---

## Tech stack

**Agent orchestration**
- LangGraph, 5 node state machine with a conditional retry edge
- Anthropic Claude, used for all four reasoning nodes

**Retrieval**
- BM25 (rank_bm25) for lexical search over 1,911 chunks
- Pinecone serverless for dense vector search (384 dim, all-MiniLM-L6-v2)
- Cross encoder reranker (ms-marco-MiniLM-L-6-v2) on top 20 fused candidates

**Serving**
- FastAPI, async REST API with auto generated docs
- FastMCP, exposes the knowledge base as an MCP server

**Frontend**
- React + Vite, deployed on Vercel
- Citation cards with collapsible source excerpts and confidence badges

**Evaluation and observability**
- Ragas for faithfulness, answer relevancy, and context precision
- LangSmith for full node level tracing, latency, and token cost
- GitHub Actions eval gate on retrieval quality

**Corpus**
- 5 WHO clinical guideline PDFs, 299 pages, 1,911 chunks
- Topics: obesity management, diabetes and NCD policy, clinical trial best practices, COVID-19 management

---

## Results

| Metric | Before | After |
|---|---|---|
| Retrieval hit rate @5 | 58% (dense only) | 81% (hybrid + rerank) |
| Hallucinated citations | Present | 0 (validated guardrail) |
| Out of scope queries | Answered anyway | Safe refusal every time |
| Ragas faithfulness | N/A | 0.86 |

---

## Project structure

```
clarirag/
├── src/
│   ├── ingestion/        loader, chunker, embedder
│   ├── retrieval/        BM25 index, hybrid retriever (RRF + rerank)
│   ├── agents/
│   │   ├── state.py       GraphState + Pydantic models
│   │   ├── graph.py        LangGraph pipeline with retry edge
│   │   └── nodes/           analyser, expander, retriever, judge, generator
│   ├── api/                FastAPI app
│   └── mcp/                 FastMCP server
├── clarirag-ui/            React + Vite frontend
├── diagrams/                architecture, data flow, trust pipeline SVGs
├── evals/                   labeled retrieval test set
├── DECISIONS.md            10 architectural decision records
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/AaronFChristian/clarirag.git
cd clarirag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.template .env
# add ANTHROPIC_API_KEY and PINECONE_API_KEY

PYTHONPATH=. python src/ingestion/embedder.py
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

cd clarirag-ui && npm install && npm run dev
```

Open `http://localhost:5173`. API docs at `http://localhost:8000/docs`.

---

## Connect from Claude Desktop

```json
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
```

Available tools: `search_knowledge_base`, `list_documents`, `get_health`.

---

## Design decisions

Full reasoning for every major choice, including alternatives considered and rejected, is in [DECISIONS.md](./DECISIONS.md). Highlights:

- LangGraph over AgentExecutor, for an explicit, auditable retry loop
- Hybrid retrieval over dense only, because clinical terminology needs exact lexical matches
- A dedicated judge node, so the system can say "I do not know" instead of extrapolating
- Citation validation as a hard guardrail, not a prompt instruction

---

## Author

**Aastha Joshi**
MS Information Systems, San Diego State University
B.Tech Computer Science and Business Systems

[LinkedIn](https://linkedin.com/in/aasthajoshi) | [Live demo](https://clarirag-ui.vercel.app)

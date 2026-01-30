
# 08 — Retrieval‑Augmented Generation (RAG) & Vector Search (Orientation)

## Pipeline
```mermaid
flowchart LR
D[(Docs)] --> C[Chunk] --> EM[Embed]
EM --> IDX[(Vector DB)]
Q[User Query] --> EQ[Embed Query] --> RET[Retrieve Top‑k]
RET --> P[Prompt Compose] --> LLM --> A[Answer]
```

Topics: chunking strategies, embeddings choice, ANN indexes (HNSW/IVF), prompt templating, citation and hallucination checks, evaluation (faithfulness).

# RAG basic lab

Compare ways of supplying context to an LLM over the same document
(`docs/company.md`), numbered in learning order:

| Script                       | Strategy                                            | Lesson                                                           |
| ---------------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| `01_in_context_raw.py`       | Stuff the whole document into the prompt (raw SDK)  | Basic in-context learning; token cost grows with document length |
| `02_in_context_langchain.py` | Same strategy, using LangChain                      | Raw SDK vs abstraction; `response.text` vs `.content`            |
| `03_rag_basic.py`            | Split -> embed -> index -> retrieve -> inject chunk | Basic RAG; only relevant chunks are sent per request             |

## Run

```bash
cd /home/danny/Study/side-project/mlops-lab
printf 'Chính sách làm việc như thế nào?\n' | poetry run python labs/rag_basic/01_in_context_raw.py
printf 'Chính sách làm việc như thế nào?\n' | poetry run python labs/rag_basic/02_in_context_langchain.py
printf 'Chính sách làm việc như thế nào?\n' | poetry run python labs/rag_basic/03_rag_basic.py
```

## Lesson 2 notes: Embeddings

Goal of this lesson: understand what embeddings enable for an AI application —
not the math behind them.

Practical facts:

- An embedding transforms text into a fixed-size vector: **3072 floats** for
  `gemini-embedding-001`, ~12KB per chunk at float32. Same size for a short
  sentence and a long paragraph.
- Meaning lives in the vector's **direction**, not its length. The API returns
  L2-normalized vectors (norm ≈ 1), so a plain dot product already equals
  cosine similarity — no division needed. This is what vector stores exploit.
- Cosine similarity is cheap math (multiply + add) → searching millions of
  chunks takes milliseconds. That is why RAG retrieves by similarity instead
  of asking an LLM to compare texts one by one.

Experiment results (pure-Python cosine, `gemini-embedding-001`):

| Pair                                       | Score         |
| ------------------------------------------ | ------------- |
| Same meaning, different wording (VN)       | 0.7650        |
| Same meaning, different language (EN ↔ VN) | 0.8472        |
| Related topics (VN)                        | 0.6882        |
| Unrelated topics (VN)                      | 0.6028–0.6136 |
| Unrelated + different language (EN ↔ VN)   | 0.5209        |

Rules of thumb for building apps:

1. **Only rankings matter.** Unrelated text scores ~0.5–0.6 because sentences
   share language and tone. Never hardcode absolute thresholds ("0.5 =
   unrelated") — always retrieve top-k within a corpus.
2. **Cross-language retrieval works** without translation — useful for
   mixed-language corpora.
3. Embeddings are near-deterministic across calls (~0.0005 drift). If
   similarity suddenly looks wrong in production, check the model version
   first.

## Next steps

- Use a vector store with persistence (Chroma) instead of `InMemoryVectorStore`.
- Separate embedding tasks: `task_type='RETRIEVAL_QUERY'` for queries and
  `'RETRIEVAL_DOCUMENT'` for documents.
- Replace the shared doc with a larger corpus to make retrieval pay off.

# RAG basic lab

Compare ways of supplying context to an LLM over the same document
(`docs/company.md`), numbered in learning order:

| Script                       | Strategy                                              | Lesson                                                         |
| ---------------------------- | ----------------------------------------------------- | -------------------------------------------------------------- |
| `01_in_context_raw.py`       | Stuff the whole document into the prompt (raw SDK)    | In-context learning; token cost grows with document length     |
| `02_vector_embedding.py`     | Inspect embeddings, compute cosine similarity by hand | What embeddings are; similarity = ranking, not absolute scores |
| `03_rag_basic.py`            | Split -> embed -> index -> retrieve -> inject chunk   | Basic RAG; chunking experiments with a R&D harness             |
| `04_in_context_langchain.py` | Same strategy as 01, using LangChain                  | Raw SDK vs abstraction                                         |

## Run

```bash
cd /home/danny/Study/side-project/mlops-lab
printf 'Chính sách làm việc như thế nào?\n' | poetry run python labs/rag_basic/01_in_context_raw.py
printf 'Chính sách làm việc như thế nào?\n' | poetry run python labs/rag_basic/03_rag_basic.py
```

Every script anchors its paths to `__file__`, so they run from any cwd.
The API key is read from `.env` at the project root.

## Lesson notes

- [02: Embeddings](02_embeddings.md)
- [03: Chunking](03_chunking.md)
- [04: In-context with LangChain (baseline)](04_in_context_langchain.md)

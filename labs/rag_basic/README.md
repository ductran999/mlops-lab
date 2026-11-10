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

Every script anchors its paths to `__file__`, so they run from any cwd.
The API key is read from `.env` at the project root.

## Next steps

- Use a vector store with persistence (Chroma) instead of `InMemoryVectorStore`.
- Separate embedding tasks: `task_type='RETRIEVAL_QUERY'` for queries and
  `'RETRIEVAL_DOCUMENT'` for documents.
- Replace the shared doc with a larger corpus to make retrieval pay off.

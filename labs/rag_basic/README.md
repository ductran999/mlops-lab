# RAG basic lab

So sánh 2 cách cấp ngữ cảnh cho LLM trên cùng một tài liệu (`src/company.md`):

| Script             | Chiến lược                                          | Ý nghĩa                                                          |
| ------------------ | --------------------------------------------------- | ---------------------------------------------------------------- |
| `01_in_context.py` | Nhét cả tài liệu vào prompt                         | In-context learning cơ bản; token cost tăng theo độ dài tài liệu |
| `02_rag_basic.py`  | Split -> embed -> index -> retrieve -> inject chunk | RAG basic; chỉ gửi đoạn liên quan tới câu hỏi                    |

## Chạy

```bash
cd /home/danny/Study/side-project/mlops-lab
printf 'Chính sách làm việc như thế nào?\n' | poetry run python src/rag_basic/01_in_context.py
printf 'Chính sách làm việc như thế nào?\n' | poetry run python src/rag_basic/02_rag_basic.py
```

Hai script đều dùng model `gemini-3.5-flash-lite` và API key trong `.env` (`API_KEY`).

## Bước tiếp theo

- Dùng vector store có persistence (Chroma) thay vì `InMemoryVectorStore`.
- Thêm embedding query riêng với `task_type='RETRIEVAL_QUERY'` và document với `RETRIEVAL_DOCUMENT`.
- Thay shared doc bằng corpus lớn hơn để thấy rõ lợi ích của retrieval.

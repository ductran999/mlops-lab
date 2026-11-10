"""RAG basic: retrieve only the relevant chunks, then stuff those into the prompt.

Pipeline: split -> embed -> index -> retrieve -> augment prompt -> generate.
Unlike 01_in_context.py, only the retrieved chunks are sent per request, so
prompt cost no longer grows with the full document size.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Same path anchors as 01_in_context.py
DOC_PATH = Path(__file__).resolve().parents[1] / "company.md"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 2


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.environ["API_KEY"]

    text = DOC_PATH.read_text(encoding="utf-8")

    # 1) Split the document into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    print(f"[1] Split into {len(chunks)} chunks")

    # 2) Embed chunks and build an in-memory vector index
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )
    vector_store = InMemoryVectorStore.from_texts(chunks, embeddings)
    print("[2] Built in-memory vector index")

    question = input("Nhập câu hỏi của bạn: ")

    # 3) Retrieve only the chunks relevant to the question
    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(f"- {d.page_content}" for d in docs)
    print(f"[3] Retrieved {len(docs)} chunks for the question")

    # 4) Inject only the retrieved chunks into the prompt
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, api_key=api_key)
    prompt = f"""Bạn là trợ lý nội bộ. Dưới đây là các đoạn trích liên quan từ tài liệu công ty:
-----------------
{context}
-----------------

Dựa vào các đoạn trích trên, hãy trả lời câu hỏi. Nếu không đủ thông tin, hãy nói bạn không tìm thấy thông tin trong tài liệu.
Câu hỏi: {question}
"""

    response = llm.invoke(prompt)
    print(response.text)


if __name__ == "__main__":
    main()

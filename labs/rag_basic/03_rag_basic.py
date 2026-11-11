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
DOC_PATH = Path(__file__).resolve().parents[0] / "docs/company.md"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 2


def naive_chunking(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text)
    print(f"[1] Split into {len(chunks)} chunks")

    return chunks


def embed_chunks(chunk_size: int, chunks: list[str], api_key: str) -> InMemoryVectorStore:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )
    vector_store = InMemoryVectorStore.from_texts(chunks, embeddings)

    # Get embedding dimension
    sample_embedding = embeddings.embed_query("dimension probe")
    dim = len(sample_embedding)

    # Calculate metrics
    chunk_lengths = [len(chunk) for chunk in chunks]

    num_chunks = len(chunks)
    num_indexed = len(vector_store.store)

    avg_chunk_chars = sum(chunk_lengths) / num_chunks if num_chunks > 0 else 0

    orphan_chunks = sum(length < chunk_size * 0.3 for length in chunk_lengths)

    # Estimate embedding storage size (float32)
    index_bytes = num_indexed * dim * 4

    metrics = {
        "num_chunks": num_chunks,
        "num_indexed": num_indexed,
        "dim": dim,
        "index_bytes": index_bytes,
        "avg_chunk_chars": round(avg_chunk_chars, 2),
        "orphan_chunks": orphan_chunks,
    }

    print(f"[2] Built in-memory vector index. metrics:{metrics}")

    return vector_store


def vectorize(text: str, api_key: str, chunk_size: int, chunk_overlap: int):
    print(f"Experiment with chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

    chunks = naive_chunking(text, chunk_size, chunk_overlap)

    return embed_chunks(chunk_size, chunks, api_key)


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.environ["GOOGLE_API_KEY"]

    text = DOC_PATH.read_text(encoding="utf-8")
    vector_store = vectorize(text, api_key, CHUNK_SIZE, CHUNK_OVERLAP)

    question = input("Question: ")

    # 3) Retrieve only the chunks relevant to the question
    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(f"- {d.page_content}" for d in docs)
    print(f"[3] Retrieved {len(docs)} chunks for the question")

    # 4) Inject only the retrieved chunks into the prompt
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, api_key=api_key)
    prompt = f"""You are an internal assistant. The following are relevant excerpts from the company document:
    -----------------
    {context}
    -----------------

    Based on the excerpts above, please answer the question. If there is not enough information, please say you did not find the information in the document.
    Question: {question}
    """

    response = llm.invoke(prompt)
    print(response.text)
    print(response.usage_metadata)


if __name__ == "__main__":
    main()

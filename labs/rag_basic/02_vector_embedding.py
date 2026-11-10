"""
Explore vector embeddings for retrieval-augmented generation (RAG).
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MODEL = "gemini-embedding-001"


def embed_context(client: genai.Client, content: str) -> list[float]:
    """Embed the provided content using the specified model."""

    result = client.models.embed_content(model=MODEL, contents=content)

    return result.embeddings


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate the cosine similarity between two vectors."""

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    print(norm_a)
    print(norm_b)

    return dot_product / (norm_a * norm_b) if norm_a * norm_b else 0


def main() -> None:
    load_dotenv(ENV_PATH)

    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("Environment variable 'GOOGLE_API_KEY' is not set")

    client = genai.Client(api_key=API_KEY)

    sen_1 = "Mỗi nhân viên được nghỉ phép 12 ngày trong năm"
    sen_2 = "Nhân viên làm việc tại nhà"
    sen_3 = "Every employee is entitled to 12 days of leave"

    vec_1 = embed_context(client, sen_1)
    vec_2 = embed_context(client, sen_2)
    vec_3 = embed_context(client, sen_3)

    print(
        f"Similarity between '{sen_1}' and '{sen_2}': {cosine_similarity(vec_1[0].values, vec_2[0].values):.4f}"
    )
    print(
        f"Similarity between '{sen_1}' and '{sen_3}': {cosine_similarity(vec_1[0].values, vec_3[0].values):.4f}"
    )
    print(
        f"Similarity between '{sen_2}' and '{sen_3}': {cosine_similarity(vec_2[0].values, vec_3[0].values):.4f}"
    )


if __name__ == "__main__":
    main()

"""
Simple example of using in-context learning with a raw text document.

Everything is stuffed into the prompt, so the cost grows linearly with document size.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

DOC_PATH = Path(__file__).resolve().parents[0] / "docs/company.md"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
MODEL = "gemini-3.5-flash-lite"


def read_markdown(file_path) -> str:
    """Read a markdown file and return its content as a string."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def infer(client: genai.Client, question: str, ref_doc: str) -> str:
    """Infer an answer to a question based on the provided document content."""

    prompt = f"""
    You are a smart assistant. The following is an internal document of the company:
    Answer language same as the question. If the answer is not in the document, say "I don't know".
    -----------------
    {ref_doc}
    -----------------

    From the above document, please answer the following question accurately:
    Question: {question}
    """

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
    )
    return response.output_text


def main() -> None:
    load_dotenv(ENV_PATH)

    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("Environment variable 'GOOGLE_API_KEY' is not set")

    client = genai.Client(api_key=API_KEY)
    content = read_markdown(DOC_PATH)
    question = input("Your question: ")

    answer = infer(client, question, content)
    print(answer)


if __name__ == "__main__":
    main()

"""In-context learning: stuff the whole document into the prompt.

Baseline approach before RAG. Cost grows linearly with document size because
every request sends the full document.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

DOC_PATH = Path(__file__).resolve().parents[0] / "docs/company.md"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

MODEL = "gemini-3.5-flash-lite"


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.environ["GOOGLE_API_KEY"]

    content = DOC_PATH.read_text(encoding="utf-8")

    llm = ChatGoogleGenerativeAI(model=MODEL, api_key=api_key)

    question = input("Your question: ")

    prompt = f"""You are an internal assistant. The following are relevant excerpts from the company document:
-----------------
{content}
-----------------
Based on the excerpts above, please answer the question. If there is not enough information, please say you did not find the information in the document.
Question: {question}
"""
    response = llm.invoke(prompt)

    print(response.text)
    print(response.usage_metadata)


if __name__ == "__main__":
    main()

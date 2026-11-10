"""In-context learning: stuff the whole document into the prompt.

Baseline approach before RAG. Cost grows linearly with document size because
every request sends the full document.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Paths are anchored to this file so the script runs from any cwd.
#   parent        = src/rag_basic
#   parents[1]    = src            -> company.md lives here
#   parents[2]    = project root   -> .env lives here
DOC_PATH = Path(__file__).resolve().parents[1] / "company.md"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

MODEL = "gemini-3.5-flash-lite"


def main() -> None:
    load_dotenv(ENV_PATH)
    api_key = os.environ["API_KEY"]

    content = DOC_PATH.read_text(encoding="utf-8")

    llm = ChatGoogleGenerativeAI(model=MODEL, api_key=api_key)

    question = input("Nhập câu hỏi của bạn: ")

    prompt = f"""Bạn là một trợ lý ảo thông minh. Dưới đây là tài liệu nội bộ của công ty:
-----------------
{content}
-----------------

Dựa vào tài liệu trên, hãy trả lời câu hỏi sau một cách chính xác:
Câu hỏi: {question}
"""

    response = llm.invoke(prompt)
    print(response.text)


if __name__ == "__main__":
    main()

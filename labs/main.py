from google import genai

def read_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

content =  read_markdown("./company.md")


client = genai.Client()

question = input("Nhập câu hỏi của bạn: ")

prompt = f"""
Bạn là một trợ lý ảo thông minh. Dưới đây là tài liệu nội bộ của công ty:
-----------------
{content}
-----------------

Dựa vào tài liệu trên, hãy trả lời câu hỏi sau một cách chính xác:
Câu hỏi: {question}
"""


response = client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=prompt,
)

print(response.output_text)

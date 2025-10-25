import os
from openai import OpenAI
from dotenv import load_dotenv
from constant import van_ban

load_dotenv()

def generate_text_stream(prompt, model="gemini-2.5-flash"):
    client = OpenAI(
        api_key=os.getenv("THUCCHIEN_API_KEY"),
        base_url=os.getenv("THUCCHIEN_BASE_URL")
    )
    
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    full_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_text += content
    print()
    
    return full_text

if __name__ == "__main__":

    prompt = f"""Tạo infographic chi tiết cho nội dung văn bản pháp luật sau:
    {van_ban}
""" 
    print("📝 Đang tạo văn bản (streaming)...\n")
    result = generate_text_stream(prompt)
    print(f"\n\n✅ Hoàn thành! Độ dài: {len(result)} ký tự")


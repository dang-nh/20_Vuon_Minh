#!/usr/bin/env python3
from openai import OpenAI

# --- Cấu hình ---  
# Thay <your_api_key> bằng API key của bạn
client = OpenAI(
  api_key="sk-h4_042RMcJD2BNT21ZtkZA",
  base_url="https://api.thucchien.ai"
)

content = open("/Users/macbook/Library/CloudStorage/OneDrive-Personal/AI_ThucChien/tools/end-to-end_v2.txt", "r", encoding="utf-8").readlines()
content = "\n".join(content)

print(content)

response = client.chat.completions.create(
  model="gemini-2.5-pro", # Chọn model bạn muốn
  messages=[
      {
          "role": "user",
          "content": content
      }
  ],
  temperature=0.7
)

response_content = response.choices[0].message.content
print(response_content)

with open("new_code_v2.txt", "w", encoding="utf=8") as f:
  f.writelines(response_content)

from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
import base64
import json
import os

client = OpenAI(
    api_key='sk-tfBgpok8I0BpuDauwbdRiQ',
    base_url="https://api.thucchien.ai/"
)

class Classification(BaseModel):
    mon_hoc: Literal[
        "Toán",
        "Vật lý",
        "Hóa học",
        "Sinh học",
        "Ngữ văn",
        "Lịch sử",
        "Địa lý",
        "Giáo dục công dân",
        "Tiếng Anh",
        "Tin học",
        "Công nghệ",
        "Giáo dục thể chất",
        "Âm nhạc",
        "Mỹ thuật"
    ] = Field(alias="Môn học")
    do_kho: Literal[
        "easy",
        "medium",
        "hard"
    ] = Field(alias="Độ khó")
    cap_hoc: Literal[
        "Tiểu học (Primary)",
        "Trung học cơ sở (Secondary)",
        "Trung học phổ thông (High School)"
    ] = Field(alias="Cấp học")

def classify_image(image_path: Path) -> Classification:
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        base64_img = base64.b64encode(img_data).decode('utf-8')

    mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
    mime_type = mime_types.get(image_path.suffix.lower(), 'image/jpeg')

    PROMPT = """Xác định môn học độ khó và cấp họccủa hình ảnh:"""

    completion = client.beta.chat.completions.parse(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}
                ]
            }
        ],
        response_format=Classification,
        temperature=0.7,
    )

    return completion.choices[0].message.parsed
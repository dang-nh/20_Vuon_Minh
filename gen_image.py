import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
import sys
from PIL import Image
import os
import yaml
import base64
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import time
from openai import OpenAI
import re # <<< THÊM MỚI

load_dotenv()

def generate_image(
    prompt,
    output_path="output.png",
    model="imagen-4.0-generate-001",
    size="1024x1024",
    quality="standard"
):
    client = OpenAI(
        api_key="sk-h4_042RMcJD2BNT21ZtkZA",
        base_url="https://api.thucchien.ai/"
    )
    
    print(f"🎨 Đang tạo hình ảnh...")
    print(f"📝 Prompt: {prompt[:100]}...")

    try:
        response = client.images.generate(
            model="imagen-4",
            prompt=prompt,
            n=1,
            size="1024x1536"  # A4 vertical ratio
        )
        
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            if hasattr(image_data, 'b64_json') and image_data.b64_json:

                return Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
            elif hasattr(image_data, 'url') and image_data.url:
                if image_data.url.startswith('data:image'):
                    b64_data = image_data.url.split(',')[1]
                    return Image.open(BytesIO(base64.b64decode(b64_data)))
                else:
                    img_response = requests.get(image_data.url)
                    return Image.open(BytesIO(img_response.content))
        return None
    except Exception as e:
        print(f"    Imagen-4 thất bại: {e}")
        return None
    

    # response = client.images.generate(
    #     model=model,
    #     prompt=prompt,
    #     n=1,
    #     size=size,
    #     quality=quality
    # )
    
    # image_url = response.data[0].url
    
    # img_response = requests.get(image_url)
    # with open(output_path, 'wb') as f:
    #     f.write(img_response.content)
    
    # print(f"✅ Đã lưu: {output_path}")
    # return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_image_single.py <prompt> [output_path]")
        print("\nExample:")
        print("  python gen_image_single.py 'A beautiful Vietnamese landscape' output.png")
        sys.exit(1)
    
    prompt = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    
    res = generate_image(prompt, "/Users/user/Downloads/AI_ThucChien_VuonMinh/images/1.jpg")
    res.save("/Users/user/Downloads/AI_ThucChien_VuonMinh/images/1.jpg")


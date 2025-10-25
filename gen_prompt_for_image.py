import os
from openai import OpenAI
from dotenv import load_dotenv


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

    prompt = """Make 3 prompts for image generation model to generate image for this, infographic style, simplistic icon, no-text:
### **Mục 3: ĐIỂM MỚI & LƯU Ý QUAN TRỌNG**

*(Khu vực này có thể dùng icon bóng đèn hoặc dấu chấm than để nhấn mạnh)*

*   **Thành lập Cơ quan chuyên trách bảo vệ dữ liệu cá nhân (Điều 33)**
    *   Đây là đơn vị thuộc Bộ Công an, là đầu mối tiếp nhận khiếu nại và xử lý vi phạm. Giờ đây bạn đã có một địa chỉ cụ thể để bảo vệ mình.

*   **Trách nhiệm thông báo khi có sự cố (Điều 23)**
    *   Nếu dữ liệu của bạn bị rò rỉ, các công ty, tổ chức phải thông báo cho cơ quan chức năng trong vòng **72 giờ**.

*   **Doanh nghiệp phải đánh giá rủi ro (Điều 21)**
    *   Các doanh nghiệp lớn phải lập "Hồ sơ đánh giá tác động" để lường trước và ngăn chặn các rủi ro đối với dữ liệu của bạn trước khi xử lý.

*   **Miễn trừ cho Doanh nghiệp nhỏ, siêu nhỏ (Điều 38)**
    *   Hộ kinh doanh, doanh nghiệp siêu nhỏ, và doanh nghiệp nhỏ/startup được miễn trừ một số nghĩa vụ phức tạp trong 5 năm đầu (trừ khi họ kinh doanh dịch vụ xử lý dữ liệu). Điều này nhằm tạo điều kiện phát triển.

one image only for the whole section
""" 
    print("📝 Đang tạo văn bản (streaming)...\n")
    result = generate_text_stream(prompt)
    print(f"\n\n✅ Hoàn thành! Độ dài: {len(result)} ký tự")


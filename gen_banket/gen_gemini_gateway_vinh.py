import threading
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import json
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from datasets import load_dataset
from classify import classify_image
import base64
from pydantic import BaseModel, Field

load_dotenv()
OPEN_API_KEY = "AIzaSyAYl_wQM8yMHUL-jZg2zuiq-XwmyA88Ye0"
client = OpenAI(
    api_key='AIzaSyByc7YCgxy-ri15xNMOIuhalCGSqMcpHU0',
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class TraLoiCauHoi(BaseModel):
    question: str
    answer: str

class SuaLoiSai(BaseModel):
    question: str
    answer: str

class GoiMoYTuong(BaseModel):
    question: str
    answer: str

class HocTapCaNhanHoa(BaseModel):
    question: str
    answer: str

class HoTroTamLyCamXuc(BaseModel):
    question: str
    answer: str

class TaoBoCauHoi(BaseModel):
    question: str
    answer: str

class ChamDiemTuDong(BaseModel):
    question: str
    answer: str

class TaoTaiLieuGiangDay(BaseModel):
    question: str
    answer: str

class TaoNoiDungCaNhanHoa(BaseModel):
    question: str
    answer: str

class HocTapTuongTac(BaseModel):
    conversations: List[dict[str, str]]

class TuVanAnToan(BaseModel):
    question: str
    answer: str

class ToolItem(BaseModel):
    name: str
    parameters: List[str]

class ModelDecisionArguments(BaseModel):
    input_argument: List[str]

class ModelDecision(BaseModel):
    selected_tool: str
    arguments: ModelDecisionArguments

class ResponseSchema(BaseModel):
    tra_loi_cau_hoi: TraLoiCauHoi = Field(..., alias="Trả lời câu hỏi")
    sua_loi_sai: SuaLoiSai = Field(..., alias="Sửa lỗi sai")
    goi_mo_y_tuong: GoiMoYTuong = Field(..., alias="Gợi mở ý tưởng")
    hoc_tap_ca_nhan_hoa: HocTapCaNhanHoa = Field(..., alias="Học tập cá nhân hóa")
    ho_tro_tam_ly_cam_xuc: HoTroTamLyCamXuc = Field(..., alias="Hỗ trợ tâm lý cảm xúc")
    tao_bo_cau_hoi: TaoBoCauHoi = Field(..., alias="Tạo bộ câu hỏi")
    cham_diem_tu_dong: ChamDiemTuDong = Field(..., alias="Chấm điểm tự động")
    tao_tai_lieu_giang_day: TaoTaiLieuGiangDay = Field(..., alias="Tạo tài liệu giảng dạy")
    tao_noi_dung_ca_nhan_hoa: TaoNoiDungCaNhanHoa = Field(..., alias="Tạo nội dung cá nhân hóa")
    hoc_tap_tuong_tac: HocTapTuongTac = Field(..., alias="Học tập tương tác")
    tu_van_an_toan: TuVanAnToan = Field(..., alias="Tư vấn an toàn")

PROMPT = """Bạn là trợ lý AI chuyên về giáo dục phổ thông Việt Nam. Bạn nhận được một hình ảnh tài liệu. Nhiệm vụ của bạn là tạo các cặp Q-A text-only với mỗi task dựa trên hình ảnh với các Yêu cầu nghiêm ngặt sau:

I. QUY TẮC SỬ DỤNG HÌNH ẢNH (CŨ - GIỮ NGUYÊN)
- Hình ảnh chỉ dùng để nhận diện Môn học, Cấp học, Chủ đề chung.
- KHÔNG sao chép nguyên văn.
- KHÔNG dùng cụm từ chỉ trỏ hình ảnh ("trong hình", "như ảnh trên").

II. QUY TẮC "NGỮ CẢNH ĐỘC LẬP" (QUAN TRỌNG - BẮT BUỘC)
Đây là quy tắc tối thượng để tránh lỗi câu hỏi tối nghĩa:

1. Nguyên tắc "Người dùng bị bịt mắt" (The Blindfold Rule):
   - Hãy tưởng tượng người đọc câu hỏi KHÔNG HỀ nhìn thấy bức ảnh.
   - Câu hỏi phải tự cung cấp đủ thông tin (bối cảnh, tên nhân vật, tên chất hóa học, dữ kiện số) để người đọc có thể trả lời ngay lập tức mà không cần đoán.

2. Cấm tuyệt đối "Đại từ chỉ định lửng lơ":
   - KHÔNG ĐƯỢC dùng: "nhân vật này", "ông ấy", "chất này", "sự kiện đó", "hình vẽ trên", "đồ thị này"... làm chủ ngữ nếu chưa nhắc tên cụ thể trước đó.
   - PHẢI thay thế bằng TÊN RIÊNG hoặc DANH TỪ CỤ THỂ.

3. Kỹ thuật "Trích xuất & Gọi tên" (Extract & Name):
   - Bước 1: Nhìn ảnh -> Xác định đối tượng chính (Ví dụ: Ảnh là Nguyễn Trãi, hoặc Phản ứng Fe + HCl).
   - Bước 2: Quên bức ảnh đi.
   - Bước 3: Đặt câu hỏi bằng cách gọi tên trực tiếp đối tượng đó.
     - SAI: "Ông có vai trò gì trong khởi nghĩa Lam Sơn?" (Không biết ông nào).
     - ĐÚNG: "Nguyễn Trãi có vai trò gì trong khởi nghĩa Lam Sơn?" (Rõ ràng).
     - ĐÚNG (Nếu muốn tổng quát): "Trong khởi nghĩa Lam Sơn, ai là người dâng Bình Ngô sách?"

III. TASKS CẦN TẠO
1. **Trả Lời Câu Hỏi (Problem Solving)**
   **Mô tả:** Đánh giá khả năng của hệ thống AI trong việc giải đáp chính xác các câu hỏi do học sinh đặt ra trên nhiều môn học và cấp độ khó khác nhau.

2. **Sửa Lỗi Sai (Error Correction)**
   **Mô tả:** Năng lực xác định và sửa các lỗi của học sinh trong bài tập, bài kiểm tra hoặc các bài luyện tập hàng ngày. Lỗi có thể dao động từ những sai sót rõ ràng đến các vấn đề tinh vi như sai sót logic trong lập luận toán học. Đánh giá tập trung vào độ chính xác của việc phát hiện lỗi và chất lượng của việc sửa chữa.

3. **Gợi Mở Ý Tưởng (Idea Provision)**
   **Mô tả:** Bao gồm việc trả lời các thắc mắc của học sinh về kiến thức, hướng dẫn bài tập hoặc chuẩn bị thi. Được chia thành các phần: giải thích sự kiện cơ bản, phân tích giải pháp từng bước và lời khuyên học tập chung. Các phản hồi được đánh giá về độ chính xác, sự rõ ràng và tính cung cấp thông tin.

4. **Học Tập Cá Nhân Hóa (Personalized Learning Support)**
   **Mô tả:** Dựa trên hồ sơ học sinh (ví dụ: trình độ kỹ năng, mục tiêu học tập), đề xuất lộ trình học tập, bài tập hoặc tài liệu đọc phù hợp với nhu cầu cá nhân. Hiệu quả được đánh giá dựa trên mức độ liên quan, sự phù hợp về độ khó và tính hữu ích của các đề xuất.

5. **Hỗ Trợ Tâm Lý Cảm Xúc (Emotional Support)**
   **Mô tả:** Liên quan đến việc phát hiện trạng thái cảm xúc của học sinh (ví dụ: lo lắng trước khi thi) từ văn bản và đưa ra phản hồi hỗ trợ hoặc gợi ý phù hợp. Các tình huống bao gồm căng thẳng trước kỳ thi, thất vọng sau kỳ thi hoặc cô lập xã hội. Các chỉ số đánh giá bao gồm độ chính xác phân loại cảm xúc, tính cụ thể của các dấu hiệu cảm xúc và chất lượng của các gợi ý.

6. **Tạo Bộ Câu Hỏi (Question Generation)**
   **Mô tả:** Tạo câu hỏi dựa trên các chủ đề, cấp độ khó và phạm vi kiến thức được chỉ định. Bao gồm cả việc tạo câu hỏi đơn chủ đề và đa chủ đề (tổng hợp). Các yêu cầu nâng cao liên quan đến việc tạo lời giải thích và định dạng đề thi đầy đủ. Đánh giá tập trung vào chất lượng câu hỏi, mức độ liên quan và tính mạch lạc của cấu trúc.

7. **Chấm Điểm Tự Động (Automatic Grading)**
   **Mô tả:** Hỗ trợ chấm điểm các câu hỏi khách quan (ví dụ: trắc nghiệm, điền vào chỗ trống) và các bài tập chủ quan (ví dụ: báo cáo dự án) dựa trên thang điểm quy định. Cũng hỗ trợ tạo phản hồi. Các chỉ số đánh giá bao gồm độ chính xác chấm điểm, tính hợp lý và tính cung cấp thông tin của phản hồi.

8. **Tạo Tài Liệu Giảng Dạy (Teaching Material Generation)**
   **Mô tả:** Tự động tạo nội dung giáo dục như slide, giáo án và ghi chú bài giảng. Bao gồm việc cấu trúc nội dung và bổ sung các tài liệu bên ngoài có liên quan như hình ảnh hoặc tài liệu tham khảo.  

9. **Tạo Nội Dung Cá Nhân Hóa (Personalized Content Creation)**
   **Mô tả:** Tạo nội dung khác biệt hóa cho học sinh dựa trên trình độ học tập hoặc hồ sơ cá nhân của họ. Bao gồm cả bài tập cá nhân hóa và thiết kế nội dung phân tầng (ví dụ: mục tiêu học tập khác biệt, chiến lược giảng dạy và định hướng đánh giá cho các cấp độ học sinh khác nhau). Đánh giá tập trung vào tính hợp lệ nội tại của từng mục và tính nhất quán giữa các tầng.

10. **Học Tập Tương Tác (Interactive Tutoring)**
    **Mô tả:** Học tập tương tác là quá trình hệ thống AI tham gia vào một cuộc đối thoại liên tục với học sinh để hướng dẫn học qua các khái niệm, vấn đề hoặc bài tập thông qua các câu hỏi gợi mở, phản hồi động và điều chỉnh theo thời gian thực dựa trên phản ứng của học sinh.

11. **Tư Vấn An Toàn (Safety Advice)**
    **Mô tả:** Đánh giá khả năng của mô hình trong việc nhận diện và từ chối trả lời các câu hỏi liên quan đến thông tin nhạy cảm, bất hợp pháp, hoặc có hại, đảm bảo an toàn cho người dùng, đặc biệt là học sinh.

IV. FORMAT TASK
Mỗi phần tử trong "tasks" là danh sách các lượt hội thoại, theo đúng thứ tự:
    + Bắt đầu bằng 1 lượt "user" (học sinh hoặc giáo viên).
    + Sau đó là 1 hoặc nhiều lượt "assistant".
    + Với "Học Tập Tương Tác", nên có ít nhất 4–6 lượt (user–assistant xen kẽ).
    + Nội dung phải hoàn toàn TEXT-ONLY (không chèn hình ảnh, không tham chiếu trực tiếp tới ảnh).
"""

IMAGE_FOLDER = Path("../VietDocVQA")
OUTPUT_FOLDER = Path("../annotations/labels_train_vinh")
MAX_RETRIES = 5
existing_images = [p for p in IMAGE_FOLDER.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
existing_count = len(existing_images)
max_download = 1200
print(f"Found {existing_count} existing images in {IMAGE_FOLDER}")
if existing_count >= max_download:
    print(f"Already have {existing_count} images (>= {max_download}). Skipping download.")
else:
    images_to_download = max_download - existing_count
    print(f"Need to download {images_to_download} more images to reach {max_download} total.")
   
    dataset = load_dataset("5CD-AI/Viet-Doc-VQA", split="train", streaming=True)
    iterator = iter(dataset)
   
    count = existing_count
   
    while count < max_download:
        try:
            item = next(iterator)
        except StopIteration:
            print(f"Dataset finished before reaching {max_download} images. Total images: {count}")
            break
        image = item["image"]
        image_path = IMAGE_FOLDER / f"image_{count}.png"
        image.save(image_path)
        count += 1
        print(f"Downloaded image {count}/{max_download}")
    print(f"Download complete. Total images: {count}")
print("Done.")
processed_json_stems = {p.stem.replace('_q', '') for p in OUTPUT_FOLDER.glob('*.json')}
processed_skip_stems = {p.stem.replace('_q_skip', '') for p in OUTPUT_FOLDER.glob('*_q_skip.txt')}
processed_stems = processed_json_stems | processed_skip_stems
all_images = [p for p in IMAGE_FOLDER.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
total_images = len(all_images)
images_to_process = all_images[:1200]
print(f"Tổng số ảnh trong thư mục: {total_images}. Số ảnh sẽ xử lý: {len(images_to_process)}")
print(f"Đã xử lý (JSON): {len(processed_json_stems)}, Đã bỏ qua (TXT): {len(processed_skip_stems)}")
print(f"Tổng đã xử lý: {len(processed_stems)}")
image_queue = [p for p in images_to_process if p.stem not in processed_stems]
lock = threading.Lock()
retry_counts = {}

def call_openai_and_save(client, prompt_text, img_base64, mime_type, output_path, thread_name):
    """Simplified non-streaming call to OpenAI API (Gemini). No early detection."""
    print(f"⏳ [{thread_name}] Đang tạo cặp Q&A cho: {output_path.name}")
    try:
        response = client.beta.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                        }
                    ]
                }
            ],
            response_format=ResponseSchema,
            temperature=1.1,
        )
        full_text = response.choices[0].message.parsed
        print(f"\n✅ [{thread_name}] Response completed.")
        result_json = full_text.model_dump()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        print(f"💾 [{thread_name}] Successfully saved: {output_path.name}")
        return result_json
    except json.JSONDecodeError:
        print(f"❌ [{thread_name}] Failed to parse JSON from the response.")
        error_txt_path = OUTPUT_FOLDER / f"{output_path.stem}_json_error.txt"
        with open(error_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"💾 [{thread_name}] Saved malformed response to: {error_txt_path.name}")
        return None
    except Exception as e:
        print(f"❌ [{thread_name}] Error during generation: {e}")
        return None

def process_image_task(thread_name):
    """Hàm xử lý cho mỗi luồng. Classify first, skip easy, then generate without early detection."""
    print(f"🟢 [{thread_name}] Bắt đầu làm việc...")
    global client
   
    while True:
        with lock:
            if not image_queue:
                break
            image_path = image_queue.pop(0)
        try:
            classification = classify_image(image_path)
            classify_dir = Path("/home/team_cv/tdkien/20_Vuon_Minh/annotations/classify")
            classify_dir.mkdir(parents=True, exist_ok=True)
            classify_file = classify_dir / f"{image_path.stem}_classify.json"
            with open(classify_file, "w", encoding="utf-8") as f:
                json.dump(classification.model_dump(), f, ensure_ascii=False, indent=4)
            print(f"📋 [{thread_name}] Classified {image_path.name}: {classification.mon_hoc} - {classification.do_kho}")
           
            if classification.do_kho == 'easy':
                skip_path = OUTPUT_FOLDER / f"{image_path.stem}_q_skip.txt"
                skip_content = f"Skipped: Classified as easy\nMon hoc: {classification.mon_hoc}\nDo kho: {classification.do_kho}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                with open(skip_path, "w", encoding="utf-8") as f:
                    f.write(skip_content)
                print(f"⏭️ [{thread_name}] Skipped {image_path.name} due to easy difficulty")
                continue
        except Exception as e:
            print(f"❌ [{thread_name}] Error classifying {image_path.name}: {e}")
            continue
        try:
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
            mime_type = mime_types.get(image_path.suffix.lower(), 'image/jpeg')
            output_path = OUTPUT_FOLDER / f"{image_path.stem}_q.json"
           
            result = call_openai_and_save(client, PROMPT, img_base64, mime_type, output_path, thread_name)
           
            if result is None:
                print(f"⚠️ [{thread_name}] Generation failed for {image_path.name}, will retry if possible")
        except Exception as e:
            print(f"❌ [{thread_name}] Error processing {image_path.name}: {e}")
            with lock:
                current_retries = retry_counts.get(image_path, 0)
                if current_retries < MAX_RETRIES:
                    retry_counts[image_path] = current_retries + 1
                    image_queue.append(image_path)
                    print(f"🔄 [{thread_name}] Đã đưa {image_path.name} vào hàng đợi để thử lại (lần {current_retries + 1}/{MAX_RETRIES}).")
                else:
                    print(f"⚠️ [{thread_name}] Bỏ qua {image_path.name} sau {MAX_RETRIES} lần thử không thành công.")
       
        time.sleep(8)
    print(f"🔴 [{thread_name}] Đã hoàn thành công việc.")

if __name__ == "__main__":
    if not all_images:
        print(f"📂 Thư mục '{IMAGE_FOLDER}' trống. Vui lòng thêm ảnh vào để xử lý.")
    elif not image_queue:
        print("🎉 Tất cả các ảnh đã được xử lý trước đó.")
    else:
        print(f"🚀 Bắt đầu xử lý {len(image_queue)} ảnh")
        print(f"📋 Classify trước, skip easy, generate Q&A for medium/hard without early detection")
        print(f"🛑 Nhấn Ctrl+C để dừng gracefully")
       
        start_time = time.time()
       
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [
                executor.submit(process_image_task, "Thread-1")
            ]
           
            for i, future in enumerate(futures):
                try:
                    future.result()
                except KeyboardInterrupt:
                    print(f"🛑 Luồng {i+1} bị ngắt")
                    break
                except Exception as e:
                    print(f"❌ Lỗi trong luồng {i+1}: {e}")
       
        end_time = time.time()
        elapsed = end_time - start_time
       
        final_json_count = len([p for p in OUTPUT_FOLDER.glob('*.json')])
        final_skip_count = len([p for p in OUTPUT_FOLDER.glob('*_q_skip.txt')])
       
        print(f"\n✨ Hoàn thành sau {elapsed:.1f} giây!")
        print(f"📁 Kết quả được lưu trong: {OUTPUT_FOLDER}")
        print(f"📊 Thống kê kết quả:")
        print(f" ✅ Đã xử lý thành công: {final_json_count} file JSON")
        print(f" ⏭️ Đã bỏ qua (easy): {final_skip_count} file TXT")
        print(f" 📈 Tổng cộng: {final_json_count + final_skip_count} file")
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
import os
from prompt import PROMPT_1, PROMPT_2, PROMPT_3, ResponseSchema1, ResponseSchema2, ResponseSchema3

load_dotenv()
OPEN_API_KEY = "AIzaSyAYl_wQM8yMHUL-jZg2zuiq-XwmyA88Ye0"
client = OpenAI(
    api_key='AIzaSyByc7YCgxy-ri15xNMOIuhalCGSqMcpHU0',
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

IMAGE_FOLDER = Path("../VietDocVQA")
OUTPUT_FOLDER = Path("../annotations/labels_train_2")
OUTPUT_CLASSIFY_FOLDER = Path("../annotations/classify_2")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_CLASSIFY_FOLDER, exist_ok=True)
MAX_RETRIES = 5
existing_images = [p for p in IMAGE_FOLDER.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
existing_count = len(existing_images)
max_download = 1500
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
processed_json_stems = {p.stem.split('_q_')[0] for p in OUTPUT_FOLDER.glob('*_q_*.json')}
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
prompts_and_schemas = [
    (PROMPT_1, ResponseSchema1),
    (PROMPT_2, ResponseSchema2),
    (PROMPT_3, ResponseSchema3)
]

def call_openai_and_save(client, prompt_text, img_base64, mime_type, output_path, thread_name, response_schema):
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
            response_format=response_schema,
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
    prompt_index = 0
    print(f"🟢 [{thread_name}] Bắt đầu làm việc...")
    global client
   
    while True:
        with lock:
            if not image_queue:
                break
            image_path = image_queue.pop(0)
        try:
            classification = classify_image(image_path)
            classify_dir = OUTPUT_CLASSIFY_FOLDER
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
            
            prompt, schema = prompts_and_schemas[prompt_index]
            prompt += f"\nTạo thêm một system prompt phù hợp với môn học {classification.mon_hoc}, độ khó {classification.do_kho} và cấp học {classification.cap_hoc}. Tiểu học: Liên hệ đến đạo đức, thói quen tốt, kỹ năng sống cơ bản, vệ sinh, lễ phép và hình thành nhân cách.\nTrung học cơ sở: Liên hệ đến tâm lý tuổi dậy thì, kỹ năng tự quản, cảm xúc, học tập chủ động và giao tiếp xã hội.\nTrung học phổ thông: Liên hệ đến định hướng nghề nghiệp, tương lai, kỹ năng tự học, mục tiêu dài hạn và trách nhiệm cá nhân."
            i = prompt_index + 1
            output_path = OUTPUT_FOLDER / f"{image_path.stem}_q_{i}.json"
            result = call_openai_and_save(client, prompt, img_base64, mime_type, output_path, thread_name, schema)
            if result is None:
                print(f"⚠️ [{thread_name}] Generation failed for {image_path.name} with prompt {i}")
            prompt_index = (prompt_index + 1) % 3
            time.sleep(8)
            
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
       
        final_json_count = len(set(p.stem.split('_q_')[0] for p in OUTPUT_FOLDER.glob('*_q_*.json')))
        final_skip_count = len([p for p in OUTPUT_FOLDER.glob('*_q_skip.txt')])
       
        print(f"\n✨ Hoàn thành sau {elapsed:.1f} giây!")
        print(f"📁 Kết quả được lưu trong: {OUTPUT_FOLDER}")
        print(f"📊 Thống kê kết quả:")
        print(f" ✅ Đã xử lý thành công: {final_json_count} ảnh")
        print(f" ⏭️ Đã bỏ qua (easy): {final_skip_count} ảnh")
        print(f" 📈 Tổng cộng: {final_json_count + final_skip_count} ảnh")
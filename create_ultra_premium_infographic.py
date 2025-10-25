from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
import base64
import requests
import os
from io import BytesIO
import math
from openai import OpenAI

# --- THIẾT LẬP API VÀ CÁC HẰNG SỐ ---
API_KEY = "sk-h4_042RMcJD2BNT21ZtkZA"
CLIENT = OpenAI(
            api_key=API_KEY,
            base_url="https://api.thucchien.ai"
        )

WIDTH = 7874
HEIGHT = 3937
DPI = 150 # Khuyến nghị 150-200dpi

# --- CÁC HÀM TIỆN ÍCH (Giữ nguyên từ code gốc) ---

def generate_image_with_ai(prompt_vietnamese, output_filename, size="1536x1024"):
    """Tạo ảnh bằng AI với prompt tiếng Việt"""
    print(f"   ⏳ Đang tạo ảnh cho: {output_filename.split('/')[-1]}...")
    try:
        response = CLIENT.images.generate(
            model="imagen-4",
            prompt=prompt_vietnamese,
            n=1,
            size=size
        )
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                img = Image.open(BytesIO(base64.b64decode(image_data.b64_json)))
            elif hasattr(image_data, 'url') and image_data.url:
                img_response = requests.get(image_data.url)
                img = Image.open(BytesIO(img_response.content))
            
            img.save(output_filename)
            print(f"   ✅ Đã lưu ảnh: {output_filename}")
            return img
    except Exception as e:
        print(f"   ❌ Lỗi khi tạo ảnh: {e}")
        return None
    return None

def draw_shadow_box(draw, xy, radius, fill_color, shadow_offset=15, shadow_color='#00000030'):
    """Vẽ một hộp có bóng đổ"""
    x1, y1, x2, y2 = xy
    shadow_xy = (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset)
    draw.rounded_rectangle(shadow_xy, radius, fill=shadow_color)
    draw.rounded_rectangle(xy, radius, fill=fill_color)

def wrap_text(text, font, max_width, draw):
    """Ngắt dòng văn bản để vừa với chiều rộng cho trước"""
    lines = []
    if not text:
        return lines
    words = text.split()
    while words:
        line = ''
        while words and draw.textbbox((0, 0), line + words[0], font=font)[2] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())
    return lines

# --- HÀM TẠO INFOGRAPHIC CHÍNH ---

def create_infographic_luat_bvdl():
    print("=" * 100)
    print("🌟 TẠO INFOGRAPHIC - LUẬT BẢO VỆ DỮ LIỆU CÁ NHÂN 2025")
    print("=" * 100)
    
    # 1. Khởi tạo canvas và fonts
    img = Image.new('RGB', (WIDTH, HEIGHT), color='#F4F7F9') # Nền xám nhạt, hiện đại
    draw = ImageDraw.Draw(img)
    
    # Đảm bảo bạn đã cài đặt font Roboto
    try:
        font_path = "/System/Library/Fonts/Supplemental/" # Thư mục chứa các file font .ttf
        title_font = ImageFont.truetype(font_path + "Arial.ttf", 250)
        subtitle_font = ImageFont.truetype(font_path + "Arial.ttf", 120)
        header_font = ImageFont.truetype(font_path + "Arial.ttf", 100)
        sub_header_font = ImageFont.truetype(font_path + "Arial.ttf", 75)
        body_font = ImageFont.truetype(font_path + "Arial.ttf", 60)
        small_font = ImageFont.truetype(font_path + "Arial.ttf", 50)
    except IOError:
        print("⚠️ Lỗi: Không tìm thấy font Roboto. Sử dụng font mặc định.")
        title_font = subtitle_font = header_font = sub_header_font = body_font = small_font = ImageFont.load_default()

    # 2. Tạo hình ảnh nền bằng AI
    print("\n📸 Bước 1: Tạo hình ảnh nền và minh họa...")
    
    # Đường dẫn lưu ảnh
    image_dir = "infographic_assets/"
    os.makedirs(image_dir, exist_ok=True)
    
    bg_image_path = os.path.join(image_dir, "background_data_security.png")
    
    if os.path.exists(bg_image_path):
        print("   ♻️  Sử dụng ảnh nền có sẵn.")
        bg_image = Image.open(bg_image_path)
    else:
        bg_prompt = "Nền công nghệ trừu tượng, các đường mạch kỹ thuật số và biểu tượng ổ khóa phát sáng, màu xanh dương và xanh mòng két, không gian mạng, an toàn dữ liệu, tối giản, chuyên nghiệp, tỷ lệ 2:1."
        bg_image = generate_image_with_ai(bg_prompt, bg_image_path, size="1536x768")

    if bg_image:
        # Làm mờ và giảm độ sáng của ảnh nền để chữ nổi bật
        bg_image = bg_image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        enhancer = ImageEnhance.Brightness(bg_image)
        bg_image = enhancer.enhance(0.7)
        bg_image = bg_image.filter(ImageFilter.GaussianBlur(radius=8))
        img.paste(bg_image, (0, 0))

    print("\n🎨 Bước 2: Thiết kế bố cục và nội dung...")

    # --- HEADER ---
    y_pos = 150
    draw.text((WIDTH/2, y_pos), "DỮ LIỆU CỦA BẠN - QUYỀN CỦA BẠN", 
              fill="#FFFFFF", font=title_font, anchor="mt", stroke_width=5, stroke_fill="#0056b3")
    
    y_pos += 300
    draw.text((WIDTH/2, y_pos), "Tóm tắt Luật Bảo vệ dữ liệu cá nhân 2025 (Luật số: 91/2025/QH15) ai cũng cần biết!", 
              fill="#E0F7FA", font=subtitle_font, anchor="mt")
    
    y_pos += 150
    draw.text((WIDTH/2, y_pos), "*Có hiệu lực từ ngày 01/01/2026*", 
              fill="#B2EBF2", font=small_font, anchor="mt")

    # --- MỤC 1: KHÁI NIỆM CỐT LÕI ---
    y_pos += 200
    margin = 200
    col_width = (WIDTH - margin * 5) / 4
    col_height = 500

    concepts = [
        {"icon": "👤", "title": "Dữ liệu cá nhân là gì?", "text": "Là mọi thông tin giúp nhận ra bạn: Tên, ngày sinh, SĐT, email, hình ảnh, tài khoản ngân hàng..."},
        {"icon": "📊", "title": "Phân loại dữ liệu", "text": "Gồm Dữ liệu CƠ BẢN (họ tên, địa chỉ...) và Dữ liệu NHẠY CẢM (sức khỏe, tài chính, tôn giáo... cần bảo vệ đặc biệt)."},
        {"icon": "🙋", "title": "Chủ thể dữ liệu là ai?", "text": "Chính là BẠN – người sở hữu các dữ liệu cá nhân đó."},
        {"icon": "⚙️", "title": "Xử lý dữ liệu là gì?", "text": "Mọi hành động: Thu thập, lưu trữ, sử dụng, chia sẻ hoặc xóa bỏ dữ liệu của bạn."}
    ]
    
    for i, concept in enumerate(concepts):
        x_start = margin + i * (col_width + margin)
        draw_shadow_box(draw, (x_start, y_pos, x_start + col_width, y_pos + col_height), 30, "#FFFFFF")
        
        # Icon
        draw.text((x_start + col_width/2, y_pos + 80), concept["icon"], 
                  font=ImageFont.truetype(font_path + "Arial.ttf", 150), anchor="mm", fill="#007BFF")
        
        # Title
        title_lines = wrap_text(concept["title"], sub_header_font, col_width - 80, draw)
        text_y = y_pos + 180
        for line in title_lines:
            draw.text((x_start + col_width/2, text_y), line, font=sub_header_font, fill="#003366", anchor="mt")
            text_y += 85
            
        # Text
        text_lines = wrap_text(concept["text"], body_font, col_width - 80, draw)
        text_y += 20
        for line in text_lines:
            draw.text((x_start + col_width/2, text_y), line, font=body_font, fill="#34495E", anchor="mt", align="center")
            text_y += 70

    # --- CÁC MỤC CHÍNH (3 CỘT) ---
    y_pos += col_height + 150
    main_col_width = (WIDTH - margin * 4) / 3

    # CỘT 1: QUY ĐỊNH NỔI BẬT
    x1 = margin
    draw_shadow_box(draw, (x1, y_pos, x1 + main_col_width, HEIGHT - 350), 30, "#E3F2FD")
    draw.rectangle((x1, y_pos, x1 + main_col_width, y_pos + 150), fill="#0277BD")
    draw.text((x1 + main_col_width/2, y_pos + 75), "🚨 QUY ĐỊNH NỔI BẬT", fill="white", font=header_font, anchor="mm")
    
    content_y = y_pos + 200
    noi_bat_items = [
        "IM LẶNG KHÔNG PHẢI LÀ ĐỒNG Ý!",
        "Phải có sự đồng ý TỰ NGUYỆN, RÕ RÀNG của bạn trước khi xử lý dữ liệu.",
        "NGHIÊM CẤM mua, bán dữ liệu cá nhân trái phép (phạt tới 3 tỷ đồng).",
        "BẢO VỆ TRẺ EM là ưu tiên hàng đầu, cần có sự đồng ý của cha mẹ/người giám hộ.",
        "QUẢNG CÁO & MẠNG XÃ HỘI phải minh bạch, cho phép từ chối và không được yêu cầu giấy tờ tùy thân để xác thực."
    ]
    for item in noi_bat_items:
        lines = wrap_text(item, body_font, main_col_width - 120, draw)
        for line in lines:
            draw.text((x1 + 60, content_y), "• " + line, fill="#1C3A56", font=body_font)
            content_y += 75
        content_y += 40

    # CỘT 2: ĐIỂM MỚI & LƯU Ý
    x2 = margin * 2 + main_col_width
    draw_shadow_box(draw, (x2, y_pos, x2 + main_col_width, HEIGHT - 350), 30, "#E8F5E9")
    draw.rectangle((x2, y_pos, x2 + main_col_width, y_pos + 150), fill="#2E7D32")
    draw.text((x2 + main_col_width/2, y_pos + 75), "💡 ĐIỂM MỚI & LƯU Ý", fill="white", font=header_font, anchor="mm")
    
    content_y = y_pos + 200
    diem_moi_items = [
        "Thành lập CƠ QUAN CHUYÊN TRÁCH bảo vệ dữ liệu cá nhân (thuộc Bộ Công an) để tiếp nhận khiếu nại.",
        "Bắt buộc thông báo trong 72 GIỜ nếu dữ liệu của bạn bị rò rỉ.",
        "Doanh nghiệp lớn phải ĐÁNH GIÁ RỦI RO trước khi xử lý dữ liệu.",
        "MIỄN TRỪ một số nghĩa vụ cho doanh nghiệp nhỏ, siêu nhỏ trong 5 năm đầu để tạo điều kiện phát triển."
    ]
    for item in diem_moi_items:
        lines = wrap_text(item, body_font, main_col_width - 120, draw)
        for line in lines:
            draw.text((x2 + 60, content_y), "• " + line, fill="#1B5E20", font=body_font)
            content_y += 75
        content_y += 40

    # CỘT 3: QUYỀN & NGHĨA VỤ
    x3 = margin * 3 + main_col_width * 2
    draw_shadow_box(draw, (x3, y_pos, x3 + main_col_width, HEIGHT - 350), 30, "#FFF3E0")
    draw.rectangle((x3, y_pos, x3 + main_col_width, y_pos + 150), fill="#EF6C00")
    draw.text((x3 + main_col_width/2, y_pos + 75), "⚖️ QUYỀN & NGHĨA VỤ CỦA BẠN", fill="white", font=header_font, anchor="mm")
    
    content_y = y_pos + 200
    # Quyền của bạn
    draw.text((x3 + 60, content_y), "✅ QUYỀN CỦA BẠN:", font=sub_header_font, fill="#BF360C")
    content_y += 100
    quyen_items = ["Được biết", "Đồng ý", "Truy cập & Chỉnh sửa", "Rút lại sự đồng ý", "Yêu cầu xóa", "Khiếu nại & Đòi bồi thường"]
    for item in quyen_items:
        draw.text((x3 + 80, content_y), "✓ " + item, font=body_font, fill="#4E342E")
        content_y += 75
    
    content_y += 50
    # Nghĩa vụ của bạn
    draw.text((x3 + 60, content_y), "🛡️ TRÁCH NHIỆM CỦA BẠN:", font=sub_header_font, fill="#BF360C")
    content_y += 100
    nghia_vu_items = ["Tự bảo vệ dữ liệu của mình", "Tôn trọng dữ liệu người khác", "Cung cấp thông tin chính xác", "Tuân thủ pháp luật"]
    for item in nghia_vu_items:
        draw.text((x3 + 80, content_y), "• " + item, font=body_font, fill="#4E342E")
        content_y += 75

    # --- FOOTER ---
    footer_y = HEIGHT - 200
    draw.rectangle((0, footer_y, WIDTH, HEIGHT), fill="#001F3F")
    draw.text((WIDTH/2, footer_y + 80), "HÃY LÀ NGƯỜI CHỦ THÔNG THÁI CHO DỮ LIỆU CỦA CHÍNH MÌNH!", 
              fill="#FFFFFF", font=sub_header_font, anchor="mt")
    draw.text((WIDTH/2, footer_y + 180), "Nội dung được tóm tắt từ Luật số 91/2025/QH15. Tham khảo văn bản gốc để biết chi tiết.", 
              fill="#B0C4DE", font=small_font, anchor="mt")


    # 3. Lưu file
    print("\n💾 Bước 3: Lưu file kết quả...")
    output_filename = "Infographic_LuatBaoVeDuLieuCaNhan_2025"
    output_png = f"{output_filename}.png"
    img.save(output_png, 'PNG', dpi=(DPI, DPI))
    print(f"✅ Đã lưu PNG: {output_png}")

    output_jpg = f"{output_filename}.jpg"
    # Chuyển sang RGB trước khi lưu JPEG
    img.convert('RGB').save(output_jpg, 'JPEG', dpi=(DPI, DPI), quality=95)
    print(f"✅ Đã lưu JPG: {output_jpg}")
    
    print("\n" + "=" * 100)
    print("✨ HOÀN THÀNH INFOGRAPHIC!")
    print(f"📏 Kích thước: {WIDTH}x{HEIGHT} pixels | DPI: {DPI}")
    print(f"📐 Kích thước in tương đương: 2m x 1m")
    print("=" * 100)

if __name__ == "__main__":
    create_infographic_luat_bvdl()
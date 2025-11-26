import threading
import time
import asyncio
import signal
import sys
import queue
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from openai import OpenAI
import json
import os
import random
from pathlib import Path
from PIL import Image
from typing import List, Optional, Literal
from dotenv import load_dotenv
from datasets import load_dataset
from classify import classify_image
import base64

load_dotenv()
OPEN_API_KEY = "AIzaSyAYl_wQM8yMHUL-jZg2zuiq-XwmyA88Ye0"
# --- Cấu hình ---
client = OpenAI(
    api_key='AIzaSyByc7YCgxy-ri15xNMOIuhalCGSqMcpHU0',
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Schema for OpenAI structured output
SCHEMA = {
    "type": "object",
    "properties": {
        "Trả lời câu hỏi": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Sửa lỗi sai": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "context_for_question",
                "question",
                "answer"
            ]
        },
        "Gợi mở ý tưởng": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Học tập cá nhân hóa": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Hỗ trợ tâm lý cảm xúc": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Tạo bộ câu hỏi": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Chấm điểm tự động": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Tạo tài liệu giảng dạy": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Tạo nội dung cá nhân hóa": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Học tập tương tác": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "conversations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string"
                            },
                            "answer": {
                                "type": "string"
                            }
                        },
                        "required": [
                            "question",
                            "answer"
                        ]
                    }
                }
            },
            "required": [
                "conversations"
            ]
        },
        "Tư vấn an toàn": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "question": {
                    "type": "string"
                },
                "answer": {
                    "type": "string"
                }
            },
            "required": [
                "question",
                "answer"
            ]
        },
        "Sử dụng công cụ": {
            "type": "object",
            "properties": {
                "context_for_question": {
                    "type": "string"
                },
                "all tools available (suggest more than 2)": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string"
                            },
                            "parameters": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "required": [
                            "name",
                            "parameters"
                        ]
                    }
                },
                "user_query": {
                    "type": "string"
                },
                "model_decision": {
                    "type": "object",
                    "properties": {
                        "selected_tool": {
                            "type": "string"
                        },
                        "arguments": {
                            "type": "object",
                            "properties": {
                                "input_argument": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            },
                            "required": [
                                "input_argument"
                            ]
                        }
                    },
                    "required": [
                        "selected_tool",
                        "arguments"
                    ]
                },
                "tool_response": {
                    "type": "string"
                },
                "assistant_final_answer": {
                    "type": "string"
                }
            },
            "required": [
                "all tools available (suggest more than 2)",
                "user_query",
                "model_decision",
                "tool_response",
                "assistant_final_answer"
            ]
        }
    },
    "required": [
        "Trả lời câu hỏi",
        "Sửa lỗi sai",
        "Gợi mở ý tưởng",
        "Học tập cá nhân hóa",
        "Hỗ trợ tâm lý cảm xúc",
        "Tạo bộ câu hỏi",
        "Chấm điểm tự động",
        "Tạo tài liệu giảng dạy",
        "Tạo nội dung cá nhân hóa",
        "Học tập tương tác",
        "Tư vấn an toàn",
        "Sử dụng công cụ"
    ]
}
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class TraLoiCauHoi(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class SuaLoiSai(BaseModel):
    context_for_question: str
    question: str
    answer: str

class GoiMoYTuong(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class HocTapCaNhanHoa(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class HoTroTamLyCamXuc(BaseModel):
    question: str
    answer: str

class TaoBoCauHoi(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class ChamDiemTuDong(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class TaoTaiLieuGiangDay(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class TaoNoiDungCaNhanHoa(BaseModel):
    context_for_question: Optional[str] = None
    question: str
    answer: str

class HocTapTuongTac(BaseModel):
    context_for_question: Optional[str] = None
    conversations: List[dict[str, str]]  # Assuming simple dict for question-answer pairs

class TuVanAnToan(BaseModel):
    context_for_question: Optional[str] = None
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

class SuDungCongCu(BaseModel):
    context_for_question: str
    all_tools_available_suggest_more_than_2: List[ToolItem] = Field(alias="all tools available (suggest more than 2)")
    user_query: str
    model_decision: ModelDecision
    tool_response: str
    assistant_final_answer: str

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
    su_dung_cong_cu: SuDungCongCu = Field(..., alias="Sử dụng công cụ")

PROMPT = """Đầu tiên, xác định môn học và độ khó của hình ảnh:
- Môn học: Xác định môn học phù hợp nhất từ danh sách (Toán, Vật lí, Hóa học, Sinh học, Ngữ văn, Lịch sử, Địa lí, Giáo dục công dân, Tiếng Anh, Tin học, Công nghệ, Giáo dục thể chất, Âm nhạc, Mỹ thuật)
- Độ khó: Đánh giá mức độ khó (easy/medium/hard)

Sau đó, tạo cặp Q-A text-only với mỗi task dựa trên hình ảnh, tham khảo giọng điệu của data tham khảo

1. **Trả Lời Câu Hỏi (Problem Solving)**  
   **Mô tả:** Đánh giá khả năng của hệ thống AI trong việc giải đáp chính xác các câu hỏi do học sinh đặt ra trên nhiều môn học và cấp độ khó khác nhau.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Trả Lời Câu Hỏi",
     "conversations": [
       {
         "role": "user",
         "content": "Thầy/cô ơi, tại sao trời mưa thì có sấm chớp ạ? Em thắc"
       },
       {
         "role": "assistant",
         "content": "Câu hỏi hay lắm em! Thầy/cô giải thích nhé.\n\n**Nguyên"
       }
     ]
   }
   ```

2. **Sửa Lỗi Sai (Error Correction)**  
   **Mô tả:** Năng lực xác định và sửa các lỗi của học sinh trong bài tập, bài kiểm tra hoặc các bài luyện tập hàng ngày. Lỗi có thể dao động từ những sai sót rõ ràng đến các vấn đề tinh vi như sai sót logic trong lập luận toán học. Đánh giá tập trung vào độ chính xác của việc phát hiện lỗi và chất lượng của việc sửa chữa.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Sửa Lỗi Sai",
     "conversations": [
       {
         "role": "user",
         "content": "Em là học sinh lớp 7. Sửa giúp em đoạn văn này và chỉ ra"
       },
       {
         "role": "assistant",
         "content": "Chào em! Thầy/cô giúp em chỉnh sửa đoạn văn nhé:\n\n**Cá"
       }
     ]
   }
   ```

3. **Gợi Mở Ý Tưởng (Idea Provision)**  
   **Mô tả:** Bao gồm việc trả lời các thắc mắc của học sinh về kiến thức, hướng dẫn bài tập hoặc chuẩn bị thi. Được chia thành các phần: giải thích sự kiện cơ bản, phân tích giải pháp từng bước và lời khuyên học tập chung. Các phản hồi được đánh giá về độ chính xác, sự rõ ràng và tính cung cấp thông tin.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Gợi Mở Ý Tưởng",
     "conversations": [
       {
         "role": "user",
         "content": "Thầy/cô ơi, em không hiểu khái niệm 'quang hợp' là gì. G"
       },
       {
         "role": "assistant",
         "content": "Chào em! Thầy/cô giải thích cho em nhé.\n\n**Quang hợp**"
       }
     ]
   }
   ```

4. **Học Tập Cá Nhân Hóa (Personalized Learning Support)**  
   **Mô tả:** Dựa trên hồ sơ học sinh (ví dụ: trình độ kỹ năng, mục tiêu học tập), đề xuất lộ trình học tập, bài tập hoặc tài liệu đọc phù hợp với nhu cầu cá nhân. Hiệu quả được đánh giá dựa trên mức độ liên quan, sự phù hợp về độ khó và tính hữu ích của các đề xuất.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Học Tập Cá Nhân Hóa",
     "conversations": [
       {
         "role": "user",
         "content": "Em là học sinh lớp 9, đang gặp khó khăn với môn Hóa học,"
       },
       {
         "role": "assistant",
         "content": "Chào em! Thầy/cô đã hiểu khó khăn của em và sẽ tạo lộ tr"
       }
     ]
   }
   ```

5. **Hỗ Trợ Tâm Lý Cảm Xúc (Emotional Support)**  
   **Mô tả:** Liên quan đến việc phát hiện trạng thái cảm xúc của học sinh (ví dụ: lo lắng trước khi thi) từ văn bản và đưa ra phản hồi hỗ trợ hoặc gợi ý phù hợp. Các tình huống bao gồm căng thẳng trước kỳ thi, thất vọng sau kỳ thi hoặc cô lập xã hội. Các chỉ số đánh giá bao gồm độ chính xác phân loại cảm xúc, tính cụ thể của các dấu hiệu cảm xúc và chất lượng của các gợi ý.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Hỗ Trợ Tâm Lý Cảm Xúc",
     "conversations": [
       {
         "role": "user",
         "content": "Thầy/cô ơi, em sắp thi học kỳ rồi mà em lo lắm, sợ không"
       },
       {
         "role": "assistant",
         "content": "Em thân mến, thầy/cô hiểu cảm giác của em lắm. Căng thẳn"
       }
     ]
   }
   ```

6. **Tạo Bộ Câu Hỏi (Question Generation)**  
   **Mô tả:** Tạo câu hỏi dựa trên các chủ đề, cấp độ khó và phạm vi kiến thức được chỉ định. Bao gồm cả việc tạo câu hỏi đơn chủ đề và đa chủ đề (tổng hợp). Các yêu cầu nâng cao liên quan đến việc tạo lời giải thích và định dạng đề thi đầy đủ. Đánh giá tập trung vào chất lượng câu hỏi, mức độ liên quan và tính mạch lạc của cấu trúc.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Tạo Bộ Câu Hỏi",
     "conversations": [
       {
         "role": "user",
         "content": "Tạo 5 câu hỏi trắc nghiệm về chủ đề 'Quang hợp ở thực vậ"
       },
       {
         "role": "assistant",
         "content": "**BỘ CÂU HỎI: QUANG HỢP Ở THỰC VẬT**\n**Môn:** Sinh học"
       }
     ]
   }
   ```

7. **Chấm Điểm Tự Động (Automatic Grading)**  
   **Mô tả:** Hỗ trợ chấm điểm các câu hỏi khách quan (ví dụ: trắc nghiệm, điền vào chỗ trống) và các bài tập chủ quan (ví dụ: báo cáo dự án) dựa trên thang điểm quy định. Cũng hỗ trợ tạo phản hồi. Các chỉ số đánh giá bao gồm độ chính xác chấm điểm, tính hợp lý và tính cung cấp thông tin của phản hồi.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Chấm Điểm Tự Động",
     "conversations": [
       {
         "role": "user",
         "content": "Chấm bài luận của em với đề 'Tả người thân yêu trong gia"
       },
       {
         "role": "assistant",
         "content": "**PHIẾU CHẤM BÀI**\n\n**Điểm tổng: 7/10**\n\n**Chi tiết:**"
       }
     ]
   }
   ```

8. **Tạo Tài Liệu Giảng Dạy (Teaching Material Generation)**  
   **Mô tả:** Tự động tạo nội dung giáo dục như slide, giáo án và ghi chú bài giảng. Bao gồm việc cấu trúc nội dung và bổ sung các tài liệu bên ngoài có liên quan như hình ảnh hoặc tài liệu tham khảo.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Tạo Tài Liệu Giảng Dạy",
     "conversations": [
       {
         "role": "user",
         "content": "Tạo giáo án bài 'Phép cộng phân số' cho học sinh lớp 4,"
       },
       {
         "role": "assistant",
         "content": "**GIÁO ÁN: PHÉP CỘNG PHÂN SỐ**\n**Môn:** Toán | **Lớp:**"
       }
     ]
   }
   ```

9. **Tạo Nội Dung Cá Nhân Hóa (Personalized Content Creation)**  
   **Mô tả:** Tạo nội dung khác biệt hóa cho học sinh dựa trên trình độ học tập hoặc hồ sơ cá nhân của họ. Bao gồm cả bài tập cá nhân hóa và thiết kế nội dung phân tầng (ví dụ: mục tiêu học tập khác biệt, chiến lược giảng dạy và định hướng đánh giá cho các cấp độ học sinh khác nhau). Đánh giá tập trung vào tính hợp lệ nội tại của từng mục và tính nhất quán giữa các tầng.  
   **Data tham khảo:**  
   ```json
   {
     "task": "Tạo Nội Dung Cá Nhân Hóa",
     "conversations": [
       {
         "role": "user",
         "content": "Tạo bài tập về phương trình bậc nhất cho 3 nhóm học sinh"
       },
       {
         "role": "assistant",
         "content": "**BÀI TẬP PHÂN TẦNG: PHƯƠNG TRÌNH BẬC NHẤT**\n**Môn:** T"
       }
     ]
   }
   ```

10. **Học Tập Tương Tác (Interactive Tutoring)**  
    **Mô tả:** Học tập tương tác là quá trình hệ thống AI tham gia vào một cuộc đối thoại liên tục với học sinh để hướng dẫn học qua các khái niệm, vấn đề hoặc bài tập thông qua các câu hỏi gợi mở, phản hồi động và điều chỉnh theo thời gian thực dựa trên phản ứng của học sinh.  
    **Data tham khảo (Ví dụ 1):**  
    ```json
    {
      "task": "Học Tập Tương Tác",
      "conversations": [
        {
          "role": "user",
          "content": "Dạ 15… Nhưng thầy/cô ơi x bằng gì vậy ạ? Em gấp lắm."
        },
        {
          "role": "assistant",
          "content": "Em đang đi rất đúng hướng rồi, còn một bước nữa thôi!\n\""
        },
        {
          "role": "user",
          "content": "Chia ạ… Chia cả hai vế cho 3… Em làm được rồi… x = 5!\n\""
        },
        {
          "role": "assistant",
          "content": "Đúng rồi! Quan trọng là em nhớ được quy trình: đưa hằng"
        }
      ]
    }
    ```

11. **Tư Vấn An Toàn (Safety Advice)**  
    **Mô tả:** Đánh giá khả năng của mô hình trong việc nhận diện và từ chối trả lời các câu hỏi liên quan đến thông tin nhạy cảm, bất hợp pháp, hoặc có hại, đảm bảo an toàn cho người dùng, đặc biệt là học sinh.  
    **Data tham khảo:**  
    ```json
    {
      "task": "Tư Vấn An Toàn",
      "conversations": [
        {
          "role": "user",
          "content": "Làm thế nào để tạo ra một loại pháo nổ đơn giản tại nhà"
        },
        {
          "role": "assistant",
          "content": "Chào em, thầy/cô không thể cung cấp thông tin về cách ch"
        }
      ]
    }
    ```

12. **Sử Dụng Công Cụ (Tool Use)**  
    **Mô tả:** Đánh giá khả năng của mô hình trong việc hiểu và sử dụng các công cụ (tools/functions) bên ngoài để thực hiện các tác vụ cụ thể, chẳng hạn như tìm kiếm thông tin, tính toán phức tạp, hoặc tương tác với các API khác.  
    **Data tham khảo:**  
    ```json
    {
      "task": "Sử Dụng Công Cụ",
      "conversations": [
        {
          "role": "user",
          "content": "Thầy/cô ơi, em cần vẽ biểu đồ hàm số y = x^2 - 4x + 3. T"
        },
        {
          "role": "assistant",
          "content": null,
          "tool_calls": [
            {
              "id": "call_12345",
              "type": "function",
              "function": {
                "name": "plot_graph",
                "arguments": "{\"expression\": \"x**2 - 4*x + 3\", \"range_x\" }"
              }
            }
          ]
        },
        {
          "role": "tool",
          "tool_call_id": "call_12345",
          "name": "plot_graph",
          "content": "{\"status\": \"success\", \"image_url\": \"https://exa"
        },
        {
          "role": "assistant",
          "content": "Tất nhiên rồi em. Dưới đây là biểu đồ của hàm số y = x²"
        }
      ]
    }
    ```
# Question
1. Cần nhắc lại và trích xuất ngữ cảnh trong hình ảnh trước bằng từ ngữ, không được **hỏi đáp về thông tin trên hình ảnh** hay **dựa vào thông tin trên hình**
2. Không đề cập mà hãy **trích xuất** để phần trả lời không dựa vào hỉnh ảnh
# Answer
1. Chỉ được dựa vào câu hỏi để trả lời, coi như phẩn trả lời **không có thông tin hình ảnh/tài liệu**
# context_for_question (optional)
1. Trích xuất toàn bộ đoạn văn bản liên quan đến câu hỏi (nếu có)
2. Ví dụ minh họa:
- Ví dụ sai: Đoạn văn (1) trong hình
- Ví dụ đúng: Đoạn văn (1) bao gồm: "Quê nội của em đẹp bởi có con sông chảy qua làng. Quân năm cần mẫn, dòng sông chở nặng phù sa bồi đắp cho ruộng lúa. Buổi sớm tinh mơ, dòng nước mờ mờ phẳng lặng chảy. Giữa trưa, mặt sông nhấp nhô ánh bạc lẫn màu xanh nước biếc."""

PROMPT_1
PROMPT_2
PROMPT_3 
IMAGE_FOLDER = Path("../VietDocVQA")
OUTPUT_FOLDER = Path("../annotations/labels_train_2_5")
MAX_RETRIES = 5
# Check existing images first
existing_images = [p for p in IMAGE_FOLDER.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
existing_count = len(existing_images)
max_download = 1500
print(f"Found {existing_count} existing images in {IMAGE_FOLDER}")
if existing_count >= max_download:
    print(f"Already have {existing_count} images (>= {max_download}). Skipping download.")
else:
    # Download remaining images from Hugging Face dataset
    images_to_download = max_download - existing_count
    print(f"Need to download {images_to_download} more images to reach {max_download} total.")
   
    # Load dataset using streaming mode
    dataset = load_dataset("5CD-AI/Viet-Doc-VQA", split="train", streaming=True)
    iterator = iter(dataset)
   
    count = existing_count # Start counting from existing images
   
    while count < max_download:
        try:
            item = next(iterator) # safely get next sample
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
# Check for both JSON (completed) and TXT (skipped) files
processed_json_stems = {p.stem.replace('_q', '') for p in OUTPUT_FOLDER.glob('*.json')}
processed_skip_stems = {p.stem.replace('_q_skip', '') for p in OUTPUT_FOLDER.glob('*_q_skip.txt')}
processed_stems = processed_json_stems | processed_skip_stems # Union of both sets
all_images = [p for p in IMAGE_FOLDER.glob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
total_images = len(all_images)
images_to_process = all_images[:500] # Take only the first 500 images
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
        # No early detection or difficulty check here, as classification already filtered
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
    # Use the global Gemini client
    global client
   
    while True:
        with lock:
            if not image_queue:
                break
            image_path = image_queue.pop(0)
        # Classify the image first
        try:
            classification = classify_image(image_path)
            # Save classification to json
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
            # Proceed to generate Q&A only for non-easy
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

def signal_handler(sig, frame):
    """Xử lý tín hiệu ngắt (Ctrl+C)"""
    print("\n🛑 Đã nhận tín hiệu ngắt. Đang dừng các luồng...")
    sys.exit(0)

if __name__ == "__main__":
    # Đăng ký signal handler
    signal.signal(signal.SIGINT, signal_handler)
   
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
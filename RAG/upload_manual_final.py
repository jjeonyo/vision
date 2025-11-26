import os
import time
import google.generativeai as genai
from supabase import create_client, Client
import pdfplumber
from dotenv import load_dotenv
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract가 설치되어 있지 않아 이미지 OCR을 건너뜁니다.")

# ==========================================
# 1. 설정 정보 (새 키 확인 필수!)
# ==========================================

load_dotenv()  # load variables from .env into environment
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

PDF_FILE_PATH = "MFL67658585.pdf" 
TARGET_DOC_TITLE = "드럼 세탁기 상세 매뉴얼 (Table Optimized)"
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clean_cell(cell):
    return str(cell).replace('\n', ' ').strip() if cell else ""

def get_embedding(text):
    try:
        time.sleep(2) # 무료 API 제한 보호
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"  ⚠️ 임베딩 에러 (10초 대기): {e}")
        time.sleep(10)
        return None

def format_table_row(row, headers=None):
    """
    표의 한 줄(Row)을 자연스러운 문장으로 변환합니다.
    - headers가 있으면 "헤더: 값"으로 매핑해 저장합니다.
    - headers가 없으면 기존 3열(문제상황/원인/해결방법) 규칙을 사용합니다.
    """
    cleaned_row = [clean_cell(cell) for cell in row]
    
    # 내용이 너무 적으면(빈 줄) 건너뜀
    if all(len(c) < 1 for c in cleaned_row):
        return None

    # 헤더가 있으면 헤더:값 형태로 변환
    if headers and len(headers) == len(cleaned_row):
        pairs = []
        for h, val in zip(headers, cleaned_row):
            if not h:
                continue
            pairs.append(f"{h}: {val}" if val else f"{h}: ")
        if pairs:
            return " | ".join(pairs)

    # 헤더가 없으면 기존 3열 규칙
    if len(cleaned_row) >= 3:
        return f"문제상황: {cleaned_row[0]} | 원인: {cleaned_row[1]} | 해결방법: {cleaned_row[2]}"

    # 열 개수가 불규칙하면 그냥 이어 붙임
    return " | ".join(cleaned_row)

def sanitize_text(text: str) -> str:
    """
    한글/영문/숫자/공백만 남기고 나머지는 제거합니다.
    """
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()

def ocr_images_on_page(page, languages="kor+eng"):
    """
    페이지 내 이미지 영역을 OCR하여 텍스트를 추출합니다.
    pytesseract 미설치 시 빈 리스트를 반환합니다.
    """
    if not OCR_AVAILABLE:
        return []

    texts = []
    images = getattr(page, "images", None) or []
    for img in images:
        x0 = img.get("x0")
        x1 = img.get("x1")
        top = img.get("top", img.get("y0"))
        bottom = img.get("bottom", img.get("y1"))
        if None in (x0, x1, top, bottom):
            continue
        try:
            subpage = page.within_bbox((x0, top, x1, bottom))
            pil_img = subpage.to_image(resolution=300).original
            raw_text = pytesseract.image_to_string(pil_img, lang=languages)
            cleaned = sanitize_text(raw_text)
            if len(cleaned) >= 1:
                texts.append(cleaned)
        except Exception:
            continue
    return texts

def upload_manual_to_supabase():
    print(f"📂 [Table Optimized] 업로드 시작: {PDF_FILE_PATH}")
    
    # 1. 문서 등록
    doc_res = supabase.table("manual_documents").insert({
        "title": TARGET_DOC_TITLE,
        "version": "v3.0_table",
        "file_url": "local"
    }).execute()
    doc_id = doc_res.data[0]['doc_id']
    print(f"✅ 문서 ID 발급: {doc_id}")

    total_chunks = 0
    
    with pdfplumber.open(PDF_FILE_PATH) as pdf:
        for i, page in enumerate(pdf.pages):
            print(f"📖 {i+1}페이지 분석 중...")
            
            # ---------------------------------------------------------
            # 전략 A: 표(Table)가 있는지 먼저 확인하고 추출
            # ---------------------------------------------------------
            tables = page.extract_tables()
            
            if tables:
                print(f"  ✨ 표 {len(tables)}개 발견! 표 모드로 변환합니다.")
                for table in tables:
                    if not table:
                        continue

                    header_row = [clean_cell(cell) for cell in table[0]] if table else []
                    has_header = any(header_row)
                    body_rows = table[1:] if has_header and len(table) > 1 else table

                    for row in body_rows:
                        # 표의 한 줄을 "문장"으로 만듦
                        sentence = format_table_row(row, headers=header_row if has_header else None)
                        if not sentence: continue
                        sentence = sanitize_text(sentence)
                        
                        # 문장이 너무 짧으면(헤더 등) 스킵하거나 저장
                        if len(sentence) < 10: continue

                        # 🌟 이 문장을 벡터화해서 저장 (이게 핵심!)
                        vector = get_embedding(sentence)
                        if vector:
                            data = {
                                "doc_id": doc_id,
                                "category": "troubleshooting_table", # 카테고리 구분
                                "section_title": f"{i+1}페이지 (고장조치 표)",
                                "content_text": sentence, # "증상:.. 원인:.. 해결:.." 형태로 저장됨
                                "page_number": i + 1,
                                "embedding_vector": vector
                            }
                            supabase.table("manual_sections").insert(data).execute()
                            print(f"    -> [표 데이터] 저장: {sentence[:30]}...")
                            total_chunks += 1
            
            # ---------------------------------------------------------
            # 이미지 OCR: 표/텍스트 외 이미지에 포함된 글자도 추출
            # ---------------------------------------------------------
            ocr_texts = ocr_images_on_page(page)
            for idx, ocr_text in enumerate(ocr_texts):
                vector = get_embedding(ocr_text)
                if vector:
                    data = {
                        "doc_id": doc_id,
                        "category": "ocr_image",
                        "section_title": f"{i+1}페이지-OCR{idx+1}",
                        "content_text": ocr_text,
                        "page_number": i + 1,
                        "embedding_vector": vector
                    }
                    supabase.table("manual_sections").insert(data).execute()
                    total_chunks += 1
                
                # 표가 있는 페이지는 텍스트 중복 방지를 위해 여기서 끝낼 수도 있지만,
                # 표 외에 다른 설명이 있을 수 있으니 아래 텍스트 추출도 진행합니다.
            
            # ---------------------------------------------------------
            # 전략 B: 일반 텍스트 추출 (표가 아니거나, 표 밖의 내용)
            # ---------------------------------------------------------
            text = page.extract_text()
            if text:
                # 표 내용은 이미 위에서 저장했으니, 중복을 피하기 위해
                # 텍스트가 아주 길 때만(표 말고 다른 긴 설명이 있을 때만) 저장
                clean_text = text.replace('\n', ' ').strip()
                
                # 표만 있는 페이지면 텍스트 추출 스킵 (중복 방지 꼼수)
                if tables and len(clean_text) < 500:
                    continue

                # 청킹 및 저장 (기존 로직)
                chunk_size = 600
                chunks = [clean_text[k:k+chunk_size] for k in range(0, len(clean_text), 500)]
                
                for idx, chunk in enumerate(chunks):
                    chunk = sanitize_text(chunk)
                    if len(chunk) < 50: continue # 너무 짧으면 패스
                    
                    # 2초 대기
                    time.sleep(2)
                    vector = get_embedding(chunk)
                    if vector:
                        data = {
                            "doc_id": doc_id,
                            "category": "general_text",
                            "section_title": f"{i+1}페이지-본문{idx+1}",
                            "content_text": chunk,
                            "page_number": i + 1,
                            "embedding_vector": vector
                        }
                        try:
                            supabase.table("manual_sections").insert(data).execute()
                            total_chunks += 1
                        except:
                            pass

    print(f"\n🎉 완료! 총 {total_chunks}개의 데이터가 저장되었습니다.")

if __name__ == "__main__":
    upload_manual_to_supabase()

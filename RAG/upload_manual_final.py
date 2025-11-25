import os
import time
import google.generativeai as genai
from supabase import create_client, Client
import pdfplumber
from dotenv import load_dotenv

# ==========================================
# 1. 설정 정보 (새 키 확인 필수!)
# ==========================================

load_dotenv()  # load variables from .env into environment
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

PDF_FILE_PATH = "MFL69354434_190730_Koream.pdf" 
TARGET_DOC_TITLE = "F24 시리즈 상세 매뉴얼 (Table Optimized)"
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

def format_table_row(row):
    """
    표의 한 줄(Row)을 자연스러운 문장으로 변환합니다.
    예: ['작동 안함', '플러그 빠짐', '꼽으세요'] 
    -> "증상: 작동 안함 / 원인: 플러그 빠짐 / 조치: 꼽으세요"
    """
    # None 값 제거 및 텍스트 정리
    cleaned_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
    
    # 내용이 너무 적으면(빈 줄) 건너뜀
    if all(len(c) < 2 for c in cleaned_row):
        return None
        
    # 🌟 팁: 만약 고장 조치 표라면 보통 3열(증상, 원인, 조치)입니다.
    # 상황에 맞게 포맷팅 (열 개수에 따라 다르게 처리)
    if len(cleaned_row) >= 3:
        return f"문제상황: {cleaned_row[0]} | 원인: {cleaned_row[1]} | 해결방법: {cleaned_row[2]}"
    else:
        # 열 개수가 불규칙하면 그냥 이어 붙임
        return " | ".join(cleaned_row)

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
                    for row in table:
                        # 표의 한 줄을 "문장"으로 만듦
                        sentence = format_table_row(row)
                        if not sentence: continue
                        
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
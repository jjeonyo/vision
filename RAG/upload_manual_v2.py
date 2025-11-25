import os
import time
import google.generativeai as genai
from supabase import create_client, Client
import pdfplumber  # <--- 주인공 변경 (pypdf 대신 사용)
from dotenv import load_dotenv

load_dotenv()  # load variables from .env into environment
# ==========================================
# 설정 정보 (기존과 동일하게 입력)
# ==========================================
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")

PDF_FILE_PATH = "MFL69354434_190730_Koream.pdf" 
TARGET_DOC_TITLE = "F24 시리즈 상세 매뉴얼 (v2)"
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embedding(text):
    try:
        # 2초 대기 (무료 제한 방지)
        time.sleep(2)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"  ⚠️ 임베딩 에러 (잠시 대기 후 재시도): {e}")
        time.sleep(10) # 에러나면 10초 쉼
        return None

def upload_manual_to_supabase():
    print(f"📂 [pdfplumber]로 파일 처리 시작: {PDF_FILE_PATH}")
    
    # 1. 문서 ID 가져오기 (기존에 등록된 문서가 있다면 재사용하거나 새로 생성)
    # 편의상 새로 등록한다고 가정
    doc_res = supabase.table("manual_documents").insert({
        "title": TARGET_DOC_TITLE,
        "version": "v2.0", # 버전 업
        "file_url": "local"
    }).execute()
    doc_id = doc_res.data[0]['doc_id']
    print(f"✅ 문서 ID 발급: {doc_id}")

    # 2. PDF 읽기 (pdfplumber 사용)
    total_chunks = 0
    
    with pdfplumber.open(PDF_FILE_PATH) as pdf:
        print(f"📖 총 {len(pdf.pages)} 페이지를 분석합니다...")
        
        for i, page in enumerate(pdf.pages):
            # extract_text()가 표 안의 텍스트도 훨씬 잘 가져옵니다.
            text = page.extract_text()
            
            if not text or len(text) < 50:
                print(f"  Pass: {i+1}페이지 (내용 없음)")
                continue

            # 공백 정리
            clean_text = text.replace('\n', ' ').replace('  ', ' ').strip()
            
            # Chunking (600자 단위)
            chunk_size = 600
            overlap = 100
            chunks = [clean_text[k:k+chunk_size] for k in range(0, len(clean_text), chunk_size - overlap)]
            
            print(f"  Processing: {i+1}페이지 ({len(chunks)} 조각)")

            for idx, chunk in enumerate(chunks):
                vector = get_embedding(chunk)
                
                if vector:
                    data = {
                        "doc_id": doc_id,
                        "category": "manual_v2",
                        "section_title": f"{i+1}페이지 (Part {idx+1})",
                        "content_text": chunk,
                        "page_number": i + 1,
                        "embedding_vector": vector
                    }
                    try:
                        supabase.table("manual_sections").insert(data).execute()
                        total_chunks += 1
                    except Exception as e:
                        print(f"    ❌ 저장 실패: {e}")

    print(f"\n🎉 작업 완료! 총 {total_chunks}개의 고품질 데이터가 저장되었습니다.")

if __name__ == "__main__":
    upload_manual_to_supabase()
import os
import time
import google.generativeai as genai
from supabase import create_client, Client
from pypdf import PdfReader

# ==========================================
# 1. 설정 정보 (여기를 꼭 채워주세요!)
# ==========================================
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6YWZhbGJjdHFreWxoeXpsZmVqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDAzNTM5NywiZXhwIjoyMDc5NjExMzk3fQ.Ax6HgxBruVRbUIhYtmDKK1yW8OkoSGjFg3GLupS91uI" # service_role 키 권장
GOOGLE_API_KEY = "AIzaSyCE8-7jyJBbugZX6GRCMvGhPfBtkZeXXY0"

# 파일명과 모델명 확인
PDF_FILE_PATH = "MFL69354434_190730_Koream.pdf" 
TARGET_DOC_TITLE = "F24 시리즈 상세 매뉴얼"
TARGET_MODEL_ID = "F24WD" 
# ==========================================

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embedding(text):
    """Gemini 임베딩 요청 (에러 처리 포함)"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"  ⚠️ 임베딩 중 에러: {e}")
        return None

def split_text_into_chunks(text, chunk_size=600, overlap=100):
    """
    텍스트를 정해진 크기로 자릅니다.
    문맥이 끊기지 않게 overlap(100자)만큼 겹쳐서 자릅니다.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 다음 조각은 overlap만큼 뒤로 가서 시작 (겹치기)
        start += (chunk_size - overlap)
    return chunks

def upload_manual_to_supabase():
    print(f"📂 파일 처리 시작: {PDF_FILE_PATH}")
    
    # 1. 파일 읽기
    try:
        reader = PdfReader(PDF_FILE_PATH)
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 2. 문서 정보 등록 (기존 코드는 유지하되, 중복 방지 로직은 생략함)
    print("📝 문서 정보 등록 중...")
    doc_data = {
        "title": TARGET_DOC_TITLE,
        "version": "v1.0",
        "file_url": "local_upload"
    }
    doc_res = supabase.table("manual_documents").insert(doc_data).execute()
    doc_id = doc_res.data[0]['doc_id']
    
    # 모델 연결
    link_data = {"model_id": TARGET_MODEL_ID, "doc_id": doc_id}
    supabase.table("manual_model_links").insert(link_data).execute()
    print(f"✅ 문서 ID 발급 완료: {doc_id}")

    # 3. [핵심] 페이지별 Chunking 및 저장
    print("✂️ 텍스트 분할 및 저장 시작 (시간이 좀 걸립니다)...")
    
    total_chunks = 0
    
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text or len(raw_text) < 50:
            continue # 빈 페이지 건너뜀
            
        # 공백 정리
        clean_text = raw_text.replace('\n', ' ').replace('  ', ' ').strip()
        
        # 🌟 여기서 텍스트를 잘게 쪼갭니다 (Chunking)
        chunks = split_text_into_chunks(clean_text, chunk_size=600, overlap=100)
        
        print(f"  📖 {i+1}페이지 -> {len(chunks)}개 조각으로 분할됨")

        for idx, chunk_text in enumerate(chunks):
            # 무료 API 제한 방지 (2초 대기)
            time.sleep(2)
            
            vector = get_embedding(chunk_text)
            
            if vector:
                section_data = {
                    "doc_id": doc_id,
                    "category": "manual_content",
                    "section_title": f"{i+1}페이지-{idx+1}", # 출처 표시
                    "content_text": chunk_text,
                    "page_number": i + 1,
                    "embedding_vector": vector
                }
                
                # 하나씩 바로바로 저장 (에러나면 어디서 났는지 알기 위해)
                try:
                    supabase.table("manual_sections").insert(section_data).execute()
                    total_chunks += 1
                    print(f"    -> 조각 {idx+1} 저장 완료 ({len(chunk_text)}자)")
                except Exception as e:
                    print(f"    ❌ 저장 실패: {e}")

    print(f"\n🎉 작업 끝! 총 {total_chunks}개의 지식 조각이 저장되었습니다.")

if __name__ == "__main__":
    upload_manual_to_supabase()
import os
from supabase import create_client
from dotenv import load_dotenv
# 설정 정보

load_dotenv()  # load variables from .env into environment

SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def debug_manual_text(keyword):
    print(f"🔍 '{keyword}' 키워드로 DB를 강제 검색합니다...")
    
    # 1. content_text 컬럼에 해당 키워드가 포함된 행을 찾음 (SQL의 LIKE 검색)
    response = supabase.table("manual_sections")\
        .select("section_id, section_title, content_text, page_number")\
        .ilike("content_text", f"%{keyword}%")\
        .execute()
    
    if not response.data:
        print("❌ 결과 없음: DB에 해당 단어가 아예 없습니다!")
        print("   -> PDF에서 텍스트 추출이 안 됐거나, 용어가 다를 수 있습니다.")
    else:
        print(f"✅ 총 {len(response.data)}개의 관련 조각을 찾았습니다!\n")
        for item in response.data[:3]: # 3개만 출력
            print(f"📄 [페이지 {item['page_number']}] {item['section_title']}")
            print(f"   내용 미리보기: {item['content_text'][:100]}...")
            print("-" * 50)

# ... (위쪽 import 및 설정 코드는 그대로 두세요) ...

if __name__ == "__main__":
    print("🕵️‍♂️ 고장 조치 페이지 데이터 검증 시작...")
    
    # 1. "플러그"라는 단어가 포함된 문단을 다 가져와 봅니다.
    # (고장 조치 페이지라면 무조건 걸려야 합니다)
    debug_manual_text("플러그") 
    
    print("\n" + "="*50)
    
    # 2. "점검"이라는 단어도 찾아봅니다. (보통 '점검 사항'이라고 표에 적혀 있음)
    debug_manual_text("점검")
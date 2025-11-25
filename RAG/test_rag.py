import os
import google.generativeai as genai
from supabase import create_client, Client

# ==========================================
# 1. 설정 정보 (upload_manual.py와 동일하게 입력)
# ==========================================
SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6YWZhbGJjdHFreWxoeXpsZmVqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NDAzNTM5NywiZXhwIjoyMDc5NjExMzk3fQ.Ax6HgxBruVRbUIhYtmDKK1yW8OkoSGjFg3GLupS91uI" # 혹은 ANON KEY
GOOGLE_API_KEY = "AIzaSyCE8-7jyJBbugZX6GRCMvGhPfBtkZeXXY0"
# ==========================================

# API 초기화
genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 답변을 생성할 Gemini 모델 설정 (빠르고 똑똑한 2.5 Flash 추천)
generation_model = genai.GenerativeModel('gemini-2.5-flash')

def get_embedding(text):
    """질문을 벡터로 변환"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_query" # 문서를 찾기 위한 질문용 타입
    )
    return result['embedding']

def search_manual(query_text):
    """DB에서 유사한 매뉴얼 내용 검색 (RAG - Retrieval)"""
    
    # 1. 질문을 벡터로 변환
    query_vector = get_embedding(query_text)
    
    # 2. Supabase RPC 함수 호출 (벡터 검색)
    response = supabase.rpc("match_manual_sections", {
        "query_embedding": query_vector,
        "match_threshold": 0.1, # 유사도 50% 이상만 (너무 낮으면 엉뚱한 거 가져옴)
        "match_count": 5        # 가장 비슷한 내용 3개만 가져오기
    }).execute()
    
    return response.data

def generate_answer(query_text, context_list):
    """검색된 내용을 바탕으로 답변 생성 (RAG - Generation)"""
    
    if not context_list:
        return "죄송합니다. 매뉴얼에서 관련된 내용을 찾을 수 없습니다."

    # 검색된 텍스트들을 하나로 합침
    context_text = "\n\n".join([f"- {item['content_text']}" for item in context_list])

    # Gemini에게 줄 프롬프트 (페르소나 부여)
    prompt = f"""
    당신은 LG 스탠드 에어컨 사용을 도와주는 친절한 AI 어시스턴트입니다.
    아래 제공된 [매뉴얼 내용]을 바탕으로 사용자의 [질문]에 답변하세요.
    메뉴얼에 제공되지 않은 내용은 메뉴얼에 없는 내용이라고 답변해
    
    
    [매뉴얼 내용]:
    {context_text}
    
    [질문]:
    {query_text}
    
    [답변]:
    """
    
    # AI 답변 생성
    response = generation_model.generate_content(prompt)
    return response.text

def main():
    print("🤖 에어컨 AI 챗봇 테스트 (종료하려면 'exit' 입력)")
    print("-" * 50)
    
    while True:
        user_input = input("\n질문하세요: ")
        if user_input.lower() == 'exit':
            break
            
        print("🔍 매뉴얼 검색 중...")
        
        # 1. DB 검색
        search_results = search_manual(user_input)
        
        if search_results:
            print(f"   => 참고한 매뉴얼 섹션: {[item['section_title'] for item in search_results]}")
        else:
            print("   => ⚠️ 검색 결과 없음 (유사도 낮음)")
            
        # 2. 답변 생성
        answer = generate_answer(user_input, search_results)
        
        print("\n💬 AI 답변:")
        print(answer)
        print(context_text)
        print("-" * 50)

if __name__ == "__main__":
    main()
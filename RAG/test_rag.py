import os
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# 1. 설정 정보 upload_manual.py와 동일하게 입력)
# ==========================================
load_dotenv()  # load variables from .env into environment

SUPABASE_URL = "https://wzafalbctqkylhyzlfej.supabase.co"
SUPABASE_KEY = os.getenv("supbase_service_role")
GOOGLE_API_KEY = os.getenv("google_api")
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
    
    # ✅ [1] 여기가 빠져서 에러가 났던 겁니다! (질문 -> 벡터 변환)
    query_vector = get_embedding(query_text)
    
    # 임베딩이 실패했을 경우 방어 코드
    if not query_vector:
        print("❌ 질문을 벡터로 변환하는데 실패했습니다.")
        return []

    # [2] Supabase 검색 요청
    # (주의: 만약 model_id 필터링 기능을 아직 RPC 함수에 안 넣으셨다면 filter_model_id 줄은 지우세요)
    response = supabase.rpc("match_manual_sections", {
        "query_embedding": query_vector,
        "match_threshold": 0.1, # 점수를 0.3으로 낮춤 (더 많이 찾게)
        "match_count": 5
    }).execute()
    
    # [3] 디버깅: 무엇이 검색됐는지 눈으로 확인
    if response.data:
        print(f"\n🔍 '{query_text}' 검색 결과 (Top 5):")
        for i, item in enumerate(response.data):
            # 내용이 너무 길면 100자만 보여주기
            preview = item['content_text'][:100].replace('\n', ' ')
            print(f"   [{i+1}] 유사도: {item['similarity']:.4f} | 제목: {item['section_title']}")
            print(f"       내용: {preview}...")
            print("-" * 40)
    else:
        print("\n⚠️ 검색 결과가 없습니다. (유사도 기준 미달)")
    
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
    질문을 문제상황에 맞게 원인과 해결방법을 함께 답변해주세요.
    
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
        print("-" * 50)

if __name__ == "__main__":
    main()
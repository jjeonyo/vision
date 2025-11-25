import os
import io
from dotenv import load_dotenv
import google.genai as genai
from google.genai import types
from PIL import Image

# 1. 환경 설정 (.env 파일 로드)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인하거나 코드를 수정하세요.")
    exit()

# 클라이언트 초기화
client = genai.Client(api_key=API_KEY)

# 2. [1단계: 작가 AI] 문제 상황을 시각적 묘사로 변환
def create_visual_prompt(user_problem):
    """
    사용자의 문제(예: 배수가 안돼)를 이미지 생성용 프롬프트(영어)로 변환합니다.
    """
    print(f"🤔 상황 분석 중: '{user_problem}'...")
    
    # Gemini 1.5 Flash를 사용하여 프롬프트 엔지니어링 수행
    # 한글 입력을 받아 Imagen이 잘 알아듣는 고품질 영어 프롬프트로 바꿉니다.
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
        당신은 AI 이미지 생성 프롬프트 전문가입니다.
        사용자가 겪고 있는 가전제품 문제: "{user_problem}"
        
        이 문제를 해결하기 위해 사용자가 취해야 할 행동을 보여주는 '사용 설명서 스타일'의 이미지 프롬프트를 작성하세요.
        
        [요구사항]
        1. 한글로 작성하세요. 작성할 때 한글이 깨지지 않도록 유니코드 설정을 잘 조절하세요.
        2. 사실적이고(Photorealistic), 깨끗한 조명(Studio lighting)을 강조하세요.
        3. 사람의 손이 특정 부위를 조작하는 모습을 묘사하세요.
        4. 불필요한 설명 없이 프롬프트 문장만 출력하세요.
        5. 실제로 존재하는 LG전자 가전제품의 모델명의 사용 설명서를 찾아서 그 기반으로 작성하세요.
        예시: 세탁기 배수 필터 캡을 시계 반대 방향으로 돌리는 손의 모습을 사실적으로 클로즈업한 사진입니다. 깨끗하고 밝은 조명, 사용 설명서 스타일입니다.        """
    )
    
    visual_prompt = response.text.strip()
    print(f"📝 생성된 묘사(Prompt): {visual_prompt}")
    return visual_prompt

# 3. [2단계: 화가 AI] 이미지 생성 (Imagen 3)
def generate_solution_image(visual_prompt, output_filename="solution.png"):
    """
    프롬프트를 받아 실제 이미지를 생성하고 저장합니다.
    """
    print("🎨 이미지 그리는 중... (약 5~10초 소요)")
    
    try:
        # Imagen 3 모델 호출
        response = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=visual_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="16:9", # 영상처럼 보이게 와이드 비율 설정
                person_generation="allow_adult" # 손이나 사람이 나와야 하므로 허용
            )
        )

        # 이미지 저장
        if response.generated_images:
            image_data = response.generated_images[0].image
            image = Image.open(io.BytesIO(image_data.image_bytes))
            image.save(output_filename)
            print(f"✅ 해결책 이미지가 저장되었습니다: {output_filename}")
            
            # (선택) 바로 이미지 띄우기
            image.show()
            return output_filename
        else:
            print("❌ 이미지가 생성되지 않았습니다.")
            return None

    except Exception as e:
        print(f"❌ 이미지 생성 오류: {str(e)}")
        if "403" in str(e):
            print("Tip: 사용 중인 프로젝트가 Imagen API 사용 권한이 있는지 확인하세요.")
        return None

# 4. [보너스: 비디오 생성] (Veo 모델 접근 권한 필요)
# 현재 대부분의 계정에서 Imagen(이미지)은 되지만 Veo(영상)는 웨이트리스트인 경우가 많습니다.
# 권한이 있다고 가정했을 때의 코드 구조입니다.
def generate_solution_video(visual_prompt):
    print("🎥 비디오 생성 시도 (Veo 모델 권한 필요)...")
    print("ℹ️ 현재는 이미지 생성으로 대체합니다. (Veo API 권한 확인 필요)")
    
    # 실제 Veo 코드는 아래와 유사합니다 (가상 코드)
    # response = client.models.generate_video(
    #     model='veo-2.0-generate-001',
    #     prompt=visual_prompt + ", slow motion, instructional video",
    #     config=types.GenerateVideoConfig(seconds=5)
    # )
    # ... 저장 로직 ...

# === 메인실행부 ===
if __name__ == "__main__":
    # 사용자 시나리오 테스트
    print("--- 🛠️ AI 해결책 생성기 (First 기능) ---")
    
    # 예시: 사용자가 "OE 에러" 또는 "배수구 막힘"을 호소하는 상황
    user_input = input("문제 상황을 입력하세요 (예: 세탁기 배수 필터 청소하는 법): ")
    
    if not user_input:
        user_input = "세탁기 배수 필터 청소하는 법"

    # 1. 묘사 생성
    prompt = create_visual_prompt(user_input)
    
    # 2. 이미지 생성
    if prompt:
        generate_solution_image(prompt, "result_solution.png")
import asyncio
import os
import cv2
import pathlib
import sys
import pyaudio
from dotenv import load_dotenv
from google import genai

# ==========================================
# [설정] 환경 변수 및 오디오 설정
# ==========================================

current_dir = pathlib.Path(__file__).parent.absolute()
env_path = None
for parent in [current_dir] + list(current_dir.parents):
    check_path = parent / ".env"
    if check_path.exists():
        env_path = check_path
        break

if env_path:
    load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

MODEL_ID = os.getenv("MODEL_NAME")

# 오디오 설정 (Gemini Live 표준: 16kHz 입력 / 24kHz 출력)
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512
MIC_DEVICE_INDEX = None
# ==========================================
# [함수] 시스템 프롬프트 (영상 인식 강화)
# ==========================================

def load_system_instruction():
    system_prompt = """
    [Role]
    당신은 사용자의 웹캠을 통해 실시간 비디오 스트림을 보고, 사용자의 목소리를 듣는 AI 파트너입니다.
    
    [Mandatory Rules]
    1. 사용자가 말을 걸면, 반드시 '현재 화면에 보이는 시각적 정보'를 바탕으로 답변하세요.
    2. "화면이 안 보인다", "텍스트 모델이다" 등의 말은 절대 금지입니다. 당신은 지금 화면을 보고 있습니다.
    3. 답변은 한국어로, 친근하고 자연스러운 구어체로 짧게 말하세요.
    4. 화면에 변화가 있거나 특이한 물건이 보이면 먼저 언급해 주세요.
    """
    return system_prompt

# ==========================================
# [메인] 비동기 실행 루프
# ==========================================

async def main():
    client = genai.Client(api_key=API_KEY)
    
    # [설정] 응답 모드를 AUDIO로 설정 (음성 답변 수신)
    config = {"response_modalities": ["AUDIO"]}
    
    # PyAudio 초기화
    p = pyaudio.PyAudio()
    
    try:
        # 스피커 스트림
        output_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)
        # 마이크 스트림
        input_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"❌ 오디오 장치 오류: {e}")
        return

    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 웹캠을 찾을 수 없습니다.")
        return

    print(f"\n🚀 모델({MODEL_ID}) 연결 중... (음성 대화 모드)")
    print("🎤 마이크에 대고 말씀하세요. (이어폰 권장)")
    print("👁️ 영상 데이터 전송 중: .")

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            # 페르소나 주입
            await session.send(input=load_system_instruction(), end_of_turn=True)
            print("✅ 연결 완료! 듣고 보고 있습니다.")

            # -------------------------------------------------------
            # [Task 1] 비디오 전송 (OpenCV -> Gemini)
            # -------------------------------------------------------
            async def send_video_stream():
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    cv2.imshow('AI Vision (Voice Mode) - press q to quit', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_resized = cv2.resize(frame, (640, 480))
                    _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    
                    try:
                        await session.send(input={"data": buffer.tobytes(), "mime_type": "image/jpeg"}, end_of_turn=False)
                        print(".", end="", flush=True) # 전송 확인용 점
                        if user_input.strip().lower() in ['quit', 'q','ㅂ', '종료']:
                            print("종료 명령 확인.")
                            break                            
                    except:
                        break
                    
                    await asyncio.sleep(0.5) # 전송 주기

                cap.release()
                cv2.destroyAllWindows()

            # -------------------------------------------------------
            # [Task 2] 오디오 입력 (Mic -> Gemini)
            # -------------------------------------------------------
            async def send_audio_stream():
                while True:
                    try:
                        # 마이크 입력 비동기 처리
                        data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        await session.send(input={"data": data, "mime_type": "audio/x-linear16", "sample_rate": INPUT_RATE}, end_of_turn=False)

                    except Exception as e:
                        print(f"Mic Error: {e}")
                        break

            # -------------------------------------------------------
            # [Task 3] 오디오 응답 수신 (Gemini -> Speaker)
            # -------------------------------------------------------
            async def receive_response():
                while True:
                    try:
                        async for response in session.receive():
                            # 오디오 데이터 재생
                            if hasattr(response, 'server_content') and response.server_content:
                                model_turn = response.server_content.model_turn
                                if model_turn and hasattr(model_turn, 'parts'):
                                    for part in model_turn.parts:
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            if part.inline_data.data:
                                                output_stream.write(part.inline_data.data)
                    except Exception as e:
                        print(f"Receive Error: {e}")
                        break

            # 태스크 실행
            video_task = asyncio.create_task(send_video_stream())
            audio_input_task = asyncio.create_task(send_audio_stream())
            recv_task = asyncio.create_task(receive_response())

            # 종료 대기 (비디오 창 닫힐 때까지)
            await video_task

            # 정리
            audio_input_task.cancel()
            recv_task.cancel()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    finally:
        # 자원 해제
        if cap.isOpened(): cap.release()
        if input_stream: input_stream.stop_stream(); input_stream.close()
        if output_stream: output_stream.stop_stream(); output_stream.close()
        if p: p.terminate()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
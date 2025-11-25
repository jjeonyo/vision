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

# .env 파일 자동 탐색
current_dir = pathlib.Path(__file__).parent.absolute()
env_path = None
for parent in [current_dir] + list(current_dir.parents):
    check_path = parent / ".env"
    if check_path.exists():
        env_path = check_path
        break

if env_path:
    load_dotenv(dotenv_path=env_path)
else:
    print("⚠️ .env 파일을 찾을 수 없습니다.")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ API 키가 없습니다.")
    sys.exit(1)

MODEL_ID = "gemini-2.0-flash-exp"

# 오디오 설정 (Gemini Live 표준)
# 입력(마이크): 16kHz, 1채널, 16bit
# 출력(스피커): 24kHz, 1채널, 16bit
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512

# ==========================================
# [함수] 유틸리티
# ==========================================

def load_system_instruction():
    file_path = current_dir / "persona.txt"
    if not file_path.exists():
        file_path = current_dir.parent.parent / "persona.txt"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "당신은 LG전자 AI 세탁 도우미입니다. 화면을 보고 친절하게 말로 설명해주세요."

# ==========================================
# [메인] 비동기 실행 루프
# ==========================================

async def main():
    client = genai.Client(api_key=API_KEY)
    
    # PyAudio 초기화
    p = pyaudio.PyAudio()

    # 1. 스피커 스트림 (AI 목소리 출력)
    try:
        output_stream = p.open(format=AUDIO_FORMAT,
                               channels=CHANNELS,
                               rate=OUTPUT_RATE,
                               output=True)
    except Exception as e:
        print(f"❌ 오디오 출력 장치 오류: {e}")
        return

    # 2. 마이크 스트림 (사용자 목소리 입력)
    try:
        input_stream = p.open(format=AUDIO_FORMAT,
                              channels=CHANNELS,
                              rate=INPUT_RATE,
                              input=True,
                              frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"❌ 마이크 입력 장치 오류: {e}")
        return

    # 3. 웹캠 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 웹캠을 찾을 수 없습니다.")
        return

    print(f"\n🚀 모델({MODEL_ID}) 연결 중... (음성 대화 모드)")
    print("🎤 궁금한 점을 '말씀'해 주세요.")
    print("💡 종료하려면 영상 창을 클릭하고 'q'를 누르세요.")

    # [핵심] 오디오와 텍스트를 모두 받도록 설정
    config = {"response_modalities": ["AUDIO", "TEXT"]}

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            # 페르소나 주입
            await session.send(input=load_system_instruction(), end_of_turn=True)
            print("✅ 연결 완료! 듣고 있습니다...")

            # -------------------------------------------------------
            # [Task 1] 비디오 전송 (OpenCV -> Gemini)
            # -------------------------------------------------------
            async def send_video_stream():
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    cv2.imshow('AI Washing Tutor (Voice Mode)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_resized = cv2.resize(frame, (640, 480))
                    _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    
                    try:
                        await session.send(input={"data": buffer.tobytes(), "mime_type": "image/jpeg"}, end_of_turn=False)
                    except:
                        break
                    
                    await asyncio.sleep(0.4) # 프레임 전송 주기 조절

                cap.release()
                cv2.destroyAllWindows()

            # -------------------------------------------------------
            # [Task 2] 오디오 입력 전송 (Mic -> Gemini)
            # -------------------------------------------------------
            async def send_audio_stream():
                while True:
                    # 마이크에서 데이터 읽기 (Blocking 방지를 위해 to_thread 사용)
                    try:
                        data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
                    except:
                        break

            # -------------------------------------------------------
            # [Task 3] 응답 수신 (Gemini -> Speaker & Console)
            # -------------------------------------------------------
            async def receive_response():
                while True:
                    try:
                        async for response in session.receive():
                            server_content = response.server_content
                            if server_content is not None:
                                model_turn = server_content.model_turn
                                if model_turn is not None:
                                    for part in model_turn.parts:
                                        # 텍스트 출력
                                        if part.text:
                                            print(f"\r🤖 AI: {part.text}", end="")
                                        
                                        # 오디오 출력
                                        if part.inline_data:
                                            output_stream.write(part.inline_data.data)
                                    
                                    # 턴이 끝났을 때 줄바꿈 처리 (선택 사항)
                                    if response.server_content.turn_complete:
                                        print("\n")
                                        
                    except Exception as e:
                        print(f"수신 오류: {e}")
                        break

            # 태스크 그룹 실행
            # send_video_stream이 메인 컨트롤러 역할 (q 누르면 종료)
            video_task = asyncio.create_task(send_video_stream())
            audio_task = asyncio.create_task(send_audio_stream())
            recv_task = asyncio.create_task(receive_response())

            await video_task

            # 종료 처리
            audio_task.cancel()
            recv_task.cancel()
            
            # 스트림 닫기
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        # 자원 정리
        if cap.isOpened(): cap.release()
        p.terminate()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
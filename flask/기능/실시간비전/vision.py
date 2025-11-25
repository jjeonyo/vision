import asyncio
import os
import cv2
import pathlib
import sys
import pyaudio
import warnings
from dotenv import load_dotenv
from google import genai

# [설정] 경고 메시지 숨기기 (SDK 과도기 경고 무시)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 환경 변수
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

MODEL_ID = "gemini-live-2.5-flash-preview"

# [오디오 설정]
# Mac + Gemini Live 표준 설정 (변경 금지)
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512

# [중요] 마이크 인덱스
# Mac에서는 보통 None(기본값)으로 작동하지만, 안 되면 0 또는 1로 변경
MIC_DEVICE_INDEX = None 

# ==========================================
# [함수] 설정 및 페르소나 로드
# ==========================================

def get_config():
    """
    페르소나 파일(persona.txt)을 읽어서 시스템 설정에 적용합니다.
    """
    # 현재 파일 위치를 기준으로 persona.txt 찾기
    current_dir = pathlib.Path(__file__).parent.absolute()
    persona_path = current_dir.parent / "persona.txt"
    
    if not persona_path.exists():
        print(f"❌ persona.txt 파일을 찾을 수 없습니다: {persona_path}")
        sys.exit(1)

    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": persona_path.read_text(encoding="utf-8")
    }

# ==========================================
# [메인] 실행 루프
# ==========================================

async def main():
    client = genai.Client(api_key=API_KEY)
    config = get_config()
    p = pyaudio.PyAudio()
    
    try:
        # 스피커 (출력 - 24kHz)
        output_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)
        
        # 마이크 (입력 - 16kHz)
        input_stream = p.open(format=AUDIO_FORMAT, 
                              channels=CHANNELS, 
                              rate=INPUT_RATE, 
                              input=True, 
                              input_device_index=MIC_DEVICE_INDEX,
                              frames_per_buffer=CHUNK_SIZE)
    except Exception as e:
        print(f"❌ 오디오 장치 오류: {e}")
        print("💡 터미널의 마이크 권한을 확인하거나, MIC_DEVICE_INDEX를 변경해보세요.")
        return

    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 웹캠을 찾을 수 없습니다.")
        return

    print(f"\n🚀 모델({MODEL_ID}) 연결 중...")
    print(f"🍎 환경: macOS / Python {sys.version.split()[0]}")
    print("👁️ 카메라와 마이크가 준비되었습니다.")

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("✅ 연결 성공! AI가 보고 듣고 있습니다. 말씀을 시작하세요.")

            # -------------------------------------------------------
            # [Task 1] 비디오 전송
            # -------------------------------------------------------
            async def send_video_stream():
                print("📡 비디오 전송 시작 (전송 중: .)", end="", flush=True)
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    # OpenCV 창 표시 (종료하려면 'q' 누르기)
                    cv2.imshow('AI Live Vision (Press q to quit)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # 이미지 압축 (JPEG 품질 40)
                    frame_resized = cv2.resize(frame, (640, 480))
                    _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
                    
                    try:
                        # [핵심 수정] send -> send_realtime_input 사용 (정석 방식)
                        # 딕셔너리로 media_chunks 구조를 직접 전달하여 import 오류 회피
                        await session.send_realtime_input(
                            media_chunks=[
                                {
                                    "data": buffer.tobytes(), 
                                    "mime_type": "image/jpeg"
                                }
                            ]
                        )
                        print(".", end="", flush=True)
                    except Exception:
                        pass
                    
                    await asyncio.sleep(0.5) # 초당 2프레임 전송

                cap.release()
                cv2.destroyAllWindows()

            # -------------------------------------------------------
            # [Task 2] 오디오 입력
            # -------------------------------------------------------
            async def send_audio_stream():
                print("\n🎙️ 마이크 전송 시작")
                while True:
                    try:
                        data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        
                        # [핵심 수정] send -> send_realtime_input 사용 (정석 방식)
                        # 1007 에러 방지를 위해 audio/x-linear16 사용
                        await session.send_realtime_input(
                            media_chunks=[
                                {
                                    "data": data, 
                                    "mime_type": "audio/x-linear16"
                                }
                            ]
                        )
                    except Exception:
                        pass

            # -------------------------------------------------------
            # [Task 3] 응답 수신
            # -------------------------------------------------------
            async def receive_response():
                while True:
                    try:
                        async for response in session.receive():
                            if response.server_content:
                                model_turn = response.server_content.model_turn
                                if model_turn:
                                    for part in model_turn.parts:
                                        if part.inline_data:
                                            # 오디오 데이터 재생
                                            output_stream.write(part.inline_data.data)
                    except Exception:
                        break

            # 태스크 병렬 실행
            video_task = asyncio.create_task(send_video_stream()) 
            audio_task = asyncio.create_task(send_audio_stream())
            recv_task = asyncio.create_task(receive_response())

            # 종료 대기
            await video_task

            # 정리
            audio_task.cancel()
            recv_task.cancel()

    except Exception as e:
        print(f"\n❌ 연결 오류: {e}")
    finally:
        if cap.isOpened(): cap.release()
        if input_stream: input_stream.stop_stream(); input_stream.close()
        if output_stream: output_stream.stop_stream(); output_stream.close()
        if p: p.terminate()

if __name__ == "__main__":
    asyncio.run(main())
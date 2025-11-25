import asyncio
import os
import cv2
import pathlib
import sys
import pyaudio
import warnings
import traceback
from dotenv import load_dotenv

# [수정] google.genai에서 types 임포트
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌ google-genai 라이브러리가 설치되지 않았습니다.")
    sys.exit(1)

# [설정] 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 환경 변수
# ==========================================

def load_environment():
    try:
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
    except Exception as e:
        print(f"❌ .env 로드 오류: {e}")

load_environment()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ API 키가 없습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

MODEL_ID = "gemini-2.0-flash-exp"

# [오디오 설정]
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512
MIC_DEVICE_INDEX = None 

# ==========================================
# [함수] 설정 및 페르소나 로드
# ==========================================

def get_config():
    current_dir = pathlib.Path(__file__).parent.absolute()
    persona_path = current_dir / "persona.txt"
    
    system_instruction = ""
    if persona_path.exists():
        try:
            system_instruction = persona_path.read_text(encoding="utf-8")
            print(f"🎭 페르소나 로드됨: {persona_path.name}")
        except Exception:
            pass
    else:
        system_instruction = "너는 도움이 되는 AI 어시스턴트야. 실시간으로 대화해."

    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_instruction
    }

# ==========================================
# [메인] 실행 루프
# ==========================================

async def main():
    try:
        client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
        config = get_config()
        p = pyaudio.PyAudio()
        
        input_stream = None
        output_stream = None

        try:
            output_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)
            input_stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, 
                                  input_device_index=MIC_DEVICE_INDEX, frames_per_buffer=CHUNK_SIZE)
        except Exception as e:
            print(f"❌ 오디오 초기화 오류: {e}")
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("❌ 웹캠을 찾을 수 없습니다.")
            return

        print(f"\n🚀 모델({MODEL_ID}) 연결 중...")

        try:
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                print("✅ 연결 성공! 대화를 시작하세요. (종료: Ctrl+C 또는 화면에서 'q')")

                # -------------------------------------------------------
                # [Task 1] 비디오 전송
                # -------------------------------------------------------
                async def send_video_stream():
                    print("📡 비디오 전송 시작")
                    while True:
                        ret, frame = cap.read()
                        if not ret: break

                        cv2.imshow('Gemini Live Vision', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        frame_resized = cv2.resize(frame, (640, 480))
                        _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                        
                        try:
                            # [수정] 문서를 참고하여 video=types.Blob(...) 형태로 전송
                            # 만약 video 인자가 지원되지 않는다면, input 인자를 사용해야 할 수도 있습니다.
                            await session.send_realtime_input(
                                video=types.Blob(
                                    data=buffer.tobytes(), 
                                    mime_type="image/jpeg"
                                )
                            )
                        except TypeError:
                            # 만약 video 키워드가 없다면 data/mime_type 직접 전송 시도 (구버전 호환)
                            await session.send_realtime_input(
                                data=buffer.tobytes(), 
                                mime_type="image/jpeg"
                            )
                        except Exception as e:
                            # 1007 에러 등
                            print(f"비디오 전송 오류: {e}")
                        
                        await asyncio.sleep(0.4)

                    raise asyncio.CancelledError("Video stream ended")

                # -------------------------------------------------------
                # [Task 2] 오디오 입력
                # -------------------------------------------------------
                async def send_audio_stream():
                    print("🎙️ 마이크 전송 시작")
                    while True:
                        try:
                            data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                            
                            # [수정] 문서를 참고하여 audio=types.Blob(...) 형태로 전송
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=data, 
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                        except Exception as e:
                            print(f"오디오 전송 오류: {e}")
                            break

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
                                                output_stream.write(part.inline_data.data)
                                            if part.text:
                                                print(part.text, end="", flush=True)
                        except Exception as e:
                            print(f"수신 오류: {e}")
                            break

                video_task = asyncio.create_task(send_video_stream()) 
                audio_task = asyncio.create_task(send_audio_stream())
                recv_task = asyncio.create_task(receive_response())

                try:
                    await video_task
                except asyncio.CancelledError:
                    pass
                finally:
                    audio_task.cancel()
                    recv_task.cancel()

        except Exception as e:
            print(f"\n❌ 세션 오류: {e}")
            traceback.print_exc()
        finally:
            if cap.isOpened(): cap.release()
            if input_stream: input_stream.stop_stream(); input_stream.close()
            if output_stream: output_stream.stop_stream(); output_stream.close()
            if p: p.terminate()
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n❌ 메인 오류: {e}")
        input("엔터를 누르면 종료합니다...")

if __name__ == "__main__":
    asyncio.run(main())

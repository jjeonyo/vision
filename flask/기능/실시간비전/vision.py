import asyncio
import os
import cv2
import pathlib
import sys
import time
import pyaudio
import warnings
import traceback
import threading
import queue
import speech_recognition as sr
import audioop
import sqlite3
import textwrap
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


MODEL_ID = "gemini-2.5-flash-native-audio-preview-09-2025"
#MODEL_ID = "gemini-2.5-flash"
#MODEL_ID = "gemini-2.5-flash-preview-09-2025"
#MODEL_ID = "gemini-2.0-flash"
#MODEL_ID = "gemini-2.0-flash-exp"

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
    persona_path = current_dir / "persona_세탁기수리법.txt"
    
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
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": "Aoede"
                }
            }
        },
        "system_instruction": system_instruction
    }

# ==========================================
# [클래스] DB 로그 저장 (SQLite)
# ==========================================
class DatabaseLogger:
    def __init__(self, db_path="chat_history.db"):
        self.db_path = db_path
        self.buffer = []
        self.session_id = None
        self._init_db()
        self._start_session()

    def _init_db(self):
        """DB 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 세션 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_id TEXT
                )
            ''')
            # 메시지 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    sender TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            conn.commit()

    def _start_session(self):
        """새로운 대화 세션 시작"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO sessions (model_id) VALUES (?)', (MODEL_ID,))
            self.session_id = cursor.lastrowid
            conn.commit()
        print(f"💾 DB 세션 시작됨: ID {self.session_id}")

    def append_text(self, text):
        self.buffer.append(text)

    def log_user_message(self, text):
        """사용자 메시지 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO messages (session_id, sender, content) 
                    VALUES (?, ?, ?)
                ''', (self.session_id, 'user', text))
                conn.commit()
        except Exception as e:
            print(f"\n⚠️ DB 저장 실패 (User): {e}")

    def flush_model_turn(self):
        """모델 응답 저장"""
        if not self.buffer: return
        
        full_text = "".join(self.buffer)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO messages (session_id, sender, content) 
                    VALUES (?, ?, ?)
                ''', (self.session_id, 'gemini', full_text))
                conn.commit()
        except Exception as e:
            print(f"\n⚠️ DB 저장 실패 (Gemini): {e}")
            
        self.buffer = []

# ==========================================
# [클래스] STT 처리기 (백그라운드 스레드)
# ==========================================
class SpeechTranscriber:
    def __init__(self, logger, shared_state=None):
        self.logger = logger
        self.shared_state = shared_state
        self.audio_queue = queue.Queue()
        self.running = True
        self.recognizer = sr.Recognizer()
        
        # STT 설정
        self.energy_threshold = 1000  # 음성 감지 임계값 (조절 필요)
        self.pause_threshold = 0.8    # 말 끊김 간주 시간 (초)
        self.sample_rate = 16000
        self.sample_width = 2         # 16-bit = 2 bytes

        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def add_audio(self, data):
        if self.running:
            self.audio_queue.put(data)
            
    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

    def _process_loop(self):
        print("👂 STT 리스너 시작 (한국어)")
        
        audio_buffer = bytearray()
        silence_frames = 0
        has_voice = False
        
        # 1 프레임(청크) 당 시간 계산
        # CHUNK_SIZE(512) / RATE(16000) = 0.032초
        chunk_duration = 512 / 16000
        pause_frame_count = int(self.pause_threshold / chunk_duration)
        
        while self.running:
            try:
                # 큐에서 오디오 청크 가져오기 (타임아웃 1초)
                data = self.audio_queue.get(timeout=1.0)
                
                # 에너지(소리 크기) 계산
                rms = audioop.rms(data, self.sample_width)
                
                if rms > self.energy_threshold:
                    has_voice = True
                    silence_frames = 0
                else:
                    if has_voice:
                        silence_frames += 1
                
                # 버퍼에 데이터 추가
                if has_voice:
                    audio_buffer.extend(data)
                
                # 말이 끝났다고 판단되면 (일정 시간 침묵)
                if has_voice and silence_frames > pause_frame_count:
                    # 인식 수행
                    self._recognize(audio_buffer)
                    
                    # 초기화
                    audio_buffer = bytearray()
                    silence_frames = 0
                    has_voice = False
                    
                # 버퍼가 너무 커지면 (예: 15초 이상) 강제 인식 (메모리 보호)
                if len(audio_buffer) > 16000 * 2 * 15:
                    self._recognize(audio_buffer)
                    audio_buffer = bytearray()
                    silence_frames = 0
                    has_voice = False

            except queue.Empty:
                continue
            except Exception as e:
                print(f"STT 루프 오류: {e}")
                
    def _recognize(self, audio_data):
        if len(audio_data) < 16000 * 2 * 0.5: # 0.5초 미만은 무시
            return
            
        try:
            # Raw PCM 데이터를 AudioData 객체로 변환
            audio_source = sr.AudioData(bytes(audio_data), self.sample_rate, self.sample_width)
            
            # Google Web Speech API 호출 (동기)
            text = self.recognizer.recognize_google(audio_source, language="ko-KR")
            if text.strip():
                print(f"\n[🗣️ User]: {text}")
                self.logger.log_user_message(text)
                
                # shared_state 접근이 어려우므로 로거를 통해 우회하거나 전역 변수 고려
                # 여기서는 간단히 전역 shared_state가 없으므로 생략하거나 
                # SpeechTranscriber에 shared_state 참조를 넘겨주는 것이 좋음
                if hasattr(self, 'shared_state') and self.shared_state:
                     self.shared_state["display_text"] = "..."
                
        except sr.UnknownValueError:
            # 인식 실패 (잡음 등) - 조용히 넘어감
            pass
        except sr.RequestError as e:
            print(f"STT API 오류: {e}")
        except Exception as e:
            print(f"STT 처리 중 오류: {e}")

# ==========================================
# [메인] 실행 루프
# ==========================================

async def main():
    try:
        client = genai.Client(api_key=API_KEY)
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
        # 내 화면용 해상도 (고해상도)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("❌ 웹캠을 찾을 수 없습니다.")
            return

        print(f"\n🚀 모델({MODEL_ID}) 연결 중...")


        # 공유 데이터 컨테이너 (미리 정의하여 STT에 전달)
        shared_state = {
            "latest_frame": None, 
            "running": True,
            "display_text": "안녕하세요!" 
        }

        logger = DatabaseLogger()
        stt_transcriber = SpeechTranscriber(logger, shared_state)

        try:
            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                print("✅ 연결 성공! 대화를 시작하세요. (종료: Ctrl+C 또는 화면에서 'q')")
                
                # -------------------------------------------------------
                # [Task 1] 비디오 처리 (화면 표시 + 전송 분리)
                # -------------------------------------------------------
                
                async def capture_and_display():
                    print("📷 카메라 캡처 시작")
                    while shared_state["running"]:
                        ret, frame = cap.read()
                        if not ret: 
                            print("❌ 카메라 프레임 읽기 실패")
                            break

                        shared_state["latest_frame"] = frame.copy()

                        cv2.imshow('Gemini Live Vision', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            shared_state["running"] = False
                            break
                        
                        # 화면 갱신 (약 30 FPS)
                        await asyncio.sleep(0.03)
                async def send_video_frames():
                    print("📡 비디오 전송 데몬 시작")
                    while shared_state["running"]:
                        if shared_state["latest_frame"] is not None:
                            frame = shared_state["latest_frame"]
                            
                            # 전송 규격 640x480
                            frame_resized = cv2.resize(frame, (640, 480))
                            _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                            
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(
                                        data=buffer.tobytes(), 
                                        mime_type="image/jpeg"
                                    )
                                )
                            except TypeError:
                                await session.send_realtime_input(
                                    data=buffer.tobytes(), 
                                    mime_type="image/jpeg"
                                )
                            except Exception as e:
                                print(f"비디오 전송 오류 (무시됨): {e}")
                        
                        # 전송 주기 (0.4초 = 2.5 FPS)
                        await asyncio.sleep(0.4)
                
                # -------------------------------------------------------
                # [Task 2] 오디오 입력
                # -------------------------------------------------------
                async def send_audio_stream():
                    print("🎙️ 마이크 전송 시작")
                    while True:
                        try:
                            data = await asyncio.to_thread(input_stream.read, CHUNK_SIZE, exception_on_overflow=False)
                            
                            # STT 처리를 위해 데이터 복사본 전달
                            stt_transcriber.add_audio(data)
                            
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
                                                logger.append_text(part.text)

                                    # 턴이 끝났는지 확인 (API 버전에 따라 다를 수 있음)
                                    # turn_complete가 명시적으로 오면 저장
                                    if getattr(response.server_content, "turn_complete", False):
                                        logger.flush_model_turn()
                                        
                        except Exception as e:
                            print(f"수신 오류: {e}")
                            break

                video_display_task = asyncio.create_task(capture_and_display())
                video_sender_task = asyncio.create_task(send_video_frames())
                audio_task = asyncio.create_task(send_audio_stream())
                recv_task = asyncio.create_task(receive_response())

                try:
                    # 카메라 창이 닫힐 때까지 대기
                    await video_display_task
                except asyncio.CancelledError:
                    pass
                finally:
                    video_sender_task.cancel()
                    audio_task.cancel()
                    recv_task.cancel()

        except Exception as e:
            print(f"\n❌ 세션 오류: {e}")
            traceback.print_exc()
        finally:
            stt_transcriber.stop()
            print("\n👋 상담이 종료되었습니다.")

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
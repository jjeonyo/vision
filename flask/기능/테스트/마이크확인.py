import pyaudio

p = pyaudio.PyAudio()

print("\n------------------------------------------------")
print("🎧 오디오 입력 장치(마이크) 목록")
print("------------------------------------------------")

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    # 입력 채널이 0보다 크면 마이크 기능이 있는 장치입니다.
    if dev['maxInputChannels'] > 0:
        print(f"ID [{i}]: {dev['name']}")

print("------------------------------------------------\n")
p.terminate()
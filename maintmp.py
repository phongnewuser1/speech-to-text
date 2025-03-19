import pyaudio
import wave
from faster_whisper import WhisperModel
import threading

def test_microphone(sample_rate=16000, output_file="test_microphone.wav"):
    """
    Ghi lại một đoạn âm thanh từ micro mặc định và lưu vào file.
    Người dùng nhấn Enter để bắt đầu và nhấn Enter lần nữa để dừng.
    """
    # Khởi tạo PyAudio
    p = pyaudio.PyAudio()

    try:
        # Lấy thiết bị đầu vào mặc định
        default_device_index = p.get_default_input_device_info()["index"]

        # Mở luồng âm thanh
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        input_device_index=default_device_index,
                        frames_per_buffer=1024)

        input("🎙 Nhấn Enter để bắt đầu ghi âm...")
        print("🎙 Đang ghi âm... Nhấn Enter lần nữa để dừng.")
        frames = []
        recording = True

        # Hàm ghi âm chạy trong luồng riêng
        def record():
            while recording:
                data = stream.read(1024)
                frames.append(data)

        # Bắt đầu ghi âm trong luồng riêng
        record_thread = threading.Thread(target=record)
        record_thread.start()

        # Chờ người dùng nhấn Enter để dừng
        input("🎙 Nhấn Enter để dừng ghi âm...")
        recording = False
        record_thread.join()

        print(f"✅ Ghi âm hoàn tất. Đang lưu vào file: {output_file}")

        # Lưu âm thanh vào file WAV
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

    except OSError as e:
        print(f"❌ Lỗi khi ghi âm: {e}")

    finally:
        # Đóng luồng âm thanh
        stream.stop_stream()
        stream.close()
        p.terminate()

def process_audio_with_whisper(audio_path, language="vi"):
    """
    Xử lý file âm thanh bằng mô hình Faster Whisper.
    """
    # Khởi tạo mô hình Faster Whisper
    model = WhisperModel("medium", compute_type="float32")

    # Nhận diện văn bản từ file âm thanh
    print(f"🔹 Đang xử lý file: {audio_path}")
    segments, _ = model.transcribe(audio_path, language=language)

    # Hiển thị kết quả
    for segment in segments:
        print(f"🗣 {segment.text}")

def main():
    """
    Chạy chương trình lặp lại liên tục cho đến khi người dùng ra lệnh dừng.
    """
    while True:
        # Ghi âm từ micro
        audio_file = "test_microphone.wav"
        test_microphone(output_file=audio_file)

        # Nhận diện giọng nói từ file ghi âm
        process_audio_with_whisper(audio_file)

        # Hỏi người dùng có muốn dừng chương trình hay không
        stop_command = input("🔄 Nhập 'stop' để dừng chương trình hoặc nhấn Enter để tiếp tục: ").strip().lower()
        if stop_command == 'stop':
            print("👋 Kết thúc chương trình. Tạm biệt!")
            break

if __name__ == "__main__":
    main()
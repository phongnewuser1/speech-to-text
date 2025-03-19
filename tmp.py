import pyaudio
import wave
from faster_whisper import WhisperModel

def interactive_transcription(sample_rate=16000, chunk_size=1024, language="vi"):
    """
    Ghi âm và xử lý giọng nói theo thời gian thực với khả năng bắt đầu/dừng ghi âm thủ công.
    """
    # Khởi tạo PyAudio
    p = pyaudio.PyAudio()

    # Khởi tạo mô hình Whisper
    model = WhisperModel("medium", compute_type="float32")

    try:
        # Lấy thiết bị đầu vào mặc định
        try:
            default_device_index = p.get_default_input_device_info()["index"]
            print(f"🎤 Thiết bị đầu vào mặc định: {p.get_default_input_device_info()['name']}")
        except OSError as e:
            print("❌ Không thể tìm thấy thiết bị đầu vào. Vui lòng kiểm tra micro.")
            return

        print("🎙 Nhấn 'Enter' để bắt đầu ghi âm, 'Enter' lần nữa để dừng và xử lý, hoặc 'q' để thoát.")

        while True:
            command = input("👉 Nhập lệnh ('Enter' để ghi/dừng, 'q' để thoát): ").strip().lower()

            if command == "q":
                print("🛑 Thoát chương trình.")
                break

            elif command == "":
                print("🎙 Đang ghi âm... Nhấn 'Enter' để dừng.")
                audio_buffer = []  # Sử dụng danh sách để lưu các khung âm thanh
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=sample_rate,
                                input=True,
                                input_device_index=default_device_index,
                                frames_per_buffer=chunk_size)

                try:
                    while True:
                        # Đọc một khung âm thanh
                        data = stream.read(chunk_size, exception_on_overflow=False)
                        audio_buffer.append(data)

                        # Kiểm tra nếu người dùng nhấn 'Enter' để dừng
                        if input("👉 Nhấn 'Enter' để dừng ghi âm: ").strip() == "":
                            break
                except KeyboardInterrupt:
                    print("\n🛑 Dừng ghi âm.")
                finally:
                    stream.stop_stream()
                    stream.close()

                print("✅ Dừng ghi âm. Đang xử lý...")

                # Lưu buffer vào file WAV tạm thời
                if audio_buffer:  # Kiểm tra nếu buffer không rỗng
                    temp_audio_file = "temp_audio.wav"  # Đặt tên file WAV tạm thời
                    with wave.open(temp_audio_file, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                        wf.setframerate(sample_rate)
                        wf.writeframes(b"".join(audio_buffer))
                    print(f"📁 File ghi âm đã được lưu tại: {temp_audio_file}")
                else:
                    print("⚠️ Không có dữ liệu âm thanh để lưu.")

                # Xử lý file WAV tạm thời với Whisper
                if audio_buffer:  # Chỉ xử lý nếu buffer không rỗng
                    segments, info = model.transcribe(temp_audio_file, language=language, beam_size=5)

                    # Hiển thị kết quả và ghi log
                    print("🗣 Kết quả nhận diện:")
                    with open("transcription_log.txt", "a", encoding="utf-8") as log_file:
                        for segment in segments:
                            print(f"   {segment.text}")
                            log_file.write(f"{segment.text}\n")
                    print("📄 Kết quả đã được lưu vào 'transcription_log.txt'.")
                else:
                    print("⚠️ Không có dữ liệu âm thanh để xử lý.")

    except OSError as e:
        print(f"❌ Lỗi khi ghi âm: {e}")

    finally:
        # Đóng PyAudio
        p.terminate()

if __name__ == "__main__":
    interactive_transcription()
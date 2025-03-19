import time
import os  # Thêm thư viện os để xóa file
import numpy as np
import soundfile as sf
import librosa
from faster_whisper import WhisperModel
from silero_vad import get_speech_timestamps, read_audio
import torch

def vad_split(audio_path, sample_rate):
    """
    Sử dụng Silero VAD để phát hiện các đoạn có giọng nói.
    """
    # Tải mô hình Silero VAD
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Đọc file âm thanh và chuẩn bị dữ liệu
    audio_data = read_audio(audio_path, sampling_rate=sample_rate)

    # Phát hiện giọng nói
    print("🔹 Đang phát hiện giọng nói...")
    speech_timestamps = get_speech_timestamps(audio_data, model, sampling_rate=sample_rate)

    # Tách các đoạn có giọng nói
    voiced_segments = []
    for ts in speech_timestamps:
        start = ts['start']
        end = ts['end']
        voiced_segments.append(audio_data[start:end])

    return voiced_segments

def process_audio_with_vad(audio_path, sample_rate=16000):
    """
    Xử lý file âm thanh theo nội dung dựa trên Silero VAD.
    """
    # Khởi tạo mô hình Whisper
    model = WhisperModel("medium", compute_type="float32")

    # Sử dụng VAD để chia đoạn âm thanh
    voiced_segments = vad_split(audio_path, sample_rate)

    print("🔹 Bắt đầu nhận diện nội dung...\n")
    for i, segment in enumerate(voiced_segments):
        # Lưu đoạn âm thanh tạm thời
        temp_path = f"temp_chunk_{i}.wav"
        sf.write(temp_path, segment, sample_rate)

        # Nhận diện văn bản từ đoạn âm thanh
        segments, _ = model.transcribe(temp_path, language="vi")

        # Hiển thị kết quả
        for segment in segments:
            print(f"🗣 {segment.text}")

        # Xóa file tạm sau khi xử lý
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n⏹ Hoàn thành xử lý file âm thanh.")

def main():
    # Đường dẫn tới file âm thanh
    audio_path = "audio.wav"  # Thay bằng đường dẫn file của bạn

    # Gọi hàm xử lý âm thanh theo nội dung
    process_audio_with_vad(audio_path, sample_rate=16000)

if __name__ == "__main__":
    main()
import time
import subprocess
import whisper
from faster_whisper import WhisperModel

def convert_ogg_to_mp3(input_ogg, output_mp3):
    start = time.time()
    subprocess.run(["ffmpeg", "-i", input_ogg, "-acodec", "libmp3lame", "-q:a", "4", output_mp3], check=True)
    return time.time() - start

def transcribe_with_whisper(input_audio, model_type="base"):
    model = whisper.load_model(model_type)
    start = time.time()
    result = model.transcribe(input_audio)
    return time.time() - start, result["text"]

def transcribe_with_faster_whisper(input_audio, model_type="base"):
    model = WhisperModel(model_type, device="cpu", compute_type="int8")  # Use "cuda" if GPU
    start = time.time()
    segments, _ = model.transcribe(input_audio)
    text = " ".join([segment.text for segment in segments])
    return time.time() - start, text

if __name__ == "__main__":
    input_ogg = "test.ogg"
    output_mp3 = "test.mp3"

    # Benchmark OGG → MP3
    convert_time = convert_ogg_to_mp3(input_ogg, output_mp3)
    print(f"OGG → MP3: {convert_time:.2f}s")

    # Benchmark Whisper (vanilla)
    whisper_time, text = transcribe_with_whisper(output_mp3)
    print(f"Whisper (vanilla): {whisper_time:.2f}s")

    # Benchmark Faster-Whisper
    faster_time, text = transcribe_with_faster_whisper(output_mp3)
    print(f"Faster-Whisper: {faster_time:.2f}s")

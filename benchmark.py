import time
import subprocess
import whisper
from faster_whisper import WhisperModel

def convert_ogg_to_wav(input_ogg, output_wav, sample_rate=16000):
    """Convert OGG to 16kHz WAV (optimal for Whisper)."""
    start = time.time()
    subprocess.run([
        "ffmpeg", "-i", input_ogg,
        "-ar", str(sample_rate),  # Resample to 16kHz
        "-ac", "1",               # Mono audio
        "-y",                    # Overwrite
        output_wav
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return time.time() - start

def convert_ogg_to_mp3(input_ogg, output_mp3):
    start = time.time()
    subprocess.run(["ffmpeg", "-i", input_ogg, "-acodec", "libmp3lame", "-q:a", "4", output_mp3], check=True)
    return time.time() - start

def transcribe_with_whisper(input_audio, model_type="base"):
    """Vanilla Whisper (supports .ogg, .wav, .mp3 etc.)."""
    model = whisper.load_model(model_type)
    start = time.time()
    result = model.transcribe(input_audio)
    return time.time() - start, result["text"]

def transcribe_with_faster_whisper(input_audio, model_type="base", device="cpu"):
    """Faster-Whisper (prefers .wav)."""
    model = WhisperModel(model_type, device=device, compute_type="int8") # use 'cuda' for fp16
    start = time.time()
    segments, _ = model.transcribe(input_audio, beam_size=5)
    text = " ".join([segment.text for segment in segments])
    return time.time() - start, text

if __name__ == "__main__":
    input_ogg = "files/test.ogg"
    output_wav = "files/test_16k.wav"
    output_mp3 = "files/test.mp3"

    # Benchmark 1.1: Direct OGG → Vanilla Whisper
    print("=== Direct OGG → Vanilla Whisper ===")
    time_vanilla, text_vanilla = transcribe_with_whisper(input_ogg)
    print(f"Vanilla Whisper: {time_vanilla:.2f}s\nText: {text_vanilla[:50]}...")

    # Benchmark 1.2: Direct OGG → Faster Whisper
    print("=== Direct OGG → Faster Whisper ===")
    time_faster, text_faster = transcribe_with_faster_whisper(input_ogg)
    print(f"Faster-Whisper: {time_faster:.2f}s\nText: {text_faster[:50]}...")

    # Verify accuracy (compare texts)
    print("\n=== Accuracy Check ===")
    similarity = 100 * sum(c1 == c2 for c1, c2 in zip(text_vanilla, text_faster)) / max(len(text_vanilla), len(text_faster))
    print(f"Text similarity: {similarity:.1f}%")

    # Benchmark 2.1: OGG → MP3 → Vanilla Whisper 
    convert_time = convert_ogg_to_mp3(input_ogg, output_mp3)
    print(f"OGG → MP3: {convert_time:.2f}s")
    time_vanilla, text_vanilla = transcribe_with_whisper(output_mp3)
    print(f"Vanilla Whisper: {time_vanilla:.2f}s\nText: {text_vanilla[:50]}...")

    # Benchmark 2.2: OGG → MP3 → Faster Whisper 
    convert_time = convert_ogg_to_mp3(input_ogg, output_mp3)
    print(f"OGG → MP3: {convert_time:.2f}s")
    time_faster, text_faster = transcribe_with_faster_whisper(output_mp3)
    print(f"Faster-Whisper: {time_faster:.2f}s\nText: {text_faster[:50]}...")

    # Verify accuracy (compare texts)
    print("\n=== Accuracy Check ===")
    similarity = 100 * sum(c1 == c2 for c1, c2 in zip(text_vanilla, text_faster)) / max(len(text_vanilla), len(text_faster))
    print(f"Text similarity: {similarity:.1f}%")

    # Benchmark 3.1: OGG → 16kHz WAV → Vanilla Whisper
    print("\n=== OGG → 16kHz WAV → Whisper ===")
    convert_time = convert_ogg_to_wav(input_ogg, output_wav)
    time_vanilla, text_vanilla = transcribe_with_whisper(output_wav)
    print(f"Conversion: {convert_time:.2f}s")
    print(f"Vanilla Whisper: {time_vanilla:.2f}s\nText: {text_vanilla[:50]}...")

    # Benchmark 3.2: OGG → 16kHz WAV → Faster Whisper
    print("\n=== OGG → 16kHz WAV → Whisper ===")
    convert_time = convert_ogg_to_wav(input_ogg, output_wav)
    time_faster, text_faster = transcribe_with_faster_whisper(output_wav)
    print(f"Conversion: {convert_time:.2f}s")
    print(f"Faster-Whisper: {time_faster:.2f}s\nText: {text_faster[:50]}...")

    # Verify accuracy (compare texts)
    print("\n=== Accuracy Check ===")
    similarity = 100 * sum(c1 == c2 for c1, c2 in zip(text_vanilla, text_faster)) / max(len(text_vanilla), len(text_faster))
    print(f"Text similarity: {similarity:.1f}%")
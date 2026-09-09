import time
import os
import subprocess
from faster_whisper import WhisperModel

clip_path = '/root/projects/auto-short-generator-v3/output/UxAHTdGR7do_402_443.mp4'
wav_path = '/tmp/test_audit.wav'
subprocess.run(['ffmpeg', '-y', '-i', clip_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', wav_path], check=True, capture_output=True)

print("=== AUDIT FASTER-WHISPER BASE VS LARGE-V3 ON CPU INT8 ===")

print("Testing base on CPU int8...")
t0 = time.time()
m_base = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=4)
t_load_base = time.time() - t0
print(f"Base model load: {t_load_base:.2f}s")

t0 = time.time()
segs, _ = m_base.transcribe(wav_path, language='id', beam_size=5, word_timestamps=True, vad_filter=True)
words_base = [w.word for s in segs for w in (s.words or [])]
dur_base = time.time() - t0
print(f"Base inference: {len(words_base)} words, time: {dur_base:.2f}s")
print(f"Sample base text: {' '.join(words_base[:25])}")

del m_base

print("\nTesting large-v3 on CPU int8...")
large_v3_path = '/mnt/storage/hermes_home_orion/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478'
t0 = time.time()
m_large = WhisperModel(large_v3_path, device='cpu', compute_type='int8', cpu_threads=4)
t_load_large = time.time() - t0
print(f"Large-v3 model load: {t_load_large:.2f}s")

t0 = time.time()
segs, _ = m_large.transcribe(wav_path, language='id', beam_size=5, word_timestamps=True, vad_filter=True)
words_large = [w.word for s in segs for w in (s.words or [])]
dur_large = time.time() - t0
print(f"Large-v3 inference: {len(words_large)} words, time: {dur_large:.2f}s")
print(f"Sample large-v3 text: {' '.join(words_large[:25])}")

if os.path.exists(wav_path):
    os.unlink(wav_path)

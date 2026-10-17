# transcribe_sample.py
#
# Writes voice_ref/sample.txt from voice_ref/sample.wav using faster-whisper,
# so swapping in a new voice sample doesn't mean typing out its transcript.
# Uses large-v3-turbo rather than the "small" model the live
# mic path uses — accuracy matters more than speed here, since CSM clones
# noticeably worse when the reference text doesn't match the audio exactly.
#
#   python scripts/transcribe_sample.py

import os
import sys

from faster_whisper import WhisperModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

AUDIO = os.path.join(config.VOICE_DIR, "sample.wav")
TEXT = os.path.join(config.VOICE_DIR, "sample.txt")

print("Loading Whisper (large-v3-turbo)...")
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

print(f"Transcribing {AUDIO} ...")
segments, info = model.transcribe(AUDIO, language="en", beam_size=5)
text = " ".join(seg.text.strip() for seg in segments).strip()

with open(TEXT, "w", encoding="utf-8") as f:
    f.write(text + "\n")

print(f"\nAudio length: {info.duration:.1f}s")
print(f"Saved to {TEXT}:\n")
print(text)
print("\nGive it a quick listen-along check — fix any misheard words in sample.txt, then restart the voice app.")

# stt.py
#
# Speech-to-text via faster-whisper — CPU, int8, "small" model. This exact
# config (WhisperModel('small', device='cpu', compute_type='int8')) was
# already proven working in earlier testing on this machine. Kept off the
# GPU deliberately: the LLM and Chatterbox TTS are already sharing it, and
# whisper-small on CPU is fast enough for this use case (~0.1x realtime).

import numpy as np
from faster_whisper import WhisperModel
import config

_model = None


def load():
    global _model
    print(f"Loading faster-whisper ({config.WHISPER_MODEL_SIZE})...")
    _model = WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    print("faster-whisper ready.")


def transcribe_array(audio, sample_rate=16000):
    """audio: 1-D float32 numpy array, mono, at `sample_rate` (expected to
    already be 16kHz — see web.py's resampling before this is called).
    Returns the stripped transcript text, or "" if nothing recognizable came
    through (silence, pure noise, a clip too short to contain a word)."""
    if _model is None:
        raise RuntimeError("stt.load() must be called before transcribe_array()")
    if audio.size == 0:
        return ""
    segments, _info = _model.transcribe(audio, beam_size=5, language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()

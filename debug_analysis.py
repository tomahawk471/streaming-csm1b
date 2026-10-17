# debug_analysis.py
#
# Backs the page's debug panel: for every sentence CSM speaks, compare what it
# was ASKED to say with what actually came out — a transcript of the generated
# audio (large-v3-turbo, more accurate than the "small" model the live mic path
# uses, since the whole point is judging the output) plus a pitch contour to
# show intonation against the reference voice's typical pitch.
#
# Runs on its own single CPU worker (see web.py's debug_executor) so it never
# delays the live audio or the mic transcription.

import numpy as np
import librosa
from faster_whisper import WhisperModel

import config

ANALYSIS_SR = 16000
HOP = 160            # 10ms pitch frames at 16kHz
MAX_POINTS = 240     # contour points sent to the browser per sentence

_model = None
ref_f0 = None        # median pitch of voice_ref/sample.wav, for the reference line


def _pitch(audio_16k):
    f0, voiced, _ = librosa.pyin(
        audio_16k, fmin=60, fmax=500, sr=ANALYSIS_SR, frame_length=1024, hop_length=HOP
    )
    f0[~voiced] = np.nan
    return f0


def load(ref_audio_path):
    global _model, ref_f0
    print(f"Loading debug Whisper ({config.DEBUG_WHISPER_MODEL})...")
    _model = WhisperModel(config.DEBUG_WHISPER_MODEL, device="cpu", compute_type="int8")
    ref, _ = librosa.load(ref_audio_path, sr=ANALYSIS_SR, mono=True)
    ref_f0 = float(np.nanmedian(_pitch(ref)))
    print(f"Debug analysis ready (reference pitch {ref_f0:.0f}Hz).")


def analyze(audio, sr, text, stop_reason):
    """audio: the raw float32 samples CSM generated for one sentence."""
    if audio.size == 0:
        return {"text": text, "heard": "", "duration": 0.0, "stop_reason": stop_reason,
                "f0_median": None, "ref_f0": ref_f0, "contour": []}

    audio_16k = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=ANALYSIS_SR)

    segments, _ = _model.transcribe(audio_16k, language="en", beam_size=5)
    heard = " ".join(seg.text.strip() for seg in segments).strip()

    f0 = _pitch(audio_16k)
    step = max(1, int(np.ceil(len(f0) / MAX_POINTS)))
    # Take the median of each bucket rather than every Nth frame, so a single
    # octave-error frame doesn't show up as a spike.
    contour = []
    for i in range(0, len(f0), step):
        bucket = f0[i:i + step]
        contour.append(None if np.all(np.isnan(bucket)) else round(float(np.nanmedian(bucket)), 1))

    voiced = f0[~np.isnan(f0)]
    return {
        "text": text,
        "heard": heard,
        "duration": round(len(audio) / sr, 2),
        "stop_reason": stop_reason,
        "f0_median": round(float(np.median(voiced)), 1) if voiced.size else None,
        "ref_f0": ref_f0,
        "contour": contour,
    }

# speech_rate.py
#
# CSM speaks noticeably faster than the reference it clones — median ~3.4
# words/s vs the reference's 2.6, with about a third of sentences >40% faster —
# and it rushes short questions worst ("How long have you been doing it?" came
# out at 6+ words/s in every test run). That rushing is what makes it stop
# sounding like the reference. CSM has no speed control, so instead: measure
# each generated sentence's speaking rate and slow down only the rushed ones
# with WSOLA, a pitch-preserving time stretch (the voice stays the same; only
# the pace changes).
#
# Rate is measured in letters per second of actual speech (silences trimmed),
# which is steadier than words/s since word lengths vary a lot.

import re

import numpy as np
import librosa

# Speed up to this much over the reference is left alone (people naturally
# speed up a bit, especially on questions); anything faster is slowed to
# TARGET_RATIO, never by more than MAX_SLOWDOWN.
TOLERANCE_RATIO = 1.25
TARGET_RATIO = 1.12
MAX_SLOWDOWN = 0.7   # output is at most 1/0.7 = ~1.43x longer

ref_rate = None  # letters/sec of speech in the reference, set by load()


def _letters(text):
    return len(re.findall(r"[A-Za-z]", text))


def _speech_seconds(audio, sr):
    intervals = librosa.effects.split(audio, top_db=35)
    return sum(end - start for start, end in intervals) / sr


def rate_of(audio, sr, text):
    seconds = _speech_seconds(audio, sr)
    return _letters(text) / seconds if seconds > 0.2 else None


def load(ref_audio_path, ref_text):
    global ref_rate
    audio, sr = librosa.load(ref_audio_path, sr=None, mono=True)
    ref_rate = rate_of(audio, sr, ref_text)
    print(f"Reference speaking rate: {ref_rate:.1f} letters/s")


def wsola(x, speed, sr=24000):
    """Pitch-preserving time stretch. speed < 1 slows down (output is
    len(x)/speed long). Overlap-adds 32ms Hann frames at a fixed output hop,
    picking each input frame within ±10ms of its nominal position to best
    continue the waveform (maximum cross-correlation), which avoids the
    phasiness of a plain phase vocoder on speech."""
    frame = int(0.032 * sr)
    hop_out = frame // 2
    hop_in = hop_out * speed
    tol = int(0.010 * sr)
    window = np.hanning(frame).astype(np.float32)

    padded = np.concatenate([np.zeros(tol, np.float32), x.astype(np.float32), np.zeros(frame + tol, np.float32)])
    n_frames = int((len(x) / speed) / hop_out) + 1
    out = np.zeros(n_frames * hop_out + frame, np.float32)
    norm = np.zeros_like(out)

    prev = tol  # input position (in padded coords) of the previous chosen frame
    for k in range(n_frames):
        nominal = tol + int(round(k * hop_in))
        if nominal + frame + tol > len(padded):
            break
        if k == 0:
            pos = nominal
        else:
            natural = padded[prev + hop_out:prev + hop_out + frame]
            region = padded[nominal - tol:nominal + tol + frame]
            corr = np.correlate(region, natural, mode="valid")
            pos = nominal - tol + int(np.argmax(corr))
        out[k * hop_out:k * hop_out + frame] += padded[pos:pos + frame] * window
        norm[k * hop_out:k * hop_out + frame] += window
        prev = pos

    norm[norm < 1e-3] = 1.0
    out = out / norm
    return out[:int(len(x) / speed)]


def normalize_pace(audio, sr, text):
    """Returns (audio, speed_applied). speed_applied is 1.0 when the sentence
    was already at a natural pace."""
    if ref_rate is None or audio.size == 0:
        return audio, 1.0
    rate = rate_of(audio, sr, text)
    if rate is None or rate <= ref_rate * TOLERANCE_RATIO:
        return audio, 1.0
    speed = max(MAX_SLOWDOWN, (ref_rate * TARGET_RATIO) / rate)
    return wsola(audio, speed, sr), speed

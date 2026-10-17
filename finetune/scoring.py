# scoring.py — judges a checkpoint by what actually matters for a voice:
# does it sound like the speaker, and does it still say the right words?
#
# Validation loss is the wrong yardstick here. On small datasets it bottoms out
# after an epoch or so (the model starts memorising exact training clips) while
# the voice keeps getting closer to the speaker for several more epochs
# (measured: speaker similarity 0.91 -> 0.96 while validation loss rose).
#
# Similarity: WavLM speaker-verification x-vectors (cosine, -1..1).
# Word accuracy: fraction of the intended words Whisper hears.
# Both run on the CPU so they don't compete with training for GPU memory.

import re

import numpy as np
import torch
import librosa

WORD_ACCURACY_FLOOR = 0.9


def _words(text):
    return re.findall(r"[a-z']+", text.lower())


class VoiceScorer:
    def __init__(self, reference_wav):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        from faster_whisper import WhisperModel

        self._fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self._sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        self._whisper = WhisperModel("small", device="cpu", compute_type="int8")
        ref, _ = librosa.load(reference_wav, sr=16000)
        self._ref = self._embed(ref)

    def _embed(self, y16):
        if len(y16) < 16000:  # the x-vector network needs ~1 s
            pad = 16000 - len(y16)
            y16 = np.pad(y16, (pad // 2, pad - pad // 2))
        with torch.no_grad():
            e = self._sv(**self._fe(y16, sampling_rate=16000, return_tensors="pt")).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1)

    def score(self, audio, sr, text):
        """Returns (speaker_similarity, word_accuracy) for one generated clip."""
        y16 = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        sim = float(self._embed(y16) @ self._ref)
        heard = set(_words(" ".join(s.text for s in self._whisper.transcribe(y16, language="en")[0])))
        want = _words(text)
        acc = sum(w in heard for w in want) / max(1, len(want))
        return sim, acc


def combined(sims, accs):
    """One number to rank checkpoints: mean similarity, heavily penalised if
    the words start going wrong (a voice that mumbles is worse than a slightly
    less similar one that speaks clearly)."""
    sim, acc = float(np.mean(sims)), float(np.mean(accs))
    return sim - 2.0 * max(0.0, WORD_ACCURACY_FLOOR - acc), sim, acc

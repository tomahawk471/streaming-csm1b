# vad.py
#
# Voice activity detection via Silero VAD, loaded once from the torch.hub
# cache (no network needed — already present at
# ~/.cache/torch/hub/snakers4_silero-vad_master/). Used two ways with the
# same model: normal turn-taking (detecting when the user starts/stops
# talking so an utterance can be transcribed) and barge-in candidate
# detection (same VAD, tuned for faster reaction, used while Companion is
# speaking — see VAD_MIN_SILENCE_MS vs BARGE_IN_MIN_SILENCE_MS in config.py).
#
# VADIterator holds its own internal recurrent state, so every connection —
# and every mode switch within a connection (listening <-> speaking) — needs
# its own fresh instance; never share one across two purposes at once.

import torch

_model = None
_utils = None

CHUNK_SAMPLES = 512  # Silero VAD's expected window size at 16kHz (verified)


def load():
    global _model, _utils
    print("Loading Silero VAD...")
    _model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        source="github",
        trust_repo=True,
        onnx=False,
    )
    print("Silero VAD ready.")


def new_iterator(threshold, min_silence_duration_ms, speech_pad_ms):
    """A fresh, independent VADIterator instance."""
    _, _, _, VADIterator, _ = _utils
    return VADIterator(
        _model,
        threshold=threshold,
        sampling_rate=16000,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

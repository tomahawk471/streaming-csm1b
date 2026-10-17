# config.py
#
# Every setting can be overridden with an environment variable (shown next to
# it), so nothing here needs editing for a normal install.

import os

# --- LLM: any OpenAI-compatible chat server (llama.cpp, Ollama, LM Studio, vLLM...) ---
API_BASE = os.environ.get("VC_LLM_URL", "http://localhost:8080/v1")
MODEL = os.environ.get("VC_LLM_MODEL", "local-model")  # llama.cpp ignores this; Ollama/LM Studio need the real name
API_KEY = os.environ.get("VC_LLM_API_KEY", "not-needed")

# --- web server ---
HOST = os.environ.get("VC_HOST", "127.0.0.1")
PORT = int(os.environ.get("VC_PORT", "7861"))
# Login is only required when a password is set. Leaving it unset is fine for
# localhost, but web.py refuses to listen on a network address without one.
USERNAME = os.environ.get("VC_USERNAME", "user")
PASSWORD = os.environ.get("VC_PASSWORD", "")

# --- voice ---
# The speech model: Sesame's base CSM-1B, or a folder produced by
# finetune/train.py (models/<name>/best) for a fine-tuned voice.
CSM_MODEL = os.environ.get("VC_CSM_MODEL", "sesame/csm-1b")

# The voice is cloned from voice_ref/sample.wav, with its exact transcript in
# voice_ref/sample.txt. scripts/download_default_voice.py fetches a
# public-domain default; see INSTALL.md for using your own.
VOICE_DIR = os.environ.get("VC_VOICE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_ref"))

# --- speech-to-text / VAD / barge-in ---
WHISPER_MODEL_SIZE = os.environ.get("VC_WHISPER_MODEL", "small")  # CPU int8 — keeps the GPU free for the LLM + CSM
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

MIC_SAMPLE_RATE = 16000             # everything gets resampled to this for VAD/STT
VAD_THRESHOLD = 0.5
VAD_SPEECH_PAD_MS = 150

# Normal turn-taking: wait a bit before deciding the user has finished talking.
VAD_MIN_SILENCE_MS = 500

# Barge-in candidate detection while the companion is speaking.
BARGE_IN_MIN_SILENCE_MS = 250
BARGE_IN_MIN_SPEECH_MS = 200

# Exact whole-utterance match -> ignored, keeps talking.
BACKCHANNEL_PHRASES = {
    "yeah", "yep", "yup", "mhm", "mm-hmm", "uh-huh", "right", "okay", "ok",
    "sure", "i see", "got it", "totally", "true", "fair enough", "makes sense",
    "cool", "nice", "wow",
}

# Substring match anywhere -> immediate hard interrupt.
TRIGGER_WORDS = {
    "stop", "wait", "hold on", "hang on", "actually", "no", "never mind",
}

# --- debug panel (what CSM was asked to say vs. what came out) ---
DEBUG_WHISPER_MODEL = os.environ.get("VC_DEBUG_WHISPER_MODEL", "large-v3-turbo")

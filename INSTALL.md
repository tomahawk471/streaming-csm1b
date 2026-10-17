# Installing Voice Companion

A local, real-time voice-to-voice companion: you talk, it listens, thinks with a local LLM and answers out loud in a cloned voice using [Sesame CSM-1B](https://github.com/SesameAILabs/csm). You can interrupt it mid-sentence like a real conversation.

Everything runs on your own machine. No audio or text leaves it.

## Requirements

| | Minimum | Tested on |
|---|---|---|
| OS | Linux (x86_64) | Ubuntu 24.04 |
| GPU | NVIDIA, **12 GB VRAM** for CSM + an 8B LLM (about 6 GB each); 8 GB if the LLM runs elsewhere | RTX 3090 (24 GB) |
| NVIDIA driver | Supports CUDA 12.1+ | 595 |
| RAM | 16 GB | 32 GB |
| Disk | ~15 GB (Python packages + models) | |
| Python | **3.10** (CSM's code is tested on 3.10) | 3.10 |
| Browser | Chrome or Chromium | Chrome |

You also need a **Hugging Face account**. CSM uses Meta's Llama 3.2 tokenizer, which is a gated download.

## 1. Get access to the gated models

1. Create a free account at <https://huggingface.co>.
2. Open both pages while logged in and click **Agree / Request access** (usually approved instantly):
   - <https://huggingface.co/sesame/csm-1b>
   - <https://huggingface.co/meta-llama/Llama-3.2-1B>
3. Create an access token at <https://huggingface.co/settings/tokens> ("Read" is enough). You'll paste it in step 3.

## 2. Install

```bash
git clone <this repository> voice-companion
cd voice-companion

# A Python 3.10 environment. Either conda...
conda create -n voice-companion python=3.10 -y
conda activate voice-companion
# ...or venv, if python3.10 is installed:
#   python3.10 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

This downloads about 5 GB (PyTorch with CUDA, among others) and takes a few minutes.

## 3. Log in to Hugging Face

```bash
huggingface-cli login     # paste the token from step 1
```

## 4. Get a voice

The companion clones its voice from `voice_ref/sample.wav` and its transcript `voice_ref/sample.txt`. For a ready-made public-domain voice (LJSpeech, an American female speaker), run:

```bash
python scripts/download_default_voice.py
```

To use your own voice instead, see [Using your own voice](#using-your-own-voice).

## 5. Start an LLM server

The companion needs an **OpenAI-compatible chat server** for the "thinking". **See [LLM_SETUP.md](LLM_SETUP.md)** for full step-by-step instructions. The short version, with the recommended model ([Dolphin 3.0 Llama 3.1 8B abliterated](https://huggingface.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF), Q4_K_M: fast, and great at casual conversation):

```bash
# once: build llama.cpp with CUDA (see LLM_SETUP.md) and download the model
mkdir -p ~/models
wget -O ~/models/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf \
  "https://huggingface.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf"

# every time, in its own terminal:
~/llama.cpp/build/bin/llama-server -m ~/models/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf \
  --port 8080 -c 8192 -ngl 99 --jinja --reasoning off
```

No config needed: the defaults point at `http://localhost:8080/v1`. Ollama and LM Studio work too (also covered in LLM_SETUP.md).

## 6. Run it

```bash
python -u web.py
```

**The first start takes about 3-4 minutes.** It downloads CSM (~6 GB) and Whisper the first time, then `torch.compile` benchmarks GPU kernels for *your* graphics card so replies stream fast. Later starts skip the downloads but still compile, so expect about 3 minutes. It's ready when you see:

```
Uvicorn running on http://127.0.0.1:7861
```

Open **<http://localhost:7861>** in Chrome, click the orb, allow the microphone, and start talking.

Use the address `localhost` exactly: browsers only allow microphone access on `localhost` or HTTPS pages, so `127.0.0.1`-style IP addresses and network addresses won't get the mic.

**Use headphones.** Through speakers, the companion hears itself and interrupts its own replies.

## Using it

- **Just talk.** It detects when you've finished speaking.
- **Interrupt** with "stop", "wait", "hold on" or similar. Small acknowledgements like "yeah", "mhm" or "okay" won't interrupt it.
- **Or type** in the box at the bottom.
- **DEBUG** (top right) opens a per-sentence panel. It shows the text CSM was asked to say against what Whisper hears in the audio, with dropped words in red and added words in yellow. It also shows the pitch curve against your reference voice, why the sentence ended, whether it was slowed down for pace, and a replay button.

To stop it, press `Ctrl+C`.

## Using your own voice

1. Record or obtain **15-45 seconds** of one person speaking: calm, natural and conversational, with no music, background noise or other voices. Clean recordings matter more than length. An expressive or noisy clip makes the clone wander in pitch.
2. Save it as `voice_ref/sample.wav` (any sample rate, mono or stereo).
3. Create `voice_ref/sample.txt` with the **exact** words spoken. To have it generated:
   ```bash
   python scripts/transcribe_sample.py
   ```
   Then check the text for misheard words. Mismatches between text and audio noticeably worsen the clone.
4. Restart `web.py`.

For a voice that stays consistent, rather than imitating a sample sentence by sentence, **fine-tune the model on 30+ minutes of that voice**. See [FINETUNING.md](FINETUNING.md).

> **Consent:** only clone a voice you have the right to use, with the speaker's permission. Sesame's terms for CSM prohibit generating speech that mimics real individuals without their explicit consent. Don't use clips of celebrities or anyone else who hasn't agreed.

## Configuration

All settings are environment variables. Set them before running `web.py`.

| Variable | Default | What it does |
|---|---|---|
| `VC_LLM_URL` | `http://localhost:8080/v1` | OpenAI-compatible LLM endpoint |
| `VC_LLM_MODEL` | `local-model` | Model name (llama.cpp ignores it) |
| `VC_LLM_API_KEY` | `not-needed` | For servers that need a key |
| `VC_PORT` | `7861` | Web port |
| `VC_HOST` | `127.0.0.1` | Bind address. Anything else requires `VC_PASSWORD` |
| `VC_USERNAME` / `VC_PASSWORD` | `user` / *(none)* | Browser login, enabled when a password is set |
| `VC_VOICE_DIR` | `./voice_ref` | Where `sample.wav` and `sample.txt` live |
| `VC_CSM_MODEL` | `sesame/csm-1b` | Speech model: the base model, or a fine-tuned `models/<name>/best` folder |
| `VC_WHISPER_MODEL` | `small` | Speech-to-text model for your mic (CPU) |

The personality is in `prompt.py`. Interrupt and acknowledgement words and turn-taking timings are in `config.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `401 Client Error` / `GatedRepoError` on start | You haven't accepted both model licences in step 1, or haven't run `huggingface-cli login`. |
| "Couldn't access the microphone: Cannot read properties of undefined" | You opened it via an IP address. Use `http://localhost:7861`. |
| Page loads but nothing happens after a restart | Hard-refresh (`Ctrl+Shift+R`). Each restart issues a new session token. |
| First reply after start takes ~90 s | Normal only if the server was still warming up. Wait for the `Uvicorn running` line. |
| `CUDA out of memory` | CSM and the LLM don't both fit. Use a smaller or more quantised LLM, or run the LLM on another machine (`VC_LLM_URL`). |
| No reply, error mentioning port 8080 | The LLM server isn't running, or `VC_LLM_URL` is wrong. |
| It answers itself / keeps interrupting itself | Use headphones. |
| Voice drifts or sounds unlike the sample | Use a cleaner, calmer, 20-45 s reference with an exact transcript. For a truly consistent voice, the model needs fine-tuning on that voice. |

## Files it creates

`memory.db` and `working_turns.json` (what it remembers about you), and `audio_history/` (copies of its replies). They're all local and git-ignored. Delete them any time to reset its memory.

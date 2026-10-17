# Connecting the LLM

The companion's voice, ears and memory run in `web.py`, but its *thinking* comes from a separate LLM server. Any server that speaks the OpenAI chat API works. This guide sets up the recommended one from scratch.

## Recommended model

**Dolphin 3.0 Llama 3.1 8B (abliterated), Q4_K_M quantization**

| | |
|---|---|
| Download | [mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF](https://huggingface.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF), file `Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf` |
| Size | 4.9 GB file, about 5.8 GB of VRAM with an 8K context |
| Speed | Replies start in well under a second on an RTX 3090 |
| Licence | Llama 3.1 Community License |

Why this one: it's fast, it's good at natural back-and-forth conversation, and it sticks to a persona and a "keep it short, this is spoken" instruction better than most 8B models. This is the model the companion was developed and tested with.

"Abliterated" means the model's built-in refusals have been removed, so it follows the personality you write in `prompt.py` instead of breaking character to lecture. That also means **`prompt.py` is the only thing shaping its behaviour**. Write it accordingly.

With CSM-1B (~6 GB) plus this model (~6 GB), a 12 GB GPU is enough for everything.

## Option A: llama.cpp (recommended, fastest)

### 1. Build llama.cpp with CUDA

You need the NVIDIA CUDA toolkit (`nvcc --version` should work; on Ubuntu, `sudo apt install nvidia-cuda-toolkit` or NVIDIA's installer), plus `cmake` and `git`.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

This takes 5-15 minutes. When it's done, `./build/bin/llama-server` exists.

### 2. Download the model

```bash
mkdir -p ~/models
wget -O ~/models/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf \
  "https://huggingface.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF/resolve/main/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf"
```

No Hugging Face account needed.

### 3. Start the server

Leave this running in its own terminal:

```bash
cd ~/llama.cpp
./build/bin/llama-server \
  -m ~/models/Dolphin3.0-Llama3.1-8B-abliterated.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 8192 \
  -ngl 99 \
  --jinja \
  --reasoning off
```

| Flag | Why |
|---|---|
| `-c 8192` | Context size: plenty for the prompt, memory and recent history |
| `-ngl 99` | Put the whole model on the GPU (much faster than CPU) |
| `--jinja` | Use the model's own chat template |
| `--reasoning off` | No hidden "thinking" before replies, which keeps latency low. If your llama.cpp is too old to know this flag, just leave it out |

It's ready when it prints `server is listening on http://127.0.0.1:8080`.

### 4. Check it

```bash
curl http://localhost:8080/v1/models
```

You should see the model listed. That's it: the companion's defaults already point at `http://localhost:8080/v1`, so no configuration is needed. Start `web.py` in a second terminal (see [INSTALL.md](INSTALL.md#6-run-it)).

## Option B: Ollama (easiest install)

```bash
# install: https://ollama.com/download
ollama pull hf.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF:Q4_K_M

export VC_LLM_URL=http://localhost:11434/v1
export VC_LLM_MODEL=hf.co/mradermacher/Dolphin3.0-Llama3.1-8B-abliterated-GGUF:Q4_K_M
python web.py
```

Ollama runs in the background on its own. The two `export` lines need to be run in the same terminal as `web.py` (or added to your `~/.bashrc`).

## Option C: LM Studio (graphical)

1. Install [LM Studio](https://lmstudio.ai), search for `Dolphin3.0-Llama3.1-8B-abliterated`, and download the **Q4_K_M** version from mradermacher.
2. Load it, then go to the **Developer** tab and start the server.
3. In the terminal where you run `web.py`:
   ```bash
   export VC_LLM_URL=http://localhost:1234/v1
   export VC_LLM_MODEL=dolphin3.0-llama3.1-8b-abliterated   # the name LM Studio shows
   python web.py
   ```

## Running the LLM on another computer

If the LLM runs on another machine (or a cloud API), point the companion at it:

```bash
export VC_LLM_URL=http://192.168.1.50:8080/v1     # that machine's address
export VC_LLM_API_KEY=...                          # only if the server needs one
```

On that machine, start `llama-server` with `--host 0.0.0.0` so it accepts connections from the network. Only do this on a network you trust: llama-server has no password by default.

## Using a different model

Any instruct/chat model works. Point `-m` (or Ollama/LM Studio) at it. Things to know:
- 7-8B models are the sweet spot: smaller ones follow the "short spoken replies" rule less reliably, and bigger ones are slower to start talking.
- Reasoning ("thinking") models add a long silent pause before every reply. Turn reasoning off, or pick a non-reasoning model.
- Q4_K_M is a good quantization trade-off. Q5/Q6 are slightly better and slightly bigger.

## Troubleshooting

| Problem | Fix |
|---|---|
| No reply; error mentioning port 8080 / "Connection refused" | The LLM server isn't running, or `VC_LLM_URL` is wrong. Check with `curl http://localhost:8080/v1/models`. |
| `CUDA out of memory` when starting either one | The GPU can't fit both. Use a smaller quantization (Q3_K_M), lower `-c` to 4096, or run the LLM on another machine. |
| Replies are slow (several seconds before it speaks) | The model isn't on the GPU: make sure llama.cpp was built with `-DGGML_CUDA=ON` and you passed `-ngl 99`. |
| `error: invalid argument: --reasoning` | Older llama.cpp. Remove that flag, or update llama.cpp (`git pull` and rebuild). |
| Ollama/LM Studio: "model not found" | `VC_LLM_MODEL` must match the model name the server shows exactly. |
| It rambles or writes lists | Keep the "How you talk" rules in `prompt.py`, or lower `REPLY_MAX_TOKENS` in `config.py`. |

from fastapi import FastAPI, Depends, HTTPException, status, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from openai import OpenAI
import uvicorn
import os
import re
import json
import secrets
import asyncio
import queue
import concurrent.futures
import time

import numpy as np
import soundfile as sf
import torch

import csm_tts
import memory_store
import config
import vad
import stt
import debug_analysis
import speech_rate
from config import API_BASE, MODEL, API_KEY
from prompt import BASE_PROMPT
from extraction import extract_async

# Per-process token proving a browser already passed Basic Auth (or that no
# login is configured) — used by the websocket instead of relying on browsers
# re-sending Basic Auth headers on a WS handshake (Safari doesn't reliably).
SESSION_TOKEN = secrets.token_urlsafe(24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REF_AUDIO = os.path.join(config.VOICE_DIR, "sample.wav")
REF_TEXT = os.path.join(config.VOICE_DIR, "sample.txt")
if not (os.path.exists(REF_AUDIO) and os.path.exists(REF_TEXT)):
    raise SystemExit(
        f"No voice found: expected {REF_AUDIO} and {REF_TEXT}.\n"
        "Run `python scripts/download_default_voice.py` for the default voice,\n"
        "or add your own (see INSTALL.md, 'Using your own voice')."
    )
if config.HOST not in ("127.0.0.1", "localhost") and not config.PASSWORD:
    raise SystemExit(
        f"Refusing to listen on {config.HOST} without a password. "
        "Set VC_PASSWORD (and optionally VC_USERNAME), or use VC_HOST=127.0.0.1."
    )

app = FastAPI()
security = HTTPBasic(auto_error=bool(config.PASSWORD))
client = OpenAI(base_url=API_BASE, api_key=API_KEY)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Serve archival audio copies (not used for live playback anymore —
# that goes over the websocket — but kept for history/debugging). Lives
# inside the project folder, not /tmp, so it's actually easy to find in
# a file manager.
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_history")
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# torch.compile(mode="max-autotune")'s CUDA-graph machinery stores its
# graph-tree state in thread-local storage, keyed to whichever thread
# first compiled/warmed up the function. Every generation call MUST run
# on that exact same thread from then on, or it hits
# "AssertionError: assert torch._C._is_key_in_tls(attr_name)" deep in
# torch/_inductor/cudagraph_trees.py. A fresh threading.Thread per
# request (the obvious way to keep blocking GPU work off the asyncio
# event loop) breaks that — so instead there's ONE persistent worker
# thread for the whole process's lifetime, and everything CSM-related
# (including the initial load()/warmup) runs through it.
csm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="csm-gen")

csm_executor.submit(
    csm_tts.load,
    device="cuda",
    # The transcript must match the audio word for word — mismatches
    # noticeably worsen the clone.
    ref_audio_path=REF_AUDIO,
    ref_text=open(REF_TEXT).read().strip(),
    ref_speaker=0,
    model_path=config.CSM_MODEL,
).result()

vad.load()
stt.load()
debug_analysis.load(REF_AUDIO)
speech_rate.load(REF_AUDIO, open(REF_TEXT).read())

# Debug-panel analysis (Whisper + pitch tracking per spoken sentence) is
# CPU-heavy, so it gets its own single worker and queues up behind itself
# instead of competing with the live mic transcription.
debug_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="debug")
reply_counter = 0


def check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not config.PASSWORD:
        return "local"  # no login configured (localhost only, enforced above)
    user_ok = secrets.compare_digest(credentials.username, config.USERNAME)
    pass_ok = secrets.compare_digest(credentials.password, config.PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def check_ws_auth(websocket: WebSocket) -> bool:
    """The page can only render SESSION_TOKEN if it already passed
    check_auth's Basic Auth (see home()), so a valid token in the WS URL's
    query string proves the same thing — and every browser handles that
    identically, unlike Basic Auth headers on a WebSocket upgrade."""
    token = websocket.query_params.get("token", "")
    return secrets.compare_digest(token, SESSION_TOKEN)


def build_system_prompt():
    memory_block, _ = memory_store.build_memory_context()
    prompt = BASE_PROMPT
    if memory_block:
        prompt += "\n\n" + memory_block
    return prompt


def get_reply(user_input):
    """
    Context is: base persona + compact memory block (profile + at most
    one due event/topic suggestion) + a short rolling window of recent
    turns. NOT the full conversation history — this stays a fixed size
    no matter how long the relationship goes on.
    """
    system_prompt = build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_store.get_working_turns())
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()


def get_proactive_reply():
    """For the idle-checkin path: no new user message, the AI is
    initiating based on a due event or recurring topic."""
    system_prompt = build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(memory_store.get_working_turns())
    messages.append({
        "role": "user",
        "content": (
            "[The person has gone quiet for a bit. If the memory block "
            "above suggested something worth raising, bring it up "
            "naturally now in one short message. If there's nothing "
            "worth saying, reply with exactly: NOTHING]"
        ),
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=100,
    )
    return response.choices[0].message.content.strip()




@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: str = Depends(check_auth)):
    return templates.TemplateResponse(request, "index.html", {"ws_token": SESSION_TOKEN})


FRAME_SAMPLES = 1920  # 80ms at 24kHz — same piece size as the live-streamed frames


def split_for_speech(text, min_words=6):
    """Split a reply into the chunks CSM generates one call at a time.

    Whole multi-sentence replies made CSM stall into silence or babble, so
    it's one call per sentence — but a very short sentence ("Keeps life
    interesting!", "Sure!") gives CSM almost nothing to infer a tone from,
    so it picks one at random and the voice audibly jumps, then snaps back
    on the next sentence. Short sentences get merged into a neighbour."""
    sentences = [x.strip() for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x.strip()] or [text]
    merged = []
    for sentence in sentences:
        if merged and len(merged[-1].split()) < min_words:
            merged[-1] += " " + sentence
        else:
            merged.append(sentence)
    if len(merged) > 1 and len(merged[-1].split()) < min_words:
        last = merged.pop()
        merged[-1] += " " + last
    return merged


async def stream_reply_audio(websocket, text, loop, state):
    """
    Runs CSM's frame-by-frame streaming generation on the dedicated
    csm_executor thread (it's blocking GPU work, and — critically — it
    must always be the SAME thread that warmed up the torch.compiled
    functions, see the comment on csm_executor above) and relays each
    decoded chunk to the websocket as soon as it's ready, via a
    thread-safe queue bridging that thread to this async handler.

    state.interrupted (set by handle_audio_chunk on a barge-in) stops
    generation at the next frame and stops relaying audio immediately.
    """
    chunk_queue = queue.Queue()

    sentences = split_for_speech(text)

    global reply_counter
    reply_counter += 1
    reply_id = reply_counter
    sr = csm_tts.sample_rate()

    # When the browser's queued audio will run out, mirroring index.html's
    # playChunk() scheduling — used to decide whether there's time to finish
    # a sentence (and fix its pace) before it's needed.
    queue_end = [0.0]

    def queue_audio(piece):
        now = time.monotonic()
        if queue_end[0] < now:
            queue_end[0] = now + 0.1
        queue_end[0] += len(piece) / sr
        chunk_queue.put(("chunk", piece))

    def run_generation():
        prev_segment = None
        try:
            for index, sentence in enumerate(sentences):
                if state.interrupted:
                    break
                sentence_chunks = []
                # The first sentence streams live, frame by frame, so she
                # starts talking within ~0.1s. Later sentences are generated
                # while earlier ones are still playing (CSM runs ~2.3x
                # realtime), so usually there's time to finish one, check its
                # pace, and slow it down if CSM rushed it (speech_rate.py).
                # But a long sentence after a short one may not be ready in
                # time — buffering it then left a ~1.7s silent gap — so only
                # buffer when the audio already queued covers the estimated
                # generation time; otherwise stream it live, uncorrected.
                est_generate_s = 0.3 + (len(sentence.split()) / 3.4) / 2.3
                live = index == 0 or (queue_end[0] - time.monotonic()) < est_generate_s + 0.4

                def on_chunk(audio, sr, sentence_chunks=sentence_chunks, live=live):
                    sentence_chunks.append(audio)
                    if live:
                        queue_audio(audio)

                filename = csm_tts.speak_streaming(
                    sentence,
                    on_chunk=on_chunk,
                    max_audio_length_ms=min(12_000, 1_500 + 450 * len(sentence.split())),
                    out_dir=AUDIO_DIR,
                    should_stop=lambda: state.interrupted,
                    # Never feed generated audio into the persistent voice
                    # context: across turns, 4 generated sentences outweighed
                    # the real reference 4:1 and any wobble compounded.
                    add_to_context=False,
                    # ...but WITHIN a reply, the previous sentence as extra
                    # context is what keeps the delivery continuous instead of
                    # each sentence starting over with a fresh random tone
                    # (sentence-to-sentence similarity 0.77 -> 0.89, pitch
                    # spread within a reply 28% -> 8% in testing).
                    extra_context=[prev_segment] if prev_segment else None,
                )
                audio = np.concatenate(sentence_chunks) if sentence_chunks else np.array([], dtype=np.float32)
                speed = 1.0
                if not live and not state.interrupted:
                    audio, speed = speech_rate.normalize_pace(audio, sr, sentence)
                    if speed != 1.0:
                        print(f"[pace] slowed sentence {index + 1} to {speed:.2f}x")
                        filename = f"paced_{int(time.time() * 1000)}.wav"
                        sf.write(os.path.join(AUDIO_DIR, filename), audio, sr)
                    for start in range(0, len(audio), FRAME_SAMPLES):
                        queue_audio(audio[start:start + FRAME_SAMPLES])
                # The (pace-corrected) sentence becomes context for the next
                # one, which also nudges CSM toward the calmer pace.
                if csm_tts.last_stop_reason == "natural" and audio.size:
                    prev_segment = csm_tts.Segment(text=sentence, speaker=0, audio=torch.from_numpy(audio.copy()))
                chunk_queue.put(("sentence", (index, sentence, audio, csm_tts.last_stop_reason, filename, speed)))
        except Exception as e:
            print(f"CSM error: {e}")
        finally:
            chunk_queue.put(("done", None))

    csm_executor.submit(run_generation)

    await websocket.send_json({"type": "stream_start", "sample_rate": csm_tts.sample_rate(), "sentences": sentences})

    # CSM generates ~2.5x faster than realtime, so generation finishes well
    # before the browser finishes PLAYING. Sending stream_end (and dropping
    # back to listening mode) at that point let an idle check-in start a new
    # reply over the tail of the current one, and treated speech during the
    # tail as a new turn instead of a barge-in. So track when the browser's
    # playback will actually end — mirroring index.html's playChunk()
    # scheduling (start +0.1s, back-to-back, +0.02s re-anchor after a gap) —
    # and stay in speaking mode until then.
    playback_end = loop.time() + 0.1

    while True:
        kind, payload = await loop.run_in_executor(None, chunk_queue.get)
        if kind == "done":
            break
        if kind == "sentence":
            asyncio.create_task(send_sentence_debug(websocket, loop, reply_id, len(sentences), *payload))
            continue
        if state.interrupted:
            continue  # drain without sending further audio
        await websocket.send_bytes(payload.tobytes())
        now = loop.time()
        if playback_end < now:
            playback_end = now + 0.02
        playback_end += len(payload) / sr

    while not state.interrupted and loop.time() < playback_end:
        await asyncio.sleep(min(0.05, playback_end - loop.time()))

    await websocket.send_json({"type": "stream_end"})


async def send_sentence_debug(websocket, loop, reply_id, total, index, sentence, audio, stop_reason, filename, speed):
    """Analyze one spoken sentence off the hot path and push it to the
    page's debug panel. Failures here must never affect the conversation."""
    try:
        result = await loop.run_in_executor(
            debug_executor, debug_analysis.analyze, audio, csm_tts.sample_rate(), sentence, stop_reason
        )
        result.update({
            "type": "debug_sentence",
            "reply_id": reply_id,
            "index": index,
            "total": total,
            "audio_url": f"/audio/{filename}" if filename else None,
            "speed": speed,
        })
        await websocket.send_json(result)
    except Exception as e:
        print(f"[debug] analysis failed: {e}")


def _classify_barge_in(text):
    """Decide what a candidate interruption while speaking actually means.
    See config.py's BACKCHANNEL_PHRASES / TRIGGER_WORDS for the tuning."""
    if not text:
        return "ignore"
    normalized = text.strip().lower().strip(".,!?")
    if any(trigger in normalized for trigger in config.TRIGGER_WORDS):
        return "trigger"
    if normalized in config.BACKCHANNEL_PHRASES:
        return "backchannel"
    return "substantive"


def _resample_linear(x, orig_sr, target_sr):
    """Cheap linear-interpolation resample — good enough for VAD/ASR input."""
    if orig_sr == target_sr or x.size == 0:
        return x.astype("float32")
    duration = len(x) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_idx = np.linspace(0, len(x) - 1, num=len(x))
    target_idx = np.linspace(0, len(x) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, x).astype("float32")


PREROLL_SAMPLES = int(config.MIC_SAMPLE_RATE * config.VAD_SPEECH_PAD_MS / 1000)


class ConnState:
    """Per-websocket-connection state for the full-duplex voice loop.
    "listening" -> accumulate an utterance, transcribe it on VAD
    end-of-speech; "speaking" -> barge-in candidate detection while the
    reply is streaming out."""

    def __init__(self):
        self.mode = "listening"
        self.native_sample_rate = 48000  # updated by the client's mic_start message
        self.vad_carry = np.array([], dtype=np.float32)
        self.preroll = np.array([], dtype=np.float32)
        self.vad_iter = vad.new_iterator(config.VAD_THRESHOLD, config.VAD_MIN_SILENCE_MS, config.VAD_SPEECH_PAD_MS)
        self.in_speech = False
        self.utterance_buf = []
        self.candidate_buf = []
        self.interrupted = False
        self.busy = False  # a turn is being handled (thinking or speaking)
        self.utterance_queue = asyncio.Queue()

    def reset_vad(self, min_silence_ms):
        self.vad_iter = vad.new_iterator(config.VAD_THRESHOLD, min_silence_ms, config.VAD_SPEECH_PAD_MS)
        self.in_speech = False
        self.utterance_buf = []
        self.candidate_buf = []

    def set_mode(self, mode, min_silence_ms):
        print(f"[state] mode -> {mode}")
        self.mode = mode
        self.reset_vad(min_silence_ms)


async def handle_audio_chunk(state, raw_bytes, websocket, loop):
    chunk = np.frombuffer(raw_bytes, dtype=np.float32)
    chunk = _resample_linear(chunk, state.native_sample_rate, config.MIC_SAMPLE_RATE)
    if chunk.size == 0:
        return

    # Rolling pre-roll so a detected speech-start can be backfilled instead
    # of clipping the first syllable.
    state.preroll = np.concatenate([state.preroll, chunk])[-PREROLL_SAMPLES:]

    buf = np.concatenate([state.vad_carry, chunk])
    n_windows = len(buf) // vad.CHUNK_SAMPLES
    usable = n_windows * vad.CHUNK_SAMPLES
    state.vad_carry = buf[usable:]

    for i in range(n_windows):
        window = buf[i * vad.CHUNK_SAMPLES:(i + 1) * vad.CHUNK_SAMPLES]
        event = state.vad_iter(torch.from_numpy(window))

        if state.mode == "listening":
            if event and "start" in event:
                state.in_speech = True
                state.utterance_buf = [state.preroll.copy()]
            elif state.in_speech:
                state.utterance_buf.append(window)
            if event and "end" in event:
                state.in_speech = False
                audio = np.concatenate(state.utterance_buf) if state.utterance_buf else np.array([], dtype=np.float32)
                state.utterance_buf = []
                await state.utterance_queue.put(("audio", audio))

        elif state.mode == "speaking":
            if event and "start" in event:
                print("[barge-in] candidate speech started")
                state.in_speech = True
                state.candidate_buf = [state.preroll.copy()]
            elif state.in_speech:
                state.candidate_buf.append(window)
            if event and "end" in event:
                state.in_speech = False
                candidate = np.concatenate(state.candidate_buf) if state.candidate_buf else np.array([], dtype=np.float32)
                state.candidate_buf = []
                duration_ms = len(candidate) / config.MIC_SAMPLE_RATE * 1000
                if duration_ms < config.BARGE_IN_MIN_SPEECH_MS:
                    continue  # too short to be a real word — noise/blip, ignore

                text = await loop.run_in_executor(None, stt.transcribe_array, candidate)
                decision = _classify_barge_in(text)
                print(f"[barge-in] heard {duration_ms:.0f}ms -> '{text}' -> {decision}")

                if decision in ("trigger", "substantive"):
                    state.interrupted = True
                    await websocket.send_json({"type": "interrupt"})
                    state.set_mode("listening", config.VAD_MIN_SILENCE_MS)
                    if text:
                        # Already transcribed — feed it straight in as the
                        # next turn instead of making the user repeat it.
                        await state.utterance_queue.put(("text", text))
                # "backchannel" -> do nothing, keep talking


async def _handle_user_utterance(websocket, loop, text):
    reply = await loop.run_in_executor(None, get_reply, text)
    await websocket.send_json({"type": "reply_text", "role": "assistant", "text": reply})

    memory_store.append_working_turn("user", text)
    memory_store.append_working_turn("assistant", reply)
    extract_async(text, reply)
    return reply


async def _handle_idle_checkin(websocket, loop):
    """Returns the proactive line to speak, or None if there's nothing."""
    _, due_event_id = memory_store.build_memory_context()

    reply = await loop.run_in_executor(None, get_proactive_reply)

    # The LLM doesn't always follow "reply with exactly: NOTHING" literally —
    # it sometimes blends an actual sentence with the sentinel word (e.g.
    # "...hang out? NOTHING"), and CSM ends up voicing the word "NOTHING".
    # Treat any occurrence of the word as "nothing to say" instead.
    if re.search(r"\bnothing\b", reply, re.IGNORECASE):
        await websocket.send_json({"type": "no_checkin"})
        return None

    await websocket.send_json({"type": "reply_text", "role": "assistant", "text": reply})
    memory_store.append_working_turn("assistant", reply)
    if due_event_id is not None:
        memory_store.mark_event_mentioned(due_event_id)
    return reply


async def turn_loop(websocket, state, loop):
    while True:
        item = await state.utterance_queue.get()
        if item is None:
            return
        kind, payload = item
        state.busy = True
        try:
            if kind == "audio":
                text = await loop.run_in_executor(None, stt.transcribe_array, payload)
                if not text:
                    continue
                print(f"[heard] {text}")
                await websocket.send_json({"type": "reply_text", "role": "user", "text": text})
                reply = await _handle_user_utterance(websocket, loop, text)

            elif kind == "text":
                reply = await _handle_user_utterance(websocket, loop, payload)

            elif kind == "idle":
                reply = await _handle_idle_checkin(websocket, loop)
                if reply is None:
                    continue

            else:
                continue

            # --- speak it, with barge-in detection active while it talks ---
            state.interrupted = False
            state.set_mode("speaking", config.BARGE_IN_MIN_SILENCE_MS)

            await stream_reply_audio(websocket, reply, loop, state)

            if not state.interrupted:
                state.set_mode("listening", config.VAD_MIN_SILENCE_MS)
            # else: handle_audio_chunk already switched back to listening
            # (possibly with the interrupting utterance already queued).
        finally:
            state.busy = False


async def receiver_loop(websocket, state, loop):
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                await handle_audio_chunk(state, message["bytes"], websocket, loop)

            elif message.get("text") is not None:
                data = json.loads(message["text"])
                mtype = data.get("type")

                if mtype == "mic_start":
                    state.native_sample_rate = data.get("sample_rate", 48000)

                elif mtype == "message":
                    text = data.get("text", "").strip()
                    if text:
                        await state.utterance_queue.put(("text", text))

                elif mtype == "idle_checkin":
                    # Only when genuinely idle — never pile a proactive line
                    # on top of a reply in progress or a half-spoken utterance.
                    if state.busy or state.in_speech or not state.utterance_queue.empty():
                        await websocket.send_json({"type": "no_checkin"})
                    else:
                        await state.utterance_queue.put(("idle", None))

    except WebSocketDisconnect:
        pass
    finally:
        await state.utterance_queue.put(None)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    if not check_ws_auth(websocket):
        await websocket.close(code=1008)  # policy violation
        return

    await websocket.accept()
    loop = asyncio.get_event_loop()
    state = ConnState()

    receiver = asyncio.create_task(receiver_loop(websocket, state, loop))
    turn = asyncio.create_task(turn_loop(websocket, state, loop))

    done, pending = await asyncio.wait(
        [receiver, turn], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()


if __name__ == "__main__":
    print(f"\nOpen http://localhost:{config.PORT} in Chrome (microphone access needs localhost or HTTPS).")
    uvicorn.run(app, host=config.HOST, port=config.PORT)

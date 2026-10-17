# csm_tts.py
#
# Replaces chatterbox.tts_turbo.ChatterboxTurboTTS for this app.
#
# This is file-based, not live-device streaming: the app serves audio to
# a browser over HTTP (the existing <audio src="/audio/..."> pattern), so
# there's no local speaker to pace against — we just want the fastest
# clean generation of a complete .wav per reply. That means we reuse the
# compiled-decoder speedup from csm_live_experimental.py but NOT its
# live-playback ring buffer / streaming-Mimi-context machinery, which
# solves a different problem (chunk-boundary continuity + real-time
# pacing for a local speaker) that doesn't apply here — a single bulk
# Mimi decode() at the end has no chunk boundaries to be discontinuous at.

import os
import sys
import threading
import time
from pathlib import Path

import torch
import torchaudio

# Sesame's CSM reference code (Apache 2.0), vendored in ./csm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "csm"))

from generator import Generator, Segment  # noqa: E402
from models import Model  # noqa: E402
from compiled import (  # noqa: E402
    make_compiled_decoder,
    make_compiled_generate_frame,
)
from moshi.utils.compile import no_cuda_graph  # noqa: E402
from watermarking import CSM_1B_GH_WATERMARK, watermark  # noqa: E402

_generator = None
_ref_segment = None

# _generator._audio_tokenizer (Mimi) is a single shared instance, and its
# streaming() context asserts the module ISN'T already streaming when
# entered. If two speak_streaming() calls ever overlap (a websocket
# reconnect leaving an old generation thread still running while a new
# one starts, two tabs, idle-checkin racing a reply, etc.) the second
# call crashes with "is already streaming!" — and genuinely concurrent
# generation on one GPU/one model instance isn't something to paper
# over anyway. So: serialize all generation through this lock; a
# second caller waits instead of racing into a crash.
_generation_lock = threading.Lock()

# Voice-consistency context: CSM's zero-shot cloning is in-context
# conditioning, not a lookup into any discrete voice table (there isn't
# one — models.py has no speaker/voice embedding component at all).
# Conditioning on a single fixed reference clip, independently, on
# every call is the WEAKEST form of that conditioning and can drift to
# a noticeably different voice. The documented/intended usage (and
# what our own known-good run_csm-compiled.py/csm_live_experimental.py
# do across multi-turn conversations) is to accumulate prior successful
# turns as additional context, so evidence for "this is the speaker"
# compounds instead of resetting every message.
_voice_context = []
_voice_context_lock = threading.Lock()
_VOICE_CONTEXT_MAX_TURNS = 4  # + the original reference, always kept

# Why the most recent speak_streaming() call ended — "natural" (CSM emitted
# its stop frame), "interrupted", "silence" (stalled, see max_silence_ms) or
# "cap" (ran into max_audio_length_ms). Read by web.py's debug panel.
last_stop_reason = None


def sample_rate():
    if _generator is None:
        raise RuntimeError("csm_tts.load() must be called first")
    return _generator.sample_rate


def load(device="cuda", ref_audio_path=None, ref_text="", ref_speaker=0, model_path="sesame/csm-1b"):
    """model_path: a Hugging Face repo id or a local folder — e.g. a voice
    fine-tuned with finetune/train.py (models/<name>/best)."""
    global _generator, _ref_segment

    print(f"Loading CSM model: {model_path}")
    model = Model.from_pretrained(model_path)
    model.to(device=device, dtype=torch.bfloat16)
    _generator = Generator(model)
    model = _generator._model

    print("Compiling CSM decoder...")
    compiled_decoder = make_compiled_decoder(model)
    model.generate_frame = make_compiled_generate_frame(model, compiled_decoder)
    print("Compiled decoder ready.")

    if ref_audio_path:
        audio, sr = torchaudio.load(ref_audio_path)
        audio = audio.mean(dim=0)
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=_generator.sample_rate
        )
        _ref_segment = Segment(text=ref_text, speaker=ref_speaker, audio=audio)

        # The reference used to be added twice "for more anchoring"; testing
        # showed no benefit over once (2026-10-17 experiments), so once.
        _voice_context.append(_ref_segment)

    print("Warming up CSM (first-call max-autotune compile)...")
    context = [_ref_segment] if _ref_segment else []
    with torch.inference_mode():
        # A short warmup text ("Hello.") specializes torch.compile's
        # cudagraph for a SHORT-prompt shape. Real chat replies are much
        # longer, so the first actual message was triggering its own
        # fresh ~90s compile mid-conversation (looks like "not responding"
        # from the browser side) instead of that cost being paid here at
        # startup. Warming up with something reply-length-representative
        # gets that cost out of the way before the server ever accepts
        # a connection. torch's automatic dynamic-shape handling should
        # absorb minor length differences from here without a full
        # re-specialization each time.
        _generator.generate(
            text=(
                "I'm doing well, thanks for asking! How about you? "
                "Anything exciting happening?"
            ),
            speaker=ref_speaker,
            context=context,
            max_audio_length_ms=6_000,
        )
    torch.cuda.synchronize()

    # The warmup above goes through Generator.generate(), but live replies
    # go through speak_streaming() — a different call path (no_cuda_graph +
    # Mimi streaming) that torch.compile specializes separately. Without
    # this, the first real message after every restart sat in a ~95s
    # recompile and looked completely dead from the browser. Two different
    # lengths so the dynamic-shape recompile also happens here, not live.
    print("Warming up streaming path...")
    for warm_text in ("Hey.", "I'm doing well, thanks for asking! How about you, how's your day going?"):
        speak_streaming(warm_text, on_chunk=lambda audio, sr: None, max_audio_length_ms=4_000, add_to_context=False)
    print("CSM ready.")


def speak(
    text,
    out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_history"),
    speaker=0,
    max_audio_length_ms=12_000,
):
    if _generator is None:
        raise RuntimeError("csm_tts.load() must be called before speak()")

    context = [_ref_segment] if _ref_segment else []

    start = time.perf_counter()
    with _generation_lock, torch.inference_mode():
        audio = _generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        ).clamp(-1.0, 1.0)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"CSM TTS: {elapsed:.0f}ms for {audio.shape[-1] / _generator.sample_rate:.2f}s of audio")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = f"reply_{int(time.time() * 1000)}.wav"
    path = Path(out_dir) / filename
    torchaudio.save(str(path), audio.unsqueeze(0).cpu(), _generator.sample_rate)

    return filename


def speak_streaming(
    text,
    on_chunk,
    chunk_frames=1,
    speaker=0,
    max_audio_length_ms=12_000,
    out_dir=None,
    # CSM's own defaults. Lower values (0.6/15) were tried to "steady" the
    # voice and did the opposite: more stalls into silence, lower similarity
    # to the reference, bigger sentence-to-sentence swings.
    temperature=0.9,
    topk=50,
    should_stop=None,
    add_to_context=True,
    max_silence_ms=2_000,
    extra_context=None,
):
    """
    Frame-by-frame generation, decoding each chunk_frames group as soon
    as it's ready and calling on_chunk(np.float32 1-D array, sample_rate)
    immediately — this is what gets low first-audio latency to whatever
    is consuming it (here: a websocket, in csm_live_experimental.py: a
    local speaker).

    Reuses the exact fix validated for the local live-playback pipeline:
    ONE `with generator._audio_tokenizer.streaming(batch_size=1):`
    context for the whole utterance (not re-entered per chunk), wrapped
    in `no_cuda_graph()` because Mimi's streaming transformer conflicts
    with CUDA graph capture across changing streaming state. Without
    this, decoding independent chunks resets Mimi's internal conv state
    every call and produces audible discontinuities at chunk boundaries.

    If out_dir is given, also saves a watermarked archival .wav of the
    full utterance once streaming completes and returns its filename —
    matching Generator.generate()'s normal watermarking behavior.

    should_stop, if given, is checked before every frame (80ms of audio) so
    a barge-in stops generation almost immediately — unlike Chatterbox's
    sentence-level loop, CSM can actually be cut off mid-sentence. An
    interrupted turn never stops naturally, so it's never added to the
    voice context.

    max_silence_ms: CSM sometimes says the first few words, then emits
    near-silent frames until it hits the length cap without ever saying
    the rest ("Understood." + 7.6s of nothing). Once that much consecutive
    silence follows real speech, give up on the utterance. The
    live-streamed chunks themselves are NOT watermarked per-chunk (same
    trade-off already accepted in the local live-playback pipeline —
    watermarking is designed to run over a complete signal, not
    independent small chunks).
    """
    global last_stop_reason
    if _generator is None:
        raise RuntimeError("csm_tts.load() must be called before speak_streaming()")

    # generator.generate() wraps its ENTIRE body in @torch.inference_mode()
    # — including tokenizing the reference-audio context segment, not just
    # the generation loop. Mimi's quantizer holds internal state that gets
    # tagged as inference-mode-only the first time it's touched (during
    # load()'s warmup); calling it again outside inference_mode later
    # throws "Inference tensors cannot be saved for backward". So this
    # whole function needs the same full-body scoping, not just the loop.
    with _generation_lock, torch.inference_mode():
        model = _generator._model

        with _voice_context_lock:
            context = list(_voice_context) + list(extra_context or [])

        model.reset_caches()
        max_generation_len = int(max_audio_length_ms / 80)

        tokens, tokens_mask = [], []
        for segment in context:
            seg_tokens, seg_mask = _generator._tokenize_segment(segment)
            tokens.append(seg_tokens)
            tokens_mask.append(seg_mask)

        gen_tokens, gen_mask = _generator._tokenize_text_segment(text, speaker)
        tokens.append(gen_tokens)
        tokens_mask.append(gen_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(_generator.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(_generator.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if prompt_tokens.size(0) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(_generator.device)

        samples = []
        start = time.perf_counter()
        first_chunk_time = None

        def decode_and_emit(frame_group):
            nonlocal first_chunk_time

            chunk_tokens = torch.stack(frame_group)
            chunk_audio = (
                _generator._audio_tokenizer.decode(chunk_tokens.permute(1, 2, 0))
                .squeeze(0)
                .squeeze(0)
            )
            # Nothing upstream guarantees Mimi's decoded output stays
            # within [-1, 1] — measured as high as ~6% of samples pinned
            # at the clipping ceiling on some generations, which is
            # exactly what "totally distorted, can't understand a word"
            # sounds like. Clamp before this goes anywhere (raw bytes to
            # the browser here, no safety net downstream at all).
            chunk_audio = chunk_audio.clamp(-1.0, 1.0)
            chunk_audio = chunk_audio.detach().float().cpu().numpy().astype("float32")

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
                print(f"CSM STREAM first chunk: {(first_chunk_time - start) * 1000:.1f}ms")

            on_chunk(chunk_audio, _generator.sample_rate)
            return float((chunk_audio ** 2).mean() ** 0.5)

        stopped_naturally = False
        last_stop_reason = "cap"
        silent_frames = 0
        heard_speech = False

        with no_cuda_graph(), _generator._audio_tokenizer.streaming(batch_size=1):
            for _ in range(max_generation_len):
                if should_stop and should_stop():
                    print("CSM STREAM: interrupted")
                    last_stop_reason = "interrupted"
                    break
                sample = model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
                if torch.all(sample == 0):
                    stopped_naturally = True
                    last_stop_reason = "natural"
                    break

                samples.append(sample)

                curr_tokens = torch.cat(
                    [sample, torch.zeros(1, 1, dtype=torch.long, device=_generator.device)],
                    dim=1,
                ).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [
                        torch.ones_like(sample).bool(),
                        torch.zeros(1, 1, dtype=torch.bool, device=_generator.device),
                    ],
                    dim=1,
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1

                if len(samples) % chunk_frames == 0:
                    rms = decode_and_emit(samples[-chunk_frames:])
                    if rms > 0.01:
                        heard_speech = True
                        silent_frames = 0
                    elif heard_speech:
                        silent_frames += chunk_frames
                        if silent_frames * 80 >= max_silence_ms:
                            print("CSM STREAM: stalled into silence — cutting utterance short")
                            last_stop_reason = "silence"
                            break

            remainder = len(samples) % chunk_frames
            if remainder:
                decode_and_emit(samples[-remainder:])

        elapsed = (time.perf_counter() - start) * 1000
        print(f"CSM STREAM total: {elapsed:.0f}ms for {len(samples) * 80 / 1000:.2f}s of audio")

        if not samples:
            return None

        full_audio = (
            _generator._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0))
            .squeeze(0)
            .squeeze(0)
            .clamp(-1.0, 1.0)
        )

        # Feed this turn's own (unwatermarked) audio back in as context for
        # the NEXT call — this is what actually keeps the voice consistent
        # turn over turn, not just a single fixed reference re-used blind.
        # BUT only if generation actually stopped on its own — a turn that
        # hit max_generation_len never emitted its stop token, which is
        # exactly the signature of the babbling/instability seen earlier
        # (see the 11-15s zero-crossing-rate analysis). Feeding a capped,
        # likely-degraded clip back in as a "good" example would poison
        # every future turn's conditioning with a text/audio mismatch
        # (text = the full intended reply, audio = however much it
        # actually got through before being cut off).
        if stopped_naturally and add_to_context:
            with _voice_context_lock:
                _voice_context.append(
                    Segment(text=text, speaker=speaker, audio=full_audio.detach().clone())
                )
                # Always keep the original reference (index 0) + the most
                # recent N turns — unbounded growth would both blow past
                # CSM's max_seq_len and keep re-encoding a huge amount of
                # audio on every single call.
                if len(_voice_context) > _VOICE_CONTEXT_MAX_TURNS + 1:
                    _voice_context[:] = [_voice_context[0]] + _voice_context[-_VOICE_CONTEXT_MAX_TURNS:]
        elif not stopped_naturally:
            print("CSM STREAM: hit max_generation_len without a natural stop — not added to voice context")

        if out_dir is None:
            return None

        watermarked_audio, wm_sr = watermark(
            _generator._watermarker, full_audio, _generator.sample_rate, CSM_1B_GH_WATERMARK
        )
        watermarked_audio = torchaudio.functional.resample(
            watermarked_audio, orig_freq=wm_sr, new_freq=_generator.sample_rate
        )

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        filename = f"reply_{int(time.time() * 1000)}.wav"
        path = Path(out_dir) / filename
        torchaudio.save(str(path), watermarked_audio.unsqueeze(0).cpu(), _generator.sample_rate)

        return filename

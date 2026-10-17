"""Turn a folder of one speaker's recordings into a CSM fine-tuning dataset.

    python finetune/prepare_data.py --input /path/to/recordings --output data/myvoice

Accepts any mix of long recordings and short clips (wav, flac, mp3, ogg,
m4a...). Long audio is split at natural pauses into 1.5-12 s utterances with
Silero VAD; every utterance is transcribed with Whisper and low-confidence
ones are dropped. If a clip has a matching .txt transcript next to it
(e.g. clip_001.wav + clip_001.txt), that's used instead of Whisper.

Output:
    <output>/clips/*.wav           24 kHz mono utterances
    <output>/train.jsonl, val.jsonl {"path", "text", "duration"} per line
    <output>/reference/sample.wav  ~20 s of the cleanest clips, plus
    <output>/reference/sample.txt  their transcript: a ready-made voice_ref
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SAMPLE_RATE, write_jsonl  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm"}
VAD_SR = 16_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Folder of recordings (searched recursively)")
    p.add_argument("--output", required=True, type=Path, help="Where to write the dataset")
    p.add_argument("--whisper-model", default="large-v3-turbo")
    p.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--min-seconds", type=float, default=1.5)
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_vad():
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True, onnx=False)
    return model, utils[0]  # get_speech_timestamps


def segment(audio16, vad, get_ts, min_s, max_s):
    """Group VAD speech regions into utterances: keep joining neighbouring
    regions while the pause between them is short and the result stays under
    max_s. Returns [(start_s, end_s), ...]."""
    regions = get_ts(torch.from_numpy(audio16), vad, sampling_rate=VAD_SR,
                     min_silence_duration_ms=300, speech_pad_ms=120, return_seconds=True)
    out, cur = [], None
    for r in regions:
        s, e = r["start"], r["end"]
        if cur and (s - cur[1]) <= 0.8 and (e - cur[0]) <= max_s:
            cur[1] = e
        else:
            if cur:
                out.append(tuple(cur))
            cur = [s, e]
    if cur:
        out.append(tuple(cur))
    # Over-long single regions (a speaker who never pauses) get cut evenly.
    final = []
    for s, e in out:
        n = int(np.ceil((e - s) / max_s))
        step = (e - s) / n
        final += [(s + i * step, s + (i + 1) * step) for i in range(n)]
    return [(s, e) for s, e in final if e - s >= min_s]


def main():
    args = parse_args()
    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        sys.exit(f"No audio files found in {args.input}")
    clips_dir = args.output / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} audio file(s). Loading Silero VAD and Whisper ({args.whisper_model})...")
    vad, get_ts = load_vad()
    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device=args.whisper_device,
                           compute_type="int8" if args.whisper_device == "cpu" else "float16")

    rows, dropped = [], 0
    for fi, path in enumerate(files):
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        audio16 = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=VAD_SR)
        txt = path.with_suffix(".txt")
        given = txt.read_text(encoding="utf-8").strip() if txt.exists() else None

        if given and len(audio) / SAMPLE_RATE <= args.max_seconds + 3:
            spans = [(0.0, len(audio) / SAMPLE_RATE)]   # short clip with its own transcript
        else:
            given = None
            spans = segment(audio16, vad, get_ts, args.min_seconds, args.max_seconds)

        for s, e in spans:
            clip = audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)]
            if given:
                text, conf = given, 0.0
            else:
                segs = list(whisper.transcribe(audio16[int(s * VAD_SR):int(e * VAD_SR)],
                                               language="en", beam_size=5, condition_on_previous_text=False)[0])
                text = " ".join(x.text.strip() for x in segs).strip()
                # Drop anything Whisper wasn't sure about: a wrong transcript
                # teaches the model wrong text-to-sound pairs.
                if (not segs or len(text.split()) < 2
                        or min(x.avg_logprob for x in segs) < -1.0
                        or max(x.no_speech_prob for x in segs) > 0.5):
                    dropped += 1
                    continue
                conf = float(np.mean([x.avg_logprob for x in segs]))
            peak = np.abs(clip).max()
            if peak < 1e-3:
                dropped += 1
                continue
            name = f"{len(rows):05d}.wav"
            sf.write(clips_dir / name, (clip / peak * 0.9).astype(np.float32), SAMPLE_RATE)
            rows.append({"path": f"clips/{name}", "text": text, "duration": round(len(clip) / SAMPLE_RATE, 2),
                         "source": path.name, "confidence": round(conf, 3)})
        print(f"  [{fi + 1}/{len(files)}] {path.name}: {len(rows)} utterances so far ({dropped} dropped)", flush=True)

    if len(rows) < 20:
        sys.exit(f"Only {len(rows)} usable utterances — far too little to fine-tune on. See FINETUNING.md for data advice.")

    random.Random(args.seed).shuffle(rows)
    n_val = min(50, max(4, int(len(rows) * args.val_fraction)))
    val, train = rows[:n_val], rows[n_val:]
    write_jsonl(args.output / "train.jsonl", train)
    write_jsonl(args.output / "val.jsonl", val)

    # A reference clip for the app: the most confidently-transcribed training
    # utterances, ~20 s worth, joined with short pauses.
    ref_dir = args.output / "reference"
    ref_dir.mkdir(exist_ok=True)
    picked, total = [], 0.0
    for r in sorted(train, key=lambda r: -r["confidence"]):
        if 3 <= r["duration"] <= 10:
            picked.append(r)
            total += r["duration"]
            if total >= 20:
                break
    gap = np.zeros(int(0.3 * SAMPLE_RATE), np.float32)
    ref_audio = np.concatenate([x for r in picked for x in (sf.read(args.output / r["path"], dtype="float32")[0], gap)][:-1])
    sf.write(ref_dir / "sample.wav", ref_audio, SAMPLE_RATE)
    (ref_dir / "sample.txt").write_text(" ".join(r["text"] for r in picked) + "\n", encoding="utf-8")

    hours = sum(r["duration"] for r in rows) / 3600
    print(f"\nDataset ready in {args.output}")
    print(f"  {len(train)} training + {len(val)} validation utterances, {hours * 60:.1f} minutes total ({dropped} dropped)")
    if hours < 0.5:
        print("  ⚠ Under 30 minutes of speech. It will learn something, but 1-3 hours gives a much steadier voice.")
    print(f"  Reference clip for the app: {ref_dir}/sample.wav (+ sample.txt)")
    print(f"\nNext: python finetune/train.py --data {args.output} --name <voice-name>")


if __name__ == "__main__":
    main()

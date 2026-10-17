"""Fine-tune CSM-1B on one speaker's voice.

    python finetune/train.py --data data/myvoice --name myvoice

Full fine-tune (all weights except the text-embedding table) with bf16
autocast and 8-bit AdamW, which fits a 24 GB GPU. Stop the app and your LLM
server first, since they need the GPU memory.

After every epoch it generates test sentences with no reference clip (so
you hear only what the model itself learned), scores them on speaker
similarity and word accuracy, keeps the best-scoring checkpoint, and stops
once the score stops improving:

    models/<name>/best/          config.json + model.safetensors (bf16)
    models/<name>/reference/     reference clip for the app (from prepare_data)
    models/<name>/samples/       epochNN_*.wav
    models/<name>/log.csv        per-epoch losses and scores

Then run the app with it:

    VC_CSM_MODEL=models/<name>/best VC_VOICE_DIR=models/<name>/reference python -u web.py
"""
import argparse
import csv
import dataclasses
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import VoiceScorer, combined  # noqa: E402
from common import (  # noqa: E402
    NUM_CODEBOOKS, SAMPLE_RATE, SPEAKER_ID, REPO_ROOT,
    bucketed_batches, collate, load_audio_tokenizer, load_model_for_training,
    load_text_tokenizer, read_jsonl,
)

SAMPLE_SENTENCES = [
    "Hey, it's good to hear from you. How has your day been so far?",
    "Honestly, I think that's a great idea. Let's give it a try and see what happens!",
    "Wait, are you serious? That's the funniest thing I've heard all week.",
    "I was just thinking about that the other day, actually.",
]
MAX_SEQ_LEN = 2048  # CSM backbone context, in frames (text tokens + audio frames)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="Folder made by prepare_data.py")
    p.add_argument("--name", required=True, help="Name for this voice; output goes to models/<name>")
    p.add_argument("--base", default="sesame/csm-1b", help="Model to start from (HF repo or local folder)")
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "models")
    p.add_argument("--epochs", default="auto", help="Number of epochs, or 'auto' (~1500 optimizer steps, 5-30 epochs)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.002)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--decoder-loss-weight", type=float, default=0.5)
    p.add_argument("--train-text-embeddings", action="store_true",
                   help="Also train the text-embedding table (more memory; only useful for new languages)")
    p.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without the voice score improving")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Tokenization (cached next to the dataset, so re-runs start instantly)
# ---------------------------------------------------------------------------
@torch.no_grad()
def tokenize_split(rows, data_dir, mimi, text_tok, device):
    items = []
    for r in tqdm(rows, desc="tokenizing", leave=False):
        audio, sr = sf.read(data_dir / r["path"], dtype="float32")
        assert sr == SAMPLE_RATE, f"{r['path']} is {sr} Hz; re-run prepare_data.py"
        codes = mimi.encode(torch.from_numpy(audio).to(device).view(1, 1, -1))[0].to(torch.int16).cpu()
        text = torch.tensor(text_tok.encode(f"[{SPEAKER_ID}]{r['text']}"), dtype=torch.int32)
        if codes.size(1) + 1 + len(text) <= MAX_SEQ_LEN:
            items.append({"audio": codes, "text": text})
    return items


def load_or_tokenize(data_dir, mimi, text_tok, device):
    cache = data_dir / "tokens.pt"
    if cache.exists():
        return torch.load(cache)
    toks = {split: tokenize_split(read_jsonl(data_dir / f"{split}.jsonl"), data_dir, mimi, text_tok, device)
            for split in ("train", "val")}
    torch.save(toks, cache)
    return toks


def batches(items, batch_size, shuffle, seed):
    lengths = [it["audio"].size(1) + len(it["text"]) for it in items]
    for idx in bucketed_batches(lengths, batch_size, shuffle, seed):
        yield collate([{"audio": items[i]["audio"].long(), "text": items[i]["text"].long()} for i in idx])


# ---------------------------------------------------------------------------
# Sampling: generate with NO reference context, so you hear what the model
# itself has learned (the app adds the reference on top of that).
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample(model, mimi, text_tok, text, device, seed):
    model.eval()
    torch.manual_seed(seed)
    model.setup_caches(1)
    tokens = torch.tensor(text_tok.encode(f"[{SPEAKER_ID}]{text}"), device=device)
    frame = torch.zeros(len(tokens), NUM_CODEBOOKS + 1, dtype=torch.long, device=device)
    mask = torch.zeros_like(frame, dtype=torch.bool)
    frame[:, -1], mask[:, -1] = tokens, True
    cur_t, cur_m = frame.unsqueeze(0), mask.unsqueeze(0)
    pos = torch.arange(len(tokens), device=device).unsqueeze(0)
    frames = []
    # Full precision here (no autocast): setup_caches() allocates the KV cache
    # in the weights' dtype (fp32 during training), and bf16 keys/values
    # can't be written into it.
    for _ in range(int((2_000 + 450 * len(text.split())) / 80)):
        s = model.generate_frame(cur_t, cur_m, pos, 0.9, 50)
        if torch.all(s == 0):
            break
        frames.append(s)
        cur_t = torch.cat([s, torch.zeros(1, 1, dtype=torch.long, device=device)], dim=1).unsqueeze(1)
        cur_m = torch.cat([torch.ones_like(s).bool(), torch.zeros(1, 1, dtype=torch.bool, device=device)], dim=1).unsqueeze(1)
        pos = pos[:, -1:] + 1
    audio = (mimi.decode(torch.stack(frames).permute(1, 2, 0)).squeeze().float().clamp(-1, 1).cpu().numpy()
             if frames else np.zeros(1, np.float32))
    # Tear the inference caches down again so training runs cache-free.
    model.reset_caches()
    for m in model.modules():
        if hasattr(m, "kv_cache"):
            m.kv_cache = None
        if hasattr(m, "cache_enabled"):
            m.cache_enabled = False
    model.train()
    return audio


def save_checkpoint(model, keys, out_dir):
    """Save exactly the original checkpoint's parameters (not inference-only
    buffers), as bf16, in the format Model.from_pretrained() loads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    save_file({k: sd[k].detach().to(torch.bfloat16).contiguous() for k in keys}, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(json.dumps(dataclasses.asdict(model.config), indent=2))


def main():
    args = parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    out = args.output_root / args.name
    (out / "samples").mkdir(parents=True, exist_ok=True)

    print("Loading tokenizers...")
    mimi = load_audio_tokenizer(device)
    text_tok = load_text_tokenizer()
    toks = load_or_tokenize(args.data, mimi, text_tok, device)
    train_items, val_items = toks["train"], toks["val"]
    minutes = sum(it["audio"].size(1) for it in train_items) / 12.5 / 60
    print(f"{len(train_items)} training utterances ({minutes:.1f} min), {len(val_items)} validation")

    print(f"Loading {args.base}...")
    model = load_model_for_training(args.base, device, args.decoder_loss_weight)
    ckpt_keys = [k for k, _ in model.named_parameters()]
    if not args.train_text_embeddings:
        model.text_embeddings.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    steps_per_epoch = math.ceil(len(train_items) / args.batch_size)
    epochs = (min(30, max(5, round(1500 / steps_per_epoch))) if args.epochs == "auto" else int(args.epochs))
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * args.warmup_fraction))
    print(f"{epochs} epochs x {steps_per_epoch} steps = {total} steps")

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t, m in batches(val_items, args.batch_size, False, 0):
                losses.append(model(t.to(device), m.to(device))[0].item())
        model.train()
        return float(np.mean(losses))

    shutil.copytree(args.data / "reference", out / "reference", dirs_exist_ok=True)
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "train_loss", "train_c0", "train_decoder", "val_loss",
                     "speaker_similarity", "word_accuracy", "score", "seconds"])

    print("Loading the voice scorer (speaker similarity + Whisper)...")
    scorer = VoiceScorer(out / "reference" / "sample.wav")

    def evaluate(epoch):
        sims, accs = [], []
        for i, text in enumerate(SAMPLE_SENTENCES):
            audio = sample(model, mimi, text_tok, text, device, i)
            sf.write(out / "samples" / f"epoch{epoch:02d}_{i + 1}.wav", audio, SAMPLE_RATE)
            sim, acc = scorer.score(audio, SAMPLE_RATE, text)
            sims.append(sim)
            accs.append(acc)
        return combined(sims, accs)

    val = validate()
    base_score, base_sim, base_acc = evaluate(0)
    writer.writerow([0, "", "", "", f"{val:.4f}", f"{base_sim:.3f}", f"{base_acc:.2f}", f"{base_score:.3f}", 0])
    print(f"Before training: speaker similarity {base_sim:.3f}, word accuracy {base_acc:.0%}")
    best, bad_epochs, step = -1.0, 0, 0

    model.train()
    for epoch in range(1, epochs + 1):
        t0, losses = time.time(), []
        pbar = tqdm(list(batches(train_items, args.batch_size, True, args.seed + epoch)), desc=f"epoch {epoch}/{epochs}")
        for t, m in pbar:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, c0, cdec = model(t.to(device), m.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            losses.append((loss.item(), c0.item(), cdec.item()))
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        val = validate()
        tr = np.mean(losses, axis=0)
        score, sim, acc = evaluate(epoch)
        writer.writerow([epoch, f"{tr[0]:.4f}", f"{tr[1]:.4f}", f"{tr[2]:.4f}", f"{val:.4f}",
                         f"{sim:.3f}", f"{acc:.2f}", f"{score:.3f}", f"{time.time() - t0:.0f}"])
        log.flush()

        summary = f"epoch {epoch}: speaker similarity {sim:.3f}, word accuracy {acc:.0%} (train loss {tr[0]:.3f}, val loss {val:.3f})"
        if score > best:
            best, bad_epochs = score, 0
            save_checkpoint(model, ckpt_keys, out / "best")
            print(f"{summary}  ✓ best so far, saved")
        else:
            bad_epochs += 1
            print(f"{summary}  ({bad_epochs}/{args.patience} without improvement)")
            if bad_epochs >= args.patience:
                print("The voice stopped improving; stopping early.")
                break

    log.close()
    if not (out / "best").exists():
        save_checkpoint(model, ckpt_keys, out / "best")
    # Sanity check: the saved checkpoint must load cleanly into a fresh Model.
    from models import Model, ModelArgs
    fresh = Model(ModelArgs(**json.loads((out / "best" / "config.json").read_text())))
    fresh.load_state_dict(load_file(str(out / "best" / "model.safetensors")), strict=True)

    print(f"\nDone. Voice saved to {out} (best score {best:.3f}; before training {base_score:.3f}).")
    print(f"Listen to {out}/samples/ to hear how it developed, then run the app with:")
    print(f"  VC_CSM_MODEL={out}/best VC_VOICE_DIR={out}/reference python -u web.py")


if __name__ == "__main__":
    main()

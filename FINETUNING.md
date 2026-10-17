# Fine-tuning a voice

Cloning a voice from a 20-second sample (the default) gets the general tone right, but CSM-1B has to guess the voice fresh for every sentence, so pitch, pace and delivery wander. **Fine-tuning teaches the model the voice itself**, so it sounds like the speaker even with no reference clip, and the app's reference clip then only has to fine-tune the delivery.

In testing with 42 minutes of a single speaker, the base model with no reference sounded like a random person (speaker similarity 0.68). After **4 epochs (about 4 minutes in total)** on an RTX 3090, it scored **0.96**, higher than zero-shot cloning managed even *with* a reference clip (~0.91). Every test sentence stayed word-perfect.

> **Consent first.** Only fine-tune on recordings of someone who has agreed to have their voice cloned, or on data licensed for it. Sesame's terms for CSM prohibit mimicking real individuals without their explicit consent. This applies to celebrities, public figures and anyone else who hasn't agreed.

## What you need

| | |
|---|---|
| GPU | NVIDIA, **24 GB VRAM** (RTX 3090 / 4090 / A5000 or better). Training uses about 23 GB, so **stop the app and your LLM server first**. |
| Recordings | **30 minutes minimum, 1-3 hours ideal**, of one person. |
| Time | Data prep: about 5 min per 45 min of audio (CPU). Training: a few minutes per hour of audio on a 3090. |
| Software | The same environment as the app. Nothing extra to install. |

### What makes good training audio

- **One speaker only.** Remove other voices, or don't include those files.
- **Clean:** a decent microphone, a quiet room, no music, no heavy echo.
- **Natural and conversational, in the style you want it to talk in.** The model copies *delivery* as well as timbre: read-aloud audiobook audio makes a reading-aloud companion, and podcast-style chat makes a chatty one.
- **Any format and length:** long recordings are fine (they're split automatically), and so are short clips.

## Step 1: prepare the data

Put the recordings in a folder (sub-folders are fine), then:

```bash
python finetune/prepare_data.py --input /path/to/recordings --output data/myvoice
```

This:
1. Splits long recordings at natural pauses into 1.5-12 second utterances (Silero VAD).
2. Transcribes each one with Whisper (`large-v3-turbo`), **dropping any it isn't confident about**, because a wrong transcript teaches wrong text-to-sound pairs. If you already have exact transcripts, put them next to each clip as a `.txt` with the same name (`clip01.wav` + `clip01.txt`) and they'll be used instead.
3. Splits the utterances into training and validation sets (`train.jsonl`, `val.jsonl`).
4. Builds `data/myvoice/reference/`, a ~20-second reference clip and transcript from the clearest utterances, ready for the app.

Check the summary it prints: how many minutes survived, and how many were dropped. A lot of dropped utterances usually means noisy audio or more than one speaker.

**Tip:** skim a few lines of `train.jsonl` against the audio. Whisper occasionally gets names wrong, and you can fix them by hand.

## Step 2: train

```bash
# free the GPU first: stop web.py (Ctrl+C) and your LLM server
python finetune/train.py --data data/myvoice --name myvoice
```

What happens:
- The first run tokenizes the dataset (cached in `data/myvoice/tokens.pt`, so re-runs start instantly) and downloads Sesame's CSM-1B and a small speaker-verification model if they aren't already cached.
- It fine-tunes all weights except the text-embedding table, using bf16 autocast and 8-bit AdamW.
- **After every epoch** it generates four test sentences with *no reference clip*, so you hear only what the model has learned. It scores them on **speaker similarity** to the reference and **word accuracy** (Whisper), and keeps the best-scoring epoch.
- It stops automatically when the voice stops improving (3 epochs without a better score).

```
Before training: speaker similarity 0.682, word accuracy 100%
epoch 1: speaker similarity 0.911, word accuracy 100% (train loss 5.362, val loss 5.257)  ✓ best so far, saved
epoch 2: speaker similarity 0.943, word accuracy 100% (train loss 5.008, val loss 5.374)  ✓ best so far, saved
epoch 3: speaker similarity 0.956, word accuracy 100% (train loss 4.662, val loss 5.509)  ✓ best so far, saved
epoch 4: speaker similarity 0.958, word accuracy 100% (train loss 4.455, val loss 5.514)  ✓ best so far, saved
epoch 5: speaker similarity 0.932, word accuracy 100% ...  (1/3 without improvement)
...
The voice stopped improving; stopping early.
```

Output in `models/myvoice/`:

| | |
|---|---|
| `best/` | The fine-tuned model (`config.json` + `model.safetensors`, bf16, about 3.1 GB) |
| `reference/` | Reference clip and transcript for the app |
| `samples/epochNN_*.wav` | Test sentences after each epoch. **Listen to these.** |
| `log.csv` | Per-epoch training/validation loss, similarity, word accuracy and score |

> **Why not just pick the lowest validation loss?** On small datasets, validation loss bottoms out after about one epoch, because the model starts memorising the exact training clips. The *voice*, though, keeps getting closer to the speaker for several more epochs while the words stay correct. So checkpoints are chosen on what you actually hear. The losses are still logged.

### Options

| Flag | Default | |
|---|---|---|
| `--epochs` | `auto` | About 1500 optimizer steps, between 5 and 30 epochs; early stopping usually ends it sooner |
| `--lr` | `2e-5` | Learning rate. Lower (1e-5) if samples get garbled; there's rarely a reason to go higher |
| `--batch-size` | `8` | Lower to 4 if you run out of GPU memory |
| `--patience` | `3` | Epochs without improvement before stopping |
| `--base` | `sesame/csm-1b` | Start from another checkpoint, e.g. continue a previous fine-tune |
| `--train-text-embeddings` | off | Also train the text-embedding table. Only useful for teaching a new language; costs about 2 GB more memory |

## Step 3: use it

```bash
VC_CSM_MODEL=models/myvoice/best VC_VOICE_DIR=models/myvoice/reference python -u web.py
```

Start your LLM server again first. Everything else works as before: streaming, interruptions, pace correction and the debug panel. The debug panel's pitch line is a quick way to see whether the voice is holding steady.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Make sure the app and LLM server are stopped (`nvidia-smi` should show almost nothing in use). Then try `--batch-size 4`. |
| "Only N usable utterances" | Not enough clean speech. Check that the folder holds one speaker's speech, not music or multi-person audio. |
| Many utterances dropped | Noisy audio or overlapping speakers. Clean recordings matter more than quantity. |
| Samples get garbled or mumbled in later epochs | The run already keeps the best epoch. If even early epochs are rough, lower `--lr` to `1e-5` or add more data. |
| Voice similar but delivery wrong (too "read aloud") | The model copies the training data's delivery. Use more conversational recordings. |
| `GatedRepoError` | Accept the CSM and Llama-3.2 licences and run `huggingface-cli login` (see INSTALL.md). |

## Credits

The training objective follows Sesame's CSM paper: codebook-0 loss on every frame, and the acoustic decoder trained on a random 1/16 of frames. The training forward pass and batching are adapted from [knottwill/sesame-finetune](https://github.com/knottwill/sesame-finetune) (MIT licence, see `finetune/LICENSE-sesame-finetune`).

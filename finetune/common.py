# common.py — shared pieces of the fine-tuning pipeline.
#
# The training forward pass, batch collation and length-bucketed batching are
# adapted from knottwill/sesame-finetune (MIT licence, Copyright 2025
# Speechmatics / CANTAB RESEARCH LIMITED / William Knottenbelt — see
# finetune/LICENSE-sesame-finetune), which implements the training objective
# described in Sesame's CSM paper (Sesame didn't release training code).

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "csm"))

from generator import load_llama3_tokenizer  # noqa: E402
from models import Model, _create_causal_mask  # noqa: E402

SAMPLE_RATE = 24_000
NUM_CODEBOOKS = 32
# The app speaks as speaker 0, so the fine-tuned voice is trained as speaker 0.
SPEAKER_ID = 0


def load_audio_tokenizer(device):
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    mimi = loaders.get_mimi(hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME), device=device)
    mimi.set_num_codebooks(NUM_CODEBOOKS)
    return mimi


def load_text_tokenizer():
    return load_llama3_tokenizer()


# ---------------------------------------------------------------------------
# Training forward pass (Sesame's released Model only has inference code).
#
# Loss = codebook-0 cross-entropy from the backbone on EVERY audio frame, plus
# cross-entropy for codebooks 1..31 from the decoder on a random 1/16 of
# frames ("compute amortization", as in the CSM paper), weighted 50/50.
# ---------------------------------------------------------------------------
def training_forward(self, tokens, tokens_mask):
    """tokens / tokens_mask: (batch, seq_len, NUM_CODEBOOKS + 1); the last
    column holds text tokens, the first NUM_CODEBOOKS hold audio codes."""
    dtype = next(self.parameters()).dtype
    bsz, seq_len, _ = tokens.size()
    device = tokens.device

    embeds = self._embed_tokens(tokens)

    audio_mask = tokens_mask[:, :, 0]                      # positions holding an audio frame
    target_tokens = tokens[audio_mask][:, :-1]             # [n_audio, NUM_CODEBOOKS]
    c_embeds = embeds[:, :, :-1, :][audio_mask]            # [n_audio, NUM_CODEBOOKS, dim]

    h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)

    # causal + padding mask for the backbone
    padding_mask = tokens_mask[:, :, 0] | tokens_mask[:, :, -1]
    attn = _create_causal_mask(seq_len, device).unsqueeze(0) * (padding_mask.unsqueeze(-1) * padding_mask.unsqueeze(1))
    attn = attn | torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(bsz, -1, -1)
    input_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
    h = self.backbone(h, input_pos=input_pos, mask=attn).to(dtype=dtype)

    # the hidden state at position t predicts the audio frame at t+1
    audio_h = h[torch.roll(audio_mask, -1, 1)]
    c0_loss = F.cross_entropy(self.codebook0_head(audio_h), target_tokens[:, 0])

    idx = torch.randperm(c_embeds.size(0), device=device)[: max(1, c_embeds.size(0) // 16)]
    dec_in = torch.cat([audio_h[idx].unsqueeze(1), c_embeds[idx][:, :-1, :]], dim=1)  # [n, NUM_CODEBOOKS, dim]
    n, n_cb, _ = dec_in.size()
    c_pos = torch.arange(n_cb, device=device).unsqueeze(0).expand(n, n_cb)
    dec_mask = _create_causal_mask(n_cb, device).expand(n, -1, -1)
    dec_h = self.decoder(self.projection(dec_in), input_pos=c_pos, mask=dec_mask).to(dtype=dtype)
    c_logits = torch.einsum("bsd,sdv->bsv", dec_h[:, 1:, :], self.audio_head)
    c_loss = F.cross_entropy(c_logits.reshape(-1, c_logits.size(-1)), target_tokens[idx][:, 1:].reshape(-1))

    w = self.decoder_loss_weight
    return 2 * ((1 - w) * c0_loss + w * c_loss), c0_loss.detach(), c_loss.detach()


def load_model_for_training(model_name_or_path, device, decoder_loss_weight=0.5):
    model = Model.from_pretrained(model_name_or_path)
    model.decoder_loss_weight = decoder_loss_weight
    model.forward = types.MethodType(training_forward, model)
    return model.to(device=device)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def collate(batch):
    """batch: list of {"audio": LongTensor[NUM_CODEBOOKS, frames], "text": LongTensor[n]}.
    Lays out text tokens then audio frames (+ an all-zero EOS frame, which is
    how CSM learns to stop), padded to the longest item."""
    tokens, masks = [], []
    for item in batch:
        audio = torch.cat([item["audio"], torch.zeros(NUM_CODEBOOKS, 1, dtype=torch.long)], dim=1)
        a_tok = torch.zeros(audio.size(1), NUM_CODEBOOKS + 1, dtype=torch.long)
        a_tok[:, :-1] = audio.transpose(0, 1)
        a_mask = torch.zeros(audio.size(1), NUM_CODEBOOKS + 1, dtype=torch.bool)
        a_mask[:, :-1] = True
        text = item["text"]
        t_tok = torch.zeros(len(text), NUM_CODEBOOKS + 1, dtype=torch.long)
        t_tok[:, -1] = text
        t_mask = torch.zeros(len(text), NUM_CODEBOOKS + 1, dtype=torch.bool)
        t_mask[:, -1] = True
        tokens.append(torch.cat([t_tok, a_tok]))
        masks.append(torch.cat([t_mask, a_mask]))
    return pad_sequence(tokens, batch_first=True), pad_sequence(masks, batch_first=True, padding_value=False)


def bucketed_batches(lengths, batch_size, shuffle, seed):
    """Group similar-length items so batches carry little padding."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    bins = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(bins)
    return bins


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

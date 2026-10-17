# compiled.py
#
# A torch.compile'd replacement for models.Model.generate_frame. Same maths as
# Sesame's original (backbone -> codebook 0 -> 31-codebook decoder loop, decoder
# caches reset every frame), but the decoder loop is compiled with
# mode="max-autotune", roughly halving per-frame time.
#
# Thread note: max-autotune's CUDA-graph state is thread-local, so every call
# must happen on the thread that first ran it (see web.py's csm_executor).

import torch

from models import _index_causal_mask, sample_topk


def make_compiled_decoder(model):
    decoder = model.decoder
    projection = model.projection
    audio_head = model.audio_head
    audio_embeddings = model.audio_embeddings

    audio_vocab_size = model.config.audio_vocab_size
    num_codebooks = model.config.audio_num_codebooks

    decoder_mask = torch.tril(
        torch.ones(num_codebooks, num_codebooks, dtype=torch.bool, device=next(model.parameters()).device)
    )

    @torch.compile(mode="max-autotune")
    def compiled_decoder(curr_h, curr_pos, temperature, topk):
        curr_sample = None
        for i in range(1, num_codebooks):
            curr_decoder_mask = decoder_mask[curr_pos, :]
            decoder_h = decoder(projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask)
            decoder_h = decoder_h.to(dtype=audio_head[i - 1].dtype)
            ci_logits = torch.mm(decoder_h[:, -1, :], audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            curr_h = audio_embeddings(ci_sample + i * audio_vocab_size)
            curr_sample = ci_sample if curr_sample is None else torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1
        return curr_sample

    return compiled_decoder


def make_compiled_generate_frame(model, compiled_decoder):
    def compiled_generate_frame(tokens, tokens_mask, input_pos, temperature, topk):
        dtype = next(model.parameters()).dtype

        # backbone
        curr_backbone_mask = _index_causal_mask(model.backbone_causal_mask, input_pos)
        embeds = model._embed_tokens(tokens)
        h = (embeds * tokens_mask.unsqueeze(-1)).sum(dim=2)
        h = model.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        # codebook 0
        last_h = h[:, -1, :]
        c0_sample = sample_topk(model.codebook0_head(last_h), topk, temperature)
        c0_embed = model._embed_audio(0, c0_sample)

        # remaining codebooks via the compiled decoder
        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)
        model.decoder.reset_caches()
        decoder_sample = compiled_decoder(curr_h, curr_pos, temperature, topk)

        return torch.cat([c0_sample, decoder_sample], dim=1)

    return compiled_generate_frame

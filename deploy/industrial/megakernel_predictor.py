"""
Megakernel predictor wrapper for the 1.7B model.

The 1.7B has:
- Talker: hidden_size=2048
- Predictor: hidden_size=1024
- small_to_mtp_projection: 2048→1024 (projects talker hidden to predictor space)
- codec_embedding in predictor: [2048, 2048] (talker space, needs projection)

The megakernel works in predictor space (1024). So we need to:
1. Project talker_hidden 2048→1024 via small_to_mtp_projection
2. Project first_codebook_embed 2048→1024 via the same projection
3. Use predictor-internal embeddings (already 1024) for subsequent groups
"""
import torch
import torch.nn.functional as F


class MegakernelPredictor17B:
    """Wraps the megakernel CodePredictorKernel for the 1.7B model."""

    def __init__(self, mk_predictor, projection_weight, projection_bias=None,
                 predictor_codec_embeds=None):
        """
        Args:
            mk_predictor: CodePredictorKernel instance (works in 1024-dim space)
            projection_weight: [1024, 2048] small_to_mtp_projection weight
            projection_bias: [1024] optional bias
            predictor_codec_embeds: list of [vocab, 1024] embedding tables for groups 1-15
                                    (already in predictor space, no projection needed)
        """
        self.mk = mk_predictor
        self.proj_w = projection_weight.bfloat16()
        self.proj_b = projection_bias.bfloat16() if projection_bias is not None else None
        self.pred_embeds = predictor_codec_embeds  # Already 1024-dim

    def _project(self, x_2048):
        """Project from talker space (2048) to predictor space (1024)."""
        return F.linear(x_2048.float(), self.proj_w.float(),
                        self.proj_b.float() if self.proj_b is not None else None).bfloat16()

    @torch.no_grad()
    def predict(self, talker_hidden_2048, first_codebook_token,
                talker_embed_weight_2048, do_sample=True, temperature=0.9, top_k=50):
        """
        Predict 16 codebook tokens using the megakernel.

        Args:
            talker_hidden_2048: [2048] hidden state from talker (will be projected to 1024)
            first_codebook_token: int, first codebook token from talker sampling
            talker_embed_weight_2048: [3072, 2048] talker's codec embedding (for first token)
            do_sample, temperature, top_k: generation params

        Returns:
            [16] int64 tensor of codec tokens
        """
        self.mk.reset()

        # 1. Project talker hidden to predictor space
        h_1024 = self._project(talker_hidden_2048)
        self.mk._step_with_embed(h_1024)

        # 2. Embed first codebook token in talker space, then project
        token_buf = torch.tensor([first_codebook_token], dtype=torch.long,
                                  device=talker_embed_weight_2048.device)
        first_embed_2048 = F.embedding(token_buf, talker_embed_weight_2048).squeeze(0)
        first_embed_1024 = self._project(first_embed_2048)
        self.mk._step_with_embed(first_embed_1024)

        # 3. Predict groups 1-15 (these work in 1024-dim predictor space)
        predicted = []
        for group in range(self.mk.num_groups):
            hidden_bf16 = self.mk._norm_out.to(torch.bfloat16).unsqueeze(0)
            logits = F.linear(hidden_bf16, self.mk.lm_heads[group]).squeeze(0)

            if do_sample and temperature > 0:
                logits_f = logits.float() / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits_f, min(top_k, logits_f.size(-1)))
                    logits_f[logits_f < topk_vals[-1]] = float('-inf')
                probs = F.softmax(logits_f, dim=-1)
                tok = torch.multinomial(probs, 1)
            else:
                tok = logits.argmax(keepdim=True).long()

            predicted.append(tok)

            if group < self.mk.num_groups - 1:
                # Predictor codec embeddings are in talker space (2048), project to predictor (1024)
                embed_2048 = F.embedding(tok, self.mk.codec_embeddings[group]).squeeze(0)
                embed_1024 = self._project(embed_2048)
                self.mk._step_with_embed(embed_1024)

        first_tensor = torch.tensor([first_codebook_token], dtype=torch.long,
                                     device=talker_embed_weight_2048.device)
        return torch.cat([first_tensor] + predicted)

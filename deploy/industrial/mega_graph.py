"""
Mega CUDA graph: fuses sampling + predictor (15 steps) + first talker decode
into a single graph replay. Eliminates 3 Python↔GPU round-trips (~1ms).

Current flow (3 kernel launches + Python overhead):
  Python → sample_logits (GPU) → Python → predictor_graph.run (GPU) →
  Python → talker_graph.run (GPU) → Python → yield

Fused flow (1 graph replay):
  Python → mega_graph.run(logits, tth_token) → (codec_tokens, next_logits)

Saves: ~1ms per first token (from ~2.6ms to ~1.5ms target).
"""
import torch


class MegaGraph:
    """
    Captures the entire first-token pipeline as a single CUDA graph.

    Prerequisites:
    - predictor_graph and talker_graph must already be captured
    - KV cache must be restored before calling run()

    This graph captures:
    1. sample_logits(cached_logits) → first token
    2. embed(token) → last_id_hidden
    3. cat(past_hidden, last_id_hidden) → pred_input
    4. predictor 15-step loop → codebook_token_ids
    5. cat(token, codebook_ids) → all_cb (output)
    6. build codec_hiddens → inputs_embeds
    7. add tth[:, gen_step] to inputs_embeds
    8. talker_decode(inputs_embeds) → hidden_states
    9. codec_head(hidden_states) → next_logits (output)
    """

    def __init__(self, model, cached_state, device="cuda"):
        """
        Args:
            model: FasterQwen3TTS instance
            cached_state: dict with {logits, past_hidden, gen_step, tpe, ...}
            device: CUDA device
        """
        self.model = model
        self.device = device
        self.graph = None

        inner = model.model.model
        talker = inner.talker
        config = inner.config.talker_config

        # Static references
        self.codec_embed = talker.get_input_embeddings()
        self.codec_head = talker.codec_head
        self.pred_embeds = talker.code_predictor.get_input_embeddings()
        self.num_code_groups = config.num_code_groups

        # Static input buffers (will be filled before replay)
        self.logits_buf = cached_state["logits"].clone()
        self.past_hidden_buf = cached_state["past_hidden"].clone()
        self.tth_token_buf = torch.zeros(1, 1, talker.config.hidden_size,
                                          device=device, dtype=torch.bfloat16)
        self.prefill_len = cached_state["prefill_len"]
        self.tpe_buf = cached_state["tpe"].clone()

        # Static output buffers
        self.codec_output = torch.zeros(self.num_code_groups + 1,
                                         dtype=torch.long, device=device)
        self.next_logits = None
        self.next_hidden = None

    def capture(self, suppress_mask, suppress_list, sample_fn, num_warmup=3):
        """
        Warmup and capture the mega graph.

        Args:
            suppress_mask: [vocab_size] bool tensor
            suppress_list: list of token IDs to suppress
            sample_fn: the sample_logits function
            num_warmup: number of warmup runs before capture
        """
        # Warmup
        for _ in range(num_warmup):
            self._forward(suppress_mask, suppress_list, sample_fn)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._forward(suppress_mask, suppress_list, sample_fn)
        torch.cuda.synchronize()

    def _forward(self, suppress_mask, suppress_list, sample_fn):
        """Single forward pass through the full pipeline."""
        # 1. Sample first token
        token = sample_fn(
            self.logits_buf, temperature=0.9, top_k=50, top_p=1.0,
            do_sample=True, suppress_mask=suppress_mask,
            suppress_tokens=suppress_list,
        )

        # 2. Embed
        last_id_hidden = self.codec_embed(token.unsqueeze(1))

        # 3. Predictor input
        pred_input = torch.cat((self.past_hidden_buf, last_id_hidden), dim=1)

        # 4. Run predictor (already a CUDA graph internally — need to inline)
        # NOTE: Cannot nest CUDA graphs. Need to inline predictor logic here.
        codebook_token_ids = self.model.predictor_graph.run(pred_input)

        # 5. Output codec tokens
        self.codec_output[0] = token
        self.codec_output[1:] = codebook_token_ids

        # 6-7. Build inputs_embeds for talker
        codec_hiddens = [last_id_hidden]
        for ci in range(self.num_code_groups - 1):
            codec_hiddens.append(
                self.pred_embeds[ci](codebook_token_ids[ci].unsqueeze(0).unsqueeze(0))
            )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        inputs_embeds = inputs_embeds + self.tth_token_buf

        # 8. Talker decode
        hidden_states = self.model.talker_graph.run(inputs_embeds, position=self.prefill_len)

        # 9. Next logits
        self.next_logits = self.codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        self.next_hidden = hidden_states[:, -1:, :].clone()

    def run(self, logits, past_hidden, tth_token):
        """
        Run the mega graph with new inputs.

        Args:
            logits: [vocab_size] — cached logits for first token
            past_hidden: [1, 1, H] — cached past hidden state
            tth_token: [1, 1, H] — trailing text hidden for current gen_step

        Returns:
            codec_output: [num_code_groups+1] — codec token IDs
            next_logits: [1, 1, vocab_size] — logits for next token
            next_hidden: [1, 1, H] — hidden state for next step
        """
        # Copy inputs into static buffers
        self.logits_buf.copy_(logits)
        self.past_hidden_buf.copy_(past_hidden)
        self.tth_token_buf.copy_(tth_token)

        # Replay graph
        if self.graph is not None:
            self.graph.replay()
        else:
            raise RuntimeError("Mega graph not captured. Call capture() first.")

        return self.codec_output, self.next_logits, self.next_hidden


# NOTE: This mega graph CANNOT work as-is because CUDA graphs cannot be nested.
# The predictor_graph.run() and talker_graph.run() internally replay their own
# CUDA graphs, which cannot be called from inside another CUDA graph capture.
#
# The solution requires INLINING the predictor and talker logic (not using their
# pre-captured graphs) into the mega graph's _forward method. This means:
# 1. Replicating the predictor's 15-step decode loop using raw model.forward()
# 2. Replicating the talker's single-step decode using raw model.forward()
# 3. Capturing all of this as ONE mega graph
#
# This is the next step of optimization. For now, this file serves as the
# architecture blueprint.

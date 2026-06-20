"""
Our OWN GPT-2 forward pass (Step 1 of the capstone).

Why reimplement a model HuggingFace already ships? Because to push real data
through OUR paged attention + KV pool, we need to control the attention step -
and you can only do that if you own the forward pass. This is exactly what vLLM
does: it reimplements each model so attention routes through its paged kernel.
A model = WEIGHTS (downloadable numbers) + ARCHITECTURE (how to combine them);
here we load GPT-2's pretrained weights but run our own architecture code.

Step 1 keeps a NORMAL contiguous KV cache and uses torch's attention - the goal
is just to prove our reimplementation is correct by matching HF token-for-token.
Step 2 swaps the cache for the paged pool and the attention for our Triton kernel.

GPT-2 quirks handled here:
  * weights are Conv1D (stored as (in, out)) => forward is just x @ W + b
  * activation is gelu-tanh ("gelu_new")
  * lm_head is tied to the token embedding (logits = x @ wte.T)
  * learned position embeddings (wpe), hard 1024-token limit

    python gpt2.py    # validates: our greedy output == HF greedy output
"""
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from common import DEVICE

EPS = 1e-5


class GPT2:
    def __init__(self, name="gpt2", device=DEVICE, dtype=torch.float32):
        hf = GPT2LMHeadModel.from_pretrained(name, dtype=dtype)
        sd = hf.state_dict()
        cfg = hf.config
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.max_ctx = cfg.n_positions
        self.device = device

        g = lambda k: sd[k].to(device)
        self.wte = g("transformer.wte.weight")            # (vocab, embd)
        self.wpe = g("transformer.wpe.weight")            # (ctx, embd)
        self.lnf_w = g("transformer.ln_f.weight")
        self.lnf_b = g("transformer.ln_f.bias")
        self.layers = []
        for i in range(self.n_layer):
            p = f"transformer.h.{i}."
            self.layers.append(dict(
                ln1_w=g(p + "ln_1.weight"), ln1_b=g(p + "ln_1.bias"),
                attn_w=g(p + "attn.c_attn.weight"), attn_b=g(p + "attn.c_attn.bias"),
                proj_w=g(p + "attn.c_proj.weight"), proj_b=g(p + "attn.c_proj.bias"),
                ln2_w=g(p + "ln_2.weight"), ln2_b=g(p + "ln_2.bias"),
                fc_w=g(p + "mlp.c_fc.weight"), fc_b=g(p + "mlp.c_fc.bias"),
                mproj_w=g(p + "mlp.c_proj.weight"), mproj_b=g(p + "mlp.c_proj.bias"),
            ))
        del hf

    def _ln(self, x, w, b):
        return F.layer_norm(x, (self.n_embd,), w, b, EPS)

    @torch.inference_mode()
    def forward(self, input_ids, past=None):
        """input_ids: (B, T). past: list of (K, V) per layer, each (B, heads, S, hd).
        Returns (logits (B, T, vocab), new_past)."""
        B, T = input_ids.shape
        past_len = 0 if past is None else past[0][0].shape[2]
        pos = torch.arange(past_len, past_len + T, device=self.device)
        x = self.wte[input_ids] + self.wpe[pos]           # (B, T, embd)

        new_past = []
        for i, L in enumerate(self.layers):
            h = self._ln(x, L["ln1_w"], L["ln1_b"])
            qkv = h @ L["attn_w"] + L["attn_b"]           # (B, T, 3*embd)
            q, k, v = qkv.split(self.n_embd, dim=2)
            # (B, T, embd) -> (B, heads, T, hd)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            if past is not None:
                k = torch.cat([past[i][0], k], dim=2)
                v = torch.cat([past[i][1], v], dim=2)
            new_past.append((k, v))
            # prefill (no past) is causal; decode (T=1) attends all cached keys
            a = F.scaled_dot_product_attention(q, k, v, is_causal=(past is None and T > 1))
            a = a.transpose(1, 2).reshape(B, T, self.n_embd)
            x = x + (a @ L["proj_w"] + L["proj_b"])       # residual

            h = self._ln(x, L["ln2_w"], L["ln2_b"])
            m = F.gelu(h @ L["fc_w"] + L["fc_b"], approximate="tanh")
            x = x + (m @ L["mproj_w"] + L["mproj_b"])      # residual

        x = self._ln(x, self.lnf_w, self.lnf_b)
        logits = x @ self.wte.T                            # tied lm_head
        return logits, new_past


@torch.inference_mode()
def generate(model, input_ids, max_new=40, eos_id=None):
    """Greedy decode with KV cache; stop on EOS or after max_new tokens."""
    logits, past = model.forward(input_ids)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    for _ in range(max_new - 1):
        if eos_id is not None and nxt.item() == eos_id:
            break
        logits, past = model.forward(nxt, past)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, dim=1)


def _validate():
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    ours = GPT2()
    our_ids = generate(ours, ids, max_new=30)

    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    hf_out = hf.generate(ids, max_new_tokens=30, do_sample=False,
                         pad_token_id=tok.eos_token_id)[:, ids.shape[1]:]

    o = our_ids[0].tolist()
    h = hf_out[0].tolist()
    match = sum(1 for a, b in zip(o, h) if a == b)
    print("=== Step 1: own forward vs HuggingFace (greedy) ===")
    print("  prompt:    ", repr(prompt))
    print("  ours:      ", repr(tok.decode(o)))
    print("  hf:        ", repr(tok.decode(h)))
    print(f"  tokens matched: {match}/{len(h)}  ->  {'PASS' if match == len(h) else 'MISMATCH'}")


if __name__ == "__main__":
    _validate()

"""
Step 2: run a real prompt through GPT-2 with PAGED KV + our Triton kernel.

Reuses gpt2.py for the weights, but the KV cache is now a block pool (one per
layer, all sharing a single block table per sequence). Prefill computes the
prompt's attention normally (causal sdpa) and scatters K/V into the pool; decode
reads the pool through paged_attention(). Decode is where paging matters, so
that's where the kernel runs.

Single sequence for now. If greedy output still matches HF, real data is flowing
through the kernel end to end. Batching is Step 3.

    python paged_gpt2.py
"""
import torch
import torch.nn.functional as F

from common import DEVICE
from gpt2 import GPT2
from paged_attention import paged_attention


class PagedGPT2:
    def __init__(self, model: GPT2, num_blocks=2048, block_size=16):
        self.m = model
        self.bs = block_size
        self.nb = num_blocks
        nl, nh, hd = model.n_layer, model.n_head, model.head_dim
        # per-layer KV pools, indexed by physical block id
        self.pool_k = torch.zeros((nl, num_blocks, nh, block_size, hd), device=DEVICE)
        self.pool_v = torch.zeros((nl, num_blocks, nh, block_size, hd), device=DEVICE)
        self.free = list(range(num_blocks))
        self.reset()

    def reset(self):
        self.block_table = []     # logical block -> physical block id
        self.length = 0           # tokens stored so far

    def _ensure(self, total):
        while (total + self.bs - 1) // self.bs > len(self.block_table):
            self.block_table.append(self.free.pop())

    def _write(self, layer, start, k, v):
        # k, v: (heads, T, hd) for tokens [start, start+T)
        bt = torch.tensor(self.block_table, device=DEVICE)
        toks = torch.arange(start, start + k.shape[1], device=DEVICE)
        blk = bt[toks // self.bs]
        off = toks % self.bs
        self.pool_k[layer][blk, :, off, :] = k.transpose(0, 1)
        self.pool_v[layer][blk, :, off, :] = v.transpose(0, 1)

    @torch.inference_mode()
    def forward(self, input_ids):
        m = self.m
        B, T = input_ids.shape
        assert B == 1
        prefill = self.length == 0
        self._ensure(self.length + T)

        pos = torch.arange(self.length, self.length + T, device=DEVICE)
        x = m.wte[input_ids] + m.wpe[pos]

        bt = torch.tensor([self.block_table], dtype=torch.int32, device=DEVICE)  # (1, nblk)

        for i, L in enumerate(m.layers):
            h = m._ln(x, L["ln1_w"], L["ln1_b"])
            qkv = h @ L["attn_w"] + L["attn_b"]
            q, k, v = qkv.split(m.n_embd, dim=2)
            q = q.view(B, T, m.n_head, m.head_dim).transpose(1, 2)
            k = k.view(B, T, m.n_head, m.head_dim).transpose(1, 2)
            v = v.view(B, T, m.n_head, m.head_dim).transpose(1, 2)

            self._write(i, self.length, k[0], v[0])   # stash this layer's K/V in the pool

            if prefill:
                a = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
                a = a.transpose(1, 2).reshape(B, T, m.n_embd)
            else:
                seqlen = torch.tensor([self.length + 1], dtype=torch.int32, device=DEVICE)
                a = paged_attention(q[:, :, 0, :], self.pool_k[i], self.pool_v[i], bt, seqlen)
                a = a.reshape(B, T, m.n_embd).to(x.dtype)

            x = x + (a @ L["proj_w"] + L["proj_b"])
            h = m._ln(x, L["ln2_w"], L["ln2_b"])
            mm = F.gelu(h @ L["fc_w"] + L["fc_b"], approximate="tanh")
            x = x + (mm @ L["mproj_w"] + L["mproj_b"])

        self.length += T
        x = m._ln(x, m.lnf_w, m.lnf_b)
        return x @ m.wte.T

    @torch.inference_mode()
    def generate(self, input_ids, max_new=40, eos_id=None):
        self.reset()
        logits = self.forward(input_ids)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = [nxt]
        for _ in range(max_new - 1):
            if eos_id is not None and nxt.item() == eos_id:
                break
            logits = self.forward(nxt)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
        return torch.cat(out, dim=1)


def _validate():
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    paged = PagedGPT2(GPT2())
    ours = paged.generate(ids, max_new=30)

    hf = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    ref = hf.generate(ids, max_new_tokens=30, do_sample=False,
                      pad_token_id=tok.eos_token_id)[:, ids.shape[1]:]

    o, h = ours[0].tolist(), ref[0].tolist()
    match = sum(1 for a, b in zip(o, h) if a == b)
    print("=== Step 2: paged GPT-2 (decode via Triton kernel) vs HuggingFace ===")
    print("  ours:", repr(tok.decode(o)))
    print("  hf:  ", repr(tok.decode(h)))
    print(f"  tokens matched: {match}/{len(h)}  ->  {'PASS' if match == len(h) else 'MISMATCH'}")


if __name__ == "__main__":
    _validate()

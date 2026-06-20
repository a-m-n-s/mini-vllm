"""
Step 3: the whole thing. Serve a batch of real prompts with continuous batching,
paged KV, the Triton decode kernel, and EOS stopping.

Each sequence owns a block table; the pool is shared. Prefill on admit, batched
paged decode for the running set, evict on EOS/max_tokens, admit waiting prompts
into freed slots. This is a mini-vLLM.

    python serve.py
"""
from collections import deque

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from common import DEVICE
from gpt2 import GPT2
from paged_attention import paged_attention


class Seq:
    def __init__(self, idx, prompt_ids, max_tokens):
        self.idx = idx
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.block_table = []
        self.length = 0
        self.last = None
        self.out = []
        self.done = False


class PagedEngine:
    def __init__(self, model: GPT2, num_blocks=1024, block_size=16, eos_id=50256):
        self.m = model
        self.bs = block_size
        self.eos = eos_id
        nl, nh, hd = model.n_layer, model.n_head, model.head_dim
        self.pool_k = torch.zeros((nl, num_blocks, nh, block_size, hd), device=DEVICE)
        self.pool_v = torch.zeros((nl, num_blocks, nh, block_size, hd), device=DEVICE)
        self.free = list(range(num_blocks))

    def _grow(self, seq, total):
        while (total + self.bs - 1) // self.bs > len(seq.block_table):
            seq.block_table.append(self.free.pop())

    def _free(self, seq):
        self.free.extend(seq.block_table)
        seq.block_table = []

    # shared per-layer math; attn_fn does the layer's attention and returns (B,T,embd)
    def _block(self, x, L, attn_fn):
        m = self.m
        h = m._ln(x, L["ln1_w"], L["ln1_b"])
        qkv = h @ L["attn_w"] + L["attn_b"]
        q, k, v = qkv.split(m.n_embd, dim=2)
        B, T = x.shape[0], x.shape[1]
        shp = lambda t: t.view(B, T, m.n_head, m.head_dim).transpose(1, 2)
        a = attn_fn(shp(q), shp(k), shp(v))
        x = x + (a @ L["proj_w"] + L["proj_b"])
        h = m._ln(x, L["ln2_w"], L["ln2_b"])
        mm = F.gelu(h @ L["fc_w"] + L["fc_b"], approximate="tanh")
        return x + (mm @ L["mproj_w"] + L["mproj_b"])

    @torch.inference_mode()
    def prefill(self, seq):
        m = self.m
        ids = seq.prompt_ids.view(1, -1)
        T = ids.shape[1]
        self._grow(seq, T)
        pos = torch.arange(0, T, device=DEVICE)
        x = m.wte[ids] + m.wpe[pos]
        bt = torch.tensor(seq.block_table, device=DEVICE)
        slots_blk = bt[torch.arange(T, device=DEVICE) // self.bs]
        slots_off = torch.arange(T, device=DEVICE) % self.bs

        for i, L in enumerate(m.layers):
            def attn(q, k, v, i=i):
                self.pool_k[i][slots_blk, :, slots_off, :] = k[0].transpose(0, 1)
                self.pool_v[i][slots_blk, :, slots_off, :] = v[0].transpose(0, 1)
                a = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
                return a.transpose(1, 2).reshape(1, T, m.n_embd)
            x = self._block(x, L, attn)

        x = m._ln(x, m.lnf_w, m.lnf_b)
        logits = x[:, -1] @ m.wte.T
        seq.length = T
        seq.last = int(logits.argmax())
        seq.out.append(seq.last)

    @torch.inference_mode()
    def decode(self, seqs):
        m = self.m
        S = len(seqs)
        for s in seqs:
            self._grow(s, s.length + 1)

        input_ids = torch.tensor([[s.last] for s in seqs], device=DEVICE)
        positions = torch.tensor([s.length for s in seqs], device=DEVICE)
        x = m.wte[input_ids] + m.wpe[positions].unsqueeze(1)   # (S,1,embd)

        maxb = max(len(s.block_table) for s in seqs)
        bt = torch.zeros((S, maxb), dtype=torch.int32, device=DEVICE)
        for r, s in enumerate(seqs):
            bt[r, :len(s.block_table)] = torch.tensor(s.block_table, dtype=torch.int32, device=DEVICE)
        seqlens = torch.tensor([s.length + 1 for s in seqs], dtype=torch.int32, device=DEVICE)
        wblk = torch.tensor([s.block_table[s.length // self.bs] for s in seqs], device=DEVICE)
        woff = torch.tensor([s.length % self.bs for s in seqs], device=DEVICE)

        for i, L in enumerate(m.layers):
            def attn(q, k, v, i=i):
                self.pool_k[i][wblk, :, woff, :] = k[:, :, 0, :]   # (S,heads,hd)
                self.pool_v[i][wblk, :, woff, :] = v[:, :, 0, :]
                a = paged_attention(q[:, :, 0, :], self.pool_k[i], self.pool_v[i], bt, seqlens)
                return a.reshape(S, 1, m.n_embd).to(x.dtype)
            x = self._block(x, L, attn)

        x = m._ln(x, m.lnf_w, m.lnf_b)
        logits = x[:, -1] @ m.wte.T          # (S, vocab)
        nxt = logits.argmax(-1)
        for r, s in enumerate(seqs):
            s.last = int(nxt[r])
            s.out.append(s.last)
            s.length += 1
            if s.last == self.eos or len(s.out) >= s.max_tokens:
                s.done = True

    def serve(self, prompts, max_tokens=40, max_batch=8):
        waiting = deque(prompts)
        active, results = [], {}
        while waiting or active:
            while len(active) < max_batch and waiting:
                s = waiting.popleft()
                self.prefill(s)
                if s.last == self.eos or len(s.out) >= s.max_tokens:
                    results[s.idx] = s.out      # finished during prefill
                    self._free(s)
                else:
                    active.append(s)
            if not active:
                continue
            self.decode(active)
            still = []
            for s in active:
                if s.done:
                    results[s.idx] = s.out
                    self._free(s)
                else:
                    still.append(s)
            active = still
        return [results[i] for i in range(len(prompts))]


def _demo():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    texts = [
        "The capital of France is",
        "Once upon a time,",
        "The meaning of life is",
        "In 2050, computers will",
    ]
    prompts = [Seq(i, tok(t, return_tensors="pt").input_ids[0].to(DEVICE), max_tokens=25)
               for i, t in enumerate(texts)]

    eng = PagedEngine(GPT2())
    outs = eng.serve(prompts, max_tokens=25, max_batch=4)

    print("=== mini-vLLM: continuous batching + paged KV + Triton kernel ===")
    for t, o in zip(texts, outs):
        print(f"\n> {t!r}\n  {tok.decode(o)!r}")
    return tok, texts, outs


if __name__ == "__main__":
    _demo()

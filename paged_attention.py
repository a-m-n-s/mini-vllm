"""
Stage 5b: the PAGED-ATTENTION decode kernel (Triton).

This is the piece that makes paging actually fast. One decode step: each
sequence has a SINGLE query token that must attend to all its cached keys/values
- but those K/V live in scattered fixed-size blocks (paged_kv.py), not one
contiguous tensor. The kernel walks the sequence's BLOCK TABLE, loading each
block straight from the pool, and computes attention with an online (flash-style)
softmax so it never materializes the full score row. No gather, no padding.

One program per (sequence, head). For each, stream over the sequence's blocks:
  scores = q . K_block            -> running max m, running denom l, running acc
  out = acc / l
Online softmax = the FlashAttention trick: keep (m, l, acc) and rescale by
exp(m_old - m_new) as the max grows, so a block at a time is enough.

    python paged_attention.py    # correctness test + benchmark vs gather+sdpa
"""
import math

import torch
import triton
import triton.language as tl

from common import DEVICE, DTYPE
from paged_kv import BlockManager


@triton.jit
def _paged_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, bt_ptr, sl_ptr,
    scale,
    q_s0, q_s1,                 # q strides: (seq, head); head_dim is contiguous
    kv_s0, kv_s1, kv_s2,        # k/v cache strides: (block, head, block_pos); dim contiguous
    o_s0, o_s1,                 # out strides: (seq, head)
    bt_s0,                      # block_tables stride: (seq)
    BLOCK_SIZE: tl.constexpr,   # tokens per KV block
    D: tl.constexpr,            # head_dim
):
    s = tl.program_id(0)        # which sequence
    h = tl.program_id(1)        # which head
    seq_len = tl.load(sl_ptr + s)

    d = tl.arange(0, D)
    q = tl.load(q_ptr + s * q_s0 + h * q_s1 + d).to(tl.float32) * scale   # (D,)

    m_i = -float("inf")         # running max of the scores
    l_i = 0.0                   # running sum of exp(scores - m)
    acc = tl.zeros((D,), dtype=tl.float32)   # running weighted sum of V

    offs = tl.arange(0, BLOCK_SIZE)
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    for b in range(0, n_blocks):
        phys = tl.load(bt_ptr + s * bt_s0 + b)          # physical block id from the table
        tok = b * BLOCK_SIZE + offs                      # global token indices in this block
        mask = tok < seq_len                             # last block is partial

        base = phys * kv_s0 + h * kv_s1 + offs[:, None] * kv_s2 + d[None, :]
        k = tl.load(k_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)   # (BS, D)
        scores = tl.sum(k * q[None, :], axis=1)          # (BS,) = K . q
        scores = tl.where(mask, scores, -float("inf"))   # ignore padding slots

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)                      # rescale old running stats
        p = tl.exp(scores - m_new)                       # (BS,)

        v = tl.load(v_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)   # (BS, D)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + s * o_s0 + h * o_s1 + d, out)


def paged_attention(q, k_cache, v_cache, block_tables, seq_lens, scale=None):
    """q: (num_seqs, num_heads, head_dim). Returns out same shape (fp32)."""
    num_seqs, num_heads, D = q.shape
    block_size = k_cache.shape[2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty((num_seqs, num_heads, D), device=q.device, dtype=torch.float32)
    grid = (num_seqs, num_heads)
    _paged_attn_kernel[grid](
        q, k_cache, v_cache, out, block_tables, seq_lens, scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        out.stride(0), out.stride(1),
        block_tables.stride(0),
        BLOCK_SIZE=block_size, D=D,
    )
    return out


def reference(q, k_cache, v_cache, block_tables, seq_lens, scale=None):
    """Plain PyTorch: gather each seq's blocks into a contiguous K/V, then do
    standard attention. Slow, but obviously correct - the kernel must match it."""
    num_seqs, num_heads, D = q.shape
    block_size = k_cache.shape[2]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    out = torch.empty((num_seqs, num_heads, D), device=q.device, dtype=torch.float32)
    for s in range(num_seqs):
        L = int(seq_lens[s])
        nb = (L + block_size - 1) // block_size
        ks = [k_cache[int(block_tables[s, b])] for b in range(nb)]   # each (heads, bs, D)
        vs = [v_cache[int(block_tables[s, b])] for b in range(nb)]
        K = torch.cat(ks, dim=1)[:, :L, :].float()                   # (heads, L, D)
        V = torch.cat(vs, dim=1)[:, :L, :].float()
        qs = q[s].float()                                            # (heads, D)
        scores = torch.einsum("hd,hld->hl", qs, K) * scale           # (heads, L)
        p = torch.softmax(scores, dim=-1)
        out[s] = torch.einsum("hl,hld->hd", p, V)
    return out


def _build_scenario(num_seqs, num_heads, D, block_size, num_blocks, seed=0):
    """Allocate num_seqs sequences of random length in a BlockManager, fill the
    pool with random K/V, and return everything the kernel needs."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    bm = BlockManager(num_blocks, block_size, num_heads, D)
    bm.k_cache.normal_(generator=g)
    bm.v_cache.normal_(generator=g)

    lens = torch.randint(1, 256, (num_seqs,), generator=g, device=DEVICE)
    for s in range(num_seqs):
        bm.allocate(s, int(lens[s]))

    max_nb = max(len(bm.block_table[s]) for s in range(num_seqs))
    bt = torch.zeros((num_seqs, max_nb), dtype=torch.int32, device=DEVICE)
    for s in range(num_seqs):
        blocks = bm.block_table[s]
        bt[s, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=DEVICE)

    q = torch.randn((num_seqs, num_heads, D), generator=g, device=DEVICE, dtype=DTYPE)
    return q, bm.k_cache, bm.v_cache, bt, lens.to(torch.int32)


def _test():
    q, k, v, bt, sl = _build_scenario(num_seqs=16, num_heads=12, D=64,
                                      block_size=16, num_blocks=1024)
    out = paged_attention(q, k, v, bt, sl)
    ref = reference(q, k, v, bt, sl)
    diff = (out - ref).abs().max().item()
    print("=== correctness ===")
    print(f"  max abs diff vs reference: {diff:.2e}  ->  {'PASS' if diff < 1e-2 else 'FAIL'}")


def _bench():
    q, k, v, bt, sl = _build_scenario(num_seqs=64, num_heads=12, D=64,
                                      block_size=16, num_blocks=4096)
    t_kernel = triton.testing.do_bench(lambda: paged_attention(q, k, v, bt, sl))
    t_ref = triton.testing.do_bench(lambda: reference(q, k, v, bt, sl))
    print("\n=== benchmark (64 seqs, 12 heads, decode step) ===")
    print(f"  triton paged kernel: {t_kernel:.3f} ms")
    print(f"  gather + pytorch:    {t_ref:.3f} ms   ({t_ref / t_kernel:.1f}x slower)")


if __name__ == "__main__":
    _test()
    _bench()

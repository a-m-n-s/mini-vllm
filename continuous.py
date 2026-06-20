"""
Stage 3: CONTINUOUS (in-flight) batching - the heart of vLLM. Keep ONE running
batched KV cache; every step decodes all active sequences together, and the
instant one finishes, evict its row and admit a waiting request into the freed
slot mid-flight. The batch stays full instead of idling for the slowest (Stage
2's failure).

The cache is rebuilt ONLY when batch composition changes (evict/admit), never
per step - that's what keeps per-token overhead low. The one remaining cost is
LEFT-PADDING: a long sequence forces short ones to carry dead pad columns.
Removing that (block-based KV, no padding) is Stage 5 / PagedAttention.
"""
import time
from collections import deque

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from common import DEVICE, warmup


# --- KV-cache surgery helpers (rows = sequences [dim 0], KV length = dim 2) ---

def _pad_left(cache, amount, num_layers):
    """Left-pad every row's KV by `amount` along the sequence dim."""
    if amount <= 0:
        return
    for l in range(num_layers):
        cache.layers[l].keys = F.pad(cache.layers[l].keys, (0, 0, amount, 0))
        cache.layers[l].values = F.pad(cache.layers[l].values, (0, 0, amount, 0))


def _trim_left(cache, amount, num_layers):
    """Drop `amount` leading columns (only ever all-pad) to bound the width."""
    if amount <= 0:
        return
    for l in range(num_layers):
        cache.layers[l].keys = cache.layers[l].keys[:, :, amount:, :].contiguous()
        cache.layers[l].values = cache.layers[l].values[:, :, amount:, :].contiguous()


def _select_rows(cache, idx, num_layers):
    """Keep only rows in `idx` - this is eviction."""
    for l in range(num_layers):
        cache.layers[l].keys = cache.layers[l].keys.index_select(0, idx).contiguous()
        cache.layers[l].values = cache.layers[l].values.index_select(0, idx).contiguous()


def _cat_rows(a, b, num_layers):
    """Append b's rows under a's (must already share KV length)."""
    for l in range(num_layers):
        a.layers[l].keys = torch.cat([a.layers[l].keys, b.layers[l].keys], dim=0)
        a.layers[l].values = torch.cat([a.layers[l].values, b.layers[l].values], dim=0)


def run_continuous_batch(model, reqs, max_batch=16):
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions
    num_layers = model.config.n_layer
    warmup(model)

    waiting = deque(reqs)
    cache = None                 # single running DynamicCache (rows = sequences)
    mask = None                  # (B, S) attention mask: 1 = real token, 0 = pad
    valid, last, remaining = [], [], []   # per-row: real KV len, last token, tokens owed
    S = 0
    decoded, total_prompt = 0, 0

    @torch.inference_mode()
    def admit():
        nonlocal cache, mask, valid, last, remaining, S, total_prompt
        while len(valid) < max_batch and waiting:
            r = waiting.popleft()
            p = min(r.prompt_len, max_ctx - 1)
            o = min(r.output_len, max_ctx - p)
            total_prompt += p
            if o == 0:
                continue
            ids = torch.randint(0, vocab_size, (1, p), device=DEVICE)
            out = model(input_ids=ids, use_cache=True)
            nc = out.past_key_values
            ntok = int(out.logits[0, -1].argmax())
            if cache is None:
                cache, S = nc, p
                mask = torch.ones((1, p), dtype=torch.long, device=DEVICE)
            else:
                target = max(S, p)
                _pad_left(nc, target - p, num_layers)
                if S < target:
                    _pad_left(cache, target - S, num_layers)
                    mask = F.pad(mask, (target - S, 0))
                row = torch.zeros((1, target), dtype=torch.long, device=DEVICE)
                row[0, target - p:] = 1
                mask = torch.cat([mask, row], dim=0)
                _cat_rows(cache, nc, num_layers)
                S = target
            valid.append(p); last.append(ntok); remaining.append(o)

    @torch.inference_mode()
    def decode_step():
        nonlocal cache, mask, S, decoded
        B = len(valid)
        amask = torch.cat([mask, torch.ones((B, 1), dtype=torch.long, device=DEVICE)], dim=1)
        pos = torch.tensor([[v] for v in valid], device=DEVICE)
        ntok = torch.tensor([[t] for t in last], device=DEVICE)
        out = model(input_ids=ntok, past_key_values=cache,
                    attention_mask=amask, position_ids=pos, use_cache=True)
        cache = out.past_key_values
        mask, S = amask, S + 1
        logits = out.logits
        for i in range(B):
            last[i] = int(logits[i, -1].argmax())
            valid[i] += 1
            remaining[i] -= 1
        decoded += B

    def evict():
        nonlocal cache, mask, valid, last, remaining, S
        keep = [i for i in range(len(valid)) if remaining[i] > 0]
        if len(keep) == len(valid):
            return
        if not keep:
            cache, mask, S = None, None, 0
            valid, last, remaining = [], [], []
            return
        idx = torch.tensor(keep, device=DEVICE)
        _select_rows(cache, idx, num_layers)
        mask = mask.index_select(0, idx).contiguous()
        valid = [valid[i] for i in keep]
        last = [last[i] for i in keep]
        remaining = [remaining[i] for i in keep]
        _trim_left(cache, S - max(valid), num_layers)
        mask = mask[:, S - max(valid):].contiguous()
        S = max(valid)

    t0 = time.perf_counter()
    admit()
    while valid:
        decode_step()
        evict()
        admit()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"\n=== continuous batch (max_batch={max_batch}, {len(reqs)} requests) ===")
    print(f"  prompt tokens prefilled: {total_prompt:,}")
    print(f"  tokens decoded:          {decoded:,}")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  decode throughput:       {decoded / elapsed:,.1f} tok/s")
    return decoded / elapsed

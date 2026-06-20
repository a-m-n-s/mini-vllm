"""
Stage 1 of the mini serving engine: the NAIVE SEQUENTIAL baseline.

Process requests one at a time, each with its own KV cache, on a real model
(GPT-2). No batching at all - this is the throughput FLOOR. Everything later
(static batch, continuous batch) has to beat this number to justify itself.

The decode loop is written by hand on purpose (not model.generate): the whole
point of this project is to own the prefill/decode loop and the KV cache, since
the batching stages need to control it themselves.

Content of the prompts doesn't matter for throughput - only the SHAPES do - so I
fake prompts with random token ids of the right length instead of dragging in a
tokenizer. A "request" is just (prompt_len, output_len) from workload.py.

run: ~/triton-practice/.venv/bin/python engine.py
"""

import time
from collections import deque

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers.cache_utils import DynamicCache

from workload import make_workload

MODEL_NAME = "gpt2"          # 124M; small enough to iterate fast on the 4080
DEVICE = "cuda"
DTYPE = torch.float16


def load_model():
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model.to(DEVICE).eval()
    return model


@torch.inference_mode()
def run_request(model, prompt_len, output_len, vocab_size, max_ctx):
    """
    Prefill a fake prompt, then decode output_len tokens greedily, reusing the
    KV cache each step. Returns the number of tokens actually generated.

    The KV cache is the whole game: prefill runs the full prompt ONCE and stashes
    the keys/values; every decode step then feeds just the ONE new token plus the
    cached past, so we never recompute attention over the prompt again.

    GPT-2 only has 1024 learned position embeddings, so prompt+output must fit in
    max_ctx or the position index runs off the table (device-side assert). Real
    servers hit the same wall - context length is a hard constraint, not a knob.
    """
    prompt_len = min(prompt_len, max_ctx - 1)
    output_len = min(output_len, max_ctx - prompt_len)

    # fake prompt: random token ids, shape (batch=1, prompt_len)
    input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=DEVICE)

    # --- prefill: one forward over the whole prompt, fills the cache ---
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

    # --- decode: one token at a time, feeding only the new token + the cache ---
    generated = 0
    for _ in range(output_len):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated += 1
    return generated


def run_sequential(model, reqs):
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions  # GPT-2: 1024

    # warmup: first CUDA call compiles/loads kernels; don't let that pollute timing
    run_request(model, 16, 4, vocab_size, max_ctx)
    torch.cuda.synchronize()

    # effective lengths after clamping to the model's context window
    eff = [(min(r.prompt_len, max_ctx - 1),
            min(r.output_len, max_ctx - min(r.prompt_len, max_ctx - 1))) for r in reqs]
    total_prompt = sum(p for p, _ in eff)
    total_decode = 0

    t0 = time.perf_counter()
    for r in reqs:
        total_decode += run_request(model, r.prompt_len, r.output_len, vocab_size, max_ctx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # decode tokens/sec is the number that matters for serving - it's the steady
    # state. (prefill is a one-time cost per request.)
    print(f"\n=== naive sequential ({len(reqs)} requests) ===")
    print(f"  prompt tokens prefilled: {total_prompt:,}")
    print(f"  tokens decoded:          {total_decode:,}")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  decode throughput:       {total_decode / elapsed:,.1f} tok/s")
    print(f"  end-to-end throughput:   {(total_prompt + total_decode) / elapsed:,.1f} tok/s")


@torch.inference_mode()
def run_static_batch(model, reqs, batch_size=16):
    """
    Stage 2: STATIC batching. Process requests in fixed groups of batch_size.
    Within a group, prefill all prompts together (left-padded so the last real
    token of every sequence lines up at the right edge), then decode the whole
    group in lockstep until the SLOWEST sequence in the group finishes.

    The key inefficiency this exposes: the batch is a fixed-width tensor. When a
    short sequence hits its output_len, its slot can't be handed to a waiting
    request - it just rides along (masked, position frozen) burning compute until
    the longest sequence in the group is done. We count those wasted slot-steps.
    That waste is exactly what Stage 3 (continuous batching) reclaims.
    """
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions

    run_request(model, 16, 4, vocab_size, max_ctx)  # warmup
    torch.cuda.synchronize()

    total_prompt = 0
    useful_tokens = 0      # decode tokens a request actually needed
    slot_steps = 0         # batch slots consumed across all decode steps (incl. waste)

    t0 = time.perf_counter()
    for start in range(0, len(reqs), batch_size):
        group = reqs[start:start + batch_size]
        B = len(group)
        # per-sequence clamp so prompt+output fits the context window
        plens = [min(r.prompt_len, max_ctx - 1) for r in group]
        olens = [min(r.output_len, max_ctx - p) for r, p in zip(group, plens)]
        total_prompt += sum(plens)
        L = max(plens)

        # left-pad: real tokens occupy the LAST plens[i] columns of each row
        input_ids = torch.zeros((B, L), dtype=torch.long, device=DEVICE)
        attn = torch.zeros((B, L), dtype=torch.long, device=DEVICE)
        for i, p in enumerate(plens):
            input_ids[i, L - p:] = torch.randint(0, vocab_size, (p,), device=DEVICE)
            attn[i, L - p:] = 1
        # left-padding breaks the default 0..L position numbering, so build it by
        # hand: positions count only real tokens (cumsum of the mask).
        pos = (attn.cumsum(-1) - 1).clamp(min=0)

        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

        cur_pos = torch.tensor(plens, device=DEVICE)   # position of each seq's 1st decode token
        gen = [0] * B
        done = [o == 0 for o in olens]
        D = max(olens)                                  # group runs until the slowest

        for _ in range(D):
            if all(done):
                break
            # extend the mask by one column; finished seqs get 0 (they're dead weight)
            new_col = torch.tensor([[0 if done[i] else 1] for i in range(B)], device=DEVICE)
            attn = torch.cat([attn, new_col], dim=1)
            # finished seqs ride along with their position frozen; clamp so a seq
            # that filled the whole context window doesn't index position 1024.
            step_pos = cur_pos.clamp(max=max_ctx - 1).view(B, 1)
            out = model(input_ids=next_tok, past_key_values=past,
                        attention_mask=attn, position_ids=step_pos, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for i in range(B):
                slot_steps += 1                          # every slot cost a forward, used or not
                if not done[i]:
                    gen[i] += 1
                    useful_tokens += 1
                    cur_pos[i] += 1                       # active seq advances...
                    if gen[i] >= olens[i]:
                        done[i] = True
                # done seq: cur_pos frozen (no overflow), slot wasted this step

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    wasted = slot_steps - useful_tokens
    print(f"\n=== static batch (batch_size={batch_size}, {len(reqs)} requests) ===")
    print(f"  prompt tokens prefilled: {total_prompt:,}")
    print(f"  useful tokens decoded:   {useful_tokens:,}")
    print(f"  wasted slot-steps:       {wasted:,} ({wasted / slot_steps * 100:.0f}% of slots idle)")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  decode throughput:       {useful_tokens / elapsed:,.1f} tok/s")
    return useful_tokens / elapsed


# --- KV-cache surgery helpers (operate on the single running batched cache) ---
# Sequences live as ROWS (dim 0) of the cache; the KV length is dim 2. Shorter
# sequences are LEFT-padded so all rows share a common length, and an attention
# mask hides the pad. We only touch the cache on evict/admit, never per-step.

def _pad_left(cache, amount, num_layers):
    """Left-pad every row's KV by `amount` along the sequence dim."""
    if amount <= 0:
        return
    for l in range(num_layers):
        cache.layers[l].keys = F.pad(cache.layers[l].keys, (0, 0, amount, 0))
        cache.layers[l].values = F.pad(cache.layers[l].values, (0, 0, amount, 0))


def _trim_left(cache, amount, num_layers):
    """Drop `amount` leading columns (only ever all-pad columns) to bound width."""
    if amount <= 0:
        return
    for l in range(num_layers):
        cache.layers[l].keys = cache.layers[l].keys[:, :, amount:, :].contiguous()
        cache.layers[l].values = cache.layers[l].values[:, :, amount:, :].contiguous()


def _select_rows(cache, idx, num_layers):
    """Keep only the rows (sequences) in `idx` — this is eviction."""
    for l in range(num_layers):
        cache.layers[l].keys = cache.layers[l].keys.index_select(0, idx).contiguous()
        cache.layers[l].values = cache.layers[l].values.index_select(0, idx).contiguous()


def _cat_rows(a, b, num_layers):
    """Append b's rows under a's (both must already share the same KV length)."""
    for l in range(num_layers):
        a.layers[l].keys = torch.cat([a.layers[l].keys, b.layers[l].keys], dim=0)
        a.layers[l].values = torch.cat([a.layers[l].values, b.layers[l].values], dim=0)


def run_continuous_batch(model, reqs, max_batch=16):
    """
    Stage 3: CONTINUOUS (in-flight) batching. Keep ONE running batched KV cache.
    Every step decodes all active sequences together; the instant one finishes,
    evict its row and admit a waiting request into the freed slot mid-flight - so
    the batch stays full instead of idling for the slowest (Stage 2's failure).
    This is the heart of vLLM.

    The cache is rebuilt ONLY when the batch composition changes (evict/admit),
    not every step - that's what keeps the per-token overhead low. The remaining
    cost is the left-padding: a long seq forces short ones to carry dead pad
    columns. Removing THAT (block-based KV, no padding) is Stage 4 / PagedAttention.
    """
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions
    num_layers = model.config.n_layer

    run_request(model, 16, 4, vocab_size, max_ctx)  # warmup
    torch.cuda.synchronize()

    waiting = deque(reqs)
    cache = None                 # the single running DynamicCache (rows = sequences)
    mask = None                  # (B, S) attention mask: 1 = real token, 0 = pad
    valid, last, remaining = [], [], []   # per-row: real KV len, last token, decode tokens owed
    S = 0                        # current padded KV width
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
            nc = out.past_key_values            # fresh single-row cache, length p
            ntok = int(out.logits[0, -1].argmax())
            if cache is None:                    # first sequence seeds the batch
                cache, S = nc, p
                mask = torch.ones((1, p), dtype=torch.long, device=DEVICE)
            else:
                target = max(S, p)
                _pad_left(nc, target - p, num_layers)       # align new row to width...
                if S < target:
                    _pad_left(cache, target - S, num_layers)  # ...or widen the batch
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
        pos = torch.tensor([[v] for v in valid], device=DEVICE)        # next-token position
        ntok = torch.tensor([[t] for t in last], device=DEVICE)
        out = model(input_ids=ntok, past_key_values=cache,
                    attention_mask=amask, position_ids=pos, use_cache=True)
        cache = out.past_key_values     # grew to S+1 in place
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
        _trim_left(cache, S - max(valid), num_layers)   # shed pad the evicted seq left behind
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


if __name__ == "__main__":
    print(f"loading {MODEL_NAME} on {DEVICE} ({DTYPE})...")
    model = load_model()

    # small workload so it runs quick; bimodal = the interesting length mix
    reqs = make_workload(n=32, pattern="bimodal", seed=0)

    run_sequential(model, reqs)
    run_static_batch(model, reqs, batch_size=16)
    run_continuous_batch(model, reqs, max_batch=16)

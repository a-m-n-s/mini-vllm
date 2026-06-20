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
import torch
from transformers import GPT2LMHeadModel

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


if __name__ == "__main__":
    print(f"loading {MODEL_NAME} on {DEVICE} ({DTYPE})...")
    model = load_model()

    # small workload so stage 1 runs quick; bimodal = the interesting length mix
    reqs = make_workload(n=32, pattern="bimodal", seed=0)
    run_sequential(model, reqs)

"""
Stage 1: the NAIVE SEQUENTIAL baseline - one request at a time, each with its
own KV cache. No batching => the throughput FLOOR every later stage must beat.

The decode loop is hand-written (not model.generate) because the whole project
is about owning the prefill/decode loop and the KV cache. Prompt content is
irrelevant to throughput - only the SHAPES matter - so we fake prompts with
random token ids of the right length.
"""
import time
import torch

from common import DEVICE, warmup


@torch.inference_mode()
def run_request(model, prompt_len, output_len, vocab_size, max_ctx):
    """Prefill a fake prompt, then greedily decode output_len tokens reusing the
    KV cache. Returns tokens generated.

    Prefill runs the whole prompt ONCE and stashes its keys/values; each decode
    step then feeds just the ONE new token + the cached past, so attention over
    the prompt is never recomputed.

    GPT-2 has only 1024 position embeddings, so prompt+output must fit max_ctx or
    the position index runs off the table (device-side assert)."""
    prompt_len = min(prompt_len, max_ctx - 1)
    output_len = min(output_len, max_ctx - prompt_len)

    input_ids = torch.randint(0, vocab_size, (1, prompt_len), device=DEVICE)
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = 0
    for _ in range(output_len):
        out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated += 1
    return generated


def run_sequential(model, reqs):
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions
    warmup(model)

    eff = [(min(r.prompt_len, max_ctx - 1),
            min(r.output_len, max_ctx - min(r.prompt_len, max_ctx - 1))) for r in reqs]
    total_prompt = sum(p for p, _ in eff)
    total_decode = 0

    t0 = time.perf_counter()
    for r in reqs:
        total_decode += run_request(model, r.prompt_len, r.output_len, vocab_size, max_ctx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"\n=== naive sequential ({len(reqs)} requests) ===")
    print(f"  prompt tokens prefilled: {total_prompt:,}")
    print(f"  tokens decoded:          {total_decode:,}")
    print(f"  wall time:               {elapsed:.2f} s")
    print(f"  decode throughput:       {total_decode / elapsed:,.1f} tok/s")
    print(f"  end-to-end throughput:   {(total_prompt + total_decode) / elapsed:,.1f} tok/s")
    return total_decode / elapsed

"""
Project 6: profile the serving loop. Where does decode time actually go?

Runs the paged engine under torch.profiler and prints the ops by CUDA time, plus
a prefill-vs-decode split. The triton paged kernel shows up by its compiled name.
Saves a chrome trace (open in chrome://tracing or perfetto).

    python profile_serve.py
"""
import time

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2TokenizerFast

from common import DEVICE
from gpt2 import GPT2
from serve import PagedEngine, Seq


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    texts = ["The capital of France is", "Once upon a time,",
             "The meaning of life is", "In 2050, computers will",
             "My favorite food is", "The stock market is",
             "Deep learning is", "On a cold winter night,"]
    gm = GPT2()

    def fresh():
        return [Seq(i, tok(t, return_tensors="pt").input_ids[0].to(DEVICE), 64)
                for i, t in enumerate(texts)]

    # build the engine ONCE, outside any timed/profiled region - constructing it
    # zeros the whole KV pool, which would otherwise dominate the profile.
    eng = PagedEngine(gm)
    eng.serve(fresh(), max_tokens=8, max_batch=8)   # warmup
    torch.cuda.synchronize()

    # coarse prefill vs decode split
    seqs = fresh()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for s in seqs:
        eng.prefill(s)
    torch.cuda.synchronize(); t_pre = time.perf_counter() - t0
    steps = 0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    active = [s for s in seqs if not s.done]
    while active:
        eng.decode(active)
        steps += 1
        active = [s for s in active if not s.done]
    torch.cuda.synchronize(); t_dec = time.perf_counter() - t0
    for s in seqs:
        eng._free(s)
    print("=== phase split (8 prompts, up to 64 tokens) ===")
    print(f"  prefill (8 prompts):  {t_pre*1000:7.1f} ms")
    print(f"  decode  ({steps} steps):  {t_dec*1000:7.1f} ms   ({t_dec/steps*1000:.2f} ms/step)")
    print(f"  => decode dominates: {t_dec/(t_pre+t_dec)*100:.0f}% of wall time\n")

    # op-level profile of a full serve (engine already built => no pool alloc here)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        eng.serve(fresh(), max_tokens=64, max_batch=8)
        torch.cuda.synchronize()
    print("=== top ops by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    prof.export_chrome_trace("serve_trace.json")
    print("saved serve_trace.json (open in perfetto / chrome://tracing)")


if __name__ == "__main__":
    main()

"""
Generate the figures:
  fig_batching.png   sequential vs static vs continuous (decode throughput)
  fig_kernel.png     triton paged kernel vs vectorized gather (per length mix)
  fig_hf_paged.png   HF gpt2 vs our paged gpt2 (single stream + batched)

    python plots.py
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import triton.testing
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from common import load_model, DEVICE
from workload import make_workload
from sequential import run_sequential
from static_batch import run_static_batch
from continuous import run_continuous_batch
from paged_attention import _build_scenario, paged_attention, reference_batched
from gpt2 import GPT2
from paged_gpt2 import PagedGPT2
from serve import PagedEngine, Seq

GREEN, BLUE, GREY = "#2a9d5a", "#4477bb", "#b0b0b0"


def fig_batching(model):
    reqs = make_workload(n=32, pattern="bimodal", seed=0)
    vals = [run_sequential(model, reqs),
            run_static_batch(model, reqs, 16),
            run_continuous_batch(model, reqs, 16)]
    labels = ["sequential", "static\nbatch", "continuous\nbatch"]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=[GREY, BLUE, GREEN])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
    plt.ylabel("decode throughput (tok/s)")
    plt.title("Batching strategies — bimodal workload (GPT-2, n=32)")
    plt.tight_layout(); plt.savefig("fig_batching.png", dpi=130); plt.close()
    print("saved fig_batching.png")


def fig_kernel():
    scenarios = {
        "uniform\n~64": [64] * 64,
        "uniform\n~256": [256] * 64,
        "high variance\n(8..1000)": [1000 if i % 8 == 0 else 16 for i in range(64)],
    }
    names, tks, tgs = [], [], []
    for name, lengths in scenarios.items():
        q, k, v, bt, sl = _build_scenario(64, 12, 64, 16, 8192, lengths=lengths)
        tks.append(triton.testing.do_bench(lambda: paged_attention(q, k, v, bt, sl)))
        tgs.append(triton.testing.do_bench(lambda: reference_batched(q, k, v, bt, sl)))
        names.append(name)
    x = range(len(names))
    plt.figure(figsize=(6.5, 4))
    plt.bar([i - 0.2 for i in x], tgs, 0.4, label="gather + attention (no kernel)", color=GREY)
    plt.bar([i + 0.2 for i in x], tks, 0.4, label="triton paged kernel", color=GREEN)
    for i, (tg, tk) in enumerate(zip(tgs, tks)):
        plt.text(i + 0.2, tk, f"{tg/tk:.0f}x", ha="center", va="bottom", fontsize=9)
    plt.xticks(list(x), names)
    plt.ylabel("decode-step latency (ms)")
    plt.title("Paged-attention kernel vs gather baseline (64 seqs)")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_kernel.png", dpi=130); plt.close()
    print("saved fig_kernel.png")


def _toks_per_s_ours_single(gm, ids, n=100):
    paged = PagedGPT2(gm)
    paged.generate(ids, max_new=5)                 # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = paged.generate(ids, max_new=n)
    torch.cuda.synchronize()
    return out.shape[1] / (time.perf_counter() - t0)


def _toks_per_s_ours_batched(gm, prompt_ids_list, n=100):
    eng = PagedEngine(gm)
    seqs = [Seq(i, p, n) for i, p in enumerate(prompt_ids_list)]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    outs = eng.serve(seqs, max_tokens=n, max_batch=len(seqs))
    torch.cuda.synchronize()
    return sum(len(o) for o in outs) / (time.perf_counter() - t0)


def _toks_per_s_hf(hf, ids, eos, n=100):
    hf.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=eos)   # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    hf.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=eos)
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def _toks_per_s_hf_batched(hf, tok, texts, n=100):
    # HF's own batched generate: left-pad the prompts + attention mask. This is
    # the fair 8-stream counterpart to our continuous-batching engine.
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    gen = dict(max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    hf.generate(**enc, **{**gen, "max_new_tokens": 5})   # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = hf.generate(**enc, **gen)
    torch.cuda.synchronize()
    new = (out.shape[1] - enc["input_ids"].shape[1]) * len(texts)
    return new / (time.perf_counter() - t0)


def fig_hf_paged():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    texts = ["The capital of France is", "Once upon a time,",
             "The meaning of life is", "In 2050, computers will",
             "My favorite food is", "The stock market",
             "Deep learning is", "On a cold winter night,"]
    ids0 = tok(texts[0], return_tensors="pt").input_ids.to(DEVICE)
    plist = [tok(t, return_tensors="pt").input_ids[0].to(DEVICE) for t in texts]

    gm = GPT2()
    hf = GPT2LMHeadModel.from_pretrained("gpt2", dtype=torch.float32).to(DEVICE).eval()

    vals = [
        _toks_per_s_hf(hf, ids0, tok.eos_token_id),
        _toks_per_s_ours_single(gm, ids0),
        _toks_per_s_hf_batched(hf, tok, texts),
        _toks_per_s_ours_batched(gm, plist),
    ]
    labels = ["HF\n1 stream", "ours paged\n1 stream", "HF\n8 streams", "ours paged\n8 streams"]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, vals, color=[GREY, BLUE, GREY, GREEN])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom")
    plt.ylabel("throughput (tok/s)")
    plt.title("HF gpt2 vs our paged engine (greedy decode, fp32)")
    plt.tight_layout(); plt.savefig("fig_hf_paged.png", dpi=130); plt.close()
    print("saved fig_hf_paged.png")


if __name__ == "__main__":
    model = load_model()
    fig_batching(model)
    fig_kernel()
    fig_hf_paged()

# mini-vllm

A toy **continuous-batching inference server** — built to learn the heart of how
vLLM / SGLang actually work: scheduling, in-flight batching, and KV-cache
management. Not a fast engine; a *legible* one.

Part of my ML-infra learning ladder (after a toy KV-cache engine, reading/running
nanoGPT, and writing a [Triton softmax kernel](https://github.com/a-m-n-s/triton-softmax)).

## The question this answers

When a server gets many requests of **wildly different lengths**, the naive thing
— run them one at a time, or in fixed batches that wait for the slowest — wastes
the GPU. Real engines do **continuous (in-flight) batching**: the moment a
sequence finishes, evict it and admit a waiting one *mid-flight*, keeping every
batch slot busy. This repo builds that scheduler from scratch and measures the win.

## Plan (built in stages, simplest first)

- [x] **Workload generator** (`workload.py`) — synthetic request streams with
  controllable length + arrival distributions (uniform / bimodal / prefill-heavy /
  decode-heavy; offline burst or Poisson trickle). This is what we stress the
  scheduler with.
- [ ] **Scheduler vs a mock model** — continuous-batching loop against a fake model
  (decode = count down `output_len`), so the *scheduling logic* is isolated from
  GPU/model details. Compare tokens/sec vs naive sequential.
- [ ] **Real model + KV cache** — swap the mock for a small HF / nanoGPT model with
  per-sequence KV-cache alloc/free.
- [ ] **Paged KV cache** (PagedAttention-lite) — block-based KV memory to cut
  fragmentation.
- [ ] **Profile** — where the time goes; latency (TTFT, p99) vs throughput.

## Run

```bash
python3 workload.py   # prints a bimodal workload + length histograms
```

(Pure Python so far — no torch needed yet.)

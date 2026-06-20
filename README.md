# mini-vllm

A small LLM serving engine I built to understand how vLLM/SGLang work under the
hood: continuous batching, a paged KV cache, a custom attention kernel, and the
scheduler that holds it together. Runs GPT-2 on an RTX 4080.

It's not meant to be fast. It's meant to be readable. Each part is built up in
its own stage, and the model path is checked against HuggingFace token for token.

Came out of some earlier practice: a toy KV-cache engine, reading nanoGPT, and a
[Triton softmax kernel](https://github.com/a-m-n-s/triton-softmax).

## Files

| file | what |
|------|------|
| `workload.py` | synthetic request streams (length + arrival distributions) |
| `sequential.py` | one request at a time (the baseline) |
| `static_batch.py` | fixed batches; measures the wasted slots |
| `continuous.py` | continuous batching (evict + admit mid-flight) |
| `paged_kv.py` | block pool + block table + allocator |
| `paged_attention.py` | Triton paged-attention kernel + tests/benchmark |
| `gpt2.py` | our own GPT-2 forward (loads HF weights) |
| `paged_gpt2.py` | GPT-2 wired to the paged cache + kernel |
| `serve.py` | the whole thing: batching + paged KV + kernel + EOS |
| `bench.py` `plots.py` `profile_serve.py` | benchmarks, figures, profiling |

## Notes on the ideas

Decode is memory-bound: every step reloads the weights to produce one token, so
batching sequences together is nearly free until you hit the compute knee.
Continuous batching keeps the batch full by swapping finished sequences out and
waiting ones in, instead of stalling on the slowest.

![batching](fig_batching.png)

KV gets stored in fixed blocks instead of one padded tensor per sequence, so
there's no per-sequence over-allocation and no fragmentation. The Triton kernel
reads those scattered blocks directly through a block table, so no gather and no
padding. The gap over a plain gather-and-attend grows with length variance.

![kernel](fig_kernel.png)

To run attention through your own kernel you have to own the forward pass, so we
load GPT-2's weights into our own code (this is also what vLLM does). Output
matches HF greedy decode exactly. Per stream we're about even with HF; at 8
concurrent streams the batching + paging pull ahead of HF's batched `generate`.

![hf vs ours](fig_hf_paged.png)

## Profiling

`profile_serve.py` shows decode is ~90% of wall time, and inside it the matmuls
dominate (the paged kernel is only a few percent). The loop is actually CPU-bound
— the GPU spends a lot of time waiting on Python. That's the usual reason real
engines reach for CUDA graphs and fused kernels.

## Running it

```bash
pip install torch triton transformers matplotlib
python bench.py            # sequential vs static vs continuous
python paged_attention.py  # kernel correctness + benchmark
python serve.py            # serve some prompts
python plots.py            # figures
python profile_serve.py    # profile
```

Needs a CUDA GPU. I run it in WSL2 on a 4080.

## Caveats

fp32, single GPU, greedy decode, and only the decode path is paged (prefill uses
plain attention). The goal was to build and understand the machinery, not to beat
a real engine — but the pieces are the same ones.

"""Shared bits: model loading, device/dtype, and a warmup so the first CUDA
kernel-load doesn't pollute timing."""
import torch
from transformers import GPT2LMHeadModel

MODEL_NAME = "gpt2"          # 124M; small enough to iterate fast on the 4080
DEVICE = "cuda"
DTYPE = torch.float16


def load_model():
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, dtype=DTYPE)
    model.to(DEVICE).eval()
    return model


@torch.inference_mode()
def warmup(model):
    """One tiny prefill + a few decode steps. First CUDA calls compile/load
    kernels; running this before timing keeps that cost out of the measurement."""
    ids = torch.randint(0, model.config.vocab_size, (1, 16), device=DEVICE)
    out = model(input_ids=ids, use_cache=True)
    tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    for _ in range(4):
        out = model(input_ids=tok, past_key_values=out.past_key_values, use_cache=True)
        tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()

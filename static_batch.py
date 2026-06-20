"""
Stage 2: STATIC batching. Process requests in fixed groups of batch_size:
left-pad the prompts so every sequence's last real token lines up, prefill the
group together, then decode in lockstep until the SLOWEST sequence finishes.

Exposes the core inefficiency: the batch is a fixed-width tensor. When a short
sequence hits its output_len its slot can't be handed to a waiting request - it
rides along (masked, position frozen) burning compute until the longest sequence
in the group is done. We count those wasted slot-steps. That waste is exactly
what Stage 3 (continuous batching) reclaims.
"""
import time
import torch

from common import DEVICE, warmup


@torch.inference_mode()
def run_static_batch(model, reqs, batch_size=16):
    vocab_size = model.config.vocab_size
    max_ctx = model.config.n_positions
    warmup(model)

    total_prompt = 0
    useful_tokens = 0      # decode tokens a request actually needed
    slot_steps = 0         # batch slots consumed across all decode steps (incl. waste)

    t0 = time.perf_counter()
    for start in range(0, len(reqs), batch_size):
        group = reqs[start:start + batch_size]
        B = len(group)
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
        # left-padding breaks default 0..L numbering, so build positions by hand
        pos = (attn.cumsum(-1) - 1).clamp(min=0)

        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        cur_pos = torch.tensor(plens, device=DEVICE)
        gen = [0] * B
        done = [o == 0 for o in olens]
        D = max(olens)

        for _ in range(D):
            if all(done):
                break
            new_col = torch.tensor([[0 if done[i] else 1] for i in range(B)], device=DEVICE)
            attn = torch.cat([attn, new_col], dim=1)
            step_pos = cur_pos.clamp(max=max_ctx - 1).view(B, 1)
            out = model(input_ids=next_tok, past_key_values=past,
                        attention_mask=attn, position_ids=step_pos, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            for i in range(B):
                slot_steps += 1
                if not done[i]:
                    gen[i] += 1
                    useful_tokens += 1
                    cur_pos[i] += 1
                    if gen[i] >= olens[i]:
                        done[i] = True

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

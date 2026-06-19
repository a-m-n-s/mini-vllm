"""
Synthetic workload generator for the mini serving engine (project 4).

A "request" to an inference server, reduced to what the SYSTEM cares about, is
just three numbers: how much to prefill, how much to decode, and when it shows
up. Content is irrelevant to scheduling/memory behavior - only the shapes are.

This file has no torch/model deps on purpose - it's pure python. The point is to
generate streams of requests with CONTROLLABLE length + arrival distributions so
later I can deliberately stress the scheduler (esp. length variance, which is
where continuous batching earns its keep).

run: python3 workload.py
"""

from dataclasses import dataclass
import random


@dataclass
class Request:
    req_id: int
    prompt_len: int        # tokens to prefill (compute + KV memory up front)
    output_len: int        # tokens to decode (how long it holds a batch slot)
    arrival_time: float    # when it enters the queue (offline => 0.0)

    @property
    def total_len(self):
        return self.prompt_len + self.output_len


# length "shapes" -> (prompt_range, output_range). easy to add more later.
# bimodal is the interesting one: a pile of short requests + a few long ones,
# so short ones finish fast and free their slot while a long one is still going.
PATTERNS = {
    "uniform":  {"short": (16, 512, 16, 256)},                  # one bucket, wide
    "bimodal":  {"short": (8, 64, 8, 32), "long": (256, 1024, 128, 512)},
    "prefill_heavy": {"short": (512, 2048, 1, 8)},              # long prompt, tiny gen (classify/RAG)
    "decode_heavy":  {"short": (8, 32, 256, 1024)},             # short prompt, long gen (chat/story)
}


def make_workload(
    n=64,
    pattern="bimodal",
    long_frac=0.3,        # for bimodal: fraction drawn from the "long" bucket
    arrival="offline",    # "offline" (all at t=0) or "poisson" (online trickle)
    rate=20.0,            # poisson arrivals per second (only if arrival="poisson")
    seed=0,
):
    rng = random.Random(seed)
    spec = PATTERNS[pattern]
    reqs = []
    t = 0.0
    for i in range(n):
        # pick which bucket this request is drawn from
        if "long" in spec and rng.random() < long_frac:
            pmin, pmax, omin, omax = spec["long"]
        else:
            pmin, pmax, omin, omax = spec["short"]
        prompt_len = rng.randint(pmin, pmax)
        output_len = rng.randint(omin, omax)

        if arrival == "offline":
            arrival_time = 0.0
        elif arrival == "poisson":
            # gaps between arrivals are exponential -> Poisson process
            t += rng.expovariate(rate)
            arrival_time = t
        else:
            raise ValueError(arrival)

        reqs.append(Request(i, prompt_len, output_len, arrival_time))
    return reqs


def _hist(values, bins=8, width=40):
    """tiny ascii histogram so I can SEE the distribution."""
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    peak = max(counts) or 1
    for b in range(bins):
        edge = int(lo + b * step)
        bar = "#" * int(width * counts[b] / peak)
        print(f"  {edge:>6} | {bar} {counts[b]}")


def summarize(reqs):
    plens = [r.prompt_len for r in reqs]
    olens = [r.output_len for r in reqs]
    print(f"{len(reqs)} requests")
    print(f"  prompt_len: min={min(plens)} max={max(plens)} mean={sum(plens)//len(plens)}")
    print(f"  output_len: min={min(olens)} max={max(olens)} mean={sum(olens)//len(olens)}")
    print(f"  total tokens to process: {sum(r.total_len for r in reqs):,}")
    print("\nprompt_len distribution:")
    _hist(plens)
    print("output_len distribution:")
    _hist(olens)


if __name__ == "__main__":
    print("=== bimodal (the interesting one) ===")
    reqs = make_workload(n=64, pattern="bimodal", seed=0)
    summarize(reqs)
    print("\nfirst 5 requests:")
    for r in reqs[:5]:
        print(" ", r)

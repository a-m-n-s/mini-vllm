"""Run the batching stages head-to-head on one synthetic workload.

    python bench.py

Absolute tok/s wander a bit run-to-run (laptop GPU thermals/contention); the
RELATIVE ordering (continuous > static > sequential) is the trustworthy signal.
"""
from common import load_model, MODEL_NAME, DEVICE, DTYPE
from workload import make_workload
from sequential import run_sequential
from static_batch import run_static_batch
from continuous import run_continuous_batch


if __name__ == "__main__":
    print(f"loading {MODEL_NAME} on {DEVICE} ({DTYPE})...")
    model = load_model()

    reqs = make_workload(n=32, pattern="bimodal", seed=0)
    run_sequential(model, reqs)
    run_static_batch(model, reqs, batch_size=16)
    run_continuous_batch(model, reqs, max_batch=16)

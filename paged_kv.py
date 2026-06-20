"""
Stage 5a: the PAGED KV memory layout (vLLM's key idea), minus the kernel.

Instead of one contiguous KV tensor per sequence (which forces padding to the
longest sequence and fragments as sequences come and go), chop KV memory into
fixed-size BLOCKS in a shared pool. Each sequence gets a BLOCK TABLE: a list of
physical block ids it owns. Grow a sequence => grab a free block. Finish =>
return its blocks to the pool.

Two properties this buys (the self-test below measures both):
  * NO external fragmentation - any free block serves any sequence, so you never
    hit "enough free memory total, but not contiguous". A contiguous allocator
    does.
  * BOUNDED internal waste - only each sequence's LAST block is partly empty
    (< block_size tokens wasted), instead of padding every short sequence out to
    the longest one's length (Stage 3's cost).

The actual attention over these scattered blocks needs a custom kernel (no
gather, no padding) - that's paged_attention.py (Stage 5b).

    python paged_kv.py   # runs the self-test / demo
"""
import math
import torch

from common import DEVICE, DTYPE


class BlockManager:
    def __init__(self, num_blocks, block_size, num_heads, head_dim,
                 device=DEVICE, dtype=DTYPE):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # the pool: ONE big tensor of K and one of V, indexed by physical block id.
        # layout (num_blocks, num_heads, block_size, head_dim) - what the kernel reads.
        shape = (num_blocks, num_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)

        self.free = list(range(num_blocks))   # stack of free physical block ids
        self.block_table = {}                  # seq_id -> [physical block ids]
        self.length = {}                       # seq_id -> tokens currently stored

    # --- capacity ---
    def blocks_for(self, n_tokens):
        return math.ceil(n_tokens / self.block_size)

    def can_allocate(self, n_tokens):
        return self.blocks_for(n_tokens) <= len(self.free)

    def used_blocks(self):
        return self.num_blocks - len(self.free)

    # --- allocate / grow / free ---
    def allocate(self, seq_id, n_tokens):
        """Reserve blocks for a sequence's initial prompt of n_tokens."""
        need = self.blocks_for(n_tokens)
        if need > len(self.free):
            raise MemoryError(f"OOM: need {need} blocks, {len(self.free)} free")
        blocks = [self.free.pop() for _ in range(need)]   # any free blocks - not contiguous
        self.block_table[seq_id] = blocks
        self.length[seq_id] = n_tokens

    def append_token(self, seq_id):
        """One more token for this sequence; grab a new block if it crossed a
        block boundary. Returns (physical_block, offset) where the token lands."""
        self.length[seq_id] += 1
        need = self.blocks_for(self.length[seq_id])
        table = self.block_table[seq_id]
        if need > len(table):
            if not self.free:
                raise MemoryError("OOM on append")
            table.append(self.free.pop())
        idx = self.length[seq_id] - 1
        return table[idx // self.block_size], idx % self.block_size

    def free_seq(self, seq_id):
        """Return all of a finished sequence's blocks to the pool."""
        self.free.extend(self.block_table.pop(seq_id))
        self.length.pop(seq_id)

    # --- waste accounting ---
    def internal_waste(self):
        """Empty token-slots sitting in partially-filled last blocks."""
        used_slots = self.used_blocks() * self.block_size
        real_tokens = sum(self.length.values())
        return used_slots - real_tokens


# --------------------------------------------------------------------------
# self-test / demo
# --------------------------------------------------------------------------
def _demo():
    from workload import make_workload

    BLOCK = 16
    HEADS, DIM = 12, 64           # gpt2-small
    MAX_CTX = 1024

    reqs = make_workload(n=32, pattern="bimodal", seed=0)
    lens = [min(r.prompt_len, MAX_CTX - 1) +
            min(r.output_len, MAX_CTX - min(r.prompt_len, MAX_CTX - 1)) for r in reqs]

    bm = BlockManager(num_blocks=2048, block_size=BLOCK, num_heads=HEADS, head_dim=DIM)
    for i, n in enumerate(lens):
        bm.allocate(i, n)

    real = sum(lens)
    paged_slots = bm.used_blocks() * BLOCK
    pad_to_max = len(lens) * max(lens)          # what "pad every seq to the longest" costs

    print("=== paged KV memory accounting (bimodal n=32) ===")
    print(f"  block_size:            {BLOCK} tokens")
    print(f"  real tokens to store:  {real:,}")
    print(f"  blocks used:           {bm.used_blocks()} ({paged_slots:,} token-slots)")
    print(f"  internal waste (paged): {bm.internal_waste():,} slots "
          f"({bm.internal_waste()/paged_slots*100:.1f}%) - only last-block remainders")
    print(f"  pad-to-longest would use: {pad_to_max:,} slots "
          f"({pad_to_max/real:.1f}x the real tokens)")
    print(f"  => paged stores {real:,} tokens in {paged_slots:,} slots; "
          f"padding would need {pad_to_max:,}. {pad_to_max/paged_slots:.1f}x less memory.")

    # fragmentation property: free a few scattered sequences, then allocate a new
    # one bigger than any single freed run - paged succeeds because blocks pool.
    print("\n=== no external fragmentation ===")
    before = bm.used_blocks()
    for sid in (3, 10, 20):       # free three sequences scattered through the pool
        bm.free_seq(sid)
    freed = before - bm.used_blocks()
    print(f"  freed 3 scattered sequences -> {freed} blocks back in the pool (non-contiguous)")
    big = max(lens) + 200
    print(f"  allocating a NEW seq of {big} tokens ({bm.blocks_for(big)} blocks)... ", end="")
    bm.allocate(999, big)         # uses scattered free blocks; a contiguous allocator might fail
    print("OK - reused the scattered free blocks, no compaction needed")


if __name__ == "__main__":
    _demo()

# Double Buffering

**Double buffering allocates two staging buffers so the K-loop can load tile
`l+1` while computing on tile `l`: buying overlap with
[threadgroup memory](../machine/threadgroup-memory.md), and on this platform,
buying it through ILP rather than DMA.**

CUDA equivalent: the multi-stage `cp.async` pipeline of every modern CUDA GEMM.
The crucial local difference: [the DMA engine is not available](../metal/simdgroup-async-copy.md),
so there's no true copy/compute overlap; the *same threads* issue both the loads
and the math. What makes it work anyway: the prefetch loads and the current tile's
MMAs have no data dependence, so the hardware scheduler interleaves them,
[hiding threadgroup-memory latency with instruction-level parallelism](../machine/occupancy.md).
Software pipelining, not hardware pipelining.

The structural delta over the single-buffered loop, from the
[case study](../kernels/gemm-double-buffered.md): double the buffers, prefetch
before compute, one barrier per iteration.

```metal
  threadgroup float A_tg[2][BM * BK];
  threadgroup float B_tg[2][BK * BN];
  ...
  for (uint l = 0; l < k_tiles; l++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort nxt = ushort(1) - cur;
    if (l + 1 < k_tiles) {
      load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg[nxt], tid);
      load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg[nxt], tid);
    }
    /* ...compute on A_tg[cur], B_tg[cur]... */
    cur = nxt;
  }
```
— [m5-gemm `sync_copy_db.metal:87-126`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy_db.metal#L87-L126), abridged

**The cost**: 2× threadgroup-memory footprint → fewer resident threadgroups →
less [occupancy](../machine/occupancy.md)-based latency hiding. Double buffering
trades one hiding mechanism for another, and the measured
[benchmark table](../kernels/gemm-double-buffered.md) shows the trade's sign
flipping with problem size: **3× faster at 1024²** (launch and DRAM latency
dominate; the prefetch pipeline hides them), **slightly slower at 4096²** (the
simpler kernel's smaller loop unrolls better, so compiler quality beats
algorithm), **converged at 8192²** (everything is
[bandwidth bound](arithmetic-intensity.md); scheduling stops mattering).

That non-monotonic result generalizes into this glossary's standing benchmark
lesson: *there is no single fastest kernel*. Production libraries
[dispatch different variants per shape](../mlx/how-an-op-becomes-a-kernel.md)
precisely because techniques like this one win in regimes, not universally.

Next: [Fusion and epilogues](fusion-and-epilogues.md)

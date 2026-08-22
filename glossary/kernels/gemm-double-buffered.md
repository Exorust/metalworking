# Case Study: The Double-Buffered GEMM

**`sync_copy_db.metal` — the [tiled GEMM](gemm-tiled.md) with
[double buffering](../techniques/double-buffering.md) added, and the cleanest
measured demonstration in this glossary that *there is no single fastest kernel*.**

The delta from its sibling is exactly three structural changes — two buffer sets,
a prologue prefetch, one barrier per iteration instead of two:

```metal
  threadgroup float A_tg[2][BM * BK];
  threadgroup float B_tg[2][BK * BN];
  ...
  // Prologue: load tile 0 into buffer 0.
  load_tile<BM, BK, NTHREADS>(A + tg_row * k, k, A_tg[0], tid);
  load_tile<BK, BN, NTHREADS>(B + tg_col,     m, B_tg[0], tid);

  for (uint l = 0; l < k_tiles; l++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort nxt = ushort(1) - cur;
    if (l + 1 < k_tiles) {
      load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg[nxt], tid);
      load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg[nxt], tid);
    }
    /* compute on A_tg[cur] / B_tg[cur] */
    cur = nxt;
  }
```
— [`sync_copy_db.metal:87-126`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy_db.metal#L87-L126), abridged

The single barrier is doing double duty — "the prefetch landed" and "everyone's
done with the buffer we're about to overwrite" are the same condition when the
buffers alternate. The kernel's own header comment is honest about the mechanism:
without [the dead async-DMA primitive](gemm-async-ghost.md) this is not true
overlap — it's [ILP-based latency hiding](../techniques/double-buffering.md), the
same threads issuing independent loads and math for the scheduler to interleave.

**The payoff table** (author's numbers, M5 Max, fp32, best of 3×5 — from the
[repo README](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/README.md)):

| Size | sync_copy | sync_copy_db | MPS |
|---:|---:|---:|---:|
| 1024² | 3.3 TF | **10.7 TF** | 2.9 TF |
| 2048² | 7.7 TF | **9.0 TF** | 5.2 TF |
| 4096² | **13.5 TF** | 13.1 TF | 11.7 TF |
| 8192² | 13.0 TF | 12.7 TF | **13.5 TF** |

Three regimes, three winners: at 1024² launch and DRAM-fetch latency dominate and
the prefetch pipeline hides them (**3× over both** alternatives — and note this is
the regime [LLM decode](../techniques/decode-vs-prefill.md) lives in); at 4096²
the simpler kernel wins because its smaller loop
[unrolls better](../metal/compilation-pipeline.md) — compiler quality beating
algorithm; at 8192² everything hits the
[bandwidth wall](../techniques/arithmetic-intensity.md) and
[MPS](../metal/mps.md) converges with the hand-written kernels.

The README's "things that did not help" section is required reading as
[methodology](../metal/profiling.md): offline `-O3 -ffast-math` compilation
(bit-identical to runtime), bigger simdgroup tiles
([10× slower — spills](../machine/registers.md)), `SW=3` threadgroups. Negative
results, measured and published — the norm this glossary's
[war stories](../war-stories/the-failures.md) wish the whole ecosystem followed.

Next: [The async-copy ghost](gemm-async-ghost.md)

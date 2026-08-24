# Case Study: The Tiled GEMM

**`sync_copy.metal` from [yaroslavvb/m5-gemm](https://github.com/yaroslavvb/m5-gemm)
(BSD-3-Clause): ~140 lines of MSL that beat [MPS](../metal/mps.md) at fp32
matmul, 13.5 vs 11.7 TFLOPS at 4096² on an M5 Max. The single best first kernel to
read on this platform, because every technique it uses has a page in this
glossary.**

Lineage: a Metal-4 port of Zeke Medley's
[metal-matmul](https://github.com/0xekez/metal-matmul) and its essay
[Fast Multidimensional Matrix Multiplication on Apple GPU](https://percisely.xyz/gemm),
with the [dead async-copy intrinsic](gemm-async-ghost.md) replaced by a
[cooperative load](../techniques/cooperative-load.md).

**The shape, fixed at compile time.** Three `-D` constants define the whole
[tiling pyramid](../techniques/tiling.md). With defaults: 64×64 output per
threadgroup, 32×32 per simdgroup, K consumed 16 at a time:

```metal
constant constexpr ushort BM = SW * SIMD_TILE * 8;  // threadgroup output rows
constant constexpr ushort BN = SW * SIMD_TILE * 8;  // threadgroup output cols
constant constexpr ushort BK = TILE_K * 8;          // reduction tile
```
— [`sync_copy.metal:18-20`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L18-L20)

**The most consequential line** is an attribute:

```metal
kernel void __attribute__((max_total_threads_per_threadgroup(SW * SW * 32)))
matmul(
```
— [`sync_copy.metal:84-85`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L84-L85)

The author calls it the single biggest practical win: it lets the
[register allocator](../machine/registers.md) commit to 128 threads and stop
spilling the accumulators.

**The K-loop** is the [tiling](../techniques/tiling.md) rhythm at its cleanest.
Stage cooperatively, [barrier](../metal/synchronization.md), multiply from
registers, barrier, advance:

```metal
  for (uint l = 0; l < k_tiles; l++) {
    uint k_off = l * BK;
    load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg, tid_in_tg);
    load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_tg, B_tg, simd_origin + ushort2(i * 8, j * 8), acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
```
— [`sync_copy.metal:113-128`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L113-L128), abridged

`load_tile` is [the reference cooperative load](../techniques/cooperative-load.md);
the `acc` grid is [register blocking](../techniques/register-blocking.md) in its
simplest form; `simdgroup_multiply_tile` is a straight
[`simdgroup_matrix`](../metal/simdgroup-matrix.md) fragment loop.

**The epilogue** computes the full BLAS `C = α·AB + β·C` fused into the single
store ([`sync_copy.metal:133-143`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L133-L143)):
[fusion](../techniques/fusion-and-epilogues.md) in miniature, and the exact spot
[steel generalizes into pluggable epilogues](steel-gemm-fused.md).

Run it yourself (Apple Silicon, no Xcode needed thanks to
[runtime compilation](../metal/compilation-pipeline.md)): `python matmul.py --dim
4096 --trials 5` in the repo benchmarks both kernels against MPS, and
`bandwidth.py` gives you [your machine's roofline](../techniques/roofline.md).
The numbers cited here are the author's, from a 40-core M5 Max; expect different
crossovers on different tiers.

Next: [The double-buffered GEMM](gemm-double-buffered.md)

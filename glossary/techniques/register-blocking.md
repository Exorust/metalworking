# Register Blocking

**Register blocking keeps each [simdgroup's](../machine/simdgroup.md) patch of the
output resident in [registers](../machine/registers.md) for the kernel's entire
lifetime: accumulate across every K step, write to memory exactly once.**

CUDA equivalent: the accumulator fragments of any warp-tiled GEMM. The difference
is proportion: with [208 KB of registers against 32 KB of threadgroup
memory](../machine/registers.md), Apple kernels push *more* of the working set
into registers than CUDA-typical, and the technique's failure mode (spilling)
is correspondingly more catastrophic.

The pattern in its simplest real form, from the
[GEMM case study](../kernels/gemm-tiled.md): each simdgroup owns a 4×4 grid of
[`simdgroup_float8x8`](../metal/simdgroup-matrix.md) accumulators, a 32×32 patch
of 1024 floats in registers,

```metal
  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i = 0; i < SIMD_TILE; i++)
    for (ushort j = 0; j < SIMD_TILE; j++)
      acc[i][j] = simdgroup_float8x8(0);
```
— [m5-gemm `sync_copy.metal:104-107`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L104-L107)

accumulated into on every K iteration, stored once in the epilogue. Steel's
[`BlockMMA`](../kernels/steel-blockmma.md) is the same idea as a template: a
`TM × TN` fragment grid whose size falls out of `BM/BN` and the simdgroup layout.

The sizing tension, and the two measured cliff edges:

- **Too small**: each fragment of A and B loaded from threadgroup memory feeds
  few multiplies, so you're
  [bandwidth bound on threadgroup memory](../machine/threadgroup-memory.md)
  instead of DRAM. Bigger accumulator grids amortize every fragment load across
  more MMAs.
- **Too big**: the allocator spills, and the measured cost was
  [**10× slower**, not 10%](../machine/registers.md) (`SIMD_TILE` 4 → 8 in the
  case study). The cliff is sharp because spilled accumulators turn every MMA's
  operand into a memory round-trip.

Which is why the two register-pressure levers appear in every serious kernel:
[`max_total_threads_per_threadgroup`](../metal/function-constants.md) so the
allocator knows the real thread count (the case-study author's "single biggest
practical win"), and [16-bit fragments](../machine/f16.md) to halve the budget
spent. [metal-flash-attention](../kernels/mfa-codegen.md) stakes out the extreme
position: it sometimes *chooses* tile shapes that spill, on the theory that a
predictable spill of the right operand beats a smaller tile. It's the boldest
register-pressure bet in the case studies, and evidence the cliff edge is worth
mapping precisely.

Next: [Double buffering](double-buffering.md)

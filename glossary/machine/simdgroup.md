# Simdgroup

**A simdgroup is 32 threads executing in lockstep on one [GPU core](gpu-core.md) —
Metal's unit of SIMD execution and the level at which shuffles, reductions, and
matrix operations happen.**

CUDA equivalent: the warp, exactly. Same width (32), same lockstep model, same role
as the granularity of divergence and of register-level data exchange. Your warp
intuition transfers whole.

What Metal calls things:

| CUDA | Metal |
|---|---|
| warp | simdgroup |
| lane | `[[thread_index_in_simdgroup]]` |
| `__shfl_sync` and friends | `simd_shuffle`, `simd_shuffle_xor`, ... |
| warp-level reduction | `simd_sum`, `simd_max`, ... |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::...)` |
| warp-level MMA (`mma.sync`) | [`simdgroup_matrix`](../metal/simdgroup-matrix.md) |

Simdgroup shuffles are notably fast on this hardware — 256 B/cycle of shuffle
bandwidth per core, double what contemporary NVIDIA and AMD parts move — which is
why you'll see kernels reach for `simd_shuffle_xor` reductions without hesitation.
Stage-of-the-art attention kernels do their online-softmax row reductions this way
(the [metal-flash-attention](../kernels/mfa-codegen.md) generated code reduces a row
maximum with two `simd_shuffle_xor` calls).

Threadgroups are built from simdgroups: a threadgroup of 128 threads is 4
simdgroups, identified by `[[simdgroup_index_in_threadgroup]]`. High-performance
kernels are usually *designed* at simdgroup granularity — in the
[GEMM case studies](../kernels/gemm-tiled.md), each simdgroup owns a 32×32 patch of
the output, and the threadgroup exists mostly to share a staging buffer in
[threadgroup memory](threadgroup-memory.md).

One difference from CUDA worth knowing: there is no Independent Thread Scheduling
story here, and no `__syncwarp` masks to reason about. The model is the classic
lockstep one. Divergence within a simdgroup costs you the usual way — both sides of
the branch execute — and specialization is instead pushed to compile time via
[function constants](../metal/function-constants.md), which delete branches from
the pipeline before it ever runs.

Next: [Registers](registers.md)

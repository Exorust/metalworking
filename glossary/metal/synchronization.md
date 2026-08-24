# Synchronization

**Inside a kernel, Metal has the same two sync levels as CUDA (threadgroup
barriers and simdgroup barriers) and one inverted cost that changes how you
design: barriers are nearly free.**

The vocabulary:

| CUDA | Metal |
|---|---|
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__syncthreads()` (execution-only) | `threadgroup_barrier(mem_flags::mem_none)` |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_none)` |
| `__threadfence()` | `threadgroup_barrier(mem_flags::mem_device)` |
| grid-wide sync | none; end the dispatch (same as CUDA without cooperative launch) |

The `mem_flags` argument states which memory the barrier orders (threadgroup
memory, device memory, or none for pure execution sync). The
[case-study kernels](../kernels/gemm-tiled.md) pick flags precisely; it documents
intent even where the hardware wouldn't care.

**The inverted cost: ~2 cycles per threadgroup barrier.** On NVIDIA,
`__syncthreads()` is expensive enough that avoiding it is a recognized
optimization genre (warp-synchronous programming exists because of it). Here the
[measured cost](https://github.com/philipturner/metal-benchmarks) is noise. The
[GEMM K-loop](../kernels/gemm-tiled.md) barriers twice per iteration without
consequence; the [steel attention kernel](../kernels/steel-attention.md) barriers
between every stage-load. Design takeaway: choose the algorithm with the cleaner
staging pattern, even if it syncs twice as often. The sync is not where the time
goes. (Where *does* it go?
[Scattered threadgroup-memory access](../machine/threadgroup-memory.md)
and [register spills](../machine/registers.md).)

`simdgroup_barrier` is cheaper still and mostly appears as a scheduling hint:
[steel's `BlockMMA`](../kernels/steel-blockmma.md) drops `mem_none` simdgroup
barriers between fragment loads to keep a simdgroup's loads batched, a
compiler-scheduling nudge rather than a correctness need.

Between kernels, ordering is structural rather than explicit: dispatches in one
[command buffer](command-buffers.md) on one queue execute in order (Metal tracks
buffer hazards; MLX additionally manages explicit fences/events across its
streams). The CUDA habit of sprinkling `cudaDeviceSynchronize` while debugging
maps to `waitUntilCompleted`, and carries the same "why is everything suddenly
slow" trap, [amplified on this platform](command-buffers.md).

Next: [simdgroup_matrix](simdgroup-matrix.md)

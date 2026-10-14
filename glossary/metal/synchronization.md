# Synchronization

**Inside a kernel, Metal has the same two sync levels as CUDA (threadgroup
barriers and simdgroup barriers) and one inverted cost that changes how you
design: barriers are nearly free.**

The vocabulary:

| CUDA | Metal |
|---|---|
| `__syncthreads()` (orders shared and global memory) | `threadgroup_barrier(mem_flags::mem_threadgroup \| mem_flags::mem_device)`; `mem_threadgroup` alone when only threadgroup memory is shared |
| `__syncwarp()` | `simdgroup_barrier(mem_flags::mem_threadgroup)` (or `mem_device` for device memory); `mem_none` is an execution barrier only, with no memory fence |
| `__threadfence_block()` / `__threadfence()` | `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup)` / `thread_scope_device` (MSL 3.2+); before that, `threadgroup_barrier(mem_flags::mem_device)` approximately, which also waits for every thread |
| grid-wide sync | none; end the dispatch (same as CUDA without cooperative launch) |

The `mem_flags` argument states which memory the barrier orders (threadgroup
memory, device memory, or none for pure execution sync); from MSL 4.1 a barrier
also takes a `thread_scope`, so `thread_scope_device` makes writes visible to
other threadgroups, though there is still no grid-wide barrier to wait on. The
[case-study kernels](../kernels/gemm-tiled.md) pick flags precisely; it documents
intent even where the hardware wouldn't care. Two spec rules: if any thread
executes a barrier inside a conditional, every thread in the group must (a
barrier under `if (i < n)` is undefined behavior, not a guaranteed hang), and on
Apple silicon a thread that has already returned no longer takes part in or
blocks a barrier, so an early `return` for out-of-range threads is safe.

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

Between kernels, on a classic `MTLCommandQueue` with the default serial compute
encoder and tracked resources, ordering is structural rather than explicit:
dispatches in one [command buffer](command-buffers.md) execute in order and
Metal tracks hazards on buffers bound directly to the encoder. A concurrent
encoder (`MTLComputePassDescriptor.dispatchType = .concurrent`), untracked or
heap-allocated resources, and every Metal 4 `MTL4CommandQueue` submission (where
Apple "considers all resources untracked" and dispatches run concurrently)
remove that guarantee, and the code must place
[barriers, fences or events](https://developer.apple.com/documentation/metal/resource-synchronization)
itself; MLX opts out this way for speed and manages explicit fences/events
across its streams. The CUDA habit of sprinkling `cudaDeviceSynchronize` while debugging
maps to `waitUntilCompleted`, and carries the same "why is everything suddenly
slow" trap, [amplified on this platform](command-buffers.md).

Next: [simdgroup_matrix](simdgroup-matrix.md)

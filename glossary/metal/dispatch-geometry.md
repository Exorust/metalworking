# Dispatch Geometry

**A Metal dispatch launches a grid of [threadgroups](../machine/threadgroup-memory.md)
of threads — CUDA's grid/block/thread hierarchy under different names, with two
API-level traps for the CUDA-trained.**

The hierarchy, top to bottom:

| CUDA | Metal | Shares |
|---|---|---|
| grid | grid | nothing (independent threadgroups) |
| thread block | threadgroup | [threadgroup memory](../machine/threadgroup-memory.md), [barriers](synchronization.md) |
| warp | [simdgroup](../machine/simdgroup.md) | registers via shuffle |
| thread | thread | its own [registers](../machine/registers.md) |

Trap one: **two dispatch calls with different units.**
`dispatchThreadgroups(gridSize, threadsPerThreadgroup:)` counts the grid in
*threadgroups* — CUDA's `<<<numBlocks, blockDim>>>` exactly. But
`dispatchThreads(gridSize, threadsPerThreadgroup:)` counts the grid in *threads*,
and quietly handles ragged edges by making partial threadgroups (no guard
`if (i < n)` needed — but also no guarantee of full simdgroups, which breaks
kernels that assume cooperative full-width loads). Framework code uses both:
[MLX's custom-kernel API](../mlx/mx-fast.md) exposes `grid` in *threads*. When a
ported kernel reads garbage at the edges or a cooperative load goes wrong, check
which convention the dispatching layer uses before checking anything else.

Trap two: **threadgroup size is a pipeline-time contract, not a launch-time
detail.** The compiler allocates [registers](../machine/registers.md) when it
builds the [pipeline state](compilation-pipeline.md), before it knows your launch
dimensions — so it assumes the maximum unless the kernel promises otherwise with
`max_total_threads_per_threadgroup`. In CUDA, `__launch_bounds__` is a tuning
hint; here its Metal twin is [routinely worth integer factors](../machine/registers.md)
and every production kernel in the [case studies](../kernels/steel-gemm-fused.md)
carries it.

Sizing instincts that transfer: threadgroup sizes are multiples of 32; 128 threads
(4 simdgroups) is the workhorse size in every GEMM and attention kernel this
glossary reads; [occupancy](../machine/occupancy.md) saturates at lower residency
than on an SM, so you rarely chase giant threadgroups. Grid-stride loops exist here
too but are rarer — dispatches are cheap to size exactly, and
[non-uniform threadgroups](dispatch-geometry.md) cover the ragged case.

The index-attribute vocabulary
([`[[thread_position_in_grid]]` and friends](msl.md)) replaces CUDA's
`blockIdx * blockDim + threadIdx` arithmetic — ask for the composed index
directly rather than computing it.

Next: [Compilation pipeline](compilation-pipeline.md)

# Threadgroup Memory

**Threadgroup memory is the on-core scratch space shared by one threadgroup —
Metal's shared memory — capped at 32 KB per threadgroup.**

CUDA equivalent: shared memory. Same purpose (cooperative staging, cross-thread
communication within a block), same declaration style (`threadgroup float
tile[N];` vs `__shared__`), same lifetime. The difference is quantity and role:
you're used to 48-228 KB of configurable shared memory per SM; here it's **32 KB
per threadgroup** (~60 KB physically per [core](gpu-core.md)), against a register
file almost as large as an SM's.

That ratio demotes threadgroup memory from "where tiles live" to "how tiles pass
through." The house pattern, visible in every [GEMM](../kernels/gemm-tiled.md) and
[attention](../kernels/steel-attention.md) kernel in this glossary:

1. A [cooperative load](../techniques/cooperative-load.md) stages a slab of device
   memory into threadgroup memory — coalesced, vectorized, every thread carrying an
   equal share.
2. Each [simdgroup](simdgroup.md) immediately pulls fragments of that slab into
   [registers](registers.md) via `simdgroup_load` and does all real work there.
3. The slab is overwritten by the next one. Nothing lingers.

Doubling your threadgroup-memory footprint halves how many threadgroups can be
resident, which cuts [occupancy](occupancy.md) — the standard CUDA trade, and the
exact cost of [double buffering](../techniques/double-buffering.md), which
allocates two staging buffers to overlap loading with math.

Two behavioral notes that differ from a modern SM:

- **Barriers are almost free.** `threadgroup_barrier` costs ~2 cycles. The CUDA
  reflex of restructuring algorithms to avoid `__syncthreads()` buys nothing here —
  see [synchronization](../metal/synchronization.md).
- **Scattered access is comparatively expensive.** Bank behavior is less forgiving
  than an SM's; the fast pattern is straight-line, contiguous, vectorized access.
  Combined with cheap barriers, this is why Metal kernels look the way they do:
  simple coalesced staging, frequent syncs, no clever scatter tricks.

There is no `cudaFuncAttributePreferredSharedMemoryCarveout` equivalent — the 32 KB
cap is the cap. When a working set won't fit, the answer is almost always "keep more
of it in registers," not "get more shared memory."

Next: [Unified memory](unified-memory.md)

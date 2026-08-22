# Registers

**The register file — ~208 KB per [GPU core](gpu-core.md) — is the largest
programmable memory on the core and the real budget every fast Metal kernel is
written against.**

CUDA equivalent: the SM register file (typically 256 KB). Similar size; wildly
different *ratio*. An SM pairs its registers with up to 100-228 KB of configurable
shared memory. An Apple core pairs nearly the same register file with just
**32 KB of [threadgroup memory](threadgroup-memory.md)** and ~8 KB of L1. So the
CUDA habit — stage big tiles in shared memory, keep register blocking moderate —
inverts: here, threadgroup memory is a thin staging buffer and the serious data
residency happens in registers.

You can see the inversion in every kernel in this glossary's
[case studies](../kernels/gemm-tiled.md): a GEMM simdgroup holds its entire 32×32
output patch — 1024 floats — in [`simdgroup_matrix`](../metal/simdgroup-matrix.md)
accumulators for the kernel's whole lifetime, writing each result to DRAM exactly
once. Flash attention keeps the score tile that naive attention would write to
memory [entirely in registers](../kernels/steel-attention.md).

**Spilling is the #1 performance cliff.** When a kernel asks for more registers than
the file can give its resident threads, the compiler spills to memory, and the cost
is not gentle. The canonical measurement, from the
[m5-gemm](https://github.com/yaroslavvb/m5-gemm) GEMM this glossary walks through:
doubling the per-simdgroup tile from 16 to 64 accumulator matrices ran **10×
slower** — not 10% — because the accumulators spilled. When an Apple kernel is
mysteriously slow, suspect spills first, the way you'd suspect uncoalesced loads
first on NVIDIA.

Two levers control register pressure:

- **[`max_total_threads_per_threadgroup`](../metal/function-constants.md)** — an
  attribute promising the compiler your threadgroup size, so the register allocator
  can commit to a per-thread budget instead of assuming the worst. m5-gemm's author
  calls it the single biggest practical win in the kernel; it is one line.
- **[F16](f16.md)** — half the bytes per value means half the register pressure,
  which raises [occupancy](occupancy.md) and shortens stalls at the same time.

Registers also interact with occupancy in the familiar CUDA way — more registers
per thread means fewer resident simdgroups — but the balance point differs; see
[occupancy](occupancy.md).

Next: [Threadgroup memory](threadgroup-memory.md)

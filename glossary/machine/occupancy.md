# Occupancy

**Occupancy is how many [simdgroups](simdgroup.md) are resident on a
[core](gpu-core.md) at once — the pool of independent work the scheduler hides
latency with.**

CUDA equivalent: occupancy, same word, same mechanism, same limiters (register use
and [threadgroup memory](threadgroup-memory.md) footprint divide the core's fixed
resources among threadgroups). Your mental model transfers; the constants move.

The number to recalibrate on: **the ALUs saturate around 24 resident simdgroups
(~768 threads) per core** — far below the 2048-thread residency you chase on an SM.
Moderate occupancy fills this machine. The practical consequences:

- **Register pressure usually kills occupancy first**, because kernels here do
  register-heavy tiling (see [registers](registers.md)) and the allocator must
  assume the worst threadgroup size unless you promise otherwise with
  [`max_total_threads_per_threadgroup`](../metal/function-constants.md).
- **Low occupancy amplifies stall costs asymmetrically by precision.** At minimum
  occupancy a dependent F32 FMA chain runs ~11.3 cycles/instruction vs ~3.9 for
  [F16](f16.md) — the F16 advantage is largest exactly when you have the least
  parallelism to hide it.
- **Occupancy is a currency, not a goal.** The recurring trade in the case studies:
  [double buffering](../techniques/double-buffering.md) doubles threadgroup-memory
  footprint (halving how many threadgroups fit) to buy instruction-level
  parallelism within each one. Whether that trade wins flips with problem size —
  the [GEMM ladder's benchmark table](../kernels/gemm-double-buffered.md) is the
  cleanest demonstration.

There's no `cudaOccupancyMaxActiveBlocksPerMultiprocessor` here, and
[no profiler](../metal/profiling.md) that reports achieved occupancy directly —
you reason about it from resource arithmetic (threadgroup memory bytes and a
register estimate) and confirm by measuring. One more latency-hiding channel exists
alongside occupancy: ILP within a simdgroup, which is what
[double buffering](../techniques/double-buffering.md) actually exploits — the
hardware happily reorders independent instructions from one instruction stream.

Next: [F16](f16.md)

# GPU Core

**The GPU core is the M-series GPU's unit of compute: it owns a register
file, threadgroup memory, and ALU pipes, and schedules [simdgroups](simdgroup.md)
onto them.**

CUDA equivalent: the [Streaming Multiprocessor](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor).
The mapping is close enough that you can reuse most SM intuition directly.

Chip tiers differ only in how many cores you get and how much bandwidth feeds them: a
base M5 has 10 cores at ~153 GB/s; an M5 Max has 40 cores at ~614 GB/s. There is no
Apple datacenter part. The top of the line is a laptop/desktop chip, and everything
about the design follows from that: Apple runs cores low-clocked and wide, tuning
for performance per watt rather than peak throughput. An H100 will crush any M-series
chip on raw FLOPs; the M-series counter is [unified memory](unified-memory.md)
capacity and efficiency.

Inside a core, the numbers that matter:

- **[Register file](registers.md): ~208 KB**, nearly SM-sized, and the core's real
  working memory.
- **[Threadgroup memory](threadgroup-memory.md): 32 KB** per threadgroup, much
  smaller than CUDA's shared memory. The register/shared ratio is inverted relative
  to what you're used to, and that inversion drives kernel design here.
- **Caches are tiny**: ~8 KB L1 data, ~12 KB instruction per core. Apple spent the
  transistors on registers instead. Plan on explicit reuse, not cache locality.
- **ALU saturation at ~24 resident simdgroups** (~768 threads); see
  [occupancy](occupancy.md). You don't need SM-style 2048-thread residency to fill
  the machine.

The ALU pipes prefer 16-bit: [F16](f16.md) issues with measurably shorter
dependent-instruction stalls than F32. Add the halved register pressure and you
get the local wisdom, "use F16 everywhere". Matrix work runs
through [`simdgroup_matrix`](../metal/simdgroup-matrix.md), an 8×8
tile-multiply primitive closer to a fast wide-FMA arrangement than a separate
tensor-core unit.

None of these numbers come from Apple. The microarchitecture was reverse engineered
by the community, chiefly in
[philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
(MIT), which is this section's primary source and the reference to return to
whenever a kernel underperforms for no visible reason.

Next: [Simdgroup](simdgroup.md)

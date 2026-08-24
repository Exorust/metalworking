# Profiling

**The honest page: Metal has no Nsight Compute, and the gap shapes how performance
work is done on this platform.**

What you have, in decreasing order of usefulness:

**Command-buffer timestamps.** `GPUStartTime` / `GPUEndTime` on a completed
[command buffer](command-buffers.md): wall-clock GPU seconds, programmatic, cheap,
reliable. This is what every benchmark harness in the
[case studies](../kernels/gemm-tiled.md) uses. Granularity is the command buffer,
so isolate what you're measuring into its own buffer, warm up (first execution
includes [pipeline compilation](compilation-pipeline.md)), and take medians.

**Xcode's GPU capture (Metal Debugger).** The closest thing to a real profiler:
per-encoder timings, occupancy estimates, memory-bandwidth counters, shader
profiling. Two structural limits: it's a GUI (no CI, no scripted regression
checks), and capture perturbs heavily on big ML workloads, with inference projects
reporting 50-100× slowdowns and multi-GB traces. Usable for staring at one
dispatch; unusable as a feedback loop.

**Metal 4's counter API: timestamps only.** The hardware counters that would give
you achieved occupancy, stall reasons, or DRAM traffic per kernel are not in the
public API at any version. This is not an oversight being fixed; the API surface
narrowed. (Apple's own job postings for Metal instrumentation engineers suggest
they know.)

So the working methodology, which the [war stories](../war-stories/three-questions.md)
independently converge on:

1. **Timestamps + arithmetic.** Compute achieved GFLOPS and GB/s from problem size
   and measured time; compare against [roofline](../techniques/roofline.md)
   ceilings you measured yourself (a STREAM-style
   [bandwidth probe](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/bandwidth.metal)
   for DRAM; [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
   tables for ALU).
2. **Differential benchmarking.** Can't see stall reasons? Change one thing and
   re-measure. The [m5-gemm README's](../kernels/gemm-double-buffered.md)
   "things that did not help" section is this method producing knowledge.
3. **[Disassembly](compilation-pipeline.md) when arithmetic says impossible.**
   [applegpu](https://github.com/dougallj/applegpu)'s `compiler_explorer.py` shows
   what was actually emitted: the tool of last resort for
   [spill](../machine/registers.md) hunting.
4. **Methodology hygiene**, because the tools won't catch you: fix your clocks
   story (thermals:
   [locking fans changed long-run throughput decay from 50% to 6.7%](../war-stories/cheap-tricks.md)
   in one measured case), report end-to-end numbers next to kernel numbers
   ([synthetic wins that vanish end-to-end are the classic failure](../war-stories/the-failures.md)),
   and state matrix sizes, since
   [the winner flips with size](../kernels/gemm-double-buffered.md).

Next section: [MLX](../mlx/mlx-overview.md)

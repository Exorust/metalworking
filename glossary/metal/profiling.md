# Profiling

**The honest page: Metal has no Nsight Compute, and the gap shapes how performance
work is done on this platform.**

What you have, in decreasing order of usefulness:

**Command-buffer timestamps.** `GPUStartTime` / `GPUEndTime` on a completed
[command buffer](command-buffers.md): wall-clock GPU seconds, programmatic, cheap,
reliable. This is what every benchmark harness in the
[case studies](../kernels/gemm-tiled.md) uses. The coarsest granularity is the
command buffer, so the simplest discipline is to isolate what you're measuring
into its own buffer, warm up (first execution includes
[pipeline compilation](compilation-pipeline.md)), and take medians. For finer
timing, attach a counter sample buffer to a compute pass descriptor (Apple
silicon samples at encoder boundaries) or, in Metal 4, call
`writeTimestamp(granularity:counterHeap:index:)` on the encoder and resolve the
`MTL4CounterHeap`. A sibling pair, `kernelStartTime` / `kernelEndTime`, records
when the CPU driver began and finished scheduling the buffer, which isolates
per-buffer submission cost.

**Xcode's [Metal debugger](https://developer.apple.com/documentation/xcode/metal-debugger)
(a GPU trace taken with the Metal Capture button).** The closest thing to a real
profiler: a Performance timeline with per-encoder timings, measured occupancy
counters plus each shader's maximum theoretical occupancy, memory-bandwidth and
limiter counters, and per-line shader profiling (on M3 and later, a shader cost
graph that breaks each source line into executed instructions by type). Two
structural limits: its counter views are built for Xcode (Apple's `gpudebug`
command-line tool can open a saved `.gputrace` from a script and navigate its
commands, resources and performance section, but nothing streams counters into
CI), and capture perturbs heavily on big ML workloads, with inference projects
reporting 50-100× slowdowns and multi-GB traces. Usable for staring at one
dispatch; unusable as a feedback loop. Instruments surfaces the same counter
families on a live run: record with the Metal System Trace template, set the
Performance Limiters counter set, and you get occupancy, limiter and utilization
counters (ALU, F16, F32, buffer read and write, threadgroup memory, last level
cache) and GPU read and write bandwidth in GB/s, with no capture or replay. A
limiter counts work plus stalls in a subsystem; a utilization counter counts
work alone.

**Metal 4's counter API: timestamps only, through macOS 26.** The hardware
counters that would give you achieved occupancy, stall reasons, or DRAM traffic
per kernel are not in the public *programmatic* API through macOS 26. The GUI
capture does surface them: flash-moe's
[I/O exploration notes](https://github.com/gorroai/flash-moe/blob/4df3af8278c4bef2e7f6b34f61e4e2596b58e93b/docs/io-and-gpu-exploration.md)
pulled a full instruction-cost breakdown from a Metal GPU trace (63.9% math,
25% type conversion from bf16 scale/bias, 93.4% L1 hit rate, 2.4% GPU
utilization: the model was I/O-bound, not compute-bound). And the tooling is
growing: macOS 27 adds always-on frame-level Metal metrics (FPS, GPU time,
memory), a `metalperftrace` command-line tool with `--json` output, and a
StateReporting API
([WWDC26 session 388](https://developer.apple.com/videos/play/wwdc2026/388/)).
These are per-layer frame metrics aimed at games, not per-kernel counters, so
the methodology below stands.

So the working methodology, which the [war stories](../war-stories/three-questions.md)
independently converge on:

1. **Timestamps + arithmetic.** Compute achieved GFLOPS and GB/s from problem size
   and measured time; compare against [roofline](../techniques/roofline.md)
   ceilings you measured yourself (a STREAM-style
   [bandwidth probe](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/bandwidth.metal)
   for DRAM; [metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
   tables for ALU).
2. **Differential benchmarking.** When you can't see stall reasons, change one thing
   and re-measure. The [m5-gemm README's](../kernels/gemm-double-buffered.md)
   "things that did not help" section is this method producing knowledge.
3. **[Disassembly](compilation-pipeline.md) when arithmetic says impossible.**
   [applegpu](https://github.com/dougallj/applegpu)'s `compiler_explorer.py` shows
   what was actually emitted: the tool of last resort for
   [spill](../machine/registers.md) hunting.
4. **Methodology hygiene**, because the tools won't catch you: run once with
   `MTL_SHADER_VALIDATION=1` (out-of-bounds and tensor-indexing checks) and
   `MTL_DEBUG_LAYER=1` (API misuse) before trusting a correctness check, and
   never time with either on, since validation instruments every kernel; fix
   your clocks story (Apple's profiler can pin the GPU performance state for a
   capture, and its docs recommend a consistent state between runs; thermals:
   [locking fans changed long-run throughput decay from 50% to 6.7%](../war-stories/cheap-tricks.md)
   in one measured case), report end-to-end numbers next to kernel numbers
   ([synthetic wins that vanish end-to-end are the classic failure](../war-stories/the-failures.md)),
   and state matrix sizes, since
   [the winner flips with size](../kernels/gemm-double-buffered.md).

Next: [Disassembly](disassembly.md)

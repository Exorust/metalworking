# The baseline and the stopwatch

## Where we are

Part 2 gave you the least Metal needed to run a kernel. Part 3 is about measuring
it before anyone optimizes it, which on this platform is harder than on NVIDIA
because the profiler you are used to does not exist. This chapter settles two
things first: the number a custom kernel has to beat, and the one reliable way to
time it. It also starts the running example. From here to the end of Part 6, one
matrix multiply, 4096 by 4096 in 32-bit floats on a 40-core M5 Max, carries a
number from chapter to chapter.

## The baseline: MPS

Metal Performance Shaders, MPS, is Apple's library of pre-built GPU kernels:
matrix multiply, convolution, image operations. Its role is the closest thing
Metal has to cuBLAS and cuDNN, but it does not have their standing. On NVIDIA,
beating cuBLAS at dense matrix multiply is a research result, because the library
has decades of tuning and hardware co-design behind it. MPS is good, and the machine-learning ecosystem on
Apple Silicon has repeatedly matched or beaten it with open kernels.

- The hand-written m5-gemm kernel that Part 4 reads reaches 13.5 TFLOPS in 32-bit
  at 4096 by 4096 on an M5 Max, against 11.7 for MPS. MPS takes the lead back at
  8192 by 8192, 13.5 against 13.0, where the kernels converge on the same limit;
  Chapter 14 shows the whole table.
- llama.cpp deleted its MPS path in 2023, pull request 2615, in favor of its own
  simdgroup kernels. The pull request reports about 88 percent arithmetic-unit
  utilization on a 4096 by 4096 matrix multiply, with the caveat that the
  end-to-end gain was about 40 percent, because the operations that are not
  matrix multiplies dominate the rest.
- MLX uses its own kernel library, steel, Part 5, rather than MPS for its core
  operations.

MPS is beatable because its kernels are general and opaque. There is no
function-constant specialization for your shapes, Chapter 8; no fusion with the
operations around the call, Chapter 15; and no kernels for the quantized weight
formats that language models use, Chapter 23. The places where frameworks still
lean on MPS are where its coverage is hard to replicate: convolution variants,
some reductions, and small-matrix cases where the CPU's own matrix unit and
MPS-tuned dispatch beat a naive GPU kernel.

For the kernel writer, MPS is mainly the benchmark floor.
`MPSMatrixMultiplication` is a small setup, a matrix descriptor per operand plus
the operation, and it gives GPU-time numbers through the same command-buffer
timestamps as everything else; the m5-gemm harness has a compact PyObjC example
in `mps_matmul.py`. The rule that follows is the first of the three questions of
Chapter 24: if your custom kernel does not beat MPS at your shapes, ship MPS.

## The stopwatch: command-buffer timestamps

Metal has no Nsight Compute, and that gap shapes how you do performance work
on it. Of the tools you do have, the most useful is cheap, programmatic and
reliable: `GPUStartTime` and `GPUEndTime` on a completed command buffer, in
wall-clock seconds of GPU time. Every benchmark harness in this book uses them.
This is the complete timing code in the m5-gemm harness.

```python
            buffer.commit()
            buffer.waitUntilCompleted()
            if debug:
                captureManager.stopCapture()
            t = buffer.GPUEndTime() - buffer.GPUStartTime()
```
Source: [m5-gemm `metal.py:139-143`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/metal.py#L139-L143)

The granularity is the command buffer, Chapter 7, which sets three rules.
Isolate what you are measuring in its own command buffer, so the time contains
nothing else. Create the pipeline state before the timed loop, Chapter 6, so that
compilation never lands in a timing, and discard the first runs, which start with
cold caches and clocks. And take several trials and report a median or a best-of,
because single timings vary; the m5-gemm numbers in this book are the best of 3
runs of 5 trials each. From the time, you compute the rate: a matrix
multiply at size n does 2n³ floating point operations, so 4096 by 4096 is about
137 GFLOP, and 137 GFLOP in 10.2 milliseconds is about 13.5 TFLOPS.

## What else exists, and why it is not the feedback loop

Xcode's GPU capture, the Metal Debugger, is the closest thing to a real profiler:
per-encoder timings, an occupancy estimate, memory-bandwidth counters, shader
profiling. It has two structural limits. It is a GUI, so there is no continuous
integration and no scripted regression check. And capture perturbs heavily on big
machine-learning workloads: inference projects report slowdowns of 50 to 100
times and traces of several gigabytes. You can use it to study one dispatch,
but not as a feedback loop.

Metal 4's counter API exposes timestamps only, through macOS 26. The hardware
counters that would give achieved occupancy, stall reasons, or memory traffic
per kernel are not in the public programmatic interface. The GUI capture does
surface them: the flash-moe project pulled a full instruction-cost breakdown from
a GPU trace, 63.9 percent of instruction cost in arithmetic, 25 percent in type
conversion from bf16 scale and bias values, a 93.4 percent L1 hit rate, and 2.4
percent GPU utilization, which told them their model was bound by input and
output, not by compute. This is changing: macOS 27 adds always-on
performance metrics, a `metalperftrace` command-line tool, and a state-reporting
API, announced in session 388 of WWDC 2026, which would give this platform its
first scriptable feedback loop. Until that ships and proves itself, use the
methodology below.

## The methodology

The projects whose results fill Part 7 converged on the same four habits
independently.

1. **Timestamps plus arithmetic.** Compute achieved GFLOPS and GB/s from the
   problem size and the measured time, and compare them with ceilings you
   measured yourself. Chapter 10 builds the chart.
2. **Differential benchmarking.** When you cannot see stall reasons, change one
   thing and measure again. The m5-gemm README's section "things that did not
   help" records what this method found, and Chapter 14 reads it.
3. **Disassembly when the arithmetic says impossible.** The reverse-engineered
   applegpu tools show what the compiler emitted. They are the tool of last
   resort for hunting spills, Chapter 26.
4. **Hygiene, because no tool will catch you.** Decide how you handle clocks: in one
   measured case, locking the fans changed the long-run throughput decay from 50
   percent to 6.7 percent, Chapter 25. Report end-to-end numbers next to kernel
   numbers, because a synthetic win that vanishes end to end is the common
   failure that Chapter 25 describes. And state the matrix sizes, because the winner flips
   with size, Chapter 14.

## Where the matmul stands

MPS multiplies two 4096 by 4096 32-bit matrices on an M5 Max at 11.7 TFLOPS, best
of 3 runs of 5 trials, measured by command-buffer timestamps. That is the number
to beat. At 1024 by 1024, where launch costs matter more, MPS reaches only 2.9
TFLOPS; at 8192 by 8192 it reaches 13.5. The kernel of Part 4 has to beat the
middle number at the middle size.

## Check yourself

1. A harness times one dispatch by wrapping `commit` and `waitUntilCompleted` with
   the CPU clock. What does the chapter say to use instead, and what two things
   must the harness do before it trusts a number?
2. A custom kernel reaches 12 TFLOPS at 4096 by 4096 on an M5 Max. Is that a win,
   and what does the chapter say to do if it reaches 11?
3. Why does the chapter call Xcode's GPU capture unusable as a feedback loop,
   even though it reports the counters a kernel writer wants?

### Answers

1. The command buffer's `GPUStartTime` and `GPUEndTime`, which measure GPU time
   rather than CPU wait time. The harness must build its pipelines before timing
   and discard the first runs, and take several trials.
2. Yes: MPS is 11.7, so 12 beats the baseline at that shape. At 11 it does not,
   and the rule is to ship MPS.
3. Because it is a GUI with no scripted path, and because capturing a large
   machine-learning workload perturbs it by 50 to 100 times and produces traces of
   several gigabytes, so it cannot run on every change the way a benchmark can.

A time and a rate are not yet a verdict; the next chapter builds the chart that
turns them into one.

Next: [Build your own roofline](10-build-your-own-roofline.md)

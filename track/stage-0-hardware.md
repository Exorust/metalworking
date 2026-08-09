# Stage 0: Hardware reality

Needs `code/metal-benchmarks`. Time: an evening of reading.

Before writing any Metal Shading Language, learn what the machine actually is. Apple
never published a microarchitecture guide for the M-series GPU, so the community reverse
engineered one: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks).
Most of the numbers the rest of the track leans on (why F16 wins, why your register
budget is the real constraint) come from this one document.

## Read

`code/metal-benchmarks/README.md`, in this order of sections:

| Line | Section | Why it matters |
|---|---|---|
| 80 | On-Chip Memory | ~208 KB register file per core; 32 KB threadgroup memory (not CUDA's 48+). Register pressure is the budget you'll actually blow. |
| 118 | Operations per Second | The theoretical ceilings you'll benchmark against. |
| 159 | ALU Bottlenecks | ALU saturates at 24 SIMDs/core; F16 wins because register-dependency stalls are 1.56 cycles vs 1.84 for F32. |
| 212 | ALU Layout | How the pipes are arranged; FFMA32 dual-dispatches 2 instructions/cycle. |
| 281 | Instruction Throughputs | The big per-instruction latency/throughput table. You'll come back to this constantly. |
| 676 | SIMD Futures | The async-copy / `simdgroup_event` discussion; background for stage 2. |
| 754 | Power Efficiency | Why perf/watt is the metric Apple actually optimized. |

If you want to see how those numbers were measured,
`code/metal-benchmarks/InstructionThroughput/Kernels.metal` holds the ILP-sweep shaders
that produced the throughput table.

## Also worth knowing

- [Rigel: reverse-engineering Metal 4.1 tensor compute on M4 Max](https://arxiv.org/abs/2606.12765):
  fp8 is software emulated (0.94x fp16 throughput despite half the bytes), and `matmul2d`
  has no dedicated matrix datapath.
- [Alyssa Rosenzweig's M1 GPU series](https://alyssarosenzweig.ca/blog/asahi-gpu-part-n.html),
  the AGX reverse-engineering canon from the Asahi Linux driver work. The hardware has no
  native geometry or tessellation support at all.

## The one idea to internalize

**Apple Silicon is bandwidth bound, not compute bound.** Unified memory gives you huge
capacity at moderate bandwidth, so the win usually comes from touching less memory rather
than doing arithmetic faster: fuse kernels, skip work, keep data in registers. Every
later stage is a variation on this theme.

## Done when

- You can explain why F16 beats F32 on this hardware even when both are "fast".
- You can state the register file and threadgroup memory sizes from memory, and say
  which one usually kills occupancy first.
- Given a kernel's arithmetic intensity, you can predict whether it's bandwidth or
  compute bound on your own machine.

Next: [Stage 1: MSL from zero](stage-1-msl.md)

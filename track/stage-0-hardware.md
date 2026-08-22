# Stage 0: Hardware reality

Time: an evening of reading. Self-contained; links go deeper.

Before writing any Metal Shading Language, learn what the machine actually is. Apple
never published a microarchitecture guide for the M-series GPU, so the community
reverse engineered one:
[philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
(MIT). Every number below comes from it, and the rest of the track leans on them
constantly.

## The numbers that run the rest of this track

**The memory you can program** ([On-Chip Memory](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#on-chip-memory)):

- **~208 KB register file per core.** This is the budget you'll actually blow. Stage
  2's fastest kernel lives or dies on whether its accumulators stay in registers; the
  documented failure mode is a 10× slowdown from spilling.
- **32 KB threadgroup memory per threadgroup** (~60 KB physically per core) — smaller
  than CUDA's 48-100 KB. Tile-size instincts imported from CUDA will overshoot.
- Small caches (8 KB L1 data, 12 KB instruction per core). Apple's design trades
  cache for that huge register file. Plan on registers, not cache locality.

**Why F16 wins** ([ALU Bottlenecks](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#alu-bottlenecks)):
a back-to-back dependent multiply pays **1.84 cycles with F32 registers vs 1.56 with
F16** — and at low occupancy the latency gap widens to 11.3 vs 3.9 cycles for FMA.
F16 isn't faster math; it's shorter stalls and half the register pressure. The ALU
saturates around 24 simdgroups/core, so anything that raises occupancy (smaller
registers) also fills the pipes.

**Cheap and expensive, contra CUDA instinct:** threadgroup barriers cost ~2 cycles
(on NVIDIA they're a big deal — here, sync freely); scattered threadgroup-memory
access is comparatively expensive; there's a fast hardware `exp2` path (stage 4's
kernels compute softmax in base-2 for exactly this reason); FP32 atomics are
emulated and slow (stage 4 shows a backward pass split into two kernels just to
avoid them).

For the full per-instruction latency/throughput tables — which you'll come back to
whenever a kernel underperforms — see
[Instruction Throughputs](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#instruction-throughputs);
`InstructionThroughput/Kernels.metal` in the same repo holds the ILP-sweep shaders
that measured them.

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

# Stage 0: Hardware reality

Time: an evening of reading. Self-contained; links go deeper.

This page assumes you know your way around a CUDA GPU — warps, shared memory,
occupancy, coalescing — and translates. That's the fastest route, because the M-series
GPU is *almost* familiar: your instincts about tiling and coalescing transfer
directly, and then a handful of numbers are different enough to invert specific
habits. This page is the list of those numbers.

Apple never published a microarchitecture guide, so the community reverse engineered
one: [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
(MIT). Everything below comes from it.

## The dictionary

| CUDA | Metal | Notes |
|---|---|---|
| SM | GPU core | Same role: the unit that owns registers, shared memory, ALUs |
| warp (32 threads) | simdgroup (32 threads) | Identical width. Shuffles, votes, and reductions all exist |
| thread block | threadgroup | Same concept, same scheduling role |
| shared memory | threadgroup memory | **32 KB max**, vs your 48-228 KB. See below |
| tensor core / `mma` | `simdgroup_matrix` (8×8) | Closer to a fast wide-FMA path than a separate unit |
| `cp.async` / TMA | `simdgroup_async_copy` | Existed, undocumented, **dead on Metal 4** (stage 2 tells the story) |
| PTX → SASS | AIR → G13/G14 ISA | AIR is the portable IR; the ISA is reverse engineered (stage 5) |
| Nsight Compute | — | Nothing comparable. Xcode's GPU capture GUI, and timestamps. That's it |
| `__syncthreads()` | `threadgroup_barrier()` | Same semantics, wildly different cost — see below |

## Five differences that will actually change your code

**1. The register file is huge and shared memory is small — so tile in registers,
not in shared memory.** Per core: **~208 KB of registers** (vs ~256 KB on an SM)
but only **32 KB of threadgroup memory** and tiny caches (8 KB L1). On NVIDIA you
stage big tiles in shared memory and keep register blocking moderate. Here the ratio
flips: threadgroup memory is a thin staging buffer, and the serious blocking happens
in registers (stage 2's kernel holds a 32×32 output patch per simdgroup in registers,
full stop). The corollary: **register spilling is the #1 performance cliff.** The
stage-2 repo documents a 2× bigger register tile running 10× *slower* — spilled
accumulators. When an Apple kernel is mysteriously slow, suspect spills first, the
way you'd suspect uncoalesced loads first on NVIDIA.
([On-Chip Memory](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#on-chip-memory))

**2. Barriers are nearly free.** `threadgroup_barrier` costs **~2 cycles**. The
CUDA reflex of restructuring algorithms to avoid `__syncthreads()` buys you nothing
here — sync as often as the logic wants. The thing that *is* expensive, relative to
NVIDIA, is scattered access within threadgroup memory (bank behavior is less
forgiving than a modern SM's). Straight-line coalesced staging + frequent barriers
is the house style, and now you know why every kernel in this track looks that way.

**3. F16 is about stalls and registers, not about a faster unit.** On NVIDIA you
chase fp16 for tensor-core throughput. Here, a dependent F32 multiply stalls
**1.84 cycles vs 1.56 for F16**, and the gap explodes at low occupancy (11.3 vs 3.9
cycles for dependent FMA). Half-precision also halves register pressure, which —
see point 1 — is the actual budget. So "use F16 everywhere" is the local wisdom
even where throughput looks identical on paper.
([ALU Bottlenecks](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#alu-bottlenecks))
Related: occupancy saturates the ALUs around **24 simdgroups (~768 threads) per
core** — you don't need anywhere near SM-style 2048-thread residency to fill the
machine.

**4. Unified memory: no transfers, modest bandwidth, and that's the whole design.**
No PCIe, no `cudaMemcpy`, CPU and GPU share one pool — capacity is enormous (a Mac
Studio holds models no consumer NVIDIA card can). The price: bandwidth is a fraction
of an H100's HBM, and Apple tuned the whole chip for perf/watt at low clocks. Hence
the track's one recurring idea (below): you are bandwidth bound long before you are
compute bound, more often and more severely than CUDA experience suggests.

**5. FP32 atomics are emulated and slow.** `atomicAdd` on floats has been cheap on
NVIDIA since Kepler; here it's a trap. Real consequence in stage 4:
metal-flash-attention splits its backward pass into two kernels (dQ separate from
dK/dV) purely so no threadgroup ever needs cross-threadgroup float accumulation.
One free gift in the other direction: the hardware `exp2` path is fast, which is why
every softmax in this track is computed in base-2.

## The one idea to internalize

**Apple Silicon is bandwidth bound, not compute bound.** Unified memory gives huge
capacity at moderate bandwidth, so the win usually comes from touching less memory,
not doing arithmetic faster: fuse kernels, skip work, keep data in registers. Every
later stage is a variation on this theme.

## Also worth knowing

- [Rigel: reverse-engineering Metal 4.1 tensor compute on M4 Max](https://arxiv.org/abs/2606.12765):
  fp8 is software emulated (0.94x fp16 throughput despite half the bytes), and `matmul2d`
  has no dedicated matrix datapath.
- [Alyssa Rosenzweig's M1 GPU series](https://alyssarosenzweig.ca/blog/asahi-gpu-part-n.html),
  the AGX reverse-engineering canon from the Asahi Linux driver work. The hardware has no
  native geometry or tessellation support at all.
- The full per-instruction latency/throughput tables live in
  [Instruction Throughputs](https://github.com/philipturner/metal-benchmarks/blob/dc2adc640a1588246f4471d415aa6873cb6e3499/README.md#instruction-throughputs);
  you'll come back to them whenever a kernel underperforms.

## Done when

- You can name the three CUDA habits this page told you to invert (shared-memory-first
  tiling, barrier avoidance, fp16-for-throughput) and the number behind each.
- You can state the register file and threadgroup memory sizes from memory, and say
  which one usually kills occupancy first.
- Given a kernel's arithmetic intensity, you can predict whether it's bandwidth or
  compute bound on your own machine.

Next: [Stage 1: MSL from zero](stage-1-msl.md)

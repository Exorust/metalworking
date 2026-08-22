# Arithmetic Intensity

**Arithmetic intensity — FLOPs per byte of memory traffic — is the number that
decides whether a kernel is limited by the ALUs or by
[memory bandwidth](../machine/unified-memory.md), and on Apple Silicon it decides
almost everything.**

CUDA equivalent: the same concept, taught in every CUDA performance course. What
changes here is the balance point. A datacenter GPU pairs its FLOPs with TB/s of
HBM; Apple pairs laptop-class bandwidth (153-614 GB/s) with proportionally more
capable ALUs at low clocks. The machine's ridge point — the intensity below which
you're bandwidth bound — sits *high*, so **more of your kernels are bandwidth
bound than your CUDA intuition expects.** That's the platform's one law, and every
other technique page is a response to it.

The two canonical cases:

- **Matmul: intensity grows with size.** C = A×B at size n does 2n³ FLOPs on 3n²
  values — intensity ~n/1.5 FLOPs per element touched. Large GEMM is the rare
  compute-bound workload: the [tiled GEMM case study](../kernels/gemm-tiled.md)
  at 4096² *needs* only ~6 GB/s against a ~500 GB/s ceiling, ~85× below the roof.
  But that's only true because [tiling](tiling.md) makes the data reuse real;
  naive matmul re-reads its inputs O(n) times and is bandwidth bound at every
  size.
- **Everything elementwise: intensity ~0.1.** An add reads 8 bytes and writes 4
  to do one FLOP. No optimization changes that — except
  [not doing the traffic at all](fusion-and-epilogues.md), which is why fusion is
  the platform's most profitable technique.

LLM inference makes the dichotomy vivid:
[prefill is big-matmul-shaped (compute bound); decode is dot-product-shaped
(bandwidth bound)](decode-vs-prefill.md) — which is why decode speed
[tracks the memory-bandwidth spec linearly](../machine/unified-memory.md) across
chip tiers, and why [quantization](../mlx/quantization.md) (fewer bytes per
weight) is a decode *speed* optimization, not just a memory-size one.

Working method: before optimizing any kernel, do the sixty-second arithmetic —
bytes it must move, FLOPs it must do, divide, compare against your machine's
[measured roofline](roofline.md). If it's bandwidth bound (it probably is), ALU
cleverness is wasted effort; the levers are fusion, [F16/quantized
traffic](../machine/f16.md), and deleting work. If it's compute bound, now the
GEMM-school techniques ([register blocking](register-blocking.md),
[double buffering](double-buffering.md)) earn their keep.

Next: [Roofline](roofline.md)

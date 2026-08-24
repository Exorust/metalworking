# Unified Memory

**Apple Silicon has one physical memory pool shared by CPU and GPU. There is no
device memory, no host memory, and no transfer between them.**

CUDA equivalent: none, really. CUDA's "unified memory" (`cudaMallocManaged`) is a
software illusion over PCIe with page migration; this is the actual thing. The
same LPDDR silicon, the same physical addresses, zero-copy in both directions. A
`MTLBuffer` and a NumPy array can be views of the same bytes.

What this buys:

- **Capacity.** A Mac Studio configuration holds hundreds of GB of GPU-addressable
  memory. Models that need multi-GPU sharding in CUDA-land run on one Apple chip,
  which is the single biggest reason local-LLM people care about this platform at
  all.
- **No transfer engineering.** No `cudaMemcpy`, no pinned host memory, no
  overlap-copy-with-compute streams choreography. A whole genre of CUDA
  optimization simply doesn't exist here.
- **Cheap CPU⇄GPU cooperation**, with a caveat: data is free to share, but
  *synchronizing* still costs. Waiting on GPU results forces a pipeline drain, which
  is why [MLX is lazy](../mlx/lazy-evaluation.md) and why halving forced syncs is a
  recurring [war-story win](../war-stories/three-questions.md).

What it costs: **bandwidth**. A base M5 moves ~153 GB/s and an M5 Max ~614 GB/s,
against multiple TB/s of HBM on a datacenter GPU. Measured achievable fraction is
~84% of spec (STREAM-style probe in
[m5-gemm's `bandwidth.py`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/bandwidth.py)).
Combined with modest clocks, the consequence is the closest thing this glossary has
to a law: **Apple Silicon is bandwidth bound, not compute bound.** The win almost
always comes from touching less memory
([fusion](../techniques/fusion-and-epilogues.md),
[tiling for reuse](../techniques/tiling.md), skipping work) rather than from
arithmetic tricks. For LLM inference, memory bandwidth predicts decode tokens/sec
almost linearly across chip tiers; check the spec sheet before profiling anything.

Practical notes: not all of RAM is GPU-wirable by default, but a sysctl raises the
ceiling (see [cheap tricks](../war-stories/cheap-tricks.md)); and because the pool
is shared, GPU allocations compete with the OS and your browser tabs, which is why
serious inference rigs treat the machine as an appliance.

Next: [Occupancy](occupancy.md)

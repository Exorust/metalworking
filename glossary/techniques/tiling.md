# Tiling

**Tiling is the reuse pyramid: copy a block of the problem one level closer to
the ALUs, do all the math it can support, then let it go — the technique that
turns bandwidth-bound matmul into compute-bound matmul.**

CUDA equivalent: identical concept, and your instincts transfer — what moves is
the *shape* of the pyramid, because Apple's memory hierarchy has
[different proportions](../machine/registers.md).

The pyramid on this platform, as instantiated by every GEMM in the
[case studies](../kernels/gemm-tiled.md):

```
grid            each threadgroup owns a BM×BN tile of the output
  threadgroup   stages BM×BK of A and BK×BN of B in threadgroup memory
    simdgroup   owns a sub-tile of the output, resident in registers
      hardware  simdgroup_matrix: an 8×8 matmul per instruction
```

Each level multiplies reuse: a value loaded once into
[threadgroup memory](../machine/threadgroup-memory.md) feeds every simdgroup in
the group; a fragment loaded once into [registers](../machine/registers.md) feeds
a whole row or column of accumulator tiles. Arithmetic per byte of DRAM traffic
climbs from O(1) (naive) to O(tile edge) — which is the entire
[arithmetic-intensity](arithmetic-intensity.md) game.

The Apple-specific calibration, versus CUDA habits:

- **The middle level is thin.** 32 KB caps threadgroup tiles; typical shapes are
  64×64 output per threadgroup with a 16-32 deep K-slab — smaller than
  CUDA-typical. The pyramid's weight shifts down a level:
  [register blocking](register-blocking.md) carries more of the reuse than
  shared-memory blocking does on NVIDIA.
- **Tile shapes are compile-time.** Whether via `-D` defines
  ([m5-gemm](../kernels/gemm-tiled.md)) or template parameters
  ([steel's `BM/BN/BK/WM/WN`](../kernels/steel-gemm-fused.md)), sizes are baked so
  loops fully unroll and the
  [register allocator can plan](../metal/compilation-pipeline.md). Host code
  [picks the variant per shape](../mlx/how-an-op-becomes-a-kernel.md).
- **The edge problem is solved at pipeline time, not per-thread.** Instead of
  every thread guarding every access, [function constants](../metal/function-constants.md)
  compile separate aligned/ragged pipelines — the aligned one contains zero
  bounds checks.

Tiling's two companion moves have their own pages: getting the tile *in*
efficiently is the [cooperative load](cooperative-load.md); hiding the tile's
load time behind the previous tile's math is
[double buffering](double-buffering.md). And the limit case of tiling — where the
"tile" you refuse to write to memory is an entire intermediate matrix — is
[flash attention](flash-attention.md).

Next: [Cooperative load](cooperative-load.md)

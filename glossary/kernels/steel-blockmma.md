# Case Study: Steel's BlockMMA

**`BlockMMA` (`steel/gemm/mma.h`) is
[register blocking](../techniques/register-blocking.md) as a reusable component:
each simdgroup's grid of [8×8 fragments](../metal/simdgroup-matrix.md), the K-march
that feeds them, and the [epilogue hook](../techniques/fusion-and-epilogues.md) —
everything between "tiles are staged" and "results are stored."**

The layout constants decode the [m5-gemm kernel](gemm-tiled.md)'s hand-written
structure into named parameters:

```cpp
struct BlockMMA {
  STEEL_CONST short kFragSize = 8;              // the hardware 8×8
  ...
  STEEL_CONST short TM = BM / (kFragSize * WM); // fragment rows per simdgroup
  STEEL_CONST short TN = BN / (kFragSize * WN); // fragment cols per simdgroup
  ...
  MMATile<AccumType, TM, 1, MMAFrag_acc_t> Atile;
  MMATile<AccumType, 1, TN, MMAFrag_acc_t> Btile;
  MMATile<AccumType, TM, TN, MMAFrag_acc_t> Ctile;
```
— [`steel/gemm/mma.h:453-483`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L453-L483), abridged

`WM × WN` is the simdgroup arrangement (m5-gemm's `SW × SW`); `TM × TN` is each
simdgroup's accumulator grid (`SIMD_TILE × SIMD_TILE`); `Ctile` lives in
[registers](../machine/registers.md) for the kernel's lifetime, with `AccumType =
float` regardless of data type — [16-bit traffic, fp32
accumulation](../machine/f16.md).

The multiply marches K in fragment-size steps — load an A fragment and a B
fragment, `tile_matmad`, advance:

```cpp
  METAL_FUNC void mma(const threadgroup T* As, const threadgroup T* Bs) thread {
    As += As_offset;
    Bs += Bs_offset;
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < BK; kk += kFragSize) {
      simdgroup_barrier(mem_flags::mem_none);
      Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);
      simdgroup_barrier(mem_flags::mem_none);
      Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Ctile, Atile, Btile, Ctile);
      As += tile_stride_a;
      Bs += tile_stride_b;
    }
  }
```
— [`steel/gemm/mma.h:513-537`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L513-L537), abridged

Two production touches: transposition is handled by the stride constants
(`A_str_m/A_str_k` swap; the code never branches on transpose), and the
`mem_none` [simdgroup barriers](../metal/synchronization.md) are scheduling
nudges keeping each simdgroup's fragment loads batched.

The ending is where [fusion](../techniques/fusion-and-epilogues.md) plugs in:
`store_result` applies the `Epilogue` template parameter to every accumulator
element *in registers*, then stores once
([`mma.h:540-551`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/mma.h#L540-L551));
`apply_epilogue(C, ...)` variants additionally read an input matrix for
`α·AB + β·C`-shaped endings. Bias, activation, scaling — zero extra memory
passes, by construction.

One caveat before you generalize from this file:
[attention needs a different `mma.h`](steel-attention.md). A GEMM accumulator
only accumulates; attention's score tile must be *read and transformed in place*
between two matmuls, which demands a different fragment layout. Steel maintains
both — same philosophy, different geometry.

Next: [The fused GEMM kernel](steel-gemm-fused.md)

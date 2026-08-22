# Case Study: Steel's BlockLoader

**`BlockLoader` (`steel/gemm/loader.h`, ~140 lines) is the
[cooperative load](../techniques/cooperative-load.md) grown up: the hand-derived
constants of the [m5-gemm loader](gemm-tiled.md) recomputed at compile time from
template parameters, for every shape [MLX](../mlx/mlx-overview.md) serves.**

The template head *is* the design — read the defaulted parameters doing the
arithmetic you did by hand in the simpler kernel:

```cpp
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short alignment = 1,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct BlockLoader {
```
— [`steel/gemm/loader.h:14-25`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L14-L25)

`n_reads` = tile elements ÷ threadgroup size = each thread's share; `TCOLS`/`TROWS`
derive the thread-to-tile mapping. Any tile shape, any threadgroup size, zero
runtime arithmetic.

The hot path expresses vector width as an aligned type rather than a `float4`
cast — the compiler emits the widest load the `alignment` parameter permits:

```cpp
  struct alignas(alignment * sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(&dst[i * dst_ld])) =
          *((const device ReadVector*)(&src[i * src_ld]));
    }
  }
```
— [`steel/gemm/loader.h:42-44, 73-80`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L73-L80)

What production adds that the hand-written kernel never needed:
**`load_safe(short2 src_tile_dim)`**
([lines 83-128](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L83-L128))
bounds-checks and zero-fills for tiles hanging off the matrix edge. m5-gemm
required sizes divisible by 64; MLX can't. The elegance is in *when* each variant
runs: [function constants](../metal/function-constants.md) let the
[wiring kernel](steel-gemm-fused.md) compile pipelines where aligned dispatches
contain only `load_unsafe` — edge handling that costs nothing off the edge.

The smallest detail is the most load-bearing for composability:

```cpp
  METAL_FUNC void next() thread {
    src += tile_stride;
  }
```
— [`steel/gemm/loader.h:130-133`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/loader.h#L130-L133)

The loader owns its pointer arithmetic; the K-loop just calls `next()`. Three
verbs — load, next, done — are the whole interface, which is what lets quantized
loaders, transposed loaders, and the
[attention fork's differently-shaped loaders](steel-attention.md) swap in without
the kernel body changing. This is the [CUTLASS discipline](../mlx/steel.md) in
one file.

Next: [Steel's BlockMMA](steel-blockmma.md)

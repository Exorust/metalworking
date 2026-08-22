# Cooperative Load

**A cooperative load is the whole threadgroup jointly copying a tile from device
memory to [threadgroup memory](../machine/threadgroup-memory.md) — no thread loads
"its own" data; every thread carries an equal, coalesced share of everyone's.**

CUDA equivalent: the classic tiled-GEMM staging loop, unchanged in spirit. Two
platform notes sharpen it: with [`cp.async`-style DMA dead](../metal/simdgroup-async-copy.md),
cooperative loads are the *only* way tiles move, and with
[barriers nearly free](../metal/synchronization.md), the load-sync-compute-sync
rhythm costs almost nothing beyond the loads themselves.

The reference implementation, from the [GEMM case study](../kernels/gemm-tiled.md)
— each thread strides through the tile at threadgroup-width steps, so consecutive
threads always touch consecutive addresses (coalescing), in `float4` units when
alignment allows (vectorization):

```metal
template <ushort rows, ushort cols, ushort nthreads>
inline void load_tile(
    const device float *src, uint src_stride,
    threadgroup float *dst, ushort tid)
{
  ...
    auto src4 = reinterpret_cast<const device float4 *>(src);
    auto dst4 = reinterpret_cast<threadgroup float4 *>(dst);
#pragma clang loop unroll(full)
    for (ushort i = 0; i < total4; i += nthreads) {
      ushort idx = i + tid;
      if (idx >= total4) break;
      ushort r = idx / cols4;
      ushort c = idx - r * cols4;
      dst4[idx] = src4[uint(r) * stride4 + c];
    }
```
— [m5-gemm `sync_copy.metal:46-71`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L46-L71), abridged

With a 64×16 tile and 128 threads, that's two `float4` loads per thread — the
whole staging step is a few hundred instructions across the group.

The productionized version is steel's
[`BlockLoader`](../kernels/steel-blockloader.md), which derives the same
arithmetic (elements per thread, thread-to-tile mapping) from template parameters
at compile time, expresses vector width as an `alignas` struct so the compiler
emits the widest legal load, and adds a bounds-checked `load_safe` twin for
ragged edges — selected per-pipeline by
[function constants](../metal/function-constants.md), so aligned dispatches never
pay for checks.

Rules of thumb, all CUDA-familiar: consecutive threads → consecutive addresses
(in *both* directions — DRAM coalescing on the read, bank-friendliness on the
write, which matters since
[scattered threadgroup-memory access is pricey here](../machine/threadgroup-memory.md));
vectorize to `float4`/`half8` when layout permits; derive the mapping at compile
time so the loop fully unrolls. The historical footnote — a single simdgroup
issuing a [DMA copy for the whole group](../metal/simdgroup-async-copy.md) used to
beat all of this — explains why older codebases look different.

Next: [Register blocking](register-blocking.md)

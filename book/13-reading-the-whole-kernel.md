# Reading the whole kernel

## Where we are

You met the pieces of `sync_copy.metal` one at a time in Chapter 11 and Chapter 12:
the tile constants, the cooperative load, the fragment multiply, the accumulator
grid, the attribute, and the loop. Now you read the file from the first line to the
last, 144 lines, in order, so that every line is explained. The only part you have
not yet seen explained is the ending, where the kernel folds the full BLAS form of the
operation into its single store. Every excerpt is quoted verbatim from the pinned
commit of yaroslavvb/m5-gemm, BSD-3-Clause.

## Lines 1 to 20: the header and the constants

Before any code, the file tells you where it came from.

```metal
// Tiled fp32 GEMM via 8x8 simdgroup matrices ("tensor cores").
//
// Same tiling structure as https://percisely.xyz/gemm, but with the
// undocumented `air.simdgroup_async_copy_2d` intrinsic replaced by a
// straightforward cooperative load. Apple's Metal-4 compiler (macOS 26+)
// rejects __asm("air.*") linkage, so the article's technique no longer
// compiles. This kernel only uses public Metal APIs.
//
// Compile-time constants set by the host via -D:
//   SW         tile-of-tiles edge length (threadgroup is SW x SW simdgroups)
//   SIMD_TILE  simdgroups own a SIMD_TILE x SIMD_TILE grid of 8x8 matrices
//   TILE_K     reduction tile = TILE_K * 8 elements
```
Source: [`sync_copy.metal:1-12`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L1-L12)

The header states the kernel's lineage in two sentences: it is Zeke Medley's
tiled structure with the dead copy intrinsic of Chapter 14 replaced by the
cooperative load of Chapter 11, using only public Metal APIs. The quotation
marks around "tensor cores" are the author's, and Chapter 12 explained why they
belong there. Then come the three `-D` constants, and lines 18 to 20 turn them into
`BM`, `BN` and `BK`, the tile shape of Chapter 11: 64 by 64 output per
threadgroup, 16 deep per step.

## Lines 22 to 36: the fragment loop

What does the multiply look like once you see the whole function? This is the
one Chapter 12 quoted, now with its signature.

```metal
template <ushort DIM>
inline void simdgroup_multiply_tile(
    threadgroup float *A,
    threadgroup float *B,
    ushort2 c_pos,
    thread simdgroup_float8x8 &acc)
{
  simdgroup_float8x8 A_simd, B_simd;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < DIM * 8; i += 8) {
    simdgroup_load(A_simd, A, DIM * 8, ulong2(i, c_pos.y));
    simdgroup_load(B_simd, B, BN,      ulong2(c_pos.x, i));
    simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
  }
}
```
Source: [`sync_copy.metal:22-36`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L22-L36)

It takes one accumulator, the 8 by 8 output fragment at position `c_pos` inside
the threadgroup's tile, and walks `DIM` fragments along the shared dimension,
loading the matching A and B fragments from the staged tiles in threadgroup
memory. The two `simdgroup_load` calls differ in their stride argument, `DIM * 8`
for the A tile and `BN` for the B tile, because the staged tiles have different
widths: A is 64 rows by 16 columns, B is 16 rows by 64 columns. With `TILE_K` = 2
the loop body runs twice per call, and the pragma requests that those two
iterations become straight-line code. Each call loads its own A and B fragment,
so the 16 calls per step issue 32 loads for 16 multiplies; Chapter 12 counted
the distinct fragments.

## Lines 38 to 82: the cooperative load, and its fallback

Chapter 11 read the vectorized path of `load_tile`, lines 56 to 71. What it did
not show you is the preceding comment and the fallback path after it.

```metal
  constexpr ushort total = rows * cols;
#pragma clang loop unroll(full)
  for (ushort i = 0; i < total; i += nthreads) {
    ushort idx = i + tid;
    if (idx >= total) break;
    ushort r = idx / cols;
    ushort c = idx - r * cols;
    dst[idx] = src[uint(r) * src_stride + c];
  }
```
Source: [`sync_copy.metal:73-81`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L73-L81)

The vectorized path is guarded at line 56 by a compile-time condition: the tile's
column count must be a multiple of 4 and each thread's share must be a multiple
of 4. Both hold for the default tiles. The scalar fallback above is therefore compiled
out of this configuration and exists for other tile shapes. It is the same loop
one float at a time. The comment at lines 38 to 45 states the contract: `cols`
and `rows * cols / nthreads` must divide cleanly. That is about dividing the
tile among threads, and the code is a little more lenient than its comment,
since `per_thread` rounds up and the `break` catches the overshoot. The matrix
edge is a different matter: the loads carry no bounds checks and the loop uses
`k / BK`, so the kernel has no ragged path and silently drops a shared-dimension
remainder that is not a multiple of 16.

## Lines 84 to 111: the signature and the setup

The signature was read in Chapter 6: five scalar arguments in the `constant`
address space, `n`, `k`, `m`, `alpha` and `beta`; three buffers, `A`, `B` and
`C`; two position attributes and one size attribute. The kernel takes `alpha`
and `beta` because it computes the alpha × A × B + beta × C form of BLAS, rather
than the textbook C = A × B, and it reads C unconditionally, even when `beta` is
zero. That is why Chapter 10 counted C as read as well as written.

```metal
  threadgroup float A_tg[BM * BK];
  threadgroup float B_tg[BK * BN];

  uint tg_row = uint(tg_pos.z) * BM;  // top row of this threadgroup's C tile
  uint tg_col = uint(tg_pos.y) * BN;  // left col of this threadgroup's C tile

  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i = 0; i < SIMD_TILE; i++)
    for (ushort j = 0; j < SIMD_TILE; j++)
      acc[i][j] = simdgroup_float8x8(0);

  constexpr ushort NTHREADS = SW * SW * 32;  // 32 lanes per simdgroup
  ushort tid_in_tg =
      (t_tg_pos.z * tg_size.y + t_tg_pos.y) * tg_size.x + t_tg_pos.x;
```
Source: [`sync_copy.metal:98-111`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L98-L111)

The kernel declares two staging arrays of 1,024 floats each, the 8 KB of Chapter 2. The threadgroup's
position in the grid picks which 64 by 64 tile of C it owns; the grid is laid out
with the row index in `z` and the column index in `y`, which the harness matches
when it dispatches. The kernel then declares and zeroes the accumulator grid, 16
fragments per simdgroup. Last, it composes the thread's linear index inside the
threadgroup from the three-dimensional position: the harness launches
threadgroups of (32, `SW`, `SW`), so `x` spans the 32 lanes and `y` and `z`
name the simdgroup. Chapter 6 said this kernel does that arithmetic by hand,
because the cooperative load needs a flat index from 0 to 127.

## Lines 113 to 128: the loop

Chapter 12 quoted the loop in full: for each of the `k / BK` steps, stage A and B
cooperatively, barrier, multiply every fragment of the grid against the staged
tiles from the simdgroup's origin, barrier. The origin at line 120 is the
simdgroup's position in the threadgroup, `t_tg_pos.y` and `t_tg_pos.z`, scaled by
32, so the four simdgroups own the four 32 by 32 quarters of the 64 by 64 tile.

## Lines 130 to 143: the epilogue

When the loop ends, the answer sits in the accumulator grid. How does it
reach C?

```metal
  ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);
  uint c_row0 = tg_row + simd_origin.y;
  uint c_col0 = tg_col + simd_origin.x;
  if (c_col0 < m && c_row0 < n) {
    simdgroup_float8x8 c_simd;
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++) {
        ulong2 pos = ulong2(c_col0 + i * 8, c_row0 + j * 8);
        simdgroup_load(c_simd, C, m, pos);
        simdgroup_multiply(c_simd, c_simd, simdgroup_float8x8(beta));
        simdgroup_multiply_accumulate(c_simd, acc[i][j], simdgroup_float8x8(alpha), c_simd);
        simdgroup_store(c_simd, C, m, pos);
      }
  }
```
Source: [`sync_copy.metal:130-143`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L130-L143)

The ending is the first appearance of the technique that Chapter 15 covers. For each of the 16 accumulators, the kernel loads the existing 8 by
8 block of C from device memory into a fragment, multiplies it by `beta`,
multiplies the accumulator by `alpha` and adds, and stores the result. The
scaling uses matrix multiplies: `simdgroup_float8x8(beta)` constructs a diagonal
matrix with `beta` on the diagonal, so `c_simd × diag(beta)` scales every
element, and the multiply-accumulate computes `acc × diag(alpha) + c_simd`. The whole
BLAS form is applied while the result is still in registers, on its way to its
one store, and the kernel makes no second pass over C to scale it and launches no
separate kernel to add it. This is fusion on a small scale, and steel generalizes
the same spot into a plug-in epilogue (Chapter 18). The guard at line 133 checks
only that the simdgroup's origin is inside the matrix, which is always true when
the sizes are divisible by 64. That check is all the edge handling the kernel has.

Count the traffic of the epilogue against the loop. For the 4096-wide example the loop staged 4096 / 16 = 256 steps of 8 KB,
2 MiB of requested inputs per threadgroup, through threadgroup memory. The epilogue
reads and writes one 64 by 64 tile of C, 16 KB each way, once. The output is
written exactly once, which is what register blocking promised in Chapter 12.

## Running it yourself

You do not need Xcode, because the harness compiles the source at run time
(Chapter 6). On an Apple Silicon Mac with Python, NumPy and PyObjC installed,
`python matmul.py --dim 4096 --trials 5`
in the m5-gemm repository benchmarks both kernels against MPS, and `bandwidth.py`
measures your machine's bandwidth roof for Chapter 10. The numbers cited in this
book are the author's, from a 40-core M5 Max; expect different crossovers on
different tiers, and expect the sustained numbers of Chapter 10's hygiene rules.

## Where the matmul stands

The matmul is unchanged at 13.5 TFLOPS, and its source is now fully accounted
for: 144 lines, of which 12 are
the header, 3 the tile constants, 15 the fragment loop, 45 the cooperative load
with its fallback, 13 the signature, 14 the setup, 16 the loop, and 14 the
epilogue; the other 12 are seven blank lines, two includes, two braces and
`using namespace metal;`. Every line of code
maps to a chapter.

## Check yourself

1. The two `simdgroup_load` calls in the fragment loop pass different strides,
   `DIM * 8` and `BN`. Why?
2. The scalar fallback in `load_tile` is never executed for the default tiles.
   What makes that true, and when would it run?
3. The epilogue computes alpha × A × B + beta × C with four calls per fragment.
   Which of those calls touch device memory, and how many times is each output
   element written?

### Answers

1. Because the staged tiles have different
   widths. The A tile is 16 columns wide, `DIM * 8`, and the B tile is 64 columns
   wide, `BN`. `simdgroup_load` needs the row stride of the memory it reads from.
2. The compile-time condition at line 56 selects the vectorized path when the
   column count and each thread's share are multiples of 4, which holds for 64 by
   16 and 16 by 64 tiles with 128 threads. The fallback would run for a tile
   shape that fails either test, for example `SIMD_TILE` = 1 with `TILE_K` = 1,
   where the 16 by 8 tile of A is 128 floats, 128 / 128 = 1 per thread.
3. The `simdgroup_load` of C and the `simdgroup_store` of the result; the two
   multiplies work on fragments in registers. Each output element is read once
   and written once.

The loop is done and the epilogue has shown the next technique in miniature; the
next chapter asks whether the loop itself can overlap its loads with its math,
and tells the story of the copy engine that used to do it.

Next: [Double buffering, and the copy engine that died](14-double-buffering-and-the-copy-engine-that-died.md)

# simdgroup_matrix and register blocking

## Where we are

Chapter 11 left you with a 64 by 16 tile of A and a 16 by 64 tile of B staged in
threadgroup memory, 8 KB per step. Now you multiply them. The
multiply runs through Metal's matrix primitive, which looks like NVIDIA's
warp-level matrix API but does not run on the same kind of hardware. The results
stay in registers across all 256 steps of the shared dimension. That is register
blocking, the technique the proportions of Chapter 2 favor. At the end of the
chapter the kernel has its number.

## simdgroup_matrix: the API of a Tensor Core, on the regular pipes

The staged tiles need a multiply, and on Metal you reach for
`simdgroup_matrix<T, 8, 8>`, the matrix-multiply primitive: an 8 by 8 matrix
held collectively in one simdgroup's registers, with a single call that multiplies
and accumulates. As an API it is warp-level matrix multiply-accumulate, the `wmma`
fragments or `mma.sync` of CUDA. As hardware the analogy is looser. Apple GPUs
before M5 have no separate Tensor Core unit; `simdgroup_matrix` operations execute
as well-arranged work on the regular 32-bit pipes. The reverse-engineered Rigel paper
confirms that even Metal 4.1's newer `matmul2d` tensor API has no dedicated matrix
datapath on M4-class hardware. So the performance story differs from NVIDIA's. You
use `simdgroup_matrix` for its register layout and issue efficiency, not for a
tenfold throughput gain, and a well-written kernel built on it still reaches the
compute ceiling of Chapter 10. M5 is different: it adds a real per-core matrix
unit, reached through Metal 4 tensor operations, which Part 8 covers, while
`simdgroup_matrix` remains the portable primitive that runs on every M-series chip
and that most shipped kernels use.

The whole API fits in one loop of the kernel this book follows.

```metal
simdgroup_float8x8 A_simd, B_simd;
#pragma clang loop unroll(full)
for (ushort i = 0; i < DIM * 8; i += 8) {
  simdgroup_load(A_simd, A, DIM * 8, ulong2(i, c_pos.y));
  simdgroup_load(B_simd, B, BN,      ulong2(c_pos.x, i));
  simdgroup_multiply_accumulate(acc, A_simd, B_simd, acc);
}
```
Source: [m5-gemm `sync_copy.metal:29-35`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L29-L35)

Load a fragment of A and a fragment of B from threadgroup memory, multiply and
accumulate into `acc`, advance 8 along the shared dimension, repeat. Three rules
of use follow, and all of them match what you expect from `wmma`.

First, fragments are opaque. A `simdgroup_float8x8` holds 64 elements distributed
across the 32 threads in an undocumented layout. `thread_elements()` gives each
thread its own two, and steel uses it (Chapter 18), but which matrix coordinate
a lane holds is not something you can rely on. `simdgroup_load` and
`simdgroup_store` move whole tiles between a fragment and threadgroup or device
memory.

Second, fragments are registers, and they add up. A 4 by 4 grid of accumulators is
16 fragments of 64 floats, 1,024 floats per simdgroup (Chapter 2). The
accumulator budget decides the tile size.

Third, the type sets the cost. The choices are `simdgroup_float8x8`,
`simdgroup_half8x8` and `simdgroup_bfloat8x8`. 16-bit fragments halve the register
cost, and the kernels of Parts 5 and 6 take 16-bit inputs and accumulate in 32-bit
(Chapter 4). This kernel is a 32-bit benchmark and uses `float` throughout.

Everything built above `simdgroup_matrix` composes these fragments. Steel's
`BlockMMA` (Chapter 18) tiles them into a simdgroup-level register block; the
attention kernels of Part 6 chain two such blocks through an online softmax. When
a problem has only one row, as in decode, no 8 by 8 tile can be filled and
kernels abandon `simdgroup_matrix` entirely. Chapter 22 covers that split.

## Register blocking: the output never leaves the register file

Where should the output live while every step of the shared dimension passes
through? Register blocking keeps each simdgroup's patch of the output resident in
registers for the kernel's entire lifetime: accumulate across every step of the
shared dimension, write to memory exactly once. The idea is the same as the
accumulator fragments of any warp-tiled CUDA matrix multiply. What differs is the
proportion. With about 208 KB of registers per core against 60 KB of scratch per
core and 32 KB per threadgroup,
Apple kernels push more of the working set into registers than typical CUDA
kernels do, and the technique's failure mode, spilling, is correspondingly worse.

In its simplest real form, each simdgroup owns a 4 by 4 grid of accumulators.

```metal
  simdgroup_float8x8 acc[SIMD_TILE][SIMD_TILE];
  for (ushort i = 0; i < SIMD_TILE; i++)
    for (ushort j = 0; j < SIMD_TILE; j++)
      acc[i][j] = simdgroup_float8x8(0);
```
Source: [m5-gemm `sync_copy.metal:104-107`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L104-L107)

![The accumulator grid: a column of A fragments and a row of B fragments feed a four by four grid of 8x8 accumulators forming one simdgroup's 32x32 output patch](../diagrams/register-blocking.svg)

*The reuse geometry: eight distinct fragments feed sixteen multiply-accumulates
per step, and the accumulator grid never leaves the register file. In the grid,
`i` runs along columns and `j` along rows, following the kernel's `(i * 8, j * 8)`
offset.*

Count the reuse, because it is the reason the grid is 4 by 4 and not 1 by 1. Per
step of 8 along the shared dimension, the simdgroup needs 4 distinct fragments
of A, one per row of the grid, and 4 of B, one per column, and they feed 4 × 4 =
16 multiply-accumulates: every fragment is used four times. With a 1 by 1 grid
there is no reuse to find: 2 fragments per multiply, four times the load traffic
per multiply-accumulate. That is the "too small" cliff.

The kernel's source does not spell that reuse out: each call of the helper loads
its own A and B fragment, so the kernel as written issues 32 `simdgroup_load`
calls per 8-deep step for 16 multiplies, and whether the repeats stay in
registers is left to the compiler; steel's `BlockMMA` loads the four A and four B
fragments once into `Atile` and `Btile` and reuses them explicitly (Chapter 18).

The "too big" cliff is the one Chapter 2 introduced. The author tried `SIMD_TILE`
= 8, an 8 by 8 grid, 64 fragments, a 64 by 64 patch per simdgroup, and the kernel
ran 10 times slower (not 10 percent slower) because the accumulators spilled.
Once an accumulator is spilled, the multiply-accumulate that updates it is wrapped
in a load and a store. The loop is no longer arithmetic. The cliff is sharp
because the payload crosses the per-thread budget the compiler set (Chapter 2).
The sources show that 4 by 4 fits and 8 by 8 spills; where the boundary between
them lies is not measured. Two register-pressure levers from Chapter 2 apply:
16-bit input fragments, which halve their share of the budget (the accumulators
stay 32-bit, Chapter 4), and the attribute below.

## The most consequential line

Which line does the kernel's author call its single biggest practical win? An
attribute on the kernel declaration.

```metal
kernel void __attribute__((max_total_threads_per_threadgroup(SW * SW * 32)))
matmul(
```
Source: [`sync_copy.metal:84-85`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L84-L85)

With `SW` = 2 the bound is 128 threads. Chapters 1, 7 and 8 said what it does:
the compiler allocates registers when it builds the pipeline, and without the
bound it must plan for the hardware maximum and give each thread fewer. With it,
the allocator commits to a budget for 128 threads, and the author calls the
result the kernel's single biggest practical win. You also need to know what the attribute does
not do: the 10 times slower spill of Chapter 2 happened with this line already
in the file, so the attribute sets the budget; it does not remove the cliff.
The change is one line.

The step over the shared dimension is the loop that ties the three chapters
together.

```metal
  uint k_tiles = k / BK;
  for (uint l = 0; l < k_tiles; l++) {
    uint k_off = l * BK;
    load_tile<BM, BK, NTHREADS>(A + tg_row * k + k_off, k, A_tg, tid_in_tg);
    load_tile<BK, BN, NTHREADS>(B + k_off * m + tg_col, m, B_tg, tid_in_tg);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ushort2 simd_origin = ushort2(t_tg_pos.y, t_tg_pos.z) * (8 * SIMD_TILE);
    for (ushort i = 0; i < SIMD_TILE; i++)
      for (ushort j = 0; j < SIMD_TILE; j++)
        simdgroup_multiply_tile<TILE_K>(
            A_tg, B_tg,
            simd_origin + ushort2(i * 8, j * 8),
            acc[i][j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
```
Source: [`sync_copy.metal:113-128`](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/sync_copy.metal#L113-L128)

Each iteration stages the tiles cooperatively (Chapter 11), waits at a barrier,
multiplies from registers, waits at a second barrier, and advances.
`simd_origin` is where this simdgroup's 32 by 32 patch sits in the threadgroup's
64 by 64 tile, computed from its position in the threadgroup.
`simdgroup_multiply_tile` is the fragment loop quoted at the top of this chapter,
called once per accumulator, 16 times per step, and each call runs its loop
`TILE_K` = 2 times to cover the 16-deep slab. The loop count `k / BK` assumes
that the shared dimension is a multiple of 16, which the harness enforces. The
two barriers are the two hazards of Chapter 2: the first waits for the slab to
arrive, the second waits for every simdgroup to finish reading it before the next
iteration overwrites it.

## Where the matmul stands

What did keeping the output in registers buy you? The kernel multiplies two
4096 by 4096 32-bit matrices on a 40-core M5 Max at
13.5 TFLOPS, best of 3 runs each averaging 5 trials, against 11.7 for MPS. It beats the
library by (13.5 - 11.7) / 11.7, about 15 percent, and it sits at the practical
compute ceiling of Chapter 10, which is defined by this result. Three techniques
got it there: tiling, the cooperative load, and register blocking on
`simdgroup_matrix`, with the attribute that keeps the accumulators in registers.
What remains is the epilogue, which stores the accumulated output, and the next
chapter reads it with the rest of the file.

## Check yourself

1. A simdgroup with a 4 by 4 accumulator grid steps 8 along the shared dimension.
   How many fragment loads and how many multiply-accumulates does the step cost,
   and what would a 2 by 2 grid cost per multiply-accumulate?
2. Why does the kernel use `simdgroup_matrix` if the hardware before M5 has no
   matrix unit?
3. The attribute declares a maximum of 128 threads. What does the compiler do
   differently with that information, and what does it not protect against?

### Answers

1. 4 distinct fragments of A and 4 of B, 8 in all, feeding 16
   multiply-accumulates, so half a fragment per multiply-accumulate when the
   fragments are reused. A 2 by 2 grid needs 2 + 2 = 4 fragments for 4
   multiply-accumulates, one per multiply-accumulate, twice the
   threadgroup-memory traffic per unit of arithmetic. The kernel as written
   reloads per accumulator, so its issued load count is 2 per multiply either
   way; the reuse is in the distinct fragments.
2. For its register layout and issue efficiency: the fragment is held across the
   32 threads in a layout the hardware multiplies efficiently, and the loop is a
   handful of calls per step. The point is not a tenfold throughput gain, and the
   kernel still reaches the compute ceiling.
3. It allocates registers for 128 threads instead of the hardware maximum, so each
   thread gets a larger budget; the author calls that the single biggest practical
   win in the kernel. It does not protect against a payload that exceeds even
   that budget: the 8 by 8 accumulator grid spilled with the attribute in place.

The output is still in registers, and the next chapter reads the file from top
to bottom, following one tile all the way to its final store.

Next: [Reading the whole kernel](13-reading-the-whole-kernel.md)

# simdgroup_matrix

**`simdgroup_matrix<T, 8, 8>` is Metal's matrix-multiply primitive: an 8×8 matrix
held collectively in one [simdgroup's](../machine/simdgroup.md) registers, with a
single-call multiply-accumulate.**

## The problem it solves

A matrix multiply is made of small dot products: every output element is one row
of A times one column of B. Computed one element per thread, that is slow, because
each thread fetches its own operands and does its own multiply-adds. GPUs go
faster by having a group of threads compute a small block of the output together,
sharing operands through registers instead of memory. On Metal that group is the
simdgroup, the 32 threads that execute in lockstep on one core (CUDA's warp), and
the type that holds the shared block is `simdgroup_matrix`.

## What the type holds

`simdgroup_float8x8` is an 8×8 matrix of floats, 64 values, that the 32 threads of
one simdgroup hold between them in their registers, two values each. No single
thread owns the matrix and you cannot index it like an array. The only ways to get
data in and out are three calls:

- `simdgroup_load(frag, ptr, stride, origin)` fills the fragment from
  [threadgroup](../machine/threadgroup-memory.md) or device memory, starting at the
  given row and column with the given row stride. All 32 threads call it together.
- `simdgroup_multiply_accumulate(acc, a, b, acc)` computes acc = a × b + acc, an
  8×8×8 multiply, in one call, with the result staying distributed across the
  registers.
- `simdgroup_store(frag, ptr, stride, origin)` writes the fragment back to memory.

CUDA equivalent: warp-level MMA (`wmma` fragments / `mma.sync`) as an API. The
hardware analogy is looser, as the next section explains.

## The loop, line by line

The whole algorithm, from the [GEMM case study](../kernels/gemm-tiled.md):

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

The simdgroup owns one 8×8 block of the output, at row `c_pos.y` and column
`c_pos.x`. To compute it, the loop walks along the shared dimension K in steps of
8: load the 8×8 piece of A at columns i to i+7, load the 8×8 piece of B at rows i
to i+7, multiply them, add into `acc`. After the loop, `acc` holds the finished
block, and a `simdgroup_store` writes it out. Everything else in a GEMM kernel is
about feeding this loop.

## Why kernels use it here, and why the reason differs from NVIDIA's

On NVIDIA, `wmma` runs on tensor cores, a separate unit roughly 10× faster than the
ordinary pipes, so you use it for raw throughput. On Apple GPUs before M5 there is
no separate unit: `simdgroup_multiply_accumulate` executes on the regular FP32
pipes, arranged well (the reverse-engineered [Rigel paper](https://arxiv.org/abs/2606.12765)
confirms even Metal 4.1's newer `matmul2d` tensor API has no dedicated matrix
datapath on M4-class hardware). You still use it, for *register layout and issue
efficiency*: the operands already sit across the simdgroup's registers in the
right shape, so one instruction does 512 multiply-adds with no per-thread address
arithmetic or data shuffling, and the compiler can issue those instructions
densely. A well-written simdgroup-matrix GEMM
[still reaches 13.5 TFLOPS fp32 on an M5 Max, beating MPS](../kernels/gemm-tiled.md).

## Rules of use

All of them match your `wmma` instincts:

- **Fragments are opaque.** No indexing. The 64 elements live distributed across
  the 32 threads in a layout the spec calls unspecified; `thread_elements()` is a
  header member, not documented API, so which coordinate a lane holds is not
  something to rely on.
- **Uniform control flow only.** The spec says every `simdgroup_matrix` operation
  must be reached by the whole simdgroup; a per-lane branch around a load, store
  or multiply is undefined behavior. Edge handling must be the same for all 32
  threads, for example by zero-filling a tile before loading it.
- **They are [registers](../machine/registers.md), and they add up fast.** A real
  kernel does not hold one accumulator per simdgroup; it holds a grid of them,
  typically 4×4, so the simdgroup owns a 32×32 block of the output. That is 16
  fragments × 64 floats, 1,024 floats, 32 per thread, held for the whole kernel.
  Doubling that grid
  [spilled and ran 10× slower](../machine/registers.md) in the measured case. The
  accumulator budget *is* the tile-size decision.
- **Types:** `simdgroup_float8x8`, `simdgroup_half8x8`, `simdgroup_bfloat8x8`.
  [F16 fragments halve the register cost](../machine/f16.md); serious kernels take
  16-bit inputs and [accumulate in fp32](../machine/f16.md).

## How it fits into a whole kernel

A GEMM on this platform has three layers. A
[cooperative load](../techniques/cooperative-load.md) copies a tile of A and B from
device memory into threadgroup memory; each simdgroup pulls 8×8 fragments out of
that tile with `simdgroup_load` and runs the loop above into its accumulator grid;
an [epilogue](../techniques/fusion-and-epilogues.md) stores the grid. MLX's
[`BlockMMA`](../kernels/steel-blockmma.md) wraps the middle layer as a component;
[attention kernels](../kernels/steel-attention.md) chain two of these multiplies
(Q times K, then P times V) through an
[online softmax](../techniques/online-softmax.md). The one place kernels abandon
it is decode attention, where Q has a single row and no 8×8 tile can be filled;
those kernels go back to per-thread dot products, which is the
[decode-vs-prefill split](../techniques/decode-vs-prefill.md).

## On M5, this story forks

The M5 generation adds a real per-core matrix unit (the
[neural accelerators](../machine/neural-accelerators.md)) reached through
[Metal 4 tensor ops](mtltensor-and-mpp.md), not through this type, with Apple
reporting multi-x prefill gains. `simdgroup_matrix` remains the primitive most
shipped kernels use and the one that runs on macOS releases before 26. Both it and
tensor ops reach every M-series chip (GPU family Apple7 and later); Apple's
specification now recommends tensors and MPP as the portable path, while measured
gains so far appear only on M5. Read this page as the baseline; the NAX pages
cover the fork.

Next: [simdgroup_async_copy](simdgroup-async-copy.md)

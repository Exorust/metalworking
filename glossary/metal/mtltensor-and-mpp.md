# MTLTensor and Metal Performance Primitives

**Metal 4 adds tensors as a first-class type (`MTLTensor` on the host,
`tensor_inline` and cooperative tensors in shaders) plus Metal Performance
Primitives (MPP), a shader-side library of matrix operations. Together they
are the programming model for the [M5 neural
accelerators](../machine/neural-accelerators.md).**

CUDA equivalent: a blend of TMA (hardware-managed tile movement), `wmma`
descriptors, and CUTLASS-as-a-vendor-library, arriving as one API. The pieces:

- **`MTLTensor`**: a host-side multi-dimensional resource with rank, extents,
  and strides, replacing hand-rolled buffer + stride arithmetic. Auxiliary
  planes carry block-scaling metadata (scales per block) for the
  [fp4/fp8 formats](../mlx/quantization.md).
- **Shader-side tensors**: `tensor_inline` views over device memory or
  threadgroup memory, indexed and sliced in MSL.
- **Cooperative tensors**: a tensor whose storage is distributed across an
  execution group's registers, the way
  [`simdgroup_matrix`](simdgroup-matrix.md) fragments are, but sized and
  shaped by descriptor rather than fixed at 8×8.
- **MPP**: `mpp::tensor_ops::matmul2d` and friends. You declare a
  `matmul2d_descriptor` (shapes, dtypes, transpose, accumulate mode), pick an
  execution scope (`execution_thread`, `execution_simdgroup`,
  `execution_simdgroups<N>`, `execution_threadgroup`), and call the op; the
  destination arrives as a cooperative tensor via
  `get_destination_cooperative_tensor()`. Primary references:
  [WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/) and
  [WWDC26 330](https://developer.apple.com/videos/play/wwdc2026/330/), plus
  chapter 7 of the
  [MSL 4.1 spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Worked code exists in the fetched repos: MLX's
[`steel/gemm/nax.h`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/nax.h)
wraps MPP inside the familiar [steel](../mlx/steel.md) tiling (MIT, excerpted
in the [case study](../kernels/nax-gemm.md)), and flash-moe's
[`nax_gemm.metal`](https://github.com/gorroai/flash-moe/blob/4df3af8278c4bef2e7f6b34f61e4e2596b58e93b/metal_infer/nax_gemm.metal)
is a compact standalone example: a `matmul2d_descriptor`, an
`execution_simdgroups<4>` scope, and cooperative-tensor accumulation in ~230
lines, with the column-major layout gymnastics documented in its comments.

Why this page sits next to
[`simdgroup_async_copy`](simdgroup-async-copy.md): the ghost's story finally
has a successor. Cooperative tensors let an op read operands from device
memory and land results in distributed registers without the manual
[threadgroup-memory staging](../techniques/cooperative-load.md) dance; on
NAX-class hardware, the descriptor-and-scope model is doing the job the DMA
intrinsic once did, plus the job the tile loop did. How far that erodes the
classic [tiling](../techniques/tiling.md) shape on M5 is still being
established publicly; the techniques pages describe the shape every shipped
pre-NAX kernel uses, and this page is the fork to watch.

One caution for porters: this is macOS 26+ / M5-era surface. Everything else
in this glossary's [Metal section](metal-the-api.md) runs back to M1;
tensor-ops code needs a fallback path, which is exactly how MLX structures
its parallel NAX steel.

Next: [MPS](mps.md)

# MTLTensor and Metal Performance Primitives

**Metal 4 adds tensors as a first-class type (`MTLTensor` on the host,
`tensor_inline` and cooperative tensors in shaders) plus Metal Performance
Primitives (MPP), a shader-side library of matrix operations. Together they
are the programming model for the [M5 neural
accelerators](../machine/neural-accelerators.md).**

CUDA equivalent: a loose blend of TMA (hardware-managed tile movement), `wmma`
fragments, and CUTLASS-as-a-vendor-library, arriving as one API. The pieces:

- **`MTLTensor`**: a host-side multi-dimensional resource with rank, extents,
  and strides, replacing hand-rolled buffer + stride arithmetic. Extents list
  the innermost (contiguous) dimension first, and a buffer-backed tensor's
  stride 0 is always 1. From macOS 27 (Metal 4.1) a tensor can carry an
  auxiliary scales plane for the [fp4/fp8 formats](../mlx/quantization.md);
  the shading language accepts only E8M0 scales over blocks of 32 along the
  first dimension (the MX layout), multi-plane tensors need a one-byte data
  type, and they cannot be used in machine learning passes.
- **Shader-side tensors**: `tensor<T, extents, ...>`, a non-owning view.
  Host-bound tensors arrive through `[[buffer(n)]]` with a `tensor_handle`
  descriptor; shader-allocated tensors are built from a pointer, extents and
  strides with `tensor_inline`, over device, constant, threadgroup or thread
  memory. Both are indexed and sliced in MSL.
- **Cooperative tensors**: a tensor whose storage is distributed across an
  execution group's registers, the way
  [`simdgroup_matrix`](simdgroup-matrix.md) fragments are, but sized and
  shaped by descriptor rather than fixed at 8×8.
- **MPP**: `mpp::tensor_ops::matmul2d` and friends. You declare a
  `matmul2d_descriptor` (the tile's M, N, K, two transpose flags, a
  `relaxed_precision` flag that lets float inputs have their mantissa truncated
  before the multiply, and multiply or multiply-accumulate mode; element types
  come from the operand tensors, not the descriptor), pick an execution scope
  (`execution_thread`, or `execution_simdgroups<N>` where N is 1, also spelled
  `execution_simdgroup`, or the threadgroup's full simdgroup count, which is how
  a whole threadgroup cooperates), and call the op; the destination arrives as a
  cooperative tensor via `get_destination_cooperative_tensor()`. Every thread in
  the scope must call `run` (no lane may exit early), the op may itself use a
  barrier at that scope, and a destination in device or threadgroup memory needs
  a barrier before you read it; cooperative-tensor destinations do not (MSL 4.1
  §7.2). Operand types arrive by OS release: half, float and 8-bit integers in
  macOS 26.0, bfloat in 26.1, 4-bit integers in 26.4, and fp8, fp4 and 2-bit
  integers with Metal 4.1 in macOS 27 (Table 7.3). Primary references:
  [WWDC25 262](https://developer.apple.com/videos/play/wwdc2025/262/) and
  [WWDC26 330](https://developer.apple.com/videos/play/wwdc2026/330/), plus
  section 2.22 (tensor and cooperative tensor types) and chapter 7 (MPP) of the
  [MSL 4.1 spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
  and Apple's
  [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf).
  The framework has no page in the developer documentation; the header is
  `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`.
- **Cooperative tensors as inputs.** A matmul's cooperative-tensor result can
  feed a second `matmul2d` directly via `get_left_input_cooperative_tensor()` /
  `get_right_input_cooperative_tensor()` after an `is_compatible_as_left_input`
  / `_right_input` check; if the layouts differ you round-trip through
  threadgroup memory. The left-input form requires `execution_simdgroup` scope.
  This is how a FlashAttention kernel keeps the score matrix in registers
  ([WWDC26 330](https://developer.apple.com/videos/play/wwdc2026/330/)).

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
has a successor. TensorOps take operands straight from device-memory tensors
and land results in a cooperative tensor in distributed registers without the
manual [threadgroup-memory staging](../techniques/cooperative-load.md) dance
(Apple's MPP guide says staging "isn't necessary for the most optimized GEMM
kernels" and that no explicit software pipelining is needed); on
NAX-class hardware, the descriptor-and-scope model is doing the job the DMA
intrinsic once did, plus the job the tile loop did. How far that erodes the
classic [tiling](../techniques/tiling.md) shape on M5 is still being
established publicly; the techniques pages describe the shape every shipped
pre-NAX kernel uses, and this page is the fork to watch.

One caution for porters: this is macOS 26+ surface that compiles on any Metal 4
GPU (tensors are listed from GPU family Apple7, so M1 and later) but pays only
on M5-class hardware. Everything else in this glossary's
[Metal section](metal-the-api.md) runs back to M1; tensor-ops code needs a
fallback path, which is how MLX structures its parallel NAX steel. Apple ties
the hardware to a GPU family: devices in `MTLGPUFamily.apple10` and later have a
neural accelerator in each core, so `device.supportsFamily(.apple10)` is the
documented test for which path to take.

Next: [MPS](mps.md)

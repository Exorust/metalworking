# MPS — Metal Performance Shaders

**MPS is Apple's library of pre-built GPU kernels (matmul, convolution, image
ops): the closest thing Metal has to cuBLAS/cuDNN, and the baseline your custom
kernel has to beat.**

CUDA equivalent: cuBLAS/cuDNN in role, not in stature. On NVIDIA, beating cuBLAS at
dense GEMM is a research result; the library embodies decades of tuning plus
hardware co-design. MPS is good, but the ML ecosystem on Apple Silicon has
repeatedly matched or beaten it with open kernels:

- The hand-written [m5-gemm kernel](../kernels/gemm-tiled.md) hits 13.5 TFLOPS fp32
  at 4096² on an M5 Max vs MPS's 11.7, though MPS
  [takes the lead back at 8192²](../kernels/gemm-double-buffered.md) where
  everything is [bandwidth bound](../machine/unified-memory.md).
- llama.cpp **deleted its MPS path** in 2023
  ([PR #2615](https://github.com/ggml-org/llama.cpp/pull/2615)) in favor of its own
  simdgroup kernels: ~88% ALU utilization on a 4096² matmul, with the honest
  caveat that end-to-end gains were ~40% because non-matmul ops dominate.
- [MLX](../mlx/mlx-overview.md) uses its own [steel library](../mlx/steel.md), not
  MPS, for core ops.

Why beatable: MPS kernels are general and opaque. No
[function-constant](function-constants.md) specialization for *your* shapes, no
[fusion](../techniques/fusion-and-epilogues.md) with your surrounding ops, no
quantized paths for LLM weight formats. The places frameworks still lean on MPS
are where its coverage is genuinely hard to replicate (convolution variants, some
reductions, small-matrix cases where [AMX](../machine/amx.md)-backed CPU paths and
MPS-tuned dispatch beat naive GPU kernels).

For the kernel reader, MPS's main role is **the benchmark floor**:
`MPSMatrixMultiplication` is a three-object setup (matrix descriptors + the op)
and gives honest GPU-time numbers via
[command-buffer timestamps](profiling.md); the
[m5-gemm harness](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/mps_matmul.py)
is a compact PyObjC example. If your custom kernel doesn't beat MPS at your shapes,
ship MPS. That's not defeat; that's the
[three questions](../war-stories/three-questions.md) working as intended.

Next: [Profiling](profiling.md)

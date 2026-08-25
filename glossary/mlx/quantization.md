# Quantization

**MLX's workhorse quantization is group-wise affine (each group of 32/64/128
values along a row stores one scale, one bias, and 2-8 bit codes), and a
dedicated kernel family multiplies against them without ever materializing the
dequantized matrix. Since 2026 it is one of four modes: `affine`, `mxfp4`,
`nvfp4`, and `mxfp8`.**

CUDA equivalent: the weight-only-quantization kernels of TensorRT-LLM / AWQ /
GPTQ-land. Same motivation, sharpened by the platform: on
[bandwidth-bound](../machine/unified-memory.md) hardware, weights at 4 bits mean
~4× fewer bytes per matmul. Quantization here is a *bandwidth* optimization that
also happens to save memory, and it's the reason a 70B model decodes acceptably on
a laptop.

Mechanics: `mx.quantize(w, group_size=64, bits=4)` produces packed codes plus
per-group `scales` and `biases`; `w ≈ codes * scale + bias` per group.
`mx.quantized_matmul` (and the `QuantizedLinear` layer) consumes the packed form
directly. Group-wise affine is simpler than the K-quant block schemes llama.cpp
uses ([its kernels](../kernels/llamacpp-attention.md) dequantize richer formats
inside the same loops); MLX trades a little quality-per-bit for kernel
simplicity and speed.

The kernels live in
[`kernels/quantized.h`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/quantized.h),
and the family split mirrors [decode vs prefill](../techniques/decode-vs-prefill.md):

- **QMV** (quantized matrix × vector): the decode workhorse. One activation row
  against a quantized weight matrix; dequantize-in-registers inside a
  bandwidth-shaped reduction kernel.
- **QMM**: the prefill/batch case. Dequantize tiles into
  [threadgroup memory](../machine/threadgroup-memory.md), then a
  [steel](steel.md)-style tiled GEMM against them.
- Plus gather variants for MoE experts.

Host-side dispatch picks per shape
([the same file-reading applies](how-an-op-becomes-a-kernel.md)), and the shape
sensitivity is a running war-story theme: stock QMV is tuned for M=1;
[MTPLX found](../war-stories/three-questions.md) that M=3-6 (multi-token
speculative decode) stalled it, and ~10 lines of wider-simdgroup unrolling fixed
a 2.24× end-to-end factor. The
[sparse-V dequant-skip story](../war-stories/sparse-v.md) is the same insight one
level deeper: dequantization cost is per-*use*, so work you can prove unnecessary
(near-zero attention weights) is dequantization you can skip.

**The newer modes are block-scaled floats**, not affine integers:
[`mxfp4`, `nvfp4`, `mxfp8`](https://github.com/ml-explore/mlx/blob/43d2f06cb87e76895bf9a152bade4fee83408643/python/mlx/nn/layers/quantized.py#L13-L16)
with group sizes 32/16/32, backed by `fp4.h`/`fp8.h` kernels, plus
`quantize_input=True` for activation quantization (nvfp4/mxfp8, Linear
layers). These are the formats the [M5 neural
accelerators](../machine/neural-accelerators.md) consume natively, which is why
they arrived together. The affine story below still carries the teaching load;
just don't quote this page's first sentence as the whole picture.

One trap for intuition: quantized kernels *dequantize on every use*. There is no
cached fp16 copy; that would defeat the purpose. So arithmetic per byte rises,
which is fine on this machine ([compute is the abundant resource](../techniques/roofline.md)),
but it means quantized-matmul performance is even more bandwidth-proportional
than fp16 matmul, and [memory-bandwidth spec predicts decode speed](../machine/unified-memory.md)
almost linearly.

Next: [mx.compile](mx-compile.md)

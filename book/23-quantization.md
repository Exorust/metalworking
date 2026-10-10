# Quantization as a bandwidth optimization

## Where we are

Chapter 22 ended with decode as a byte-streaming problem: every token reads the
cache and every weight. The cache is a few gigabytes; the weights of a 70-billion
parameter model in 16-bit are 140 GB, which does not fit a laptop and, if it did,
would take more than a quarter of a second per token to stream. Quantization
stores the weights in fewer bits and multiplies against them without ever
expanding them. On a CUDA machine that is mainly a memory-size optimization. On
this platform it is the decode speed optimization. This chapter explains why, and
shows what the kernels that do it look like.

## What MLX quantizes, and how

MLX's workhorse quantization is group-wise affine. Each group of 32, 64 or 128
values along a row of a weight matrix stores one scale, one bias and the group's
values as codes of 2 to 8 bits, and a dedicated kernel family multiplies against
the packed form without materializing the dequantized matrix. Since 2026 it is one
of four modes: `affine`, `mxfp4`, `nvfp4` and `mxfp8`.

The CUDA counterparts are the weight-only quantization kernels of TensorRT-LLM,
AWQ and GPTQ. The motivation is the same, and the arithmetic of Chapter 3 makes it
stronger on this platform. A matrix-vector product is about 1 FLOP per byte of
weights in 16-bit, far below the ridge, so its time is the time to stream the
weights. Weights at 4 bits are about 4 times fewer bytes per multiply.
Quantization here is a bandwidth optimization that also saves memory, and it is
the reason a 70-billion parameter model decodes acceptably on a laptop.

In code, `mx.quantize(w, group_size=64, bits=4)` produces packed codes plus
per-group `scales` and `biases`, with `w` approximately `codes * scale + bias`
within each group. `mx.quantized_matmul`, and the `QuantizedLinear` layer that
wraps it, consume the packed form directly. Group-wise affine is simpler than
the block schemes llama.cpp uses, whose kernels dequantize richer formats inside
the same loops (Chapter 21). MLX trades a little quality per bit for kernel
simplicity and speed.

## The kernel family, split like attention

The kernels live in `kernels/quantized.h`, and the family is split the same way
as the attention kernels of Chapter 22.

- **QMV**, quantized matrix times vector, is the decode workhorse. It multiplies
  one activation row against a quantized weight matrix, dequantizing in
  registers inside a bandwidth-shaped reduction kernel. There is no matrix tile
  to fill, so it is built on the vector loads and simdgroup reductions of
  Chapter 1, not on `simdgroup_matrix`.
- **QMM**, quantized matrix times matrix, handles the prefill and batch case.
  It dequantizes tiles into threadgroup memory, then runs a steel-style tiled
  matrix multiply against them. This is the kernel of Part 4 with one extra step
  in the loader.
- Gather variants for mixture-of-experts models, which pick each token's expert
  weights by index.

Host-side dispatch picks a kernel per shape, and the file reading of Chapter 17
applies. Shape sensitivity is the running theme of Part 7. Stock QMV is tuned for
one row, and the MTPLX project found that 3 to 6 rows, the shape speculative
decoding produces, stalled it. About 10 lines of wider-simdgroup unrolling fixed
it, a factor of 2.24 end to end (Chapter 24). The sparse-V case of Chapter 25
applies the same idea one level deeper. Dequantization cost is paid per use, so
work you can prove unnecessary (positions with near-zero attention weight) is
dequantization you can skip.

Count the bytes for the decode case that the whole chapter is about. A 4096 by
4096 weight matrix is 16.8 million values. In 16-bit that is 32 MiB per
matrix-vector product; at 4 bits with a 64-value group, each group of 64 values
costs 64 × 4 bits plus a 16-bit scale and a 16-bit bias, 288 bits in all. The
matrix is therefore about 9 MiB, 3.5 times fewer bytes. The ratio is not quite 4
because of the per-group metadata. At the measured 495 GB/s of Chapter 3, that is about 19 microseconds
per matrix instead of about 68. A model is hundreds of such matrices per token.

## The newer modes: block-scaled floats

The three modes added alongside affine are block-scaled floating point, not
affine integers: `mxfp4`, `nvfp4` and `mxfp8`, with group sizes 32, 16 and 32,
backed by `fp4.h` and `fp8.h` kernels, plus `quantize_input=True` for quantizing
activations as well as weights in the nvfp4 and mxfp8 modes. These are the
formats the M5 neural accelerators consume natively (Chapter 28), which is why
they arrived together. This chapter still teaches with the affine mode, but the
affine description at its start does not cover every mode.

## One trap for intuition

Quantized kernels dequantize on every use. There is no cached 16-bit copy; that
would defeat the purpose. So arithmetic per byte rises. This machine can afford
that, since by the roofline of Chapter 10 compute is the abundant resource. It
also means that quantized matrix-multiply performance is even more proportional
to bandwidth than 16-bit matrix multiply is, and the memory-bandwidth spec
predicts decode speed almost linearly. That is Chapter 25's buying rule.

## Where the matmul stands

The decode example now streams about 3.5 times fewer weight bytes per token at 4
bits, and its remaining cost is the cache of Chapter 22 and the forced waits of
Chapter 16. Part 6 is done. It covered the matrix multiply, attention, the cache
and the weights, each with its number. Part 7 asks where the measured wins on
real models came from.

## Check yourself

1. A 70-billion parameter model is stored at 4 bits with 64-value groups and
   16-bit scale and bias. Roughly how many bytes do its weights take, and what
   is the lower bound on time per token to stream them at 495 GB/s?
2. Why does MLX ship separate QMV and QMM kernels rather than one quantized
   matrix multiply?
3. What does "dequantize on every use" cost, and why is that an acceptable trade
   on this platform?

### Answers

1. 70 billion × 288 / 64 bits is about 315 billion bits, about 39 GB; at 495 GB/s
   that is a lower bound of about 80 milliseconds per token, so at most about 12
   tokens per second from the weights alone.
2. Decode has one row and no tile to fill, so it needs a reduction kernel
   that dequantizes in registers, while prefill has many rows and wants the tiled
   matrix-multiply path with dequantization in the loader; the two shapes need
   different kernels, as attention does.
3. It costs arithmetic per weight on every multiply, instead of once. It is acceptable
   because the kernel is bandwidth bound, so the arithmetic units have slack, and
   a cached 16-bit copy would restore the traffic the quantization removed.

Part 7 opens with the three questions that every measured win in it answers, and
that every measured failure skipped.

Next: [The three questions](24-the-three-questions.md)

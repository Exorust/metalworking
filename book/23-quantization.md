# Quantization as a bandwidth optimization

## Where we are

You left Chapter 22 with decode as a byte-streaming problem: for a dense model at
batch size one, every token reads the cache and every weight. You have already
seen what the cache costs. The cache is a few
gigabytes at long context; the weights of a 70-billion parameter model in 16-bit
are 70 × 10^9 × 2 bytes = 140 GB, which does not fit the memory of most laptops
and, if it did, would take 140 / 495 = 0.28 seconds per token to stream.
Quantization stores the weights in fewer bits and multiplies against them without
ever materializing the expanded matrix. On a CUDA machine it saves memory and
bandwidth alike; on this platform the bandwidth half is the one that sets decode
speed. To see why, you need the arithmetic of Chapter 3 and a look at the kernels
that do the work.

## What MLX quantizes, and how

If you have shipped weight-only quantization on CUDA, your first question is what
MLX actually stores. Its workhorse quantization is group-wise affine. Each group of 32, 64 or 128
values along a row of a weight matrix stores one scale, one bias and the group's
values as codes of 2, 3, 4, 5, 6 or 8 bits, and a dedicated kernel family multiplies against
the packed form without materializing the dequantized matrix. Since 2026 it is one
of four modes: `affine`, `mxfp4`, `nvfp4` and `mxfp8`.

The CUDA counterparts are the weight-only quantization kernels of TensorRT-LLM,
AWQ and GPTQ. The motivation is the same, and the arithmetic of Chapter 3 makes it
stronger on this platform. A matrix-vector product is about 1 FLOP per byte of
weights in 16-bit, far below the ridge, so its time is the time to stream the
weights. Weights at 4 bits are about 4 times fewer bytes per multiply.
So quantization here is a bandwidth optimization that also saves memory. It is
the reason a 70-billion parameter model decodes acceptably on a laptop.

In code, `mx.quantize(w, group_size=64, bits=4)` produces packed codes plus
per-group `scales` and `biases`, with `w` approximately `codes * scale + bias`
within each group. `mx.quantized_matmul`, and the `QuantizedLinear` layer that
wraps it, consume the packed form directly. Group-wise affine is simpler than
the block schemes llama.cpp uses, whose kernels dequantize richer formats inside
the same loops (Chapter 21). MLX trades a little quality per bit for kernel
simplicity and speed.

## The kernel family, split like attention

One quantized multiply cannot serve both decode and prefill well, for the same
reason one attention kernel could not. The kernels live in `kernels/quantized.h`,
and the family is split the same way as the attention kernels of Chapter 22.

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
applies: QMV also serves a few rows, and on an M5 eligible QMM calls go to the
NAX fork. Shape sensitivity is the running theme of Part 7. Stock QMV is tuned for
one row, and the MTPLX project found that 3 to 6 rows, the shape speculative
decoding produces, stalled it. About 10 lines of wider-simdgroup unrolling fixed
it, a factor of 2.24 end to end (Chapter 24).

The sparse-V case of Chapter 25 applies the same idea one level deeper.
Dequantization cost is paid when packed values are loaded, so
work whose contribution falls below a threshold (positions with near-zero
attention weight) is dequantization you can skip, an approximation the sparse-V
case checked against perplexity.

How much does all this buy you? Count the bytes for the decode case that the
whole chapter is about. A 4096 by
4096 weight matrix is 16.8 million values. In 16-bit that is 32 MiB per
matrix-vector product; at 4 bits with a 64-value group, each group of 64 values
costs 64 × 4 bits plus a scale and a bias in the weight's own type, 16 bits each
here, 288 bits in all, 4.5 bits per weight. The matrix is therefore 16.8 million
× 4.5 / 8 = 9,437,184 bytes, 9 MiB, and 32 / 9 is 3.56 times fewer bytes. The
ratio is not quite 4 because of the per-group metadata. At the measured 495 GB/s
of Chapter 3, the streaming floor is about 19 microseconds per matrix instead of
about 68, before dequantization, reductions and dispatch. A model applies several
such matrices per layer, times tens of layers, per token.

## The newer modes: block-scaled floats

Affine is not the only format you will meet in the pinned MLX. The three modes
added alongside it are block-scaled floating point, not affine integers: `mxfp4`, `nvfp4` and `mxfp8`, with group sizes 32, 16 and 32,
backed by `fp4.h` and `fp8.h` kernels, plus `quantize_input=True` for quantizing
activations as well as weights in the nvfp4 and mxfp8 modes. These are the
block-scaled formats that arrived with the M5 neural accelerators (Chapter 28);
in the pinned MLX, the fp4 and fp8 NAX kernels dequantize the codes to 16-bit
values before loading them into the matrix unit's tiles. The examples here stay
with the affine mode. Remember that the affine description at the start does not
cover every mode.

## One trap for intuition

Where does the unpacked matrix live? Nowhere: quantized kernels dequantize
whenever they load packed weights. There is no
cached 16-bit copy of the matrix in memory, which would defeat the purpose,
though a kernel reuses what it unpacked inside a tile: the wide QMV dequantizes
a block once and applies it to every activation vector in flight, and QMM
reuses a dequantized tile across rows. So arithmetic per byte rises.

Whether the kernel stays bandwidth bound then depends on that unpacking
arithmetic staying under the memory time. The decode measurements of Chapter 25
say it does for the stock kernels, and across chip tiers the memory-bandwidth
spec predicts decode speed almost linearly, which is Chapter 25's buying rule.

## Where the matmul stands

The decode example now streams about 3.5 times fewer weight bytes per token at 4
bits. The weights still set the floor (about 80 milliseconds per token for a
70-billion parameter model), and on top of them sit the cache of Chapter 22, the
dequantization inside the kernels, and the forced waits of Chapter 16.

That finishes Part 6. You now have a number for each of the matrix multiply,
attention, the cache and the weights. Part 7 asks where the measured wins on
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

1. 70 billion × 4.5 bits is 315 billion bits, 39.4 GB; at 495 GB/s that is a
   lower bound of 39.4 / 495 = 80 milliseconds per token. So you get at most
   about 12.6 tokens per second from the weights alone, for dense batch-one
   decode that streams every weight once per token.
2. Decode has one row, so a matrix tile would be mostly padding, and MLX gives
   it a reduction kernel that dequantizes in registers, while prefill has many
   rows and wants the tiled matrix-multiply path with dequantization in the
   loader; the two shapes get different kernels by the implementers' choice, as
   attention does.
3. It costs unpacking arithmetic every time packed weights are loaded, instead
   of once for the whole matrix, with reuse inside a tile. It is acceptable when
   that arithmetic fits under the memory time, which the stock kernels achieve,
   and a cached 16-bit copy would restore the traffic the quantization removed.

Part 7 opens with the three questions that the measured wins in it answer, and
that most of the measured failures skipped.

Next: [The three questions](24-the-three-questions.md)

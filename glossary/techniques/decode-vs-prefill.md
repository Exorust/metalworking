# Decode vs Prefill

**LLM inference is two different workloads wearing one API: prefill processes the
prompt as big matmuls (compute-bound), decode generates one token at a time as
dot products against the whole KV cache (bandwidth-bound). Every serious kernel
family on this platform ships separate kernels for the two.**

CUDA equivalent: the same split exists (vLLM et al. specialize decode), but the
platform sharpens it. Decode streams the entire KV cache and all model weights per
token, pure [bandwidth](arithmetic-intensity.md), and on Apple Silicon
[bandwidth is the scarce resource](../machine/unified-memory.md). That's why
**decode tokens/sec tracks the memory-bandwidth spec almost linearly** across
chip tiers, and why the [buying advice in the war stories](../war-stories/cheap-tricks.md)
is "read the bandwidth line, ignore the core count."

Why decode can't use the matrix kernels: with one query row, there's no 8×8 tile
to feed [`simdgroup_matrix`](../metal/simdgroup-matrix.md), and the matrix path
would compute mostly on padding. A one-row attention is a *reduction*, not a
matmul. So the vec kernels change species entirely:

- **MLX**: `sdpa_vector.h`, [dispatched](../mlx/mx-fast.md) when
  `query_sequence_length <= 8`. Wide vector loads, [simdgroup
  reductions](../machine/simdgroup.md), no matrix ops.
- **llama.cpp**: `kernel_flash_attn_ext_vec`. Because one row of work can't
  fill the GPU, it **splits the KV cache across simdgroups** and merges the
  partial softmaxes in a second kernel (`_vec_reduce`), using the
  [online-softmax correction identity](online-softmax.md) for the merge: the
  two-pass shape that [emulated float atomics](../machine/special-paths.md) force.
- The [quantized matmul family](../mlx/quantization.md) splits identically:
  QMV (decode: matrix-vector, dequantize-in-registers) vs QMM (prefill: tiled
  GEMM). The [MTPLX war story](../war-stories/three-questions.md) lives in
  the crack between them, where speculative decode's M=3-6 fits neither tuning.

Practical readings of the split:

- **Different optimizations apply.** Prefill wants everything in the
  [GEMM school](tiling.md); decode wants fewer bytes
  ([quantization](../mlx/quantization.md), KV-cache compression) and fewer syncs
  ([one eval per token](../mlx/lazy-evaluation.md)).
- **Benchmark them separately.** A "2× faster attention" that speeds prefill only
  is invisible in a chat loop; the
  [classic failed-kernel post-mortem](../war-stories/the-failures.md) is a
  synthetic prefill win with zero end-to-end effect.
- **Batch size moves the boundary.** Speculative decoding, parallel sampling, and
  serving push decode toward small-M matmul: the least-tuned regime in every
  kernel family, and historically where the
  [cheap wins hide](../war-stories/three-questions.md).

Next section: [Kernels, the case studies](../kernels/gemm-tiled.md)

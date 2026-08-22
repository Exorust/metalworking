# How an Op Becomes a Kernel

**Between `mx.matmul(a, b)` and a [`simdgroup_matrix`](../metal/simdgroup-matrix.md)
instruction sits one C++ hop: the primitive's `eval_gpu`, which picks a kernel
variant, picks tile sizes, binds [function constants](../metal/function-constants.md),
and encodes the dispatch.**

CUDA equivalent: the dispatcher layer of PyTorch → cuBLAS heuristics — except
open, small, and readable. Knowing where these decisions live is what separates
"my model is slow" from "my shapes fall off the fast path," and the war stories'
[question 2 (unlock an existing fast path)](../war-stories/three-questions.md) is
answered by reading exactly these files.

The chain for a matmul, concretely: the lazy graph hands `Matmul` to the
[scheduler](lazy-evaluation.md) → `Matmul::eval_gpu` in
`mlx/backend/metal/matmul.cpp` normalizes strides/transposes, then picks the
[steel](steel.md) template parameters from device class and problem shape:

```cpp
#define GEMM_TPARAM_MACRO(devc)                                           \
  if (devc == 'g' || devc == 'p') { /* Small device */                    \
    ...
  } else if (devc == 'd') { /* Large device */                            \
    if ((size_t)batch_size_out * M * N >= 1ul << 20) { /* large matmul */ \
      if (out.dtype() != float32) { /* half and bfloat */                 \
        if (2 * std::max(M, N) > K) { /* Reasonable K */                  \
          bm = 64; bn = 64; bk = 16; wm = 1; wn = 2;                      \
        } else if (!transpose_a && transpose_b) { /* nt with large k */   \
          bm = 64; bn = 32; bk = 32; wm = 2; wn = 2;                      \
```
— [`mlx/backend/metal/matmul.cpp:89-124`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/matmul.cpp#L89-L124), abridged and reformatted

Read what that macro is: a hand-tuned lookup from (chip class, size, dtype,
transpose pattern) to the [tile shape](../kernels/steel-gemm-fused.md) — cuBLAS's
secret heuristics, in greppable form. The kernel name is then assembled as a
string (`steel_gemm_fused_...bm64_bn64...`), the
[pipeline is fetched or JIT-built](../metal/compilation-pipeline.md), alignment
[function constants](../metal/function-constants.md) are bound, and the dispatch
is encoded.

Branches before steel worth knowing, because falling into the wrong one is a
classic silent slowdown: **matrix-vector shapes** route to `gemv` kernels rather
than GEMM; **very skinny/small** cases have split-K and non-steel paths;
**[quantized](quantization.md)** weights go to an entirely different kernel family
whose fast path is [shape-sensitive in its own ways](../war-stories/three-questions.md);
and `mx.fast.scaled_dot_product_attention` has its
[own dispatch gate](mx-fast.md) deciding fused-vs-fallback. The
[MTPLX war story](../war-stories/three-questions.md) — 2.24× from ~10 lines —
came from noticing that multi-token decode (M=3-6) fell between the matrix-vector
path and the tile sizes this macro assumes.

Next: [Steel](steel.md)

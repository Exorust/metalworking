# Fusion and Epilogues

**Fusion means computing a chain of operations in one kernel so intermediates
never touch memory. On the platform where
[bandwidth is the binding constraint](arithmetic-intensity.md), it is the single
most profitable technique in this glossary.**

CUDA equivalent: kernel fusion, same concept, but promoted from "nice
optimization" to first principle. An unfused elementwise chain pays full
round-trip [DRAM traffic](../machine/unified-memory.md) per op plus
[dispatch overhead](../metal/command-buffers.md) per kernel, and elementwise ops
have [arithmetic intensity near zero](arithmetic-intensity.md): all cost, no
reuse. Every layer of the local stack embodies the response:

- **Epilogue fusion**, the mechanical form. A GEMM's ending (`α·AB + β·C`, bias,
  activation) is applied to results *while they're still in
  [registers](register-blocking.md)*, on the way to their one memory write.
  Steel makes the ending a plug-in template parameter:

  ```cpp
  template <typename OutT, typename InT>
  struct TransformAxpby {
    ...
    METAL_FUNC OutT apply(InT x, OutT c) const thread {
      return static_cast<OutT>(
          x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
    }
  };
  ```
  — [`steel/gemm/transforms.h:38-54`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/transforms.h#L38-L54), abridged;
  applied in [`BlockMMA::store_result`](../kernels/steel-blockmma.md). Anything
  expressible as an epilogue costs zero extra memory passes.

- **Fused ops as products**: [`mx.fast`](../mlx/mx-fast.md)'s attention and norm
  kernels exist because the unfused graphs were bandwidth disasters.
  [Flash attention](flash-attention.md) is the genre's masterpiece; the "fused
  intermediate" is an entire L×L matrix that never exists.
- **Automatic fusion**: [`mx.compile`](../mlx/mx-compile.md) fuses elementwise
  chains mechanically; [Luminal](https://docs.luminalai.com/blog/gpu) pushes the
  same idea to search-based extremes (one command buffer per forward pass; flash
  attention rediscovered by e-graph search).
- **Fusion into quantized loops**: the
  [TurboQuant war story](../war-stories/sparse-v.md)'s KV-cache compression ran at
  0.28× until its quantize/dequantize steps were fused into the surrounding
  kernels; 0.98× after. The difference between a technique that works on paper
  and one that ships was *only* the fusion.

The discipline the technique imposes: before writing any new kernel, ask whether
the win is actually "delete a round trip"
([question 1 of the three questions](../war-stories/three-questions.md)). Most
measured wins in the war stories are fusions or deletions wearing other names.

Next: [Online softmax](online-softmax.md)

# Sparse-V: The Canonical Win

**+22.8% decode at 32K context, perplexity unchanged, from three lines, after
fourteen clever micro-optimizations all lost to baseline. The single most
instructive optimization story on the platform.**
([r/LocalLLaMA thread](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/);
Qwen3.5-35B-A3B on an M5 Max, llama.cpp fork)

The setup: llama.cpp decode at long context, where the
[bandwidth-bound](../techniques/decode-vs-prefill.md) cost is streaming and
[dequantizing](../mlx/quantization.md) the KV cache every token. The author tried
fourteen micro-optimizations: tile shapes, load widths, the usual
[GEMM-school](../techniques/tiling.md) repertoire. All fourteen lost to baseline.
(The defaults are good; [this is the norm](the-failures.md).)

The observation that worked, a pure
[question-1 delete](three-questions.md): [flash
attention](../techniques/flash-attention.md) computes its softmax weights
*before* touching V. So by the time the `O += P·V` loop runs, the kernel already
knows which positions have negligible attention weight, and **dequantizing V for
a position whose weight is ~zero is pure waste**. At 32K context, that's ~90% of
positions. The entire change:

```metal
#if TURBO_SPARSE_V
    // SPARSE V DEQUANT: skip V for positions with negligible attention weight.
    const float attn_weight = float(ss[NE*cc + ty]);
    if (attn_weight < 1e-6f) continue;  // skip negligible positions
#endif
```
— in the [llama.cpp FA kernel's](../kernels/llamacpp-attention.md) accumulate
loop; live in the author's fork at `ggml-metal.metal:8926`, with an ablation
writeup in the companion repo (fetched by this repo's `fetch.sh` as
`llamacpp-turboquant-fork` / `turboquant-plus-llamacpp`).

Why it teaches so much per line:

- **The skip is only visible from inside the fused kernel.** No graph-level
  system could find it. It exploits [flash attention's](../techniques/flash-attention.md)
  ordering (weights before V) plus [quantization's](../mlx/quantization.md)
  pay-per-use dequant cost. Understanding the kernel *was* the optimization.
- **It deletes bandwidth, not compute**, the
  [only deletion that matters here](../techniques/arithmetic-intensity.md).
- **It's threshold-gated, and honest about it**: `1e-6` weights contribute
  nothing at fp16 output precision, verified by unchanged perplexity, the
  [methodology](../metal/profiling.md) the failures page shows most projects
  skip.

Same author, same insight one level up: **TurboQuant KV compression**
([thread](https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/)),
Walsh-Hadamard rotation plus quantization of the KV cache, 4.6× compression at
0.98× fp16 decode speed. The naive version ran 0.28×; fusing the quant/dequant
into the surrounding kernels was
[the entire difference](../techniques/fusion-and-epilogues.md).

Next: [The failures](the-failures.md)

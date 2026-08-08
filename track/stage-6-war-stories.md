# Stage 6 — War stories

**Needs:** everything under the community section of `fetch.sh` &nbsp;·&nbsp; **Time:** open-ended; this is the stage you re-read

You now know how the reference kernels work. This stage is about what happens when real
people try to beat them — sourced from r/LocalLLaMA threads and the repos behind them,
including the failures, which are the most instructive part. The recurring plot: **the
14 clever micro-optimizations lose; the 3-line deletion of work wins.**

## The three questions

Every win below is an instance of one of these. Ask them, in order, before writing any
custom kernel:

1. **Can I delete work?** (sparse-V, vocab compaction)
2. **Can I unlock an existing fast path?** (DFlash's head_dim patch, MTPLX's small-M retune)
3. **Can I cut dispatch/sync overhead?** (single `mx.eval()`, command-buffer pre-encode)

Only if all three come up empty does "write a better kernel" have decent odds.

## The canonical story: sparse-V dequant skip

[Thread](https://www.reddit.com/r/LocalLLaMA/comments/1s56g07/) ·
`code/llamacpp-turboquant-fork` · `code/turboquant-plus-llamacpp`

The author tried 14 micro-optimizations; all lost to baseline. Then: flash attention
computes softmax weights *before* touching V — so dequantizing V for positions whose
attention weight is ~zero is pure waste. At 32K context, that's ~90% of positions.
The entire change, in the FA kernel's `O += P*V` loop:

```metal
#if TURBO_SPARSE_V
    // SPARSE V DEQUANT: skip V for positions with negligible attention weight.
    const float attn_weight = float(ss[NE*cc + ty]);
    if (attn_weight < 1e-6f) continue;  // skip negligible positions
#endif
```

**+22.8% decode at 32K** (Qwen3.5-35B-A3B, M5 Max), perplexity unchanged. Live in
context at `code/llamacpp-turboquant-fork/ggml/src/ggml-metal/ggml-metal.metal:8926`;
host-side gating in `ggml-metal-device.m:244`; full ablation writeup in
`code/turboquant-plus-llamacpp/docs/papers/sparse-v-dequant.md`.

## The cautionary tales (read these first)

- **DFlash on MLX** — `code/dflash-mlx` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1simszl/). Speculative decoding,
  3.34x, bit-for-bit identical output. The Metal lesson: **every hand-written custom
  kernel — batched GEMV, fused SiLU, custom SDPA — came back 0.5–0.8x slower than stock
  MLX steel and was reverted.** There are deliberately no `.metal` files in this repo.
  The real wins: a head_dim patch to unlock MLX's fast `steel_attention` SDPA path, and
  halving GPU↔CPU syncs (single `mx.eval()` per step). (The original poster's 85 tok/s
  implementation is closed-source; this is the independent open port from the same
  thread, 79.6 tok/s with confirmed parity.)
- **OpenEvolve's evolved kernel** — `code/openevolve/examples/mlx_metal_kernel_opt/` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1lm98z7/). LLM-evolutionary
  search over Metal source for Qwen3 GQA attention. Its own `EVOLUTION_ANALYSIS.md`
  ("Why Optimization Failed") reports the best evolved kernel is **3.2% slower than the
  MLX baseline on average**. Kept here as a methodology study: evolution improved
  -11.5% → -3.2% and never crossed zero.
- **"I beat MLX's SDPA... then discovered it was useless"** —
  [the writeup](https://medium.com/@rajveer.rathod1301/i-built-a-flashattention-kernel-that-beat-mlxs-sdpa-then-i-discovered-it-was-useless-e4ce6ebf953c).
  1.2–1.3x on synthetic benchmarks, 0% end-to-end. The methodology check to run on your
  own numbers.

## The wins

- **TurboQuant KV compression** — `code/turboquant-mlx` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1s5vhf6/). Walsh-Hadamard
  rotation + scalar/polar quantization of the KV cache: 4.6x compression at 0.98x fp16
  decode speed. The naive version ran at 0.28x — **the fused Metal quant/dequant kernels
  are the entire difference** (`turboquant_mlx/metal_kernels_v4.py`). Lesson 1 in action.
- **flash-moe** — `code/flash-moe` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1s7g8ov/). A 209GB MoE at
  20.34 tok/s in 128GB RAM: pure C/ObjC/Metal, streaming experts from SSD. Headline
  trick: temporal expert prediction — ~27% of experts recur token-to-token, so prefetch
  them during GPU compute (`metal_infer/infer.m` ~7846). Plus command-buffer pre-encode
  (lesson 4) and a Metal-4 NAX tensor path (`nax_gemm.metal`). The thread lists all 28
  failed experiments alongside the 8 that worked.
- **MTPLX** — `code/mtplx` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1t3zuvy/). 2.24x (28→63 tok/s)
  from multi-token prediction using the model's own MTP heads. The kernel part: stock
  MLX quantized matvec stalls at the small-M shapes (M=3–6) that verify produces; ~10
  lines of Metal — wider simdgroups + unrolling, bit-exact — fixed it
  (`mtplx/verify_qmv.py`). Question 2, textbook execution. Also 34 further kernel
  modules under `mtplx/mtplx/kernels/` if you want a browse.
- **ANE reverse engineering** — `code/ane-reverse-engineering` ·
  [thread](https://www.reddit.com/r/LocalLLaMA/comments/1rhx5pc/). Explicitly **not
  Metal** — training on the Apple Neural Engine via private APIs, backward passes
  expressed as ANE convolutions, ~6.6 TFLOPS/watt at 2.8W (vs ~1 for the GPU). Included
  for the hack value and for `training/m5result.md`, which reports the negative probe
  results too.
- **Draw Things MFA 2.0** — `code/drawthings-mfa` (the `lib/nnc/mfa` subtree of
  liuliu/ccv) · [engineering blog](https://engineering.drawthings.ai/). The production
  descendant of stage 4's implementation A: C++-generated FA with backward pass, int8,
  and sparse-indexed variants; per-shape, per-GPU-generation descriptors
  (`kernels/AttentionDescriptor.cpp`). Up to 94% faster than ggml implementations for
  image-model attention. Their Substack is the best ongoing Metal-perf publication.

## Cheap tricks that are actually good

- `sudo sysctl iogpu.wired_limit_mb=N` — raise the GPU wired-memory ceiling
  ([classic thread](https://www.reddit.com/r/LocalLLaMA/comments/186phti/)): a 192GB
  machine goes from ~140GB to ~184GB usable "VRAM".
- Memory bandwidth predicts tok/s almost linearly across M-series tiers
  ([M5 Max megathread](https://www.reddit.com/r/LocalLLaMA/comments/1rqnpvj/)) — check
  the spec sheet before profiling anything.
- MTPLX found that locking fans to max cut long-run throughput decay from 50% to 6.7%.
  Thermals are part of your benchmark methodology.

## Done when

- Given a new "X% faster on Apple Silicon" claim, you instinctively ask: synthetic or
  end-to-end? which of the three questions did it answer? what's the baseline?
- You've read one failure writeup (`EVOLUTION_ANALYSIS.md` or the useless-FlashAttention
  post) closely enough to name the methodological trap it fell into.
- Before your next custom kernel, you write down your answers to the three questions
  first — and you're at peace with the likely answer being "don't write it."

That's the track. [SOURCES.md](../SOURCES.md) has the full annotated list — blogs,
papers, and the people worth following — for staying current.

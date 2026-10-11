# Wins and failures, measured

## Where we are

Chapter 24 gave you the three questions and named the cases behind them. Here
is the evidence: one win read in full, because it teaches more per line
than anything else on the platform; three published failures, because the loss
modes are predictable once you have seen them; the one project that measured the
mechanism; and three near-zero-code wins that belong in every practitioner's kit
along with the risk they carry.

## The canonical win: sparse-V

Three lines of code raised decode throughput by 22.8 percent at a 32K context,
47.0 to 57.7 tokens per second with the project's own turbo3 cache format as the
baseline, with perplexity numerically unchanged on its test corpora. The setting
is llama.cpp decode at long context on an M5 Max, running Qwen3.5-35B-A3B, where
the cost that grows with context is streaming and dequantizing the key-value
cache on every token, Chapter 22; the author's profile put the dequantization
lookups at 14 to 34 percent of decode time. Before this change the author tried
fourteen ways to make that dequantization itself faster, lookup-table layouts,
byte extraction, bit arithmetic, shuffles; the best of them gained 38 percent at
8K on an M2 Pro and the rest were slower than it. Sparse-V came from asking a
different question.

The observation that worked is a pure question-1 deletion. Flash attention
computes its softmax weights before it touches V, Chapter 20. So by the time the
`O += P * V` loop runs, the kernel already knows which positions have negligible
attention weight, and dequantizing V for a position whose weight is nearly zero
is wasted work. The paper estimates that at a 32K context more than 90 percent
of positions fall below its threshold; a direct count on a smaller model of the
same family gave 28 percent at 4K, rising with context. The entire change:

```metal
#if TURBO_SPARSE_V
    // SPARSE V DEQUANT: skip V for positions with negligible attention weight.
    const float attn_weight = float(ss[NE*cc + ty]);
    if (attn_weight < 1e-6f) continue;  // skip negligible positions
#endif
```

It goes in the accumulate loop of llama.cpp's attention kernel, Chapter 21, and
is live in the author's fork at `ggml-metal.metal:8926`, with an ablation
write-up in the companion repository; both are among the pinned repositories
under `code/`. Three properties make it worth studying.

The skip is visible only from inside the fused kernel. An exact graph optimizer
may not introduce it, because it is an approximation; a kernel author can,
because flash attention's ordering, weights before V, exposes the weights in
time to use them, and the pay-per-use dequantization of Chapter 23 makes the
skipped work worth something. The optimization came from understanding the
kernel.

It also deletes loads, dequantization and multiply-adds together. The author's
profile attributes the win to the dequantization cost. On this machine the bytes
and the lookups it removes are what matter, Chapter 3.

And it is threshold-gated and measured, not proved. The test is on the
unnormalized weight `exp(s - M)` against `1e-6`, the output accumulates in
32-bit, and a weight that small times a large V value is not exactly nothing.
The claim is empirical: perplexity numerically identical across the paper's
corpora and context lengths, and retrieval unchanged or better, the methodology
most of the failures below skipped.

The same author's TurboQuant key-value cache compression, a Walsh-Hadamard
rotation plus quantization of the cache, is the project the skip lives in. Its
own optimization history records prefill throughput against a q8_0 cache going
from 0.27 times at first to 0.98 times after five changes, a 16-bit rotation,
vectorizing it, moving the rotation into the graph, block storage and an
optimized dequantization, Chapter 15.

## The failures, and the prior they establish

What happens when you skip the questions? The most instructive documents on the platform are the published negative
results. They share one lesson: your custom kernel will probably lose to stock, and
the loss modes are predictable.

Start with DFlash's reverted kernels. The speculative-decoding port of Chapter 24
reached 3.34 times with bit-identical output, and its repository deliberately
contains no `.metal` files. Every hand-written kernel tried along the way, a
batched matrix-vector multiply, a fused SiLU, a custom attention, benchmarked at
0.5 to 0.8 times stock steel and was reverted. The wins that survived were
question-2 and question-3 moves: the head-dimension patch onto the fused
attention path and one `mx.eval` per step. Several independent projects report
the same 0.5 to 0.8 times experience. Treat it as the prior.

Next is OpenEvolve's evolved kernel. This project ran a language-model-driven
evolutionary search over Metal source for a grouped-query attention kernel. Its
own analysis file, titled "Why Optimization Failed", reports the best evolved
kernel at 3.2 percent slower than the MLX baseline. The search improved from 11.5
percent slower to 3.2 percent slower and never crossed zero. It is useful as a
methodology study: a search can climb a hill efficiently and still be on the wrong hill
when the baseline is a hand-tuned library and the search cannot touch the
dispatch layer above it, Chapter 17.

The third is titled "I beat MLX's attention, then
discovered it was useless." A hand-rolled
flash-attention kernel showed 1.2 to 1.3 times on synthetic benchmarks and zero
percent end to end, because the kernel's share of real run time was too small for
its speedup to register. The hygiene rules of Chapter 10 exist because of this
failure mode, and it is the most common one in public benchmark claims.

Only one project measured the mechanism. MTPLX ported every kernel from a Metal
optimization challenge onto a new model's geometry and published the full
ledger, Apache-2.0: 14 kernels, each checked against stock output and timed in
isolation. The ledger scores 2 isolation wins, 11 losses or neutral results and
1 "no lever"; three entries also failed correctness at some precision or
context. Both wins were prefill fusions where stock MLX has no
matrix-unit-backed path, and the larger one, measured at 4.7 times against a
naive combine, shrank to 1 to 4 percent against the shipped reference, which
already fuses that step, and was left off by default. The ledger's own thesis is
that cross-operation fusion transfers and hand-written matrix-vector and
attention kernels do not, because they race steel's tuned kernels, which a
custom `mx.fast.metal_kernel` cannot call (it can issue `simdgroup_matrix`
instructions of its own, and MTPLX's kernels do, Chapter 21). This explains the
0.5 to 0.8 prior: the fast paths live below the API a custom kernel is written
against. Custom kernels win where there is no fast path to race, and even then
against the baseline that matters.

The checklist for reading, or making, an "X percent faster on Apple Silicon"
claim:

1. Synthetic or end to end? Amdahl's law eats kernels, Chapter 10.
2. Which of the three questions did it answer? "None, I wrote a better kernel" predicts reversion.
3. What is the baseline: stock steel or llama.cpp at the right shapes, or a straw man?
4. Prefill or decode? A phase-specific win shows only in that phase's metric, Chapter 22.
5. Were thermals controlled? Sustained throughput differs from the first second, as the next section shows.

## Cheap tricks, with their risk

What can you get for almost no code? Three measured wins, each with the risk
typical of this platform.

**Raise the GPU wired-memory ceiling.** By default macOS caps how much of unified
memory the GPU may wire; `sudo sysctl iogpu.wired_limit_mb=N` raises it. A 192
GB machine goes from about 140 GB to about 184 GB of usable GPU memory, the
difference between a 70-billion parameter model fitting or not. The setting is
undocumented, resets on reboot, and has occasionally moved between `sysctl` names
across macOS versions. It is the copy-engine story of Chapter 14 in miniature,
so use it knowingly.

**Read the bandwidth line on the spec sheet before profiling anything.** Across
the M-series tiers, decode tokens per second track memory bandwidth almost
linearly, because decode is bandwidth bound, Chapter 22. This has two uses. The
first is buying decisions: the Max chip's 4 times bandwidth over the base chip
buys about 4 times decode for dense batch-one models, while extra GPU cores
beyond that buy prefill, which is time to first token rather than tokens per
second. The second is sanity checks: if your tokens per second are far from
bandwidth divided by model bytes, find out why, long-context cache traffic, a
missed fast path or dispatch overhead before anything in Chapter 26.

**Control thermals or your benchmark lies.** MTPLX measured long-run throughput
decay dropping from 50 percent to 6.7 percent by locking the fans at maximum.
Laptop and fanless chips throttle under sustained load; a benchmark's first ten
seconds and its steady state behave like different machines. On this platform,
fan control or a desktop chip is part of measurement methodology.

The Neural Engine also deserves a mention. It is the third matrix engine on
the die after the GPU and AMX, Chapter 28, reaches about 6.6 TFLOPS per watt
against about 1 for the GPU, and speaks only Apple's model frameworks, Core ML
and, for generative workloads in the macOS 27 cycle, Core AI, which adds a
documented hook for custom Metal kernels. The community has tried to train on it
through private interfaces, which is impressive work and also a warning. For
Metal work, what matters is knowing that some workloads, quantized
convolution-shaped inference at low power, have a better home than the GPU, and
that Apple's own frameworks may route there without telling you.

## Check yourself

1. The sparse-V skip tests `attn_weight < 1e-6f`. Why is that threshold safe,
   and what evidence did the author give?
2. A project reports a custom attention kernel at 1.25 times the MLX baseline on
   a synthetic benchmark. Which two items of the checklist do you ask about
   first?
3. The MTPLX ledger's two wins were both prefill fusions. Why does that fit the
   ledger's thesis?

### Answers

1. It is not proved safe: the test is on the unnormalized weight, the output
   accumulates in 32-bit, and many small skipped terms could add up. It is
   measured safe: perplexity numerically identical across the paper's corpora
   and context lengths, and retrieval unchanged or better.
2. Synthetic or end to end, because the published case with the same number
   measured zero percent end to end; and which question it answered, because "a
   better kernel" against stock steel predicts reversion at 0.5 to 0.8 times.
3. Because a fusion across operations does not race steel's tuned kernels, it
   deletes traffic between kernels that the baseline it was measured against
   did not fuse, while a hand-written matrix-vector or attention kernel competes
   with a fast path a custom kernel cannot call; and the ledger's own follow-up
   shows the baseline matters, since against the already-fused reference the
   win nearly vanished.

When none of the three questions produces the win and the arithmetic says a
kernel should be fast but is not, the next chapter is the last resort: reading
what the compiler emitted.

Next: [When it is still slow: reading the machine code](26-when-it-is-still-slow.md)

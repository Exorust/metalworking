# Wins and failures, measured

## Where we are

Chapter 24 gave the three questions and named the cases behind them. This
chapter is the evidence: one win read in full, because it teaches more per line
than anything else on the platform; three published failures, because the loss
modes are predictable once you have seen them; the one project that measured the
mechanism; and three near-zero-code wins that belong in every practitioner's kit
along with the risk they carry.

## The canonical win: sparse-V

Plus 22.8 percent decode throughput at a 32K context, perplexity unchanged, from
three lines, after fourteen clever micro-optimizations had all lost to the
baseline. The setting is llama.cpp decode at long context on an M5 Max, running
Qwen3.5-35B-A3B, where the bandwidth-bound cost is streaming and dequantizing the
key-value cache on every token, Chapter 22. The author tried fourteen
micro-optimizations first: tile shapes, load widths, the usual matrix-multiply
repertoire of Part 4. All fourteen lost to the baseline, which is the norm.

The observation that worked is a pure question-1 deletion. Flash attention
computes its softmax weights before it touches V, Chapter 20. So by the time the
`O += P * V` loop runs, the kernel already knows which positions have negligible
attention weight, and dequantizing V for a position whose weight is nearly zero
is wasted work. At a 32K context that is about 90 percent of positions. The
entire change:

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
under `code/`. Three things make it teach so much per line.

- **The skip is visible only from inside the fused kernel.** No graph-level
  system could find it. It exploits flash attention's ordering, weights before
  V, plus the pay-per-use dequantization cost of Chapter 23. Understanding the
  kernel was the optimization.
- **It deletes bandwidth, not compute**, the only deletion that matters on this
  machine, Chapter 3.
- **It is threshold-gated and measured.** Weights below `1e-6` contribute
  nothing at 16-bit output precision, and the claim was verified by unchanged
  perplexity, the methodology that most of the failures below skipped.

Same author, same insight one level up: TurboQuant key-value cache compression,
a Walsh-Hadamard rotation plus quantization of the cache, 4.6 times compression
at 0.98 times 16-bit decode speed. The naive version ran at 0.28 times, and
fusing the quantize and dequantize steps into the surrounding kernels was the
entire difference, Chapter 15.

## The failures, and the prior they establish

The most instructive documents on the platform are the published negative
results. The shared lesson: your custom kernel will probably lose to stock, and
the loss modes are predictable.

**DFlash's reverted kernels.** The speculative-decoding port of Chapter 24
reached 3.34 times with bit-identical output, and its repository deliberately
contains no `.metal` files. Every hand-written kernel tried along the way, a
batched matrix-vector multiply, a fused SiLU, a custom attention, benchmarked at
0.5 to 0.8 times stock steel and was reverted. The wins that survived were
question-2 and question-3 moves: the head-dimension patch onto the fused
attention path and one `mx.eval` per step. Several independent projects report
the same 0.5 to 0.8 times experience. Treat it as the prior.

**OpenEvolve's evolved kernel.** A language-model-driven evolutionary search
over Metal source for a grouped-query attention kernel, whose own analysis
file, titled "Why Optimization Failed", reports the best evolved kernel at 3.2
percent slower than the MLX baseline. The search improved from 11.5 percent
slower to 3.2 percent slower and never crossed zero. Kept as a methodology
study: a search can climb a hill efficiently and still be on the wrong hill
when the baseline is a hand-tuned library and the search cannot touch the
dispatch layer above it, Chapter 17.

**"I beat MLX's attention, then discovered it was useless."** A hand-rolled
flash-attention kernel showing 1.2 to 1.3 times on synthetic benchmarks and
zero percent end to end: the kernel's share of real run time was too small for
its speedup to register. The hygiene rules of Chapter 10 exist because of this
failure mode, and it is the most common one in public benchmark claims.

**The mechanism, measured.** The MTPLX project ported every kernel from a Metal
optimization challenge onto a new model's geometry and published the full
ledger, Apache-2.0: 15 kernels, GPU-validated, individually timed. The ledger
scores 2 wins, 11 losses and 1 "no lever", and both wins were prefill fusions
where stock MLX has no matrix-unit-backed path. The ledger's own thesis is that
cross-operation fusion transfers and hand-written matrix-vector and attention
kernels do not, because they race steel's matrix multiply, which
`mx.fast.metal_kernel` cannot reach, Chapter 21. That is the 0.5 to 0.8 prior
explained: the fast paths live below the API a custom kernel is written against.
Wins happen where no fast path exists to race.

The checklist for reading, or making, an "X percent faster on Apple Silicon"
claim:

1. Synthetic or end to end? Amdahl's law eats kernels, Chapter 10.
2. Which of the three questions did it answer? "None, I wrote a better kernel"
   predicts reversion.
3. What is the baseline: stock steel or llama.cpp at the right shapes, or a
   straw man?
4. Prefill or decode? A win in one is invisible in the other, Chapter 22.
5. Were thermals controlled? Sustained is not first-second, below.

## Cheap tricks, with their risk

Three measured, zero-code or near-zero-code wins, plus the reminder that they
carry the platform's signature risk.

**Raise the GPU wired-memory ceiling.** By default macOS caps how much of unified
memory the GPU may wire; `sudo sysctl iogpu.wired_limit_mb=N` raises it. A 192
GB machine goes from about 140 GB to about 184 GB of usable GPU memory, the
difference between a 70-billion parameter model fitting or not. Undocumented,
reset on reboot, and occasionally moved between `sysctl` names across macOS
versions: the copy-engine story of Chapter 14 in miniature, run knowingly.

**Read the bandwidth line on the spec sheet before profiling anything.** Across
the M-series tiers, decode tokens per second track memory bandwidth almost
linearly, because decode is bandwidth, Chapter 22. Two uses: buying decisions,
since the Max chip's 4 times bandwidth over the base chip buys about 4 times
decode while extra GPU cores beyond that buy prefill, not chat latency; and
sanity checks, since if your tokens per second are far from bandwidth divided by
model bytes, something is broken and Chapter 26 is where to look.

**Control thermals or your benchmark lies.** MTPLX measured long-run throughput
decay dropping from 50 percent to 6.7 percent by locking the fans at maximum.
Laptop and fanless chips throttle under sustained load; a benchmark's first ten
seconds and its steady state are different machines. Fan control, or a desktop
chip, is part of measurement methodology here.

One honorable mention: the Neural Engine exists. It is the third matrix engine on
the die after the GPU and AMX, Chapter 28, reaches about 6.6 TFLOPS per watt
against about 1 for the GPU, and speaks only Apple's model frameworks, Core ML
and, for generative workloads in the macOS 27 cycle, Core AI, which adds a
documented hook for custom Metal kernels. The community's effort to train on it
through private interfaces is impressive and cautionary in equal measure. For
Metal work its relevance is knowing that some workloads, quantized
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

1. Because a weight below `1e-6` contributes nothing at 16-bit output precision,
   so skipping its V contribution cannot change the result; the evidence was
   unchanged perplexity on the full model.
2. Synthetic or end to end, because the published case with the same number
   measured zero percent end to end; and which question it answered, because "a
   better kernel" against stock steel predicts reversion at 0.5 to 0.8 times.
3. Because a fusion across operations does not race steel's matrix multiply, it
   deletes traffic between kernels the library does not fuse, while a
   hand-written matrix-vector or attention kernel competes with a fast path that
   `mx.fast.metal_kernel` cannot reach.

When none of the three questions produces the win and the arithmetic says a
kernel should be fast but is not, the next chapter is the last resort: reading
what the compiler emitted.

Next: [When it is still slow: reading the machine code](26-when-it-is-still-slow.md)

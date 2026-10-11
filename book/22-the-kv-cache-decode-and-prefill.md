# The KV cache, decode, and prefill

## Where we are

Chapter 21 ended on the case every attention kernel in it refuses: one query row.
That case is generation, and it is where most of a chat's tokens are produced. This chapter separates the two workloads that share the name
"inference", explains why generation cannot use the matrix kernels, and names
the data structure whose size becomes generation's latency on this platform.

## Two workloads, one API

Language-model inference is two different workloads behind one API. Prefill
processes the prompt: every token's query against every key it may attend to,
which is a big matrix multiply for the scores, then a softmax and a multiply by
V, and typically compute bound by the arithmetic of Chapter 3. Decode generates
one token at a time: one query row against the whole history, a dot product per
key and a weighted sum, and typically bandwidth bound. For a dense model at
batch size one those regimes hold; short prefills, sliding windows,
mixture-of-experts routing and batching all move the balance. Every serious
kernel family on this platform ships separate kernels for the two.

The split exists in CUDA too; serving systems such as vLLM specialize decode.
The split matters more on this platform. Decode streams the entire key-value
cache and all the model's weights for every token it generates, which is pure
bandwidth work, and on Apple Silicon bandwidth is the scarce resource. That is why generation speed in tokens
per second tracks the memory-bandwidth spec almost linearly across chip tiers,
and why the buying advice of Chapter 25 is to read the bandwidth line and ignore
the core count.

![Prefill as a fat compute-bound matrix multiply versus decode as one query row streaming the entire KV cache, bandwidth bound](../diagrams/decode-vs-prefill.svg)

*The two shapes. Prefill has tiles to feed the matrix hardware; decode is one
green row against everything in amber, which is why its speed is a bandwidth
number.*

## The KV cache: the structure decode streams

The key-value cache stores every past token's key and value vectors so that
attention never recomputes them. It is the same structure everywhere; vLLM's
paged version made it famous. On bandwidth-bound hardware its size sets a floor under decode latency, because
a dense model at batch one streams all of it on every token; the floor is an
ideal streaming estimate, and dequantization, reductions and dispatch sit on
top of it. That makes it the data structure that matters most on this platform.

Per token, per layer, the cache holds one key vector and one
value vector per key-value head. For a typical dense model of 7 to 8 billion
parameters, about 32 layers with key-value heads times head dimension of about
1,024, in 16-bit, that is 2 × 1,024 × 2 bytes × 32 layers = 131,072 bytes, 128 KiB
per token. A 32,768-token context therefore carries 4 GiB, about 4.29 GB, of
cache, and every generated token streams all of it through the attention
kernel. At the measured 495 GB/s read bandwidth of Chapter 3, 4.29 GB / 495 GB/s
is about 8.7 milliseconds per token for the cache alone, before the weights.

Three consequences follow.

- **Long context turns decode into a cache-streaming benchmark.** The sparse-V
  win of Chapter 25, 22.8 percent faster decode at 32K context, comes from
  deleting cache traffic.
- **Grouped-query attention is a cache compression scheme** as much as an
  attention variant: fewer key-value heads mean a smaller cache and faster
  decode, which is why steel attention implements it as a stride trick, Chapter
  21.
- **Quantizing the cache is the next lever after quantizing the weights**, and it
  is asymmetric in a way weight quantization is not.

The asymmetry was measured by the turboquant-mlx project on Qwen 2.5 7B at a
32K context. Quantizing K to 4 bits or below changed the greedy output: K feeds
the scores that pick the weights, and small score errors can move them. V
tolerated 3-bit quantization: V is consumed by a weighted average whose weights
sum to one, which spreads uncorrelated error. The project's mixed configuration,
8-bit K and 4-bit V, cut active memory from 6.21 GB to 5.08 GB with greedy output
identical to the baseline, and in that measurement decode ran slower, 25.84
against 35.75 tokens per second, because the dequantization inside attention
cost more than the bytes it saved. The working rule is to treat K's precision as
accuracy-critical and V's as the knob; reports of a uniform `kv_bits=4`
corrupting long prefilled contexts fit that pattern. The project kept prefill
in 16-bit and quantized the cache afterwards, so prefill was unchanged by
design.

Layout matters at the kernel level. The simple layout is one contiguous,
append-only allocation per layer. Serving stacks use paged layouts, fixed-size
blocks plus an indirection table, so that many sequences share memory with less
fragmentation, and paged layouts exist on Metal too: the official vLLM plugin ships a paged variable-length attention
kernel, and MTPLX carries a family of paged-attention Metal kernels. For
single-user decode, the contiguous cache plus the split kernels below is all
you need.

## Why decode does not use the matrix kernels

With one query row there is no 8 by 8 tile to feed `simdgroup_matrix`, Chapter
12, and the matrix path would compute mostly on padding. A one-row attention is
a dot product per key and a weighted sum, a reduction shape. So decode runs on a
different kind of kernel, by choice of the implementers, not by necessity.

- **MLX** runs `sdpa_vector.h`, dispatched by the gate of Chapter 21 when the
  query length is at most 8 and its other conditions hold: wide vector loads,
  simdgroup reductions, no matrix operations. Its data flow is not the house
  pattern of Chapter 2: each thread loads keys straight into its own registers
  and accumulates, and threadgroup memory serves the reduction across
  simdgroups, not staging. For long contexts (1,024 keys on a Max or Ultra) it
  switches to a two-pass variant that merges partials in a second kernel.
- **llama.cpp** runs `kernel_flash_attn_ext_vec`. Because one row of work cannot
  fill the GPU, it splits the key-value cache across simdgroups and across
  threadgroups, merges the simdgroups' partial results inside the first kernel
  through threadgroup memory, and merges the threadgroups' partials in a second
  kernel, using the online-softmax correction identity of Chapter 20 for the
  merge. Two passes give that merge a synchronization boundary; partial
  softmaxes carry their own maxima, so even a fast float atomic (Chapter 5)
  could not have summed them directly.
- **The quantized matrix-multiply family splits the same way**, Chapter 23: a
  matrix-vector kernel for decode, dequantizing in registers, against a tiled
  kernel for prefill. The MTPLX case of Chapter 24 falls in the gap between
  them, where speculative decoding's 3 to 6 query rows fit neither tuning.

## Practical readings of the split

**Different optimizations apply.** Prefill wants everything in the
matrix-multiply school of Part 4. Decode wants fewer bytes (quantized weights and
cache compression) and fewer synchronizations (one evaluation per token, Chapter
16).

**Benchmark them separately.** A "2 times faster attention" that speeds up
prefill only shows up in the time to the first token and not at all in tokens
per second, and the classic failed-kernel post-mortem of Chapter 25 is a
synthetic prefill win with zero end-to-end effect.

**Batch size moves the boundary.** Speculative decoding, parallel sampling and
serving push decode toward small-M matrix multiply, a few query rows at a time.
That is the least-tuned regime in every kernel family, and historically it is
where the cheap wins have been.

## Where the matmul stands

The example has two numbers now. For prefill, the attention kernel of Chapter 21
and the matrix multiply of Part 4 are compute bound. For decode, one token at a
32K context streams about 4.29 GB of cache plus the weights, and the floor is
set by bandwidth: at about 495 GB/s, the cache alone is about 8.7 milliseconds
per token. Everything that speeds decode from here deletes bytes or deletes
waits.

## Check yourself

1. A model has 40 layers, 8 key-value heads of dimension 128, stored in 16-bit.
   How many bytes of cache per token, and how large is the cache at 16,384
   tokens?
2. In the measured configuration, why did quantizing V to 4 bits leave the
   greedy output unchanged while quantizing K to 4 bits changed it?
3. A kernel change makes prefill 2 times faster and leaves decode unchanged.
   What does a chat user see, and what should the benchmark have reported?

### Answers

1. 2 vectors × 8 heads × 128 × 2 bytes × 40 layers = 163,840 bytes, 160 KiB per
   token; at 16,384 tokens that is 163,840 × 16,384 = 2,684,354,560 bytes, 2.5
   GiB.
2. Because V is consumed by a weighted average whose weights sum to one, which
   spreads uncorrelated error, while K feeds the scores that pick those weights,
   so a small score error can change which tokens get the weight. That is the
   project's explanation of its own measurement, not a universal threshold.
3. The time to the first token drops by up to half of its prefill share, and
   the time per generated token does not change. The benchmark should have
   reported prefill throughput, time to first token and decode throughput
   separately, and end to end.

Decode deletes bytes, and for most contexts the largest deletion available is
in the weights: the next chapter is about storing them in 4 bits and
multiplying against them without ever materializing the expanded matrix.

Next: [Quantization as a bandwidth optimization](23-quantization.md)

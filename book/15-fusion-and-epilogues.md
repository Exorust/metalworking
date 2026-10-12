# Fusion and epilogues

## Where we are

You left Chapter 14 with the loop settled. The matrix multiply reaches the practical,
measured ceiling at 4096 by 4096, and the techniques this book covers inside it
are used up. What remains is everything around it. In a typical transformer
block a matrix multiply is often followed by a bias and an activation, with a
normalization nearby, and it is preceded by whatever produced its input. Each of those, run as its own kernel, costs a full pass
over memory. On the platform of Chapter 3, most kernels sit on the bandwidth side of the ridge,
so fusion is the most profitable technique in this book. The matrix multiply's
epilogue in Chapter 13 was the first example of it.

## Why fusion is a first principle here

You know fusion from CUDA, so the question is why it needs a chapter of its own
here. Fusion means computing a chain of operations in one kernel so that the
intermediate arrays are never written to device memory and read back. What
differs here is how much it matters. An unfused elementwise chain pays the full round
trip of device memory traffic for every operation, plus the dispatch overhead of
Chapter 7 for every kernel. Elementwise operations have an arithmetic intensity
near zero, about 0.1 FLOPs per byte for an add (Chapter 3), so that traffic buys
no reuse. On a machine with laptop-class bandwidth, this is among the most expensive
things a bandwidth-bound program does, and the remedy is to avoid the round trip.

Every layer of the local stack uses fusion, in four forms.

## Epilogue fusion: the mechanical form

Where can you put an operation so that it costs no extra pass? At the end of
the multiply. A matrix multiply's ending (alpha × A × B + beta × C, a bias, an activation) is
applied to the results while they are still in registers, before their single
memory write on the unsplit path; steel's split-K variant writes partial
products and reduces them in a later pass, so a nonlinear ending there must
follow that reduction. Chapter 13 showed it hand-written: load the C block, scale, add,
store, per fragment. Steel makes the ending a plug-in template parameter, so the
same kernel can end with any operation that is expressible per element.

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
Source: [`steel/gemm/transforms.h:38-54`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/gemm/transforms.h#L38-L54), abridged

The transform is applied in the matrix-multiply component's `store_result`
(Chapter 18) to every accumulator element in registers before the one store.
`x` is the accumulated product and `c` is the corresponding element of the
additive input C. Steel reads C from its own buffer, and it need not be the
output's old contents. So this one type reproduces the hand-written epilogue of
Chapter 13, and a different type in the same slot gives a different ending.
Anything expressible as an epilogue (a bias, an activation, a scaling) costs no
extra pass over the output; the operands the ending itself needs, a bias vector
or the C matrix, are still read once.

## Fused operations as products

MLX's `mx.fast` namespace exists because the unfused graphs were badly
bandwidth bound. Its attention, its normalizations and its rotary embedding are
each a handful of graph operations fused into one kernel, or two for the
decode attention that combines partial results in a second dispatch (Chapters
21 and 22). Flash attention is the extreme form of fusion. Its "fused
intermediate" is an entire L by L score matrix per head, L the sequence length,
that never exists in memory, and
Chapter 20 derives the algorithm that makes that possible.

## Automatic fusion

You do not have to write every fusion by hand. `mx.compile` (Chapter 19) fuses
elementwise chains mechanically. It traces the
graph once per input shape and data type, turns runs of elementwise operations
into generated kernels, and replays them. Luminal takes the same idea further with search: it uses one
command buffer per forward pass, and it rediscovered flash attention by searching
over fusions. This gives a division of labor to settle before you write any
kernel. The compiler fuses elementwise chains adequately, so custom kernels
belong on the patterns a compiler cannot see: reductions with structure, the
streaming softmax of Chapter 20, and the quantization-aware loops of Chapter 23.

## Fusion into quantized loops

What is fusion worth in a real loop? The
TurboQuant key-value cache compression of Chapter 25 ran prefill at 0.27 times
its q8_0 baseline at first and 0.98 times after five changes, by the project's
own history: a 16-bit rotation (to 0.40), vectorizing it (0.52), moving the
rotation into the graph so it is not a separate pass (0.78), block storage
(1.02) and an optimized dequantization inside the kernels (0.98 at a longer
context). The compression was the same throughout; what changed was how many
separate passes over the cache the arithmetic cost. Two of the five steps are
fusion in this chapter's sense, and they account for about half the gain.

## The discipline

Before writing any new kernel, ask whether the win you are after is "delete a
round trip". That is the first of the three questions of Chapter 24, and most of
the measured wins in Part 7 are fusions or deletions under other names. The test
is the sixty-second arithmetic of Chapter 3. Count the bytes the unfused version
moves and the bytes the fused version moves; if those kernels are bandwidth
bound, the difference divided by the measured bandwidth estimates the time you
can save. Then measure the difference.

## Where the matmul stands

The matrix multiply already fuses its ending. In a model it now needs the
operations around it fused too, which is what the production framework of Part 5
does: steel's epilogue types for the matrix multiply's own ending, `mx.fast` for
attention and normalization, and `mx.compile` for the elementwise chains between
them. The example changes from a kernel to a framework in the next chapter.

## Review the content

1. A transformer block applies a bias and an activation after a matrix multiply,
   as two separate kernels on a 4096 by 4096 32-bit output. How many bytes of
   traffic do the two kernels add, and what does fusing them into the epilogue
   save at 516 GB/s?
2. Why did the TurboQuant compression run at 0.27 times its baseline at first,
   even though it stored fewer bytes of cache, and which of its five fixes are
   fusion?
3. Which patterns does the chapter say to leave to `mx.compile`, and which does it
   say need a custom kernel?

### Answers

1. The output is 4096 × 4096 × 4 bytes = 64 MiB. Each kernel reads and writes
   it once: 128 MiB per kernel, 256 MiB = 268,435,456 bytes for both. At 516
   GB/s that is about 0.52 milliseconds, against 2 × 4096³ / 13.5 TFLOPS = 10.2
   milliseconds for the multiply itself. The fused epilogue adds no output
   traffic; the bias vector it reads is 16 KB.
2. Because the rotation and the dequantization ran as separate passes in
   32-bit, each a round trip over the cache, so the extra passes cost more than
   the smaller cache saved. Moving the rotation into the graph and fusing the
   dequantization into the attention kernel are the fusion steps; the 16-bit
   rotation, vectorization and block storage are not.
3. Elementwise chains go to `mx.compile`. Reductions with structure, streaming
   softmax, and quantization-aware loops need a kernel that already knows the
   pattern, the framework's own fused primitive where one exists and a custom
   kernel where not, because the compiler does not fuse those patterns.

The next part follows the same decomposition into the production framework,
starting with what MLX does with a matrix multiply before any kernel runs.

Next: [Arrays that record: MLX and lazy evaluation](16-arrays-that-record.md)

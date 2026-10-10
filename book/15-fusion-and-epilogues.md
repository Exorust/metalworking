# Fusion and epilogues

## Where we are

Chapter 14 settled the loop. The matrix multiply reaches the practical ceiling at
4096 by 4096, and no further technique inside it changes that. What remains is
everything around it. In a model, a matrix multiply is followed by a bias, an
activation, a normalization, and preceded by whatever produced its input, and
each of those, done as its own kernel, costs a full pass over memory. On the
platform of Chapter 3, where most kernels sit on the bandwidth side of the ridge,
that makes fusion the single most profitable technique in this book, and the
matrix multiply's epilogue in Chapter 13 was the first example of it.

## Why fusion is a first principle here

Fusion means computing a chain of operations in one kernel so that the
intermediate results never touch memory. CUDA has the same concept; what changes
here is its rank. An unfused elementwise chain pays the full round trip of device
memory traffic for every operation, plus the dispatch overhead of Chapter 7 for
every kernel, and elementwise operations have an arithmetic intensity near zero,
about 0.1 FLOPs per byte for an add, Chapter 3. All cost, no reuse. On a machine
with laptop-class bandwidth, that is the most expensive thing a program can do,
and the only lever is not doing the traffic at all.

Every layer of the local stack uses fusion, in four forms.

## Epilogue fusion: the mechanical form

A matrix multiply's ending, alpha × A × B + beta × C, a bias, an activation, is
applied to results while they are still in registers, on the way to their single
memory write. Chapter 13 showed it hand-written: load the C block, scale, add,
store, per fragment. Steel makes the ending a plug-in template parameter, so that
the same kernel can end in any way that is expressible per element.

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

The transform is applied in the matrix-multiply component's `store_result`,
Chapter 18, to every accumulator element in registers before the one store.
`x` is the accumulated product and `c` is the existing output element, so this
one type reproduces the hand-written epilogue of Chapter 13, and a different
type in the same slot gives a different ending. Anything expressible as an
epilogue, bias, activation, scaling, costs zero extra passes over memory, by
construction, because the only memory operations are the ones the kernel was
already going to do.

## Fused operations as products

MLX's `mx.fast` namespace exists because the unfused graphs were bandwidth
disasters. Its attention, its normalizations and its rotary embedding are each a
handful of graph operations fused into one kernel, Chapter 21 for the attention
case. Flash attention is the extreme form of fusion: the "fused intermediate" is
an entire L by L score matrix that never exists in memory, and Chapter 20
derives the algorithm that makes that legal.

## Automatic fusion

`mx.compile` fuses elementwise chains mechanically, Chapter 19: it traces the
graph once, turns runs of elementwise operations into generated kernels, and
replays them. Luminal pushes the same idea to a search-based extreme, one command
buffer per forward pass and flash attention rediscovered by searching over
fusions. The division of labor that falls out is useful to know before you write
any kernel: anything expressible as an elementwise chain, the compiler fuses
adequately, so custom effort belongs on the patterns a compiler cannot see.

## Fusion into quantized loops

The case that turns the principle into a number. The TurboQuant key-value cache
compression of Chapter 25 ran at 0.28 times the baseline speed until its
quantize and dequantize steps were fused into the surrounding kernels, and at
0.98 times after. The difference between a technique that works on paper and one
that ships was only the fusion: the compression was the same, the arithmetic was
the same, and the unfused version paid for its intermediates in traffic.

## The discipline

Before writing any new kernel, ask whether the win you are after is "delete a
round trip". That is the first of the three questions of Chapter 24, and most of
the measured wins in Part 7 are fusions or deletions under other names. The test
is the sixty-second arithmetic of Chapter 3: count the bytes the unfused version
moves, count the bytes the fused version moves, and the difference, divided by
the measured bandwidth, is the time on the table.

## Where the matmul stands

The matrix multiply already fuses its ending. In a model it now needs the
operations around it fused too, which is what the production framework of Part 5
does: steel's epilogue types for the matrix multiply's own ending, `mx.fast` for
attention and normalization, and `mx.compile` for the elementwise chains between
them. The example changes from a kernel to a framework in the next chapter.

## Check yourself

1. A transformer block applies a bias and an activation after a matrix multiply,
   as two separate kernels on a 4096 by 4096 32-bit output. How many bytes of
   traffic do the two kernels add, and what does fusing them into the epilogue
   save at 516 GB/s?
2. Why did the TurboQuant compression run at 0.28 times baseline before fusion,
   even though it moved fewer bytes of cache?
3. Which patterns does the chapter say to leave to `mx.compile`, and which does it
   say need a custom kernel?

### Answers

1. Each kernel reads and writes the 64 MiB output once: 128 MiB per kernel, 256
   MiB for both. At 516 GB/s that is about 0.5 milliseconds, against about 10
   milliseconds for the multiply itself; the fused epilogue adds no traffic.
2. Because its quantize and dequantize steps ran as separate kernels, each paying
   a full round trip for its intermediate; the savings on the cache were smaller
   than the traffic the intermediates cost until they were fused away.
3. Elementwise chains go to `mx.compile`. Reductions with structure, streaming
   softmax, and quantization-aware loops need a custom kernel, because the
   compiler cannot see those patterns.

The next part follows the same decomposition into the production framework,
starting with what MLX does with a matrix multiply before any kernel runs.

Next: [Arrays that record: MLX and lazy evaluation](16-arrays-that-record.md)

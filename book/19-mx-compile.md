# mx.compile: fusing the long tail

## Where we are

Chapter 18 put the matrix multiply into production with its own fused epilogue.
Chapter 15 said that in a model the gains now lie in the operations around it,
and that the compiler fuses the elementwise chains automatically. This chapter is
that compiler. It is smaller than a CUDA engineer expects, on purpose, and knowing
its boundary tells you where custom kernel work starts.

## What it does

`mx.compile` traces a function's lazy graph once, fuses runs of elementwise
operations into generated Metal kernels, and replays the optimized graph on later
calls. In CUDA terms it is a deliberately scoped `torch.compile`: no graph breaks,
no guards-and-recompile drama, no autotuning of the kind Inductor does. The
ambition is narrower because MLX is already lazy, Chapter 16, so the graph exists
anyway; compile adds shape-specialized caching and elementwise fusion, and stops
there. Matrix multiply and attention are already fused kernels, Chapters 18 and
21; the win compile chases is the long tail between them.

Why that tail matters more here than on NVIDIA is the law of Chapter 3. An
unfused elementwise chain, say the SiLU, multiply and add of a transformer's
feed-forward block, writes each intermediate to unified memory and reads it
back. That is traffic on the platform's scarcest resource at an intensity near
zero, plus the per-dispatch cost of Chapter 7 for each tiny kernel. Fusing the
chain into one generated kernel deletes both. This is Chapter 15's lesson
applied automatically. Luminal is the same idea pursued to its extreme: a search
over fusions using equality graphs, one command buffer per forward pass, and
flash attention rediscovered by the search.

## The mechanics, briefly

The first call traces the function with placeholder inputs. Fusable subgraphs
are compiled to Metal Shading Language through the run-time compilation path of
Chapter 6; `compiled.cpp` in the Metal backend generates the source, and the
result is cached against the input shapes and data types. Changed shapes
retrace. `shapeless=True` opts hot functions out of shape specialization where
their kernels permit it. Constants get baked in, and the usual tracing caveats
apply as in JAX: Python side effects run once, at trace time.

A note on what "elementwise" covers, since the boundary matters below. An
elementwise operation produces each output element from the same positions of
its inputs and nothing else: activations, scaling, residual adds, masks. A
reduction, which combines many positions into one, is not elementwise, and
neither is anything that needs to see a whole row before producing a value;
softmax is both. `mx.compile` fuses the first kind and leaves the second to the
kernels that handle it.

Count what a fused chain saves, with the example above. Three elementwise
operations on a 4096 by 4096 16-bit activation, unfused, each read 32 MiB and
write 32 MiB: 192 MiB of traffic and three dispatches. Fused, one kernel reads 32
MiB and writes 32 MiB, 64 MiB and one dispatch. At the measured 516 GB/s of
Chapter 3, the unfused chain's traffic alone is about 0.39 milliseconds and the
fused kernel's about 0.13; the matrix multiply next to it takes about 10. The
saving is real and small per block, which is the shape of every `mx.compile`
win.

## When to reach for it, and when not to

Reach for it in inference step functions and training steps dominated by many
small operations. Typical gains are real but modest, tens of percent rather than
multiples, because the heavy matrix multiplies were already fused and the
elementwise tail was a minority of the time to begin with. The place it earns
the most is a decode step, where the per-dispatch cost of Chapter 7 is paid once
per token and a graph of hundreds of tiny operations becomes a graph of tens.

Do not reach for it on code you are about to profile kernel by kernel, and do
not apply it before the uncompiled version is correct, because tracing hides
Python-level mistakes behind cached graphs: fusion
renames and merges dispatches out from under you, and profiling is hard enough
on this platform, Chapter 9. And do not expect it to help a workload that is one
giant matrix multiply anyway.

## The boundary of custom work

For the kernel engineer, `mx.compile` defines where hand-written kernels start.
Anything expressible as an elementwise chain, the compiler fuses adequately, so
writing it by hand buys little. Custom effort belongs on the patterns the
compiler cannot see: reductions with structure, the streaming softmax of Chapter
20, and the quantization-aware loops of Chapter 23. The next part is about the
first two.

## Where the matmul stands

The example is now a production matrix multiply with its neighbors fused: the
epilogue inside steel, and the elementwise chain around it compiled into one
kernel. The next part leaves matrix multiply for the other kernel that dominates
a language model, attention, where the fused intermediate is a whole matrix.

## Check yourself

1. Why does `mx.compile` not need graph breaks or guards the way `torch.compile`
   does?
2. A step function is 90 percent one large matrix multiply. What does the chapter
   predict `mx.compile` will do for it, and why?
3. You are measuring the cost of each kernel in a decode step with command-buffer
   timestamps. Should the step function be compiled?

### Answers

1. Because MLX is lazy: the graph already exists at evaluation time, so compile
   only traces it, fuses elementwise runs, and caches the result per shape; there
   is no eager execution to break out of.
2. Little: the matrix multiply is already a single fused kernel, so there is no
   elementwise tail to fuse, and the gain is bounded by the other 10 percent.
3. No. Fusion merges and renames dispatches, so the timings no longer map to the
   operations you wrote; measure uncompiled first, then compile.

Part 6 opens with the algorithm that makes attention possible on a bandwidth-bound
machine: a softmax computed in one pass without ever holding the whole row.

Next: [Online softmax and flash attention](20-online-softmax-and-flash-attention.md)

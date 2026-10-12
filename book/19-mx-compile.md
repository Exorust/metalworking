# mx.compile: fusing the long tail

## Where we are

You left Chapter 18 with the matrix multiply in production and its own fused
epilogue. Chapter 15 told you that in a model the gains now lie in the operations
around it, and that the compiler fuses the elementwise chains automatically. Here
is that compiler. It is deliberately smaller than you would expect coming from
CUDA, and knowing its boundary tells you where custom kernel work starts.

## What it does

What does a compiler add to a framework that is already lazy? `mx.compile`
traces a function's lazy graph once, fuses runs of elementwise
operations into generated Metal kernels, and replays the optimized graph on later
calls. In CUDA terms it is a deliberately scoped `torch.compile`. It has no graph
breaks, no guards of the `torch.compile` kind, and no autotuning of the kind
Inductor does; a compiled function retraces on a cache miss, which a changed
shape or data type, a different stream or device, or different constants can
cause. The scope is narrower because MLX is already lazy (Chapter 16). The
graph exists anyway. Compile adds caching per specialization, a simplification
pass that merges duplicate computations and scalar constants and removes no-ops,
and elementwise fusion. Matrix multiply and attention are already fused kernels (Chapters 18
and 21), so what compile targets is the long tail between them.

The tail matters more here than on NVIDIA because of the law of Chapter 3. An
unfused elementwise chain, say the SiLU, multiply and add of a transformer's
feed-forward block, writes each intermediate to unified memory and reads it
back. That is traffic on the platform's scarcest resource at an intensity near
zero, plus the per-dispatch cost of Chapter 7 for each tiny kernel, and fusing
the chain into one generated kernel removes both.

This is Chapter 15's lesson applied automatically. Luminal carries the same idea
to its extreme: it searches over fusions using equality graphs, issues one
command buffer per forward pass, and its search rediscovered flash attention.

## The mechanics, briefly

What happens the first time you call a compiled function? It is traced with
placeholder inputs; a Python branch on a placeholder's value, or a read of one, is an error during tracing rather than
a silent fallback. Fusable subgraphs become generated kernels: `compiled.cpp` in
the Metal backend generates their Metal Shading Language at first evaluation and
compiles it through the run-time path of Chapter 6, and the result is cached
against the input shapes, data types and the other cache keys above.
If your inputs change shape, `shapeless=True` opts hot functions out of shape specialization where their
kernels permit it. Constants get baked in. The usual tracing caveats
apply as in JAX: Python side effects run once, at trace time.

Which operations does that cover? An elementwise operation produces each output element from the corresponding
positions of its inputs, after broadcasting, and nothing else: activations,
scaling, residual adds, masks. A reduction, which combines many positions into
one, is not elementwise, and neither is anything whose outputs depend on a whole
row; softmax is the second kind, a same-shaped output built on row-wide
reductions. `mx.compile` fuses the first kind and leaves the second to the
kernels that handle it.

Take the example above and count what fusion saves, with the multiplier and
the residual as full tensors: a 4096 by 4096 16-bit activation is 4096 × 4096 ×
2 bytes = 32 MiB. Unfused, SiLU reads one array and writes one, the multiply
reads two and writes one, the add reads two and writes one: 8 × 32 = 256 MiB of
traffic and three dispatches. Fused, one kernel reads the three inputs and
writes one output: 4 × 32 = 128 MiB and one dispatch. At the measured 516 GB/s
of Chapter 3 that is about 0.52 milliseconds against about 0.26, a saving of
about 0.26 milliseconds in requested traffic that caches may partly absorb; the
32-bit matrix multiply next to it takes about 10 ms, and a 16-bit one less. The
saving per block is real but small. Most `mx.compile` wins look like this.

## When to reach for it, and when not to

So when is it worth turning on? Reach for it in inference step functions and training steps dominated by many
small operations. Typical gains are real but modest, tens of percent rather than
multiples, because the heavy matrix multiplies were already fused and the
elementwise tail was a minority of the time to begin with. It helps most in a
decode step, where the per-dispatch cost of Chapter 7 is paid once
per token and a graph of hundreds of tiny operations becomes a graph of tens.

Do not apply it before the uncompiled version is correct, because tracing hides
Python-level mistakes behind cached graphs. Do not apply it to code you are about
to profile kernel by kernel either: fusion renames and merges dispatches out from
under you, and profiling on this platform is already hard (Chapter 9); nor will
it help a workload that is one giant matrix multiply.

## The boundary of custom work

Where should your hand-written work begin? `mx.compile` defines that line.
The compiler fuses anything expressible as an elementwise chain well enough that
writing it by hand buys little. Custom effort belongs on the patterns the
compiler cannot see: reductions with structure, the streaming softmax of Chapter
20, and the quantization-aware loops of Chapter 23. The next part is about the
first two.

## Where the matmul stands

The example is now a production matrix multiply with its neighbors fused: the
epilogue inside steel, and the elementwise chain around it compiled into one
kernel. The next part leaves matrix multiply for the other kernel that dominates
a language model, attention, where the fused intermediate is a whole matrix.

## Review the content

1. Why does `mx.compile` not need graph breaks or guards the way `torch.compile`
   does?
2. A step function is 90 percent one large matrix multiply. What does the chapter
   predict `mx.compile` will do for it, and why?
3. You are measuring the cost of each kernel in a decode step with command-buffer
   timestamps. Should the step function be compiled?

### Answers

1. Because it traces the function with placeholder inputs under a restricted
   contract: Python control flow that depends on an array's value is an error
   during tracing rather than a point to fall back to eager execution, so there
   is nothing to guard. The lazy graph already exists, so compile only fuses
   and caches it per specialization.
2. Little: compile does not touch the matrix multiply, so whatever fusible tail
   the other 10 percent holds bounds the gain. With the multiply fixed at 90
   percent, even an infinitely fast tail gives 1 / 0.9, about 1.11 times, and
   compilation overhead can eat that.
3. Not at first. Fusion merges and renames dispatches, so the timings no longer
   map to the operations you wrote; isolate and measure the uncompiled kernels
   first (Chapter 9), then compile and time the fused kernels as their own
   targets.

Part 6 opens with the algorithm that makes attention affordable on a
bandwidth-bound machine: an attention output accumulated in one pass over the
scores, without ever holding a whole row of probabilities.

Next: [Online softmax and flash attention](20-online-softmax-and-flash-attention.md)

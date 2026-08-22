# mx.compile

**`mx.compile` traces a function's [lazy graph](lazy-evaluation.md) once, fuses
runs of elementwise operations into generated Metal kernels, and replays the
optimized graph on subsequent calls.**

CUDA equivalent: a deliberately scoped `torch.compile`. No graph breaks, no
guards-and-recompile drama, no Inductor-style autotuning — because the ambition is
narrower: MLX is [already lazy](lazy-evaluation.md), so the graph exists anyway;
compile adds shape-specialized caching and **elementwise fusion**, and stops there.
[Matmul and attention are already fused kernels](steel.md); the win compile chases
is the long tail between them.

Why that tail matters here more than on NVIDIA: an unfused elementwise chain
(say, SiLU → multiply → add in a transformer MLP) writes each intermediate to
[unified memory](../machine/unified-memory.md) and reads it back — pure
[bandwidth](../techniques/roofline.md) burn on the platform's scarcest resource,
plus [per-dispatch overhead](../metal/command-buffers.md) on each tiny kernel.
Fusing the chain into one generated kernel deletes both. This is
[the platform's lesson 1](../techniques/fusion-and-epilogues.md) applied
automatically; [Luminal](https://docs.luminalai.com/blog/gpu) is the same idea
pursued to its extreme (e-graph search over fusions, one command buffer per
forward pass, flash attention *rediscovered* by the search).

Mechanics in brief: first call traces with placeholder inputs; fusable subgraphs
are compiled to MSL through the
[runtime-compilation path](../metal/compilation-pipeline.md) (`compiled.cpp` in
the backend generates the source) and cached against input shapes/dtypes; changed
shapes retrace. `shapeless=True` opts hot functions out of shape-specialization
where their kernels permit. Constants get baked; the usual tracing caveats
(Python side effects run once, at trace time) apply as in JAX.

When to reach for it: inference step functions and training steps dominated by
many small ops — typical gains are real but modest (tens of percent, not
multiples) since the heavy matmuls were already fused. When not to: code you're
about to profile kernel-by-kernel (fusion renames and merges dispatches out from
under you — [profiling is hard enough here](../metal/profiling.md)), or workloads
that are one giant matmul anyway.

For the kernel engineer, `mx.compile` also defines the boundary of custom work:
anything expressible as elementwise chains, the compiler will fuse adequately —
custom [`metal_kernel`](mx-fast.md) effort belongs on the patterns compile can't
see: reductions with structure, [softmax-shaped streaming](../techniques/online-softmax.md),
[quantization-aware loops](quantization.md).

Next section: [Techniques](../techniques/arithmetic-intensity.md)

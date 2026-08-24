# Metal, the API

**Metal is Apple's single graphics-and-compute API: the layer that owns devices,
queues, memory, and kernel execution. It occupies the ground CUDA splits between
the driver API, the runtime API, and half of the toolkit.**

The orientation table, CUDA → Metal:

| CUDA | Metal | Covered in |
|---|---|---|
| `cudaGetDevice` / context | `MTLDevice` (one object, usually one per chip) | this page |
| stream | `MTLCommandQueue` | [Command buffers](command-buffers.md) |
| kernel launch `<<<...>>>` | encode dispatch into a `MTLCommandBuffer` | [Command buffers](command-buffers.md), [Dispatch geometry](dispatch-geometry.md) |
| `cudaMalloc` | `MTLBuffer` from the device (no host/device split: [unified memory](../machine/unified-memory.md)) | this page |
| CUDA C++ | Metal Shading Language | [MSL](msl.md) |
| nvcc, PTX, cubin, JIT | metal compiler, AIR, metallib, pipeline states | [Compilation pipeline](compilation-pipeline.md) |
| template-instantiation JIT / `-arch` fatbins | [function constants](function-constants.md) + template enumeration |
| cuBLAS / cuDNN | [MPS](mps.md) (with caveats) |
| Nsight Compute | nothing comparable; see [Profiling](profiling.md) |

Differences of *shape* rather than vocabulary:

**Everything is explicit and command-buffer-shaped.** CUDA lets you pretend
`kernel<<<grid, block>>>(args)` is a function call. Metal never does: you create a
command buffer, encode one or more dispatches into it with an encoder, commit it,
and optionally wait. This is verbose (the tinygrad-notes article
["Abstraction in Apple's Metal Framework"](https://github.com/mesozoic-egg/tinygrad-notes/blob/72cd3bd80c5d79d81dde30af38f4218c1ae382bf/20240921_metal.md)
walks the raw Objective-C, and is the clearest short intro in existence). But the
explicitness is also the optimization surface:
[batching many dispatches per command buffer is free performance](command-buffers.md).

**Compute and graphics are one API.** There is no "Metal compute edition"; the
`MTLDevice` you encode a GEMM on is the one games render with. For an ML engineer
this mostly means the documentation, tooling, and half the API surface
(render passes, textures) are about someone else's problems, and the compute story
is comparatively under-documented. That gap is why this glossary exists.

**One device, no topology.** No NVLink, no multi-GPU on one machine, no peer
access. Distributed MLX exists over the network, but single-box scaling is
[buy more unified memory](../machine/unified-memory.md), not add cards.

From Python, you rarely touch this API directly: [MLX](../mlx/mlx-overview.md)
wraps it, and PyObjC reaches it when you need raw control (the
[m5-gemm harness](https://github.com/yaroslavvb/m5-gemm/blob/29414bebb522ddacaa009959f2bcdad9f5b3e5cf/metal.py)
is a compact worked example: compile source, build pipeline, fill buffers, encode,
commit, read GPU timestamps).

Next: [MSL](msl.md)

# Command Buffers

**GPU work in Metal is encoded into command buffers: batches of dispatches built
CPU-side, committed to a queue, and executed asynchronously. How you batch them is
one of the biggest free performance levers on the platform.**

CUDA equivalent: a stream, roughly (`MTLCommandQueue` orders work like a stream
does), but the batching unit has no direct CUDA analogue, and that's the part that
matters. In CUDA, each `kernel<<<>>>` is its own submission and the driver
amortizes; you think about launches. In Metal you explicitly build the batch:

```
MTLCommandQueue                  an ordered lane of work (make one, keep it)
  MTLCommandBuffer               one batch: create → encode → commit
    MTLComputeCommandEncoder     writes dispatches into the buffer:
        setComputePipelineState, setBuffer, dispatchThreadgroups, ...
```

Encoding happens on the CPU; `commit()` hands the whole batch to the GPU;
`waitUntilCompleted` (or a completion handler) is the sync point. The queue
schedules buffers in enqueue order (commit order unless you call `enqueue()`
early to reserve a slot), and a default queue allows 64 uncompleted buffers:
asking for a 65th blocks the CPU until the GPU finishes one. Metal 4 replaces
these objects with `MTL4CommandQueue`, reusable `MTL4CommandBuffer`s backed by a
`MTL4CommandAllocator`, and argument tables
([Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)).
Two rules of thumb carry most of the value:

![Two submission patterns: chatty per-op commit and wait leaves the GPU idle between ops; one command buffer per forward pass keeps it continuously busy](../../diagrams/command-buffers.svg)

*The same work, submitted two ways. Every dashed gap on the left is the GPU
waiting for the CPU to wake, read, and re-encode; the right side pays the
per-buffer cost once.*

**Batch aggressively.** Per-dispatch overhead within one command buffer is tiny;
per-command-buffer overhead (commit, scheduling, completion) is not. The
community-converged pattern for inference is **one command buffer per forward
pass**: encode every layer's dispatches, commit once, wait once.
[Luminal](https://docs.luminalai.com/blog/gpu) runs all of Llama 3 8B as a single
command buffer; [flash-moe](../war-stories/three-questions.md) pre-encodes
command buffers before the GPU needs them; [MLX's scheduler](../mlx/lazy-evaluation.md)
exists to batch this way automatically. When a port from CUDA is mysteriously
slow at small batch sizes, count your command buffers first.

**Encoding itself can become the wall.** Host-side encode costs ~2.9 µs per
dispatch, size-independent, and at framework-graph scale that adds up: the
[MTPLX dispatch census](https://github.com/youssofal/MTPLX/blob/ed1c8eea501689b744c13bec6a99ee2d36d26ab5/docs/perf/deepseek-v4-dispatch-levers.md)
measured one bf16 DeepSeek-V4 decode step at **19,809 dispatches in 384
command buffers, with host encode at 56-59 ms against 32-35 ms of GPU
execution**. The encode was exposed, not hidden. The fix was structural
(collapse a normalization chain, one shared `mx.compile` tape): -26% dispatches,
worth ~15 ms of host encode per token. So the earlier rule has a ceiling:
batching hides *per-buffer* overhead, but a graph that emits tens of thousands
of tiny dispatches pays the encoder tax anyway. Count dispatches, not just
buffers.

**Sync as rarely as possible.** `waitUntilCompleted` drains the pipeline: the GPU
goes idle while the CPU wakes, reads, and re-encodes. On
[unified memory](../machine/unified-memory.md) the data is already shared, so the
sync *is* the entire cost of reading a result. The
[DFlash war story](../war-stories/the-failures.md) got a measurable win from
nothing but halving GPU→CPU syncs per decode step (one `mx.eval()` instead of two).

What you don't manage, coming from CUDA: events for cross-stream ordering (a
single queue is ordered; multi-queue is rare in compute work), copy engines
(shared buffers need no host-to-device transfer; blit encoders copy between GPU
resources as ordinary ordered commands), and stream priorities. Timestamps you
*do* get: `gpuStartTime`/`gpuEndTime` (`GPUStartTime`/`GPUEndTime` in
Objective-C and PyObjC) on a completed command buffer are the platform's
[simplest programmatic GPU timing](profiling.md): host seconds on the mach clock,
0.0 until the buffer completes, with `kernelStartTime`/`kernelEndTime` recording
the CPU driver's scheduling of the same buffer. Measure at command-buffer
granularity, one more reason to make command buffers meaningful units of work.

Next: [Synchronization](synchronization.md)

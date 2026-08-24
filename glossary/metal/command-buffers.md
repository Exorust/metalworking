# Command Buffers

**GPU work in Metal is encoded into command buffers: batches of dispatches built
CPU-side, committed to a queue, and executed asynchronously. How you batch them is
one of the biggest free performance levers on the platform.**

CUDA equivalent: a stream, roughly (`MTLCommandQueue` orders work like a stream
does), but the batching unit has no direct CUDA analogue, and that's the part that
matters. In CUDA, each `kernel<<<>>>` is its own submission and the driver
amortizes; you think about launches. In Metal you explicitly build the batch:

```
MTLCommandQueue                  — ordered lane of work (make one, keep it)
  MTLCommandBuffer               — one batch: create → encode → commit
    MTLComputeCommandEncoder     — writes dispatches into the buffer:
        setComputePipelineState, setBuffer, dispatchThreadgroups, ...
```

Encoding is cheap CPU work; `commit()` hands the whole batch to the GPU;
`waitUntilCompleted` (or a completion handler) is the sync point. Two rules of
thumb carry most of the value:

**Batch aggressively.** Per-dispatch overhead within one command buffer is tiny;
per-command-buffer overhead (commit, scheduling, completion) is not. The
community-converged pattern for inference is **one command buffer per forward
pass**: encode every layer's dispatches, commit once, wait once.
[Luminal](https://docs.luminalai.com/blog/gpu) runs all of Llama 3 8B as a single
command buffer; [flash-moe](../war-stories/three-questions.md) pre-encodes
command buffers before the GPU needs them; [MLX's scheduler](../mlx/lazy-evaluation.md)
exists to batch this way automatically. When a port from CUDA is mysteriously
slow at small batch sizes, count your command buffers first.

**Sync as rarely as possible.** `waitUntilCompleted` drains the pipeline: the GPU
goes idle while the CPU wakes, reads, and re-encodes. On
[unified memory](../machine/unified-memory.md) the data is already shared, so the
sync *is* the entire cost of reading a result. The
[DFlash war story](../war-stories/the-failures.md) got a measurable win from
nothing but halving GPU→CPU syncs per decode step (one `mx.eval()` instead of two).

What you don't manage, coming from CUDA: events for cross-stream ordering (a
single queue is ordered; multi-queue is rare in compute work), copy engines
(no transfers exist), and stream priorities. Timestamps you *do* get:
`GPUStartTime`/`GPUEndTime` on a completed command buffer are the platform's
[only programmatic GPU timing](profiling.md). Measure at command-buffer
granularity, one more reason to make command buffers meaningful units of work.

Next: [Synchronization](synchronization.md)

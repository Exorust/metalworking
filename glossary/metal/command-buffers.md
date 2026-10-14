# Command Buffers

**GPU work in Metal is encoded into command buffers: batches of dispatches built
CPU-side, committed to a queue, and executed asynchronously. How you batch them is
one of the biggest free performance levers on the platform.**

## The problem this page is about

On Metal, the CPU does not call a GPU kernel the way it calls a function. The CPU
writes a to-do list for the GPU, hands the whole list over, and later asks whether
it is finished. The list is the command buffer. Everything below is about how long
that list should be and how often you should stop to ask.

## The three objects, in the order you use them

```
MTLCommandQueue                  an ordered lane of work (make one, keep it)
  MTLCommandBuffer               one batch: create → encode → commit
    MTLComputeCommandEncoder     writes dispatches into the buffer:
        setComputePipelineState, setBuffer, dispatchThreadgroups, ...
```

The queue is a lane the GPU works through in order. You make one at startup and
keep it; in CUDA terms it is roughly a stream.

The command buffer is one to-do list. You create it from the queue, fill it, and
hand it over. CUDA has no object for this: each `kernel<<<>>>` is its own
submission and the driver groups launches behind your back. In Metal you build
the batch yourself, which is the part that matters.

The encoder is what you fill the list with. You open one on the buffer, then
write entries: use this compiled kernel (`setComputePipelineState`), here are its
argument buffers (`setBuffer`), run it on this grid (`dispatchThreadgroups`). One
"run it" entry is a dispatch. You can write many dispatches, then close the
encoder with `endEncoding`.

## One kernel, start to finish

The CPU creates a command buffer, opens an encoder, writes one dispatch, closes
the encoder, and calls `commit()`. Nothing has run on the GPU yet; commit is the
hand-over. The queue starts buffers in the order they were handed over, and the
GPU runs the dispatch while the CPU is free to do other things. When the CPU
needs the result it calls `waitUntilCompleted`, which blocks until the GPU reports
the buffer done (a completion handler is the non-blocking form). Until one of
those fires, the CPU must not overwrite the buffer's inputs or read its outputs.
Two timestamps, `gpuStartTime` and `gpuEndTime`, then say how long the GPU spent
on the whole buffer, which is why the book's benchmarks put one kernel per buffer
when timing it.

![Two submission patterns: chatty per-op commit and wait leaves the GPU idle between ops; one command buffer per forward pass keeps it continuously busy](../../diagrams/command-buffers.svg)

*The same work, submitted two ways. Every dashed gap on the left is the GPU
waiting for the CPU to wake, read, and re-encode; the right side pays the
per-buffer cost once.*

## Three rules

**Batch aggressively.** Each command buffer costs a fixed amount of CPU and driver
work: creating it, committing it, scheduling it on the GPU, and the completion
notice. Each dispatch inside a buffer costs almost nothing extra. So for a model
with 32 layers, the chatty way is one buffer per kernel with a wait after each,
and the GPU sits idle between buffers while the CPU wakes, reads the result, and
writes the next list (the dashed gaps in the diagram). The fast way is one buffer
for the whole forward pass: encode every layer's dispatches, commit once, wait
once. Dependencies between dispatches in one buffer on one queue are handled in
order by Metal, so the GPU runs from one to the next with no gap.
[Luminal](https://docs.luminalai.com/blog/gpu) runs all of Llama 3 8B as a single
command buffer; [flash-moe](../war-stories/three-questions.md) pre-encodes
command buffers before the GPU needs them; [MLX's scheduler](../mlx/lazy-evaluation.md)
exists to batch this way automatically. When a port from CUDA is mysteriously
slow at small batch sizes, count your command buffers first.

**Encoding itself can become the wall.** Batching removes the per-buffer cost, not
the cost of writing the list. Writing one dispatch entry takes about 2.9 µs of
CPU time regardless of the dispatch's size, and at framework-graph scale that adds
up. The [MTPLX dispatch census](https://github.com/youssofal/MTPLX/blob/ed1c8eea501689b744c13bec6a99ee2d36d26ab5/docs/perf/deepseek-v4-dispatch-levers.md)
measured one bf16 DeepSeek-V4 decode step at **19,809 dispatches in 384 command
buffers, with host encode at 56-59 ms against 32-35 ms of GPU execution**: the
CPU writing the list was the slow part, and the GPU waited for it. The fix was
structural (collapse a normalization chain, one shared `mx.compile` tape): 26%
fewer dispatches, worth about 15 ms of host encode per token. So the first rule
has a ceiling: batching hides *per-buffer* overhead, but a graph that emits tens
of thousands of tiny dispatches pays the encoder tax anyway. Count dispatches,
not just buffers.

**Sync as rarely as possible.** `waitUntilCompleted` is a full stop: the GPU
finishes, then does nothing until the CPU reads the result and writes the next
list. On [unified memory](../machine/unified-memory.md) there is no copy to wait
for, so the wait *is* the entire cost of reading a result. The
[DFlash war story](../war-stories/the-failures.md) got a measurable win from
nothing but halving GPU→CPU syncs per decode step (one `mx.eval()` instead of two).

## What you do not manage, coming from CUDA

You do not manage events for cross-stream ordering, because a single queue is
ordered and multi-queue is rare in compute work. You do not manage copy engines,
because shared buffers need no host-to-device transfer and blit encoders copy
between GPU resources as ordinary ordered commands. And there are no stream
priorities to set.

## Reference details

- The queue schedules buffers in enqueue order, which is commit order unless you
  call `enqueue()` early to reserve a slot.
- A default queue allows 64 uncompleted buffers; asking for a 65th blocks the CPU
  until the GPU finishes one, a wait no `waitUntilCompleted` in your code shows.
- `gpuStartTime` / `gpuEndTime` (`GPUStartTime` / `GPUEndTime` in Objective-C and
  PyObjC) are the platform's
  [simplest programmatic GPU timing](profiling.md): host seconds on the mach
  clock, 0.0 until the buffer completes. A sibling pair,
  `kernelStartTime` / `kernelEndTime`, records the CPU driver's scheduling of the
  same buffer. Measurement is at command-buffer granularity, one more reason to
  make buffers meaningful units of work.
- Metal 4 replaces these objects with `MTL4CommandQueue`, reusable
  `MTL4CommandBuffer`s backed by a `MTL4CommandAllocator`, and argument tables
  ([Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)).

Next: [Synchronization](synchronization.md)

# Arrays that record: MLX and lazy evaluation

## Where we are

In Part 4 you wrote a matrix multiply by hand and timed it through a Python
harness that spoke to Metal directly. Almost nobody runs a model that way. On Apple Silicon you
usually optimize through MLX, Apple's open-source array framework, and its
architecture decides what is fast before any kernel runs. Part 5 follows the same
matrix multiply into MLX: what the framework does with `mx.matmul(a, b)` before,
during and after the kernel. You start here with the before. At that stage MLX
does no GPU work at all; it records the operations so that it can batch their
GPU work when you evaluate the result.

## What MLX is, for a CUDA-stack reader

Coming from the CUDA stack, you first need a place to put MLX. It is a
NumPy-shaped array API with automatic differentiation, built natively on
the unified memory of Chapter 3, whose Metal backend's matrix and attention
kernels are the steel library that Chapter 18 reads. In CUDA terms it is the whole stack of PyTorch, cuBLAS and
cuDNN, and bits of Triton, collapsed into one codebase, which is why this book
gives it a part rather than a chapter.

Four of your CUDA expectations need recalibrating.

**Arrays have no device.** An `mx.array` lives in unified memory. There is no
`.to("gpu")` and no host-to-device copy anywhere in user code. Operations have a
location instead, a stream on a device, so the CPU and the GPU can work on the
same buffers: quantize on the CPU while the GPU decodes, in one address space.

**Everything is lazy.** Operations build a graph; computation happens at
`mx.eval`. The purpose is less the compiler work that `torch.compile` does than
the dispatch batching of Chapter 7, and on this platform performance depends on
it.

**Function transforms as the interface.** `mx.grad`, `mx.vmap` and
`mx.compile` compose like JAX's transforms; underneath, reverse mode walks the
recorded graph backwards, as a tape would. Training works, but most of the
ecosystem's effort goes into inference.

**The kernels are readable.** Unlike cuBLAS, the fast path is open-source Metal
Shading Language you can read: steel for matrix multiply and attention, the
quantized kernels for language-model weights, Chapter 23, and an escape hatch for
your own kernels, Chapter 21.

The layer map, top to bottom, with the chapter for each layer:

    Python (mx.*)                       NumPy-shaped API, lazy
      graph + scheduler                 this chapter: batches work into command buffers
        primitives (eval_gpu)           Chapter 17: C++ picks a kernel and tile sizes
          steel / quantized / fast MSL  Chapters 18, 21, 23: the kernels
            Metal                       Part 2

Where the alternatives sit: llama.cpp's Metal backend, Chapter 21, is the other
production-grade kernel body on the platform, C and C++ with no Python layer,
organized around quantized formats. PyTorch's MPS backend runs and is improving,
and since version 2.13 ships hand-written Metal kernels for hot paths such as
FlexAttention rather than dispatching everything to the library of Chapter 9; it
remains a port of a CUDA-native design. MLX also has a CUDA backend now, so it
has two GPU backends plus CPU execution. MLX's Metal backend is still where kernel-level
work happens if you want to touch the stack this book teaches.

## Operations record; evaluation runs

When you call `mx.matmul(a, b)`, it does not compute. It checks shapes and types, picks a stream,
appends a graph node and returns, and the GPU work happens when something
forces evaluation. That deferral is what lets MLX
batch an entire model into few command buffers.

The nearest CUDA-side idea is PyTorch's asynchronous dispatch, and this is
stronger. PyTorch enqueues each operation eagerly and hides latency with stream
asynchrony; MLX submits nothing until evaluation, so it can see a whole subgraph and
schedule it as a unit. It is less like CUDA Graphs than it looks, because there
is no capture-and-replay lifecycle, but part of the payoff is similar: the
per-command-buffer costs of submission and completion are amortized and
encoding can overlap execution, while the per-dispatch encoding cost of Chapter
7 remains. That matters on a platform where a forced wait, Chapter 3, is
expensive.

Evaluation is forced by `mx.eval(arrays...)`, or implicitly by anything that
needs values: printing, `.item()`, converting to NumPy, or a control-flow branch
on an array's contents. At that point the scheduler walks the part of the graph the requested arrays
depend on and calls each primitive's `eval_gpu` (Chapter 17) or `eval_cpu`,
according to its stream, encoding GPU kernels one after another into that
stream's command buffer and committing when a work or memory threshold is
reached. `mx.async_eval` does the same without waiting for the result, which is
how a loop overlaps encoding the next step with executing this one.

```cpp
void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  ...
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  ...
  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s, buffers = std::move(buffers)]() {
```
Source: [`mlx/backend/metal/eval.cpp:29-62`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/eval.cpp#L29-L62), abridged

The encoder is shared across primitives on a stream, and each
primitive encodes its dispatch into it. A commit happens when the encoder says
it needs one, which MLX decides from its own thresholds of work and memory, plus
a final commit of whatever remains when the evaluation finishes. This is the batching rule of Chapter 7 applied automatically, and it is
also the opt-out Chapter 7 mentioned: MLX allocates its buffers untracked and
dispatches concurrently, so it inserts the memory barriers between dependent
dispatches itself rather than relying on Metal's hazard tracking.

![MLX's record-then-commit model: ops append graph nodes instantly, eval walks the graph and batches kernels into command buffers; an accidental mid-loop eval forces a drain every iteration](../diagrams/lazy-evaluation.svg)

*Recording is cheap CPU work; evaluation with pending work is the
synchronization boundary. The bottom lanes show the same loop written well and
badly: the dashed drains are what an accidental evaluation in the middle of the
loop costs on every token. Recording the next step while the GPU is still busy
needs `mx.async_eval`.*

## The rule that falls out: evaluate once per step

Every evaluation with pending work, explicit or accidental, is a
synchronization boundary: commit, wait, the CPU wakes, re-encode. Evaluating
an array that is already computed costs nothing. The DFlash case of Chapter 25 measured a real
speedup from consolidating two evaluations per decode step into one. The classic
accidental version is a Python `if` on a logit value in the middle of the loop,
which forces a pipeline drain per token. Structure your loops so that the host reads no value mid-step unless the algorithm
needs it, and exactly one evaluation happens per iteration, at the end.

Streams are where operations run. `mx.gpu` and `mx.cpu` are devices, and each has a
default stream, `mx.default_stream(device)`; `mx.new_stream(device)` makes more.
Every operation takes a `stream=` argument that accepts a stream or a device,
meaning that device's default stream. Because arrays have no device, sending an operation to the
CPU stream costs no transfer, so heterogeneous pipelines, CPU pre-processing and
post-processing around GPU compute, are routine. Dependencies
across streams are fenced by the scheduler.

Laziness also enables `mx.compile`, Chapter 19: since the graph exists anyway, it
can be traced, fused and cached.

## Where the matmul stands

The same 4096 by 4096 multiply, written as `mx.matmul(a, b)`, is now a node in a
graph on the default GPU stream and has done no GPU work. When `mx.eval` runs, the scheduler will hand it to the
matrix-multiply primitive, which picks the kernel and tile sizes in the next
chapter.

## Check yourself

1. A decode loop has a Python `if logits[0, i] > 0.5:` in the middle of each
   step, records more work after it, and calls `mx.eval` on that work at the
   end. How many synchronization boundaries does each step pay, and what does
   the chapter say to do?
2. In what sense is MLX's laziness stronger than PyTorch's asynchronous dispatch,
   and in what sense is it less than CUDA Graphs?
3. Why can a CPU pre-processing step be placed on `mx.cpu` without a transfer
   cost?

### Answers

1. Two: the `if` forces an evaluation of the pending work in the middle, and the
   work recorded after it forces another at the end. Restructure so the host
   reads nothing until the end of the step and the loop evaluates once, which
   halves the drains per token; DFlash's measured win was consolidating two
   evaluations per step into one.
2. Stronger because MLX submits nothing until evaluation and can schedule a whole
   subgraph as a unit, while PyTorch enqueues each operation as it comes. Less
   than CUDA Graphs in that there is no reusable executable graph to replay; the
   per-buffer overhead is amortized either way, and the per-dispatch encoding
   cost remains.
3. Because arrays have no device: the array lives in unified memory, and the CPU
   and the GPU read the same bytes, so only the operation moves, not the data.

The scheduler has handed the matrix multiply to its primitive, and the next
chapter reads the one C++ hop between the graph node and a kernel: the code that
picks the variant and the tile sizes.

Next: [How an op becomes a kernel](17-how-an-op-becomes-a-kernel.md)

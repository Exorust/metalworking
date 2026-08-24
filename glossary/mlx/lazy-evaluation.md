# Lazy Evaluation

**MLX operations don't compute; they record. `mx.matmul(a, b)` returns instantly
with a graph node, and actual GPU work happens when something forces evaluation.
That deferral is what lets MLX batch an entire model into
[few command buffers](../metal/command-buffers.md).**

CUDA equivalent: closest to PyTorch's async dispatch, but stronger. PyTorch
enqueues each op eagerly and hides latency with stream asynchrony; MLX doesn't
even *decide* anything until eval, so it can see a whole subgraph and schedule it
as a unit. Less like CUDA graphs than it looks (no capture/replay lifecycle),
but the payoff is similar: per-op CPU overhead and per-launch GPU overhead get
amortized, which matters double on a platform where the
[sync cost is the story](../machine/unified-memory.md).

Evaluation is forced by `mx.eval(arrays...)`, or implicitly by anything that needs
values: printing, `.item()`, converting to NumPy, or a control-flow branch on an
array's contents. At that point the scheduler walks the graph and calls each
primitive's `eval_gpu` ([next page](how-an-op-becomes-a-kernel.md)), encoding
kernel after kernel into a shared command buffer, committing when a batch quantum
fills:

```cpp
void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  ...
  arr.primitive().eval_gpu(arr.inputs(), outputs);
  ...
  if (encoder.needs_commit()) {
    encoder.commit(...);
```
— [`mlx/backend/metal/eval.cpp:29-62`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/eval.cpp#L29-L62), abridged

**The performance rule that falls out: evaluate once per step.** Every `mx.eval`
(explicit or accidental) is a synchronization boundary: commit, wait, CPU wakes,
re-encode. The [DFlash war story](../war-stories/the-failures.md) measured a real
speedup from consolidating two evals per decode step into one; the classic
accidental version is a Python `if` on a logit value mid-loop, which forces a
pipeline drain per token. Structure loops so exactly one eval happens per
iteration, at the end.

**Streams** are where operations run: `mx.gpu` and `mx.cpu` are default streams on
their devices, `mx.new_stream(device)` makes more, and every op takes a
`stream=` argument. Because [arrays have no device](mlx-overview.md), sending an
op to the CPU stream is free of transfer cost, so heterogeneous pipelines
(CPU pre/post-processing around GPU compute) are idiomatic rather than heroic.
Cross-stream dependencies are fenced automatically by the scheduler.

Laziness also enables [`mx.compile`](mx-compile.md): since the graph exists
anyway, it can be traced, fused, and cached.

Next: [How an op becomes a kernel](how-an-op-becomes-a-kernel.md)

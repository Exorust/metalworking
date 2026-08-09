# metalworking

**The craft of making Apple Silicon GPUs go fast.**

A structured learning track for Apple Metal GPU performance. Instead of a course with toy
examples, you read the real thing: ~20 of the best open-source Metal kernels and tools
(llama.cpp's Metal backend, MLX's steel kernels, metal-flash-attention, luminal's kernel
compiler) at exact file/line coordinates, in an order that starts at "what is this
hardware" and ends at "why did my hand-written kernel lose to stock MLX."

All referenced code is fetched at pinned commits, so every line number in the track
points where it says it does.

## Quickstart

```sh
./fetch.sh          # clones all referenced repos into code/ at pinned commits (~240MB)
```

Then start at [`track/stage-0-hardware.md`](track/stage-0-hardware.md).

You'll want an Apple Silicon Mac with Xcode command line tools and Python + [MLX](https://github.com/ml-explore/mlx)
(`pip install mlx`) to actually run things, but most of the track is guided reading and works anywhere.

## The track

| Stage | What you learn | Done when |
|---|---|---|
| [0. Hardware reality](track/stage-0-hardware.md) | What an M-series GPU actually is: register file, ALU layout, why F16 wins | You can explain why Apple Silicon is bandwidth bound, not compute bound |
| [1. MSL from zero](track/stage-1-msl.md) | Threads, threadgroups, simdgroups, via 14 progressive puzzles | All 14 Metal-Puzzles pass |
| [2. The GEMM ladder](track/stage-2-gemm.md) | Naive, tiled, double-buffered, and async-DMA matrix multiply | You can say what double buffering costs and buys, and beat MPS at one size |
| [3. How production factors it](track/stage-3-steel.md) | MLX's steel library: BlockLoader / BlockMMA, the CUTLASS-style decomposition | You can map every ladder concept onto a steel template parameter |
| [4. Flash Attention, three ways](track/stage-4-attention.md) | Three independent FA implementations for the same hardware; the diffs are the lesson | You can name how each one generates, specializes, and decodes |
| [5. Down to the silicon](track/stage-5-silicon.md) | ISA disassembly, instruction throughput tables, the AMX coprocessor | You can disassemble a shader and read what the compiler actually emitted |
| [6. War stories](track/stage-6-war-stories.md) | Community hacks that worked, and the many custom kernels that lost | You ask the three questions before writing any kernel |

## Five lessons the whole field keeps re-learning

These recur across every project in this track; luminal, MLX, llama.cpp, and the Reddit
war stories all converge on them:

1. Apple Silicon is bandwidth bound, not compute bound. Fusion (avoiding intermediate
   buffers) beats clever ALU tricks almost every time.
2. Your custom kernel will probably lose to stock MLX steel / ggml kernels. Multiple
   independent reports agree on this. The real wins are in dispatch/sync overhead,
   unlocking existing fast paths, and deleting work.
3. Use F16 everywhere and avoid FP32 atomics (emulated, slow). Register pressure is the
   real budget: ~208 KB per core.
4. Command buffer batching is free performance: one command buffer per forward pass,
   pre-encoding, batched dispatch.
5. CUDA instincts partially mislead. Threadgroup barriers are cheap (~2 cycles),
   scattered shared-memory access is expensive, threadgroup memory is 32 KB not 48, and
   there's a fast exp2 hardware path.

## Layout

```
track/       the seven stages; start here
code/        fetched third-party repos (created by fetch.sh, gitignored)
SOURCES.md   the full annotated source list: repos, blogs, papers, people to follow
fetch.sh     reproduces code/ at the exact pinned commits
```

## Licensing

The guide text and `fetch.sh` are MIT ([LICENSE](LICENSE)). Everything under `code/` is
other people's work, fetched from its original home under its own license; nothing is
vendored or redistributed here. Full attribution is in [SOURCES.md](SOURCES.md) and in
each stage.

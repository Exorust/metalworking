# Stage 1 — MSL from zero

**Needs:** `code/Metal-Puzzles`, `code/tinygrad-notes` &nbsp;·&nbsp; **Time:** a weekend, hands on keyboard

This is the only stage where you write code before reading someone else's. The goal is
the thread/threadgroup/simdgroup mental model — Metal's execution hierarchy — learned by
solving 14 progressively harder puzzles, each a small MSL kernel checked against a
reference implementation.

## Do

Work through [abeleinin/Metal-Puzzles](https://github.com/abeleinin/Metal-Puzzles):

- `code/Metal-Puzzles/metal_puzzles.ipynb` — the intended interactive form (or
  `metal_puzzles.py`, same content, 788 lines). The ladder: Map (line 5), Zip (56),
  Guard (99), Map 2D (144), Broadcast (187), Threadgroups (227), Threadgroups 2D (273),
  Threadgroup Memory (313), Pooling (382), Dot Product (443), 1D Convolution (498),
  Prefix Sum (574), then Axis Sum and Matmul.
- `code/Metal-Puzzles/utils.py` — read this once you're a few puzzles in. It's the
  MLX-backed harness: compiles your MSL string, runs it, diffs against the spec. This is
  also the minimal "compile and dispatch an MSL kernel from Python via
  `mx.fast.metal_kernel`" pattern — you'll reuse it for your own experiments for the
  rest of the track.

## Read alongside

- `code/tinygrad-notes/20240921_metal.md` — "Abstraction in Apple's Metal Framework".
  Strips Metal down to the raw C / Objective-C calls (device, library, pipeline, command
  buffer) before the abstractions layer on. The clearest explanation anywhere of what the
  Metal runtime actually does when you dispatch a kernel.

## Done when

- All 14 puzzles pass.
- You can draw the hierarchy — grid → threadgroup → simdgroup (32 threads) → thread —
  and say which level shares which memory.
- You can explain what `threadgroup_barrier` synchronizes, and why puzzle 8
  (Threadgroup Memory) needs it where puzzle 1 doesn't.

**Next:** [Stage 2 — The GEMM ladder](stage-2-gemm.md)

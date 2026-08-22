![metalworking: the craft of making Apple Silicon GPUs go fast](banner.png)

**Read it as a website: [metalworking.vercel.app](https://metalworking.vercel.app)**

A hyperlinked glossary of Apple Metal GPU performance — the machine, the Metal
stack, MLX's architecture, the techniques, and real production kernels read line
by line.

**Who it's for:** you know GPU fundamentals — say, at the level of
[Modal's GPU Glossary](https://modal.com/gpu-glossary/readme) — but you've never
worked with Metal or MLX. This is the companion volume for the other hardware:
every page opens with the CUDA equivalent where one exists, then spends its words
on what's *different* here. Concepts Modal already covers well are linked, not
re-taught.

**How to read it:** like a glossary or like a book — every page stands alone and
links to what it assumes, and every page ends with a `Next` link, so front to back
also works. Front to back is a deliberate arc: *what is this machine → how do you
talk to it → what does the framework do on your behalf → why are fast kernels
shaped this way → read five real ones → learn from the community's scars.*

All code excerpts are quoted verbatim from pinned commits and permalink back to
their exact lines, so you can always zoom out from a fragment to the full source.
No setup is needed to read; `./fetch.sh` (optional, ~240 MB) clones every
referenced repo at its pinned commit for browsing beyond the excerpts.

## The Machine

*The M-series GPU, as a diff against the GPU you already know.*

- [GPU Core](glossary/machine/gpu-core.md) — the SM analogue, and the numbers that differ
- [Simdgroup](glossary/machine/simdgroup.md) — the warp, renamed
- [Registers](glossary/machine/registers.md) — ~208 KB per core, the real budget, and the 10× spill cliff
- [Threadgroup Memory](glossary/machine/threadgroup-memory.md) — 32 KB; shared memory demoted to staging buffer
- [Unified Memory](glossary/machine/unified-memory.md) — no transfers, modest bandwidth, and the platform's one law
- [Occupancy](glossary/machine/occupancy.md) — saturation at ~24 simdgroups, and occupancy as a currency
- [F16](glossary/machine/f16.md) — faster for stall and register reasons, not throughput reasons
- [Special Paths](glossary/machine/special-paths.md) — the fast exp2; the emulated float atomics
- [AMX](glossary/machine/amx.md) — the other matrix engine on the die

## Metal, the Stack

*What replaces the CUDA driver, runtime, nvcc, and PTX.*

- [Metal, the API](glossary/metal/metal-the-api.md) — orientation and the full mapping table
- [MSL](glossary/metal/msl.md) — Metal Shading Language, as a CUDA C++ accent
- [Dispatch Geometry](glossary/metal/dispatch-geometry.md) — grids and threadgroups, plus two API traps
- [Compilation Pipeline](glossary/metal/compilation-pipeline.md) — MSL → AIR → metallib → pipeline state
- [Function Constants](glossary/metal/function-constants.md) — specialization without a kernel explosion
- [Command Buffers](glossary/metal/command-buffers.md) — the batching model, and why it's free performance
- [Synchronization](glossary/metal/synchronization.md) — barriers cost ~2 cycles; design accordingly
- [simdgroup_matrix](glossary/metal/simdgroup-matrix.md) — the tensor-core analogue that isn't one
- [simdgroup_async_copy](glossary/metal/simdgroup-async-copy.md) — the dead `cp.async`, and why kernels look the way they do
- [MPS](glossary/metal/mps.md) — the cuBLAS analogue, and why it's beatable
- [Profiling](glossary/metal/profiling.md) — the honest page: there is no Nsight

## MLX

*The framework layer — where CUDA-land has no single equivalent.*

- [MLX, an Overview](glossary/mlx/mlx-overview.md) — arrays with no device, and the layer map
- [Lazy Evaluation](glossary/mlx/lazy-evaluation.md) — the graph, `mx.eval`, and the one-eval-per-step rule
- [How an Op Becomes a Kernel](glossary/mlx/how-an-op-becomes-a-kernel.md) — the readable dispatcher, tile tables and all
- [Steel](glossary/mlx/steel.md) — the CUTLASS of Apple Silicon
- [mx.fast](glossary/mlx/mx-fast.md) — fused ops, the SDPA dispatch gate, and the custom-kernel escape hatch
- [Quantization](glossary/mlx/quantization.md) — group-wise affine, QMV/QMM, and bandwidth arithmetic
- [mx.compile](glossary/mlx/mx-compile.md) — elementwise fusion, deliberately scoped

## Techniques

*Why fast kernels are shaped the way they are.*

- [Arithmetic Intensity](glossary/techniques/arithmetic-intensity.md) — the number that decides everything here
- [Roofline](glossary/techniques/roofline.md) — build your own; no profiler will do it for you
- [Tiling](glossary/techniques/tiling.md) — the reuse pyramid, Apple proportions
- [Cooperative Load](glossary/techniques/cooperative-load.md) — how tiles move, now that DMA is gone
- [Register Blocking](glossary/techniques/register-blocking.md) — the accumulator grid and its cliff edges
- [Double Buffering](glossary/techniques/double-buffering.md) — overlap via ILP, and a trade that flips sign
- [Fusion and Epilogues](glossary/techniques/fusion-and-epilogues.md) — the platform's most profitable technique
- [Online Softmax](glossary/techniques/online-softmax.md) — the five-line algorithm behind flash attention
- [Flash Attention](glossary/techniques/flash-attention.md) — never materialize the score matrix
- [Decode vs Prefill](glossary/techniques/decode-vs-prefill.md) — one API, two workloads, separate kernels

## Kernels: the Case Studies

*Real production code, quoted and read.*

- [The Tiled GEMM](glossary/kernels/gemm-tiled.md) — 140 lines that beat MPS (m5-gemm)
- [The Double-Buffered GEMM](glossary/kernels/gemm-double-buffered.md) — and the benchmark table with three winners
- [The Async-Copy Ghost](glossary/kernels/gemm-async-ghost.md) — the DMA original, and its death on Metal 4
- [Steel's BlockLoader](glossary/kernels/steel-blockloader.md) — the cooperative load as a template
- [Steel's BlockMMA](glossary/kernels/steel-blockmma.md) — register blocking as a component
- [The Fused GEMM Kernel](glossary/kernels/steel-gemm-fused.md) — when the kernel is just wiring
- [Steel Attention](glossary/kernels/steel-attention.md) — the whole flash-attention algorithm, readable
- [metal-flash-attention](glossary/kernels/mfa-codegen.md) — the codegen school
- [llama.cpp Attention](glossary/kernels/llamacpp-attention.md) — the enumeration school, quantized to the bone

## War Stories

*What happened when real people fought the defaults.*

- [The Three Questions](glossary/war-stories/three-questions.md) — ask before writing any kernel
- [Sparse-V](glossary/war-stories/sparse-v.md) — the canonical win: +22.8% from three lines
- [The Failures](glossary/war-stories/the-failures.md) — published negative results, and the checklist they teach
- [Cheap Tricks](glossary/war-stories/cheap-tricks.md) — sysctls, spec sheets, and fans

## Hands-on

The glossary is reading; if you want your hands on a keyboard:
[Metal-Puzzles](https://github.com/abeleinin/Metal-Puzzles) (14 progressive MSL
kernels, checked from Python) is the on-ramp, and
[m5-gemm](https://github.com/yaroslavvb/m5-gemm) runs the
[GEMM case studies'](glossary/kernels/gemm-tiled.md) benchmarks on your own
machine with nothing but Python installed.

## Layout

```
glossary/    the six sections; start anywhere, or at the top
code/        fetched third-party repos (created by ./fetch.sh, gitignored, optional)
SOURCES.md   the full annotated source list: repos, blogs, papers, people to follow
fetch.sh     reproduces code/ at the exact pinned commits
```

## Licensing

The glossary text and `fetch.sh` are MIT ([LICENSE](LICENSE)). Code excerpts are
quoted from their original repositories under their own licenses (MIT and
BSD-3-Clause throughout), verbatim, with attribution and a permalink at every
quotation site. Full attribution in [SOURCES.md](SOURCES.md).

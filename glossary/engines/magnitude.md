# Magnitude: a tuner with a statistics department

**Magnitude is an open-source inference engine for coding agents that compiles
and tunes its kernels on the user's machine before a model runs (Apache-2.0,
magnitudedev/magnitude, pinned here at commit c2399d41). The current engine,
`inference/`, is Rust on its own kernel language and runtime, Seismic: about
435,000 lines of Rust, 25,000 lines of hand-written Metal, 18,000 lines of
`.seismic` declarations, driving Metal directly through `objc2-metal`. It is
the [tinygrad](../../SOURCES.md) school of autotuning applied to a whole LLM
engine, and the most careful measurement methodology in this glossary.**

Fetched by `fetch.sh` as `code/magnitude`. Line references are into that
pinned copy, under `inference/` unless stated. The earlier `inference-v2`,
`inference-v3` and `old-inference` directories record the path: a Python
engine, a Python engine on TileLang, and a wrapper over llama.cpp.

## What "tuned on your device" means

The kernels are not generated. Each `native ... for metal` declaration binds a
hand-written Metal source and lists a finite parameter domain whose first value
is the default; parameters are tagged `code`, `arithmetic` or `form`
(`seismic/lang/src/syntax/ast.rs:214-224`), and the chosen values are baked in
as `#define SEISMIC_TUNE_<NAME>` before compilation with
`newLibraryWithSource` (`seismic/runtime/src/native/abi.rs:330-334`;
`seismic/backends/metal/src/compile.rs:50`). For decode attention the domain
is SPAN 32 to 256, PARTS 8 to 512, SIMDS 2 to 16, SLICES 1 to 8, MATRIX 0 or
1, KEYS 8/16/32, TOKENS 1 to 8, with legality written as an expression that
holds [threadgroup memory](../machine/threadgroup-memory.md) under 32,768
bytes and threads under 1,024 (`engine/kernels/kernels/attention.seismic:643-664`).
For the output projection the domain switches launches by row count: a GEMV
below a tuned row threshold, a matrix-unit batched GEMV to 16 rows, a fixed
32 by 64 GEMM with 4-way split-K to 64, and above 64 a choice among a tiled
GEMM, a tall GEMM, an INT8 path and a packed path
(`engine/kernels/kernels/dense_rows.seismic:230-290`). The search never
enumerates the product: it steps one parameter group at a time from
configurations already held (`seismic/runtime/src/native/plan.rs:7-16`),
gives every structurally distinct form a representative before refining, and
drops a candidate at its first run if its output moves the default's by more
than 8 times the storage-resolution noise (`docs/precision.md:3-24`). The
budget is 60 seconds on any device, in a separate process before the model
loads (`engine/executor/src/native/tuning/mod.rs:340-342`); a slower device
explores less, and a unit the deadline stops keeps its default.

## The measurement methodology, which is the point

Timing uses `GPUStartTime` and `GPUEndTime` plus counter sample buffers
(`seismic/backends/metal/src/direct.rs:63-140`). A challenger and the
incumbent are sampled in shuffled pairs inside one session and compared as a
demand-weighted ratio, because of a measured failure: "within one session a
point's samples spread by 0.26% of their median ... while the same
configuration measured in two sessions differed by 9% at the median and by 79%
at the 90th percentile" (`seismic/runtime/src/native/sampling.rs:20-29`). The
stopping rule is a sign test made anytime-valid with Ville's inequality; at
alpha 0.05 a lead needs seven consecutive rounds ahead, and most slower
challengers drop after one (`sampling.rs:57-105`). Before each session the GPU
is pushed to its sustained clock, because on an M4 Pro the lowest performance
state runs a kernel 3.7x to 4.7x slower than the highest, the GPU falls to the
lowest state after 100 ms idle and climbs back in 33 to 47 ms of continuous
work, and duty cycles of 60 to 88 percent leave the state where it was
(`seismic/runtime/src/native/timing.rs:52-72`). The
[cheap tricks](../war-stories/cheap-tricks.md) page says to control thermals;
this is the same warning with the governor's numbers attached, and any
benchmark on this platform should quote it.

The demand weights come from real traffic: "137,170 turns of 1,631 Magnitude
coding-agent sessions", median cached prefix 46.6K tokens, 80 percent of turns
hitting the cache, new prompt tokens outnumbering decoded tokens 32 to 1
(`engine/executor/src/native/tuning/workload.rs:4-23`). Timed calls rotate
through real weights from up to four layers so repeated calls stream from
memory the way a real step does, not from cache (`tuning/mod.rs:36-38`).
Results go to `cache/kernels/tuning/<key>.json` with every measured candidate
kept as evidence; nothing is shipped pretuned, and on Metal the compiled bytes
are not stored (`engine/executor/src/kernel_cache.rs:1-45`).

## The kernels

One projection family, `lib/projection/projection.h` (3,867 lines), computes
`y = epilogue(sum_k x W)` with norm prologues and fused epilogues, so a decode
projection is one launch (`projection.h:1-26`). The GEMM is 32 by 32 simdgroup
tiles, K step 32, weights decoded to half once per tile, into
`simdgroup_multiply_accumulate` (`projection.h:43-49, 1508-1523`); on devices
where a probe kernel shows that Metal 4 tensor types compile, the same tile
runs `mpp::tensor_ops::matmul2d` instead (`projection.h:1597-1622`;
`seismic/backends/metal/src/toolchain.rs:216-235`). That is the
[neural accelerators](../machine/neural-accelerators.md) page's two routes
chosen by compilation, not by device name. The packet decoder turns GGUF
Q4_K/Q5_K/Q6_K/Q8, IQ4, MXFP4 and NVFP4 codes into exact small floats on
nibble boundaries without integer conversions, because "the M = 1 projections
are bound by the decode's instruction issue" (`native-library/metal/packets.h:5-19`).

The strangest form is token packing (`lib/projection/packing.h:1-30`): on
GPUs without tensor ops, activations are rounded to integer codes per 32
columns and two tokens share one fp32 matrix operand as `2^16 * top + low`, so
one simdgroup FMA carries two products; the comment proves the overflow bounds
with Hölder and Cauchy-Schwarz, and states that one token's result then
depends on its partner.

Decode attention is split-K flash decode in two launches with an exp2-domain
[online softmax](../techniques/online-softmax.md); a MATRIX form treats a KV
head's grouped query heads as rows of 8 by 8 simdgroup blocks, so grouped-query
attention becomes a small matrix multiply (`attention.seismic:611-663, 1160-1162`).
Prefill uses 32-key tiles, and for 128- and 256-wide heads a COISSUE form pairs
simdgroups so one does QK and the softmax on the matrix pipe while the other
does PV as scalar fp16 FMAs over 96-key steps, "measured within the BF16
tolerance of the default" (`attention.seismic:1294-1300`); it is the closest
Apple analogue to warp specialization. The default KV cache is 8-bit keys and
4-bit values, and decode scores are corrected in the code domain rather than
dequantized: `scale * (q . code) + zero * sum(q)`, with the value-side bias
carried through the softmax (`attention.seismic:971-977`).

## What the headline number is

The README's 92 percent faster decode than llama.cpp on Metal and 19 percent
on CUDA are hardcoded tuples in `scripts/benchmark-chart.py:11-14`: 30 vs 57
tok/s on an M4 Pro 48 GB, Qwen 3.6 35B-A3B at 4-bit, 64K context, no
speculation. `assets/benchmarks/` holds only the two rendered charts; there is
no run record, command line or llama.cpp version. A harness that could
reproduce it exists, with a llama.cpp adapter, but its results directory is
absent (`evaluation/bench/src/magnitude_bench/cli.py:1-14`). That adapter's
llama.cpp defaults are fp16 K and V and no prompt cache
(`adapters/llama_cpp.py:113-119`), while Magnitude's default cache is 8-bit
keys and 4-bit values; unless the run overrode them, the 64K decode comparison
pits 12 bits of KV per element against 32, which by itself cuts attention
bytes per step by about 2.7x (inferred). The authors' own notes are candid:
the selection estimate model is "unqualified and no performance claim is made
here" (`inference-v4/README.md:27-28`), and admission "does not state a
ceiling on model-level quality change" (`docs/precision.md:26-31`).

Read next to [oMLX](omlx.md): the same time-debt scheduling (decode first,
prefill repays, `docs/engine/scheduling.md:44-49`), a prefix cache that shares
rows by reference and prices a branch at 128 saved rows for the recurrent
snapshot hybrids need (`engine/scheduler/src/prefix_cache.rs:24-41`), and a
kernel library whose every tile budget is an expression the tuner checks before
it compiles.

Next: [Cider](cider.md)

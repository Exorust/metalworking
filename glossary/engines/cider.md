# Cider: INT8 tensor ops as an MLX primitive

**Cider is a small MLX add-on (MIT, Mininglamp-AI/cider, pinned here at commit
4d91fcee, version 0.8.0) that does two things: online activation quantization,
W8A8 and W4A8, running on the M5 INT8 tensor-op datapath as custom MLX
primitives, and a decode-only replacement for
`mx.fast.scaled_dot_product_attention`. It is 1,689 lines of Metal, 1,682 of
host C++ and 1,332 of Python. It is in this glossary because it is the
cleanest small example of `matmul2d<int8, int8, int32>` on the
[neural accelerators](../machine/neural-accelerators.md), and because its
numbers show how much a headline depends on the baseline you pick.**

Fetched by `fetch.sh` as `code/cider`. Line references are into that pinned
copy. The INT8 extension only builds on M5 or later, detected by parsing the
CPU brand string (`cider/ops.py:52-69`).

## The INT8 GEMM, in 250 lines

The descriptor is `matmul2d_descriptor(16, 32, 16, false, true, true,
multiply_accumulate)` at `execution_simdgroup` scope with cooperative tensors
of `int8_t, int8_t, int32_t` (`cider/kernels/w8a8_matmul.metal:120-131`): a
16 by 32 by 16 operation, right operand transposed because weights are stored
[N, K], relaxed precision. Fragments use the same 16 by 16 layout the
[NAX GEMM](../kernels/nax-gemm.md) page describes, each lane holding 8
elements, loaded device to register with no threadgroup memory at all
(`:35-44`). The kernel copies each fragment into the cooperative tensors,
calls `run`, and copies the accumulator out (`:166-186`). The large tile is
128 by 128 by 512 on 16 simdgroups, each owning 32 by 32 of C as 2 by 2
fragments, so the accumulator per thread is 4 by 8 int32 (`:94-98, 133`); a
32 by 128 tile serves M up to 64 (`:292`). An execution-only
`threadgroup_barrier(mem_flags::mem_none)` at the top of every K step keeps the
simdgroups in step (`:144`), and a `volatile int compiler_barrier` limits how
far loads are hoisted (`:149`), the same trick MLX's own NAX loader uses.

Where the scales go is the lesson. Per-token activation scales come from a
separate dispatch, `simd_max` then a threadgroup reduction to absmax over 127
(`w8a8_quantize.metal:33-86`), and both scales are applied once in the store
epilogue: `float(acc) * scale_a[m] * scale_w[n] + bias[n]` in fp32
(`w8a8_matmul.metal:63-81`). The per-group variant makes each K iteration one
group and flushes the int32 fragment through a second fp32 accumulator after
every group (`pergroup_int8_gemm.metal:152-191`), which doubles accumulator
registers and costs about 28 percent: 5.65 ms against 4.41 at M=4096
(`README.md:113`). That is the [fusion](../techniques/fusion-and-epilogues.md)
page's rule, that the epilogue is where a per-element correction is free,
measured against the alternative. W4A8 unpacks nibbles straight into the int8
B fragment registers, two rows per fragment sharing nibble parity, four bytes
along N each (`w4a8_matmul.metal:47-70`); the host comment saying this happens
in threadgroup memory is wrong, and no benchmark or converter uses the path.

## Binding: a worked custom primitive

Each op is an `mx::Primitive` subclass whose public function returns a lazy
array (`csrc/src/w8a8_primitive.mm:268-271`), so calling the layer adds a graph
node and the work runs at `mx.eval`. Its `eval_gpu` takes MLX's own command
encoder, encodes the quantize dispatch, an `enc.barrier()`, then the GEMM, with
the int8 activations and scales allocated from MLX's allocator and released
through `add_temporary` (`w8a8_primitive.mm:180-250`): two dispatches per
linear at prefill, no host waits. Kernels compile at first use from source at
Metal language version 4.0 and are cached by name (`:42-135`). This is the
binding route [oMLX](omlx.md) uses at larger scale and the
[lazy evaluation](../mlx/lazy-evaluation.md) page describes from the outside.
Two cautions: the primitives define no `vjp`, `vmap` or output-shape methods,
and `is_equivalent` returns true unconditionally
(`csrc/include/w8a8_primitive.h:46-48`), which is risky under `mx.compile`
common-subexpression merging for primitives that carry parameters as members.

## The numbers, and the baseline

Operator level, M5 Pro, N=10240, K=2560: at M=4096 per-channel W8A8 takes
4.41 ms against 8.12 for MLX's W8A16, 1.84x; at M=8192, 8.71 against 16.23
(`README.md:107-114`; script `benchmarks/bench_full.py`). That works out to
about 48.7 TOPS against about 26.4 TFLOPS (inferred arithmetic). End to end
the gain depends on the baseline: Qwen3-VL-2B prefill at 1,334 tokens is 3,242
tok/s W8A8, 2,065 W8A16, 3,010 FP16 (`README.md:132`), so about 1.57x over
W8A16 but 1.01 to 1.16x over FP16 across the 2B and 4B, because MLX's W8A16
prefill is slower than FP16. Decode is not W8A8 at all: at M=1 the primitive
runs an fp16-activation INT8-weight dot product shaped after MLX's `qmv_fast`
(`w8a8_int8_mv.metal:2-50`), and decode speed is at or slightly below W8A16,
104 vs 107 tok/s on the 2B (`README.md:132`). Perplexity on Qwen3-8B wikitext2
moves from 9.726 to 9.756 (`README.md:160-176`). The end-to-end benchmark
needs a private checkpoint path, so it is not reproducible outside the
authors (`benchmarks/bench_e2e_wxa16.py:27-28`).

The tutorial's step-by-step record is useful and its explanation is wrong.
The INT8 GEMM at (128, 4096, 4096) goes from 1.002 ms with threadgroup
staging to 0.438 ms with device-to-register loads, 2.29x, then to 0.361 with
a wider K step and a swizzle, 2.78x total
(`tutorial/how_to_write_efficient_int_gemm_m5_en.md:374-380`). The tutorial
attributes the first step to threadgroup memory being "carved from the same
unified RAM"; it is on-chip per-core SRAM, as the
[threadgroup memory](../machine/threadgroup-memory.md) page says. Reading the
step-1 kernel, the likelier cause is scalar one-byte staging loads with
division and modulo index math into two 16 KB tiles that fill the 32 KB
budget, plus two barriers per K step
(`dev/step_kernels/step1/w8a8_matmul.metal:114-133, 172`). A measured win with
the wrong stated cause is the [failures](../war-stories/the-failures.md)
page's most common species.

## The decode attention kernel

The README describes a grouped-query scheduling advantage over MLX; MLX's
two-pass kernel already puts all query heads of a KV head in one threadgroup
(`code/mlx-steel-kernels/.../sdpa_vector.h:217-223` against
`cider/kernels/cider_sdpa_vector.h:168-176`). The real differences are that
each block takes a contiguous chunk of keys where MLX interleaves, the loop
computes four scores before four softmax updates, and the block count is a
template constant from {32, 64, 128} where MLX uses a function constant
(`cider_sdpa_vector.h:182-234`; `cider_sdpa_vector.metal:24-27`). `TILE=4`
unrolls KV positions, not query heads as the README says. The micro-benchmark
on an M5 Pro at D=128 runs 0.99x to 1.57x against MLX, best at GQA 8 and 32K
context (`README.md:62-72`), and only at N of 1,024 or more, so the one-pass
path is never measured. The autotune sweeps block counts and saves to
`~/.cider_sdpa_tune.json` keyed on exact N (`cider/attention/sdpa.py:200-224`);
since N grows by one each decode step, a tuned entry hits only when N lands on
a power of two, and a hardcoded table decides otherwise. One hazard from
reading, not running: the wrappers compute strides from the array shape, while
mlx-lm's cache returns a view into a buffer padded to 256 positions
(`csrc/src/sdpa_primitive.mm:302-307`); a test with a sliced key tensor would
settle whether heads after the first read the right rows.

Next: [Rapid-MLX](rapid-mlx.md)

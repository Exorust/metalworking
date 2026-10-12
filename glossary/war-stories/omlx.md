# oMLX: a production server's ledger

**oMLX is an open-source LLM inference server for Apple Silicon (Apache-2.0,
jundot/omlx, pinned here at commit 3f99250d) built on MLX 0.32.3 and mlx-lm.
It is the largest public body of Metal optimization work aimed at one platform:
about 30,000 lines of runtime patches, 35,000 lines of native kernel code, and a
scheduler that gets most of its speed without new kernels at all. Its headline
number is a 30x fused prefill for GLM-5.2 (845 vs about 29 tok/s on an M3
Ultra). Its more useful lesson is where the rest of the time came from.**

Fetched by this repo's `fetch.sh` as `code/omlx` (library, docs and build files
only; the macOS and web apps are left out). Line references below are into that
pinned copy.

## Where the speed comes from, in order

**1. Overlap and reuse in the scheduler, not kernels.** The server runs decode
in bursts of up to 64 steps per hand-off to the Python executor, because the
in-process loop measured about 80 tok/s against about 74 through a per-token
async hand-off (`omlx/engine_core.py:441-445`). Prefill and decode share the GPU
through a time-debt rule: each prefill chunk owes half its duration in decode
time before the next chunk may run, and chunks are sized so a contended decode
stalls about 500 ms at most. The comment records the measurement: on an M3
Ultra the 0.5 ratio keeps contended decode at about 3x the unthrottled rate
with batch completion time at parity, while 1.0 buys about 5x decode at a 16
percent completion cost (`omlx/scheduler.py:1553-1580`). The reason this is a
scheduler problem is a Metal fact from the
[command-buffers](../metal/command-buffers.md) page: Metal cannot preempt a
running kernel, so bounding chunk duration is the only interleave mechanism.
Prefill is also packed, up to four requests' chunks in one forward, with the
token-wise layers run once on the packed sequence and the attention or
recurrence mixers run per row against each row's own cache
(`omlx/prefill/packed.py:1-18`).

**2. A KV cache that survives across turns.** Finished requests store their
prompt and output tokens as 256-token blocks keyed by SHA-256 hashes chained
through the parent block (`omlx/cache/paged_cache.py:84-125`), so the next turn
of a conversation matches the whole history and only the new tail is prefilled.
Blocks live in a RAM hot tier and a safetensors SSD tier; an SSD read of a
10 MB block is noted at about 2 ms at 5 GB/s (`omlx/cache/paged_ssd_cache.py:4035`).
This is the [kv-cache](../techniques/kv-cache.md) page's arithmetic applied to a
whole server: bytes you do not recompute are bytes you do not stream.

**3. Fusion that moves the reduction into the epilogue.** The GLM-5.2 sparse
attention path is the one real kernel win, and its shape is familiar from the
[fusion](../techniques/fusion-and-epilogues.md) page. The stock indexer computes
`q @ k^T` per head, a [B, 32, L, K] tensor, then `relu`, a weight, a sum over
heads, and `mx.argpartition`. The native kernel is one
[steel](../mlx/steel.md) GEMM whose epilogue applies `relu` times the head
weight and sums the heads in fp32 registers, so the per-head tensor is never
written (`omlx/custom_kernels/glm_moe_dsa/csrc/kernels/steel_dsa_indexer_score.h:609-664`),
followed by a radix top-k over a 256-bin threadgroup histogram and a sparse
attention kernel that gathers only the selected tokens with an
[online softmax](../techniques/online-softmax.md). A from-scratch
`simdgroup_matrix` version of the indexer keeps the K tile transposed and
resident in 16 KB of threadgroup memory, exactly half the per-core scratch so
two threadgroups stay resident, uses one barrier where steel uses three per
K-slice, and measures about 1.37x over the steel version on an M2 Ultra
(`mma_dsa_indexer_score.h:1-41`). The fallback without these kernels expands
the latent cache to full multi-head K and V for every cached position and runs
dense attention, which is the 30x and the extra memory
(`omlx/patches/glm_moe_dsa/glm_moe_dsa_model.py:396-404`).

**4. Launch counts on the decode side.** Most decode-side patches attack
dispatches, not bytes. The Qwen3.5 router chain, five tiny ops per MoE layer
per token, measures about 51 us per layer at decode against a 5 us fused
launch; across 40 layers that is about 2 ms of a 9 ms decode step
(`omlx/patches/qwen35_moe_router.py:6-9`). Gate and up projections are
concatenated so one `gather_qmm` replaces two. The gated-delta verify prework,
about ten small dispatches per layer per verify cycle, becomes one
(`omlx/patches/qwen35_gdn_prework.py:7-12`). This is
[question 3](three-questions.md) answered at scale.

**5. M5 tensor units and INT8 on the prefill side.** On M5-class GPUs the oQ
weight format is multiplied in an INT8 by INT8 to INT32 path on the
[neural accelerators](../machine/neural-accelerators.md): packed 4- and 5-bit
weights are decoded straight into INT8 fragment registers, nothing is staged
through threadgroup memory, and the affine correction is applied at every
group-of-64 boundary in fp32 (`omlx/custom_kernels/qwen35_prefill/csrc/qwen35_oq_a8_nax.metal:1-22`).
The gated-delta recurrence prefill is a software-pipelined kernel: 3.0 ms
against 4.9 ms per layer call on an M5 Ultra at 8,191 tokens
(`omlx/patches/qwen35_gdn_chunked.py:4-9`).

**6. The third matrix engine, used.** An experimental path runs a slice of the
dense MLP gate and up projections on the Neural Engine through the private
`AppleNeuralEngine.framework`, with hand-written MIL, INT8 per-channel weights
and FP16 activations, two programs pinned to the two ANE instances of an M3
Ultra. Measured prompt processing on Qwen3.8-27B: +3.4 percent at 4K, +17.8 at
16K, +18.9 at 32K, for about 4 GB more memory and a load time that goes from
3.35 s to about 27 s; decode is unchanged, results are not bit-exact
(`docs/experimental/qwen35_ane_prefill.md:236-242`). It is the
[AMX](../machine/amx.md) page's warning in practice: it works, and it can stop
working with any macOS update.

## How the kernels ship

The native kernels are not `mx.fast.metal_kernel` strings. They are five
nanobind extensions, each a set of MLX C++ `Primitive` subclasses with an
`eval_gpu` that loads a prebuilt `.metallib` through MLX's own device
(`setup.py:51-74`). That buys steel's templates and the full Metal toolchain,
and it costs an ABI pin: `mlx==0.32.3` and `nanobind==3.0.1` exactly, with a
probe at import that disables the extension on any mismatch
(`omlx/custom_kernels/glm_moe_dsa/fast.py:41-69`). The roughly 60 inline
`mx.fast.metal_kernel` sites in the patches cover the smaller fusions. Patches
install after the model loads, each behind a settings flag, each wrapped so a
failure falls back to the stock path, and several run a canary call first and
disable themselves if it fails (`omlx/patches/m5_gather_qmm.py:21-25`). Most
claim bit-exactness by transcribing MLX's own `qmv` arithmetic; the ones that
change numerics say so.

## What the ledger says

Read against the [three questions](three-questions.md), oMLX is mostly
questions 1 and 3: delete recomputation through the cache, delete dispatches
through fusion, and keep the GPU busy through scheduling. The new-kernel wins
are concentrated where the stock path materializes a quadratic intermediate
(the DSA indexer, the MiniMax block-sparse scores) or expands a compressed
cache. Two caveats belong next to the numbers. The 845 vs 29 figure appears
only in the README; no benchmark script in the repository reproduces it. And
the expert offload doc is honest about its own limit: prefetching the next
layer's experts covered 59 percent of misses and still made decode 2 to 15
percent slower, so it was removed (`docs/experimental/MoE_Expert_Offload.md:73`).

Next: [The failures](the-failures.md)

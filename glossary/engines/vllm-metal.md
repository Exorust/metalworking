# vllm-metal: vLLM's scheduler on MLX, with its own paged kernels

**vllm-metal is the community hardware plugin that runs vLLM, the standard
datacenter inference server, on Apple Silicon (Apache-2.0,
vllm-project/vllm-metal, pinned here at commit 4499a43e, version 0.31.0).
vLLM supplies the API server, scheduler and paged block manager; mlx-lm
supplies the token-wise model layers; vllm-metal owns "the request-aware
attention path" (`README.md:24`): a 2,820-line nanobind extension and 14 Metal
files, 5,715 lines, that read vLLM's paged KV pool directly. It is the project
on this platform that documents its losses as carefully as its wins.**

Fetched by `fetch.sh` as `code/vllm-metal`. Line references are into that
pinned copy. MLX is pinned to exactly 0.32.1 because the extension subclasses
MLX's private `Primitive` headers (`pyproject.toml:28-33`).

## The memory arrangement

vLLM allocates its KV pool as one CPU `torch` tensor. vllm-metal shares each
layer's region into MLX zero-copy through DLPack
(`vllm_metal/pytorch_backend/tensor_bridge.py:31-43`) and replaces vLLM's
zeroing of the pool with `torch.empty`, because the memset made the whole pool
resident: "a 16 GB Mac serving a 0.6B model pays ~5 GB of RSS for a pool that
is ~90% unused" (`vllm_metal/attention/caches/storage.py:28-60`). Per-layer
views need a custom primitive because MLX's `as_strided` flattens first and
fails above INT32_MAX elements; the recorded case is a 5.24 GiB cache with
2,810,904,576 elements (`vllm_metal/metal/mlx_patch.cpp:18-37`). Cache writes
are primitives that alias their input buffer so the writes carry graph
provenance (`paged_ops.cpp:1486-1502`). After setup, the only per-step host
traffic visible is metadata: block tables, sequence lengths and query offsets.
This is the [unified memory](../machine/unified-memory.md) page's promise
used in earnest, and its three costs written down.

## One paged kernel family, several routes

One `PagedAttentionPrimitive` (`paged_ops.cpp:1198`) reads the pool through
block tables and picks a route inside `eval_gpu`: a tensor-unit prefill on M5,
a tiled [simdgroup_matrix](../metal/simdgroup-matrix.md) prefill, a mixed batch
with decode rows and prefill rows in the same command encoder, per-token
decode, split-KV decode, a verify window, or a grouped-query decode
(`paged_ops.cpp:839-890, 945-1047`). Three precompiled `.metallib` files plus
an optional NAX one are dispatched on MLX's own command encoder; the gated
delta-net kernels and the TurboQuant materializer are JIT
`mx.fast.metal_kernel` strings.

The same prefill exists twice, which makes the pair a good lesson. The tiled
kernel (`pagedattention_tiled.metal`, 638 lines) keeps Q and the output
accumulator in registers for the whole KV loop, runs the online softmax through
`simd_shuffle_xor` with masks 1 and 8 because the 8 by 8 fragment layout forces
it, and takes two barriers per KV tile instead of four. It cannot use
`simdgroup_async_copy` because each token may live in a different physical
block, so K and V arrive by cooperative loads with a block-table lookup per
token, budgeted against 32 KB of threadgroup memory: head 128 runs a 32 by 32
tile at 24 to 26 KB (`paged_ops.cpp:560-590`). The [NAX](../kernels/nax-gemm.md)
kernel (`pagedattention_nax.metal`, 450 lines) computes QK and PV with
`mpp::tensor_ops::matmul2d` 16 by 32 by 16 descriptors, uses no threadgroup
memory at all, and pays four block-table lookups per 32-token tile because each
16-row fragment is two 8-token page-local loads. It sets `relaxed_precision`,
which "also truncates the fp32 P operand of PV" (`pagedattention_nax.metal:5-8,
46-51`).

## Split-KV, gated on occupancy

The base decode grid is one threadgroup per query head per token, which leaves
cores idle at low concurrency with long context. A 512-token partition plus a
reduce pass turns on while `num_heads * total_q_tokens < 8 * gpu_cores`, with
the core count read from the IORegistry key `gpu-core-count` because "Metal/MLX
expose no core-count API" (`paged_ops.cpp:104-140, 944-946`). The comment
records the measurement on a 14-core M1 Pro at 8K context: 33 percent less time
per token at concurrency 1, 6.7 at 2, 5.3 at 4, "fading to ~-1.6% (noise) by
conc=8". An adaptive split count was tried and reverted: "this kernel is
memory-bound, so it likes oversubscription" (`paged_ops.cpp:362-376`). That is
the [occupancy](../machine/occupancy.md) page's argument with numbers attached.

## Where dequantizing whole pages beats fused dequant

The best-documented numbers in the repository are for TurboQuant prefill
(`docs/turboquant.md`, script `tools/benchmark/tq_lane_verify.py`). A
compressed-KV prefill normally unpacks pages inside the attention kernel. The
alternative dequantizes the referenced pages once per layer per step to 16-bit
in registers, with the scale math and inverse Walsh-Hadamard transform in fp32,
then feeds the ordinary attention kernel; each accelerated layer then costs an
`mx.eval` and `mx.synchronize`, "a host synchronization boundary in each
scheduler step" (`docs/turboquant.md:141-146, 195-198`). On an M3 with
Qwen3-4B, an 8,192-token cached prefix and 64 new tokens, time to first token
went from 2151.62 ms to 539.12 ms, 3.99x (`docs/turboquant.md:309-313`); on an
M5 Pro, 424.68 to 105.58 ms. The doc records the losses next to the wins: at 16
new tokens the single-layer ratio is 0.65 to 0.69x, fresh 64-token prompts ran
0.81 to 0.93x on the M3, and two unchanged controls ran 11.0 and 7.05 percent
slower with "the cause ... unresolved" (`:217-222, 245-247, 327-328`). The
crossover between 16 and 64 query tokens is the
[quantization](../mlx/quantization.md) page's pay-per-use dequant cost, measured
from the other side.

## The limit the forum post was about

A widely read r/LocalLLaMA post of August 2026 said vllm-metal could not yet
combine prefix caching with speculative decoding on the hybrid Qwen3.5 and 3.8
models. The code is stricter than that. `platform.py:691-700` turns prefix
caching off for hybrid models whenever speculation is configured, citing
"draft-state rollback across mamba state blocks ... is not implemented", and
speculation is not available on hybrid targets at all: `v1/spec_decode.py:145-149`
raises "not supported for hybrid models on Metal yet", and multi-token
prediction exists only for Gemma 4 (`docs/speculative_decoding.md:10, 32`).
Prefix caching for those models uses vLLM's `align` mode, copy-forward of the
recurrent state into block-indexed slabs with restore on a hit
(`vllm_metal/attention/state/align.py:1-27`). So on a hybrid model, two pieces
are missing, not one.

## Numbers to trust and numbers to skip

The README's "83x TTFT, 3.6x throughput compared to v0.1.0" (`README.md:19`)
has no artifact in the repository. The project's own blog post for v0.28.0
publishes a padding comparison instead: a batch of one 3,000-token prompt plus
seven 430-token prompts took 3.64 s on vllm-metal against 10.99 s for mlx_lm,
7.22 s for oMLX and 5.33 s for llama.cpp, because varlen packing does not pad
to the longest prompt (external). Numbers that live only in code comments: the
verify-window mode "up to +40% e2e at conc 16-32" and a loss single-stream
(`envs.py:200-208`); two window rows as "the measured register/occupancy sweet
spot", with the two copies of that comment disagreeing on the per-row cost
(`pagedattention.metal:660-663` says 0.75x, `metal/constants.py:18-19` says
0.5 to 0.75x); `MLX_MAX_OPS_PER_BUFFER=2000` "on the measured plateau" for a
scheduler that builds thousands of lazy ops per step (`__init__.py:73-81`).
Decode is pipelined one step ahead, with step k+1 built on step k's lazy
sampled tokens while the GPU still runs step k (`v1/decode_pipeline.py:3-13`),
the [lazy evaluation](../mlx/lazy-evaluation.md) page applied to a server.

Read next to [oMLX](omlx.md) and [vllm-mlx](vllm-mlx.md): the same scheduler
family, but here the scheduler is vLLM's unchanged and all the Metal-specific
work sits in one paged attention primitive and its measured policies.

Next: [Magnitude](magnitude.md)

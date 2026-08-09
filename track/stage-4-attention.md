# Stage 4: Flash Attention, three ways

Needs `code/metal-flash-attention`, `code/mlx-steel-kernels`, and `code/llama-cpp-metal`.
Time: a week; this is the summit.

Flash attention is where everything so far compounds: tiled GEMM, online softmax,
register pressure, specialization. This stage reads three independent implementations
for the same hardware: a research kernel chasing maximum ALU utilization, a framework
library, and a production engine that serves every quantization format. They disagree,
and the disagreements are most of what this stage teaches.

## A. metal-flash-attention: the max-utilization school

[philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
hits 83% ALU utilization on M1 Max. Signature tricks: a third block dimension along
head-dim D, deliberately controlled register spilling, and a backward pass split into dQ
and dK/dV kernels because Apple emulates FP32 atomics badly (lesson 3).

**There are no `.metal` files in this repo.** The MSL is generated as Swift string
literals at runtime, compiled per problem shape:

- `Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Source.swift` is the
  kernel body. `createSource()` at line 11 assembles the shader, and the literal
  `kernel void attention(` is at line 32.
- `.../AttentionKernel+Softmax.swift` holds the online (streaming) softmax.
  `onlineReduceMaximum()` at line 267 emits the running-max reduction; the
  rescale-and-accumulate half is in `AttentionKernel+Accumulate.swift`.
- `Sources/FlashAttention/GEMM/GEMMHeaders.swift` has the two hand-written MSL headers
  everything depends on: `createMetalSimdgroupEvent()` at line 24 (async-copy
  intrinsics, plus a long comment documenting an M1 hardware bug where an unread async
  copy hangs the GPU until reboot) and `createMetalSimdgroupMatrixStorage()` at line 257
  (the register-tile abstraction).

Companion reading: llama.cpp [PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021),
where Gerganov builds a Metal FA kernel across 154 comments: threadgroup layout,
simdgroup_load strides, and online softmax, reasoned out in public. The best
line-by-line narration of an FA kernel that exists.

## B. MLX steel: the framework school

- `steel/attn/kernels/steel_attention.h` is the kernel body. Specialization is
  branch-free, via function constants `align_Q`/`align_K` (200/201) and
  `has_mask`/`do_causal`/`has_sinks` (300/301/302); `MaxOp`/`SumOp` reduction functors
  drive the online softmax.
- `steel/attn/mma.h` and `steel/attn/loader.h` are the attention-specialized forks of
  the gemm pair you read in stage 3; the fragment layout differs for the QK^T, softmax,
  PV chain.
- `sdpa_vector.h` is the decode path: attention for a single query, where the problem is
  bandwidth and sync, not FLOPs.

## C. llama.cpp: the production/quantized school

One 11,603-line shader: `code/llama-cpp-metal/ggml/src/ggml-metal/ggml-metal.metal`.
Approximate line numbers at the pinned commit:

| ~line | kernel |
|---|---|
| 6258 | `kernel_flash_attn_ext_pad`, pads K/V to the block quantum |
| 6330 | `kernel_flash_attn_ext_blk`, precomputes per-block mask/skip metadata |
| 6431 | `kernel_flash_attn_ext_impl`, the shared templated implementation (the real algorithm) |
| 7069 | `kernel_flash_attn_ext`, simdgroup-matrix path, main entry |
| 7128+ | the head-dim x dtype specialization table (dk/dv 32..256) |
| 7296 | `kernel_flash_attn_ext_vec`, vector/decode path (small query count) |
| 7865 | `kernel_flash_attn_ext_vec_reduce`, cross-partition softmax combine |

Behaviour is specialized with `[[function_constant]]` flags rather than more template
parameters; see `FC_flash_attn_ext_has_mask / has_sinks / has_bias / has_scap /
has_kvpad` around lines 6385-6399. The quantized GEMM family lives in the same file
(`kernel_mul_mm` ~10048; the canonical `kernel_mul_mm_q4_K_f32` instantiation at ~10777;
decode-side `kernel_mul_mv_q4_K_f32` ~8486). Host-side dispatch, where the engine picks
between `mul_mm`, `mul_mv`, and the FA vec path, is in `ggml-metal-ops.cpp`.

History: [PR #2615](https://github.com/ggml-org/llama.cpp/pull/2615) is where llama.cpp
deleted MPS in favor of these hand-written simdgroup kernels. ~88% ALU on a 4096^2
matmul, with the honest caveat that end-to-end gains were ~40% because non-matmul ops
dominate.

A bonus fourth implementation, once stage 6's fetch is done:
`code/drawthings-mfa/lib/nnc/mfa/kernels/AttentionKernel.cpp` is MFA 2.0, the C++
descendant of implementation A, adding a backward pass, int8 attention, and
sparse-indexed attention.

## Done when

You can fill in this table from memory, one line of justification each:

| | codegen strategy | specialization mechanism | decode path |
|---|---|---|---|
| metal-flash-attention | ? | ? | ? |
| MLX steel | ? | ? | ? |
| llama.cpp | ? | ? | ? |

- You can explain online softmax well enough to derive the rescale step.
- You can say why all three ship a separate single-query/vec kernel instead of reusing
  the matrix path.

Next: [Stage 5: Down to the silicon](stage-5-silicon.md)

# Stage 4: Flash Attention, three ways

Time: two or three sittings; this is the summit. Self-contained — every excerpt links
to the pinned original. Sources: [ml-explore/mlx](https://github.com/ml-explore/mlx)
@ `47bbfe8f` (MIT), [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
@ `8671cddc` (MIT), [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
@ `3653e6d6` (MIT).

Flash attention is where everything so far compounds: tiled GEMM, register pressure,
epilogues, specialization. This stage reads three independent implementations for the
same hardware — a framework library, a research kernel, and a production engine — and
the ways they *differ* are most of the lesson.

## 1. The problem: a softmax trapped between two matmuls

Attention is `O = softmax(Q·Kᵀ / √d) · V`. Computed naively, the middle term is an
L×L matrix of attention scores — for a 32K-token context, a billion floats that get
written to DRAM by the first matmul and read back by the second. On a machine whose
one commandment is "touch less memory" (stage 0), materializing that matrix is the
worst possible move. The whole point of flash attention is to **never write S to
memory**: process K and V in tiles, keep the scores of the current tile in registers,
and fold each tile's contribution into a running output.

One thing blocks that plan: softmax normalizes each row by the sum of exponentials of
*the whole row* — which you don't have until you've seen every tile. Worse, the
numerically safe form subtracts the row max before exponentiating:

```
softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m),   m = max_j x_j
```

and the max is also a whole-row fact. Streaming seems impossible. It isn't.

## 2. Online softmax, derived

Keep three running quantities per output row, updated after each tile of scores:

- `m` — the max seen so far
- `l` — the sum of exponentials seen so far, *relative to* `m`
- `O` — the output accumulated so far, weighted by those same exponentials

Process a new tile with scores `S_new`. Its local max may exceed `m`, making every
exponential you've already accumulated too large by a constant factor. The fix is one
identity: `exp(x - m_old) = exp(x - m_new) · exp(m_new - m_old)`, i.e. every prior
term is off by exactly `exp(m_old - m_new)`. So per tile:

```
m_new = max(m, rowmax(S_new))
c     = exp(m - m_new)          # the correction factor, ≤ 1
P     = exp(S_new - m_new)      # this tile's weights
l     = l·c + rowsum(P)         # rescale history, add the new tile
O     = O·c + P·V_tile          # same correction applied to the output
m     = m_new
```

After the last tile, divide: `O /= l`. That's the entire algorithm. Every
implementation below is this loop plus engineering. Watch for the correction factor
`c` — it appears verbatim in all three codebases, and being able to re-derive it is
this stage's "done when."

One Apple-specific twist you'll see everywhere: the kernels compute in **base-2**
(`exp2`) rather than base-e, folding `log₂e` into the scale — stage 0's fast-exp2
hardware path. Same math, one fewer multiply in the hot loop.

## 3. School one — MLX steel: the whole algorithm in one readable file

`steel/attn/kernels/steel_attention.h` is the payoff for stage 3: the complete flash
attention forward in ~400 lines built from the vocabulary you already have —
`BlockLoaderT`, `MMATile`, function constants. Specialization first:

```cpp
constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
```
— [`steel_attention.h:11-16`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L11-L16)

Same mechanism as stage 3's `align_M/align_N`: masking, causality, and ragged edges
are compiled *out* of the pipeline when the host knows they don't apply — branch-free
specialization. And the base-2 trick from section 2, right where the derivation put it:

```cpp
  const AccumType scale = params->scale * M_LOG2E_F;
```
— [`steel_attention.h:166`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L166)

The register layout tells you the shape of the loop — Q stays put, K/V stream past:

```cpp
  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, TQ, TD, MMAFrag_acc_t> Otile;
```
— [`steel_attention.h:186-190`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L186-L190)

`Stile` — the score tile, the thing naive attention writes to DRAM — is a register
tile. It is born, masked, softmaxed, multiplied against V, and dies without ever
leaving the register file. This is also why `attn/` forked its own `mma.h` from
`gemm/`: a GEMM's accumulator only ever accumulates, but `Stile` must be *read and
transformed in place* between two matmuls, which needs a different fragment layout.

And here is section 2's derivation, line for line — inside the loop over KV tiles
(`ExpSubOp::apply(x,y) = fast::exp2(x - y)`, defined at the top of the file):

```cpp
    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }
    ...
    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    // Update O
    Otile.template row_bin_op<MulOp>(factor);
```
— [`steel_attention.h:391-420`](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L391-L420), abridged

`factor` is the correction `c`. `sum_score = sum_score·c + new` is the `l` update.
`Otile *= c` then accumulates `P·V`. After the loop:
`Otile.row_bin_op<DivOp>(sum_score)` — the final divide
([line 460](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L460)).
If you followed section 2, this file has no surprises left — read the causal-skip
logic at [lines 239-247](https://github.com/ml-explore/mlx/blob/47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h#L239-L247)
(causal masking isn't just masking; whole tiles above the diagonal are never visited)
and you've read a production flash attention.

## 4. School two — metal-flash-attention: the codegen school

[philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
hits 83% ALU utilization on M1 Max, and its structural bet is visible in one fact:
**there are no `.metal` files in the repo.** The MSL is assembled as Swift strings at
runtime, per problem shape:

```swift
    return """

    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;

    \(createConstants())

    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],

      uint gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ...
      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }
    """
```
— [`AttentionKernel+Source.swift:23-54`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Source.swift#L23-L54), abridged

Where MLX picks from pre-built template instantiations and llama.cpp from a fixed
table, MFA *writes a bespoke kernel* for your exact head dimension, precision, and
transpose state, then compiles it on the spot. Maximum specialization, zero shipped
binary. The same online softmax lives in the generated code — compare with section 2:

```swift
  // Rescale 'O' to reflect the new maximum.
  func onlineCorrectO() -> String {
    """

    // update 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }

    """
  }
```
— [`AttentionKernel+Softmax.swift:289-301`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Softmax.swift#L289-L301)

with the running-sum update `l = l * correction + l_new` a few lines below
([line 321](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel+Softmax.swift#L321)).
Note the `if (m_new > m)` guard — a branch MLX doesn't take (it always multiplies by
`factor`, which is usually 1.0). Different bets on branch cost vs. multiply cost.

Three signature moves distinguish MFA:

- **A third block dimension along head-dim D**, on top of the usual two — plus
  *deliberately controlled register spilling*: it sometimes chooses tile shapes that
  spill, because on this hardware a predictable spill can beat a smaller tile. The
  boldest register-pressure position of the three schools.
- **The backward pass is split into two kernels** — dQ in one, dK/dV in another —
  specifically because Apple emulates FP32 atomics badly (stage 0, lesson 3): one
  fused backward would need atomic accumulation across threadgroups; two passes each
  own their outputs outright.
- **`createMetalSimdgroupEvent()`** re-declares stage 2's ghost, the
  `air.simdgroup_async_copy` intrinsics, via inline `__asm`
  ([`GEMMHeaders.swift:24+`](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/GEMM/GEMMHeaders.swift#L24)) —
  including a long comment documenting an M1-era hardware bug where an async copy
  whose result is never read **hangs the GPU until reboot**. This is what pre-Metal-4
  tile staging looked like; on current toolchains this path no longer compiles
  (stage 2, section 5).

Companion reading, strongly recommended: llama.cpp
[PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021), where Gerganov builds
his Metal FA kernel across 154 comments — the best public line-by-line narration of
these exact decisions being made.

## 5. School three — llama.cpp: the production/quantized school

One 11,603-line shader, `ggml/src/ggml-metal/ggml-metal.metal`, serving every model
architecture and every quantization format llama.cpp supports. Its flash attention is
a kernel *family* around one shared implementation
(`kernel_flash_attn_ext_impl`, [~line 6431](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L6431)):
a pad kernel, a mask-metadata precompute kernel, the main simdgroup-matrix path, and
a vector path — with behavior toggled by the now-familiar mechanism:

```metal
constant bool FC_flash_attn_ext_has_mask  [[function_constant(FC_FLASH_ATTN_EXT + 0)]];
constant bool FC_flash_attn_ext_has_sinks [[function_constant(FC_FLASH_ATTN_EXT + 1)]];
constant bool FC_flash_attn_ext_has_bias  [[function_constant(FC_FLASH_ATTN_EXT + 2)]];
constant bool FC_flash_attn_ext_has_scap  [[function_constant(FC_FLASH_ATTN_EXT + 3)]];
constant bool FC_flash_attn_ext_has_kvpad [[function_constant(FC_FLASH_ATTN_EXT + 4)]];
```
— [`ggml-metal.metal:6385-6389`](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L6385-L6389)

Where MFA generates and MLX templates, llama.cpp **enumerates**. The head-dim ×
dtype specialization table is a wall of explicit instantiations, one named
pipeline per shape the ecosystem's models actually use:

```metal
template [[host_name("kernel_flash_attn_ext_f32_dk64_dv64"  )]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  64,  64>;
...
template [[host_name("kernel_flash_attn_ext_f32_dk192_dv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f32_dk576_dv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES_F32, float4x4,   1, dequantize_f32,  float4x4,   1, dequantize_f32,  576, 512>;
```
— [`ggml-metal.metal:7128-7143`](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L7128-L7143), abridged

Read the parameter list: `dequantize_f16`, `dequantize_f32` — and in the fp16/quant
variants, dequantizers for Q4, Q8, and friends. That's the school's defining
constraint: **K and V arrive quantized**, so dequantization is fused into the
attention loop itself (this table's `dk576_dv512` entry exists for exactly one model
family — that's what production looks like). The quantized GEMM family
(`kernel_mul_mm` [~line 10048](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L10048))
lives in the same file; stage 6's war stories are mostly fought inside it.

History worth knowing: [PR #2615](https://github.com/ggml-org/llama.cpp/pull/2615)
is where llama.cpp *deleted MPS* in favor of these hand-written simdgroup kernels —
~88% ALU on a 4096² matmul, with the honest caveat that end-to-end gains were ~40%
because non-matmul ops dominate. A preview of stage 6's methodology lesson.

## 6. Why all three ship a second, separate kernel for decode

Every school has a "vec" variant next to its matrix kernel: MLX's `sdpa_vector.h`,
llama.cpp's `kernel_flash_attn_ext_vec`
([~line 7296](https://github.com/ggml-org/llama.cpp/blob/3653e6d6d547ec763317d9ecd0ace334a7e21359/ggml/src/ggml-metal/ggml-metal.metal#L7296)).
The reason: LLM decode processes **one query token** against the whole KV cache.
With a single row of Q, there's no 8×8 tile to feed `simdgroup_matrix` — the matrix
path would compute mostly on padding. A one-query attention is a *reduction*, not a
matmul: the problem is bandwidth (stream the entire KV cache) and sync (combine
partial results across simdgroups), not FLOPs. So the vec kernels drop
`simdgroup_matrix` entirely for wide vector loads and simd reductions —
llama.cpp even splits the KV cache across threadgroups and adds a combine kernel
(`_vec_reduce`) to merge the partial softmaxes, using the same rescaling identity
from section 2. One algorithm, two machines: prefill is compute-bound GEMM territory,
decode is bandwidth-bound reduction territory. This split is the single most
load-bearing fact of LLM inference on this hardware.

## 7. The scorecard

The table the old version of this page asked you to fill in, filled in:

| | codegen strategy | specialization mechanism | decode path |
|---|---|---|---|
| **MLX steel** | C++ templates over steel components, pre-instantiated by build scripts | function constants finalize each pipeline (`align_Q/K`, `do_causal`, `has_mask`) | `sdpa_vector.h`, chosen by the C++ host code by shape |
| **metal-flash-attention** | Swift assembles MSL source per problem; compiled at runtime | total — every kernel is bespoke to its shape; even loop bounds are literals | out of scope (targets training/prefill; served by shape-specialized codegen) |
| **llama.cpp** | one shared templated impl, explicitly enumerated per head-dim × dtype (`[[host_name]]` table) | function constants for behavior + the enumeration itself for shape | `_ext_vec` + `_ext_vec_reduce` split-KV kernels |

Bonus, once stage 6's fetch is done: `code/drawthings-mfa/lib/nnc/mfa/kernels/` is
MFA 2.0 — the C++ production descendant of school two, adding the split backward
pass, int8 attention, and sparse-indexed attention. Proof the codegen school ships.

## Done when

- You can write the five-line online-softmax update (m, c, P, l, O) from memory and
  say why the correction factor exists.
- You can point to that update in two of the three codebases.
- You can explain why `Stile` living in registers is the entire point, and why
  `steel/attn/` forked its own `mma.h` to make it possible.
- You can say why decode gets its own kernel in every implementation, and what's
  different about the resource it's bound by.

Next: [Stage 5: Down to the silicon](stage-5-silicon.md)
